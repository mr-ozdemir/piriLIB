"""Direct shipping: the simplest plan, as a reference for every other solver.

Every (day, origin, destination) line is served on its own, by vehicles that go
straight to the destination; nothing is combined on multi-stop routes. A
vehicle may serve a line if its type is not banned at the origin and the
destination, it has vehicles left that day, and, for an owned group, the line
is the group's own line. Vehicles are compared by their dispatch cost
(fixed plus per km), whatever the objective; the objective only scores the
result.

- With split demand, a line is carried by full vehicles of the type with the
  lowest cost per carried share of the line, until the rest fits into one
  vehicle; the rest goes into the cheapest vehicle that fits it, preferring
  one that reaches its minimum fill.
- Without split demand, the demands of a line are packed first-fit
  decreasing; a new vehicle is the type with the lowest cost per capacity that
  fits the demand, and each packed vehicle is finally replaced by the cheapest
  type that fits its load, again preferring one that reaches its minimum fill.

Demand that no vehicle left can carry stays uncarried, so minimum fill and
coverage are the only rules a direct plan can break.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.plan import SPOT
from pirilib.solvers._colony import Shared
from pirilib.solvers._sequential import RouteList, SequentialSolver
from pirilib.solvers.decompose import Routes, Subproblem

_EPSILON = 1e-12


class DirectPlanner:
    """Direct trips from the origin of one subproblem to single destinations.

    Attributes:
        available: Vehicles left per option; trips planned here take from it
    """

    def __init__(self, problem: Subproblem, shared: Shared, available: NDArray[np.int64]) -> None:
        self.problem = problem
        self.shared = shared
        self.available = available

    def cost(self, option: int, node: int) -> float:
        """Dispatch cost of a direct trip of ``option`` to ``node``."""
        problem = self.problem
        return float(problem.fixed[option] + problem.per_km[option] * problem.origin_km[node])

    def options(self, node: int) -> list[int]:
        """Options with vehicles left that may go straight to ``node``."""
        problem = self.problem
        return [
            option
            for option in range(problem.options)
            if self.available[option] > 0
            and problem.allowed[option, node]
            and (problem.kind[option] == SPOT or problem.line_end[option] == node)
        ]

    def share(self, option: int, total: NDArray[np.float64]) -> float:
        """The largest share of ``total`` that one vehicle of ``option`` can carry."""
        capacity = self.problem.capacity[option]
        positive = total > 0
        if not positive.any():
            return 1.0
        return float(min(1.0, float(np.min(capacity[positive] / total[positive]))))

    def reaches_fill(self, option: int, load: NDArray[np.float64]) -> bool:
        problem, shared = self.problem, self.shared
        capacity = float(problem.capacity[option, shared.primary])
        needed = float(shared.min_fill[problem.kind[option]]) * capacity
        return float(load[shared.primary]) >= needed - TOLERANCE * capacity

    def cheapest_fitting(self, options: Sequence[int], node: int, load: NDArray[np.float64]) -> int:
        """The cheapest option that fits ``load``, preferring one that reaches its minimum fill.

        Returns -1 when none fits.
        """
        fitting = [option for option in options if self.fits(option, load)]
        if not fitting:
            return -1
        filled = [option for option in fitting if self.reaches_fill(option, load)]
        pool = filled or fitting
        return min(pool, key=lambda option: (self.cost(option, node), option))

    def fits(self, option: int, load: NDArray[np.float64]) -> bool:
        capacity = self.problem.capacity[option]
        return bool(np.all(load <= capacity + TOLERANCE * capacity))

    def take(self, option: int) -> None:
        self.available[option] -= 1

    def full_truck(self, node: int, remaining: float, routes: RouteList) -> float:
        """Send one full vehicle to ``node``, of the type with the lowest cost per carried share.

        ``remaining`` is the share of the node's demand still to carry. Returns
        the share left afterwards, unchanged when no vehicle is left.
        """
        total = self.problem.demand[node]
        options = self.options(node)
        if not options or remaining <= _EPSILON:
            return remaining
        shares = {option: self.share(option, total) for option in options}
        best = min(options, key=lambda option: (self.cost(option, node) / shares[option], option))
        carried = min(shares[best], remaining)
        self.take(best)
        routes.add(best, (node,), (total * carried,))
        return remaining - carried

    def ship_split(self, node: int, routes: RouteList) -> None:
        """Ship the demand of ``node``, splitting it over vehicles."""
        total = self.problem.demand[node]
        remaining = 1.0
        while remaining > _EPSILON:
            options = self.options(node)
            if not options:
                return
            last = self.cheapest_fitting(options, node, total * remaining)
            if last >= 0:
                self.take(last)
                routes.add(last, (node,), (total * remaining,))
                return
            before = remaining
            remaining = self.full_truck(node, remaining, routes)
            if remaining >= before:
                return

    def ship_items(self, node: int, routes: RouteList) -> None:
        """Ship the demands of ``node`` whole, packed first-fit decreasing."""
        problem = self.problem
        slots = [int(slot) for slot in np.flatnonzero(problem.item_row[node] >= 0)]
        primary = self.shared.primary
        slots.sort(key=lambda slot: -float(problem.item_quantity[node, slot, primary]))
        bins: list[tuple[int, NDArray[np.float64], list[int]]] = []
        for slot in slots:
            quantity = problem.item_quantity[node, slot]
            placed = False
            for position, (option, load, items) in enumerate(bins):
                if self.fits(option, load + quantity):
                    bins[position] = (option, load + quantity, [*items, slot])
                    placed = True
                    break
            if placed:
                continue
            options = [option for option in self.options(node) if self.fits(option, quantity)]
            if not options:
                continue
            opened = min(
                options,
                key=lambda option: (
                    self.cost(option, node) / float(problem.capacity[option, primary]),
                    option,
                ),
            )
            self.take(opened)
            bins.append((opened, quantity.copy(), [slot]))
        for option, load, items in bins:
            self.available[option] += 1
            chosen = self.cheapest_fitting(self.options(node), node, load)
            chosen = option if chosen < 0 else chosen
            self.take(chosen)
            routes.add(chosen, (node,), (load,), [(0, slot) for slot in items])

    def nodes(self) -> list[int]:
        """Nodes with demand, largest first."""
        problem = self.problem
        primary = problem.demand[:, self.shared.primary]
        return [int(node) for node in np.argsort(-primary, kind="stable") if primary[node] > 0]


class DirectShipping(SequentialSolver):
    """Direct trips only: full vehicles per line, then the cheapest vehicle that fits.

    A baseline for benchmarks; see the module documentation. It accepts any
    objective and constraints, which only score its plan.
    """

    mode = "direct"

    def plan_subproblem(
        self,
        problem: Subproblem,
        shared: Shared,
        coefficients: Mapping[str, float],
        available: NDArray[np.int64],
        seed: int,
        stats: dict[str, float | int | str],
    ) -> Routes:
        planner = DirectPlanner(problem, shared, available)
        routes = RouteList(problem)
        for node in planner.nodes():
            if shared.rules.split_demand:
                planner.ship_split(node, routes)
            else:
                planner.ship_items(node, routes)
        return routes.build()
