"""Ant colony optimization for line-haul plans, vectorized with numpy.

The colony works on (day, origin) subproblems (see
:mod:`pirilib.solvers.decompose`). Subproblems of similar size run side by
side, and all their ants build solutions together: each step is a few array
operations over every (subproblem, ant) pair, never a Python loop over ants.

An ant repeatedly picks a vehicle option and a first stop, loads what fits, and
then either extends the route to a further stop or ends it, until the demand is
carried or no vehicle can carry more. Choices are random, weighted by
``pheromone ** alpha * heuristic ** beta``, where the heuristic is

- for a vehicle and first stop: the quantity carried per cost of the trip;
- for a next stop: the cost of serving it with its own vehicle, divided by the
  extra cost of the detour, times the share of capacity it fills;
- for ending the route: 1, or 0.1 while the minimum fill is not reached.

Every built-in rule except minimum fill and coverage holds by construction.

Pheromone follows the MAX-MIN hypercube framework: values stay in
``[tau_min, 1]`` and ``tau = (1 - rho) * tau + rho * used`` does not depend on
the scale of the objective.

With an objective that has a linear form (see :mod:`pirilib.objective`) and no
constraints of your own, every subproblem is optimized on its own, which is
fast. Otherwise each ant builds a whole plan, and plans are compared with
:func:`~pirilib.evaluate` and your objective.

On scheduled instances (times, deferral, handling capacity or vehicle limits)
subproblems are planned one after another, in the order of
:class:`~pirilib.solvers.decompose.Coordinator`, against a
:class:`~pirilib.solvers.ledger.Ledger` of what the centers can still handle:
ants keep within it, routes are timed (hourly costs, lateness), and demand left
uncarried that may wait joins the subproblem of the next day. Every route starts
as soon as its loads are ready.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import Constraint
from pirilib.evaluation import evaluate
from pirilib.objective import Objective, linear_form
from pirilib.plan import SPOT, Plan
from pirilib.solvers._colony import Batch, Build, Colony, Pheromone, Shared, scores
from pirilib.solvers._timing import timeline
from pirilib.solvers.base import SolveResult, default_objective, linear_coefficients
from pirilib.solvers.decompose import (
    UNBOUNDED,
    Coordinator,
    PlanBuilder,
    Routes,
    Subproblem,
    subproblems,
)
from pirilib.solvers.ledger import Ledger, Room
from pirilib.solvers.local_search import improve
from pirilib.spec.instance import UNLIMITED, LineHaulInstance

MAX_ROWS: Final = 40_000
"""Upper bound on (subproblem, ant) rows built side by side, to bound memory."""
MAX_CELLS: Final = 2_000_000
"""Upper bound on (subproblem, option, node, node) weights held for one batch."""


@dataclass(frozen=True, slots=True)
class ACOConfig:
    """Parameters of the colony. The ABC tuner searches over these.

    Attributes:
        iterations: Per subproblem, or per whole plan without a linear objective
        patience: Stop after this many iterations without improvement
        alpha: Weight of pheromone
        beta: Weight of the heuristic
        evaporation: Rho
        exploitation: Q0: chance of taking the best choice instead of sampling
        tau_min: Lower pheromone bound; the upper bound is 1
        local_search: Improve the best routes of every subproblem (linear objectives)
        exact_nodes: Solve subproblems with at most this many destinations exactly
            (scipy); ignored on scheduled instances
    """

    ants: int = 16
    iterations: int = 60
    patience: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.1
    exploitation: float = 0.0
    tau_min: float = 0.01
    update: Literal["iteration_best", "best_so_far"] = "iteration_best"
    local_search: bool = True
    exact_nodes: int = 0

    def __post_init__(self) -> None:
        problems = [
            message
            for broken, message in (
                (self.ants < 1, "ants must be at least 1"),
                (self.iterations < 1, "iterations must be at least 1"),
                (self.patience < 1, "patience must be at least 1"),
                (self.alpha < 0, "alpha must not be negative"),
                (self.beta < 0, "beta must not be negative"),
                (not 0 < self.evaporation <= 1, "evaporation must be in (0, 1]"),
                (not 0 <= self.exploitation <= 1, "exploitation must be in [0, 1]"),
                (not 0 < self.tau_min <= 1, "tau_min must be in (0, 1]"),
                (self.exact_nodes < 0, "exact_nodes must not be negative"),
                (
                    self.update not in ("iteration_best", "best_so_far"),
                    "update must be 'iteration_best' or 'best_so_far'",
                ),
            )
            if broken
        ]
        if problems:
            raise ValueError("invalid ACOConfig: " + "; ".join(problems))


class ACOSolver:
    """Line-haul planning with an ant colony; see the module documentation."""

    def __init__(self, config: ACOConfig | None = None) -> None:
        self.config = config or ACOConfig()

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        goal = default_objective() if objective is None else objective
        started = time.perf_counter()
        search = _Search(instance, self.config, seed)
        form = linear_form(goal)
        if form is not None and not constraints:
            builder = search.linear(linear_coefficients(instance, form))
        else:
            builder = search.general(goal, constraints)
        plan = builder.build()
        evaluation = evaluate(instance, plan, constraints)
        if search.limited:
            search.stats["spot_limits_reached"] = _limits_reached(instance, plan)
        return SolveResult(
            plan=plan,
            evaluation=evaluation,
            objective=float(goal(evaluation)),
            seconds=time.perf_counter() - started,
            stats=search.stats,
        )


def _limits_reached(instance: LineHaulInstance, plan: Plan) -> int:
    """How many (day, spot offer) pairs with a daily limit used every vehicle allowed.

    A count above zero means the greedy sharing of daily limits across origins
    may have bound the plan.
    """
    used = np.zeros((len(instance.days), instance.spot_available.size), dtype=np.int64)
    spot = plan.kind == SPOT
    np.add.at(used, (plan.day[spot], plan.source[spot]), 1)
    limited = instance.spot_available != UNLIMITED
    return int(np.count_nonzero(limited[None, :] & (used >= instance.spot_available[None, :])))


def _improves(value: NDArray[np.float64], best: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True where ``value`` is better than ``best`` by more than rounding."""
    finite = np.isfinite(best)
    margin = np.where(finite, 1e-12 * np.abs(np.where(finite, best, 0.0)), 0.0)
    return np.asarray(value < best - margin)


@dataclass(slots=True)
class _Outcome:
    """The best solution the colony found for every subproblem of a batch.

    Attributes:
        value: (subproblems,) linear objective, without constants
        usage: (subproblems, options) vehicles used
    """

    value: NDArray[np.float64]
    routes: list[Routes]
    remaining: list[NDArray[np.float64]]
    item_left: list[NDArray[np.bool_]]
    usage: NDArray[np.int64]


class _Search:
    """Runs the colonies of one solve; shared by both search modes."""

    def __init__(self, instance: LineHaulInstance, config: ACOConfig, seed: int) -> None:
        self.instance, self.config, self.seed = instance, config, seed
        self.shared = Shared.of(instance)
        self.scheduled = instance.scheduled
        self.pieces = [] if self.scheduled else subproblems(instance)
        self.limits = np.where(
            instance.spot_available == UNLIMITED, UNBOUNDED, instance.spot_available
        ).astype(np.int64)
        self.limited = bool(np.any(instance.spot_available != UNLIMITED))
        self.stats: dict[str, float | int | str] = {"subproblems": len(self.pieces)}
        self.iterations = self.constructions = 0

    def _batches(self, pieces: list[Subproblem]) -> list[Batch]:
        batches = [Batch(group, self.shared) for group in self._groups(pieces)]
        self.stats["batches"] = len(batches)
        return batches

    def _groups(self, pieces: list[Subproblem]) -> list[list[Subproblem]]:
        """Subproblems built side by side.

        Subproblems of similar size go together (padded to the largest). When spot
        offers have daily limits, a group holds at most one subproblem per day, so
        that the subproblems of a day take their spot vehicles one after another.
        """
        buckets: dict[int, list[Subproblem]] = {}
        for problem in pieces:
            size = max(4, 1 << max(problem.size - 1, 0).bit_length())
            buckets.setdefault(size, []).append(problem)
        groups: list[list[Subproblem]] = []
        for size in sorted(buckets, reverse=True):
            members = buckets[size]
            options = max(problem.options for problem in members)
            per_group = max(
                1, min(MAX_ROWS // self.config.ants, MAX_CELLS // (options * size * size))
            )
            if self.limited:
                rounds: list[list[Subproblem]] = []
                seen: dict[int, int] = {}
                for problem in members:
                    turn = seen.get(problem.day, 0)
                    seen[problem.day] = turn + 1
                    if turn == len(rounds):
                        rounds.append([])
                    rounds[turn].append(problem)
            else:
                rounds = [members]
            for members in rounds:
                groups += [
                    members[start : start + per_group]
                    for start in range(0, len(members), per_group)
                ]
        return groups

    def _build(
        self,
        batch: Batch,
        pheromone: Pheromone,
        available: NDArray[np.int64],
        active: NDArray[np.bool_],
        rng: np.random.Generator,
        room: Room | None = None,
    ) -> Colony:
        config = self.config
        return Build(
            batch,
            pheromone,
            self.shared,
            ants=config.ants,
            alpha=config.alpha,
            beta=config.beta,
            exploitation=config.exploitation,
            rng=rng,
            available=available,
            active=active,
            room=room,
        ).run()

    def _spot_rows(
        self, batch: Batch
    ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.int64]]:
        """(subproblem, option) positions of the spot options of a batch, and their offers."""
        which, option = np.nonzero((batch.kind == SPOT) & (batch.available > 0))
        offers = np.array(
            [batch.problems[w].source[o] for w, o in zip(which, option, strict=True)],
            dtype=np.int64,
        )
        return which, option, offers

    def linear(self, coefficients: Mapping[str, float]) -> PlanBuilder:
        """Optimize every subproblem on its own, with the linear objective."""
        builder = PlanBuilder(self.instance)
        spot_left = {day: self.limits.copy() for day in range(len(self.instance.days))}
        if self.scheduled:
            score = self._one_by_one(coefficients, spot_left, builder)
        else:
            score = self._side_by_side(coefficients, spot_left, builder)
        owned_fixed = float(
            np.sum(
                self.instance.owned_count
                * self.instance.owned_fixed
                * self.instance.owned_mandatory
            )
            * len(self.instance.days)
        )
        self.stats |= {
            "mode": "linear",
            "iterations": self.iterations,
            "constructions": self.constructions,
            "score": score + coefficients.get("owned_fixed", 0.0) * owned_fixed,
        }
        return builder

    def _side_by_side(
        self,
        coefficients: Mapping[str, float],
        spot_left: dict[int, NDArray[np.int64]],
        builder: PlanBuilder,
    ) -> float:
        """Subproblems of similar size in batches; return the summed objective."""
        config = self.config
        score = 0.0
        pieces = self.pieces
        if config.exact_nodes and self.instance.rules.split_demand:
            small = [
                p
                for p in pieces
                if np.count_nonzero(p.demand.max(axis=1) > 0) <= config.exact_nodes
            ]
            score += self._exact(small, coefficients, spot_left, builder)
            pieces = [p for p in pieces if p not in small]
        for index, batch in enumerate(self._batches(pieces)):
            available = batch.available.copy()
            which, option, offers = self._spot_rows(batch)
            days = np.array([problem.day for problem in batch.problems])
            left = np.array(
                [spot_left[day][offer] for day, offer in zip(days[which], offers, strict=True)]
            )
            available[which, option] = np.minimum(available[which, option], left)
            best = self._colonies(batch, index, coefficients, available)
            for slot, problem in enumerate(batch.problems):
                routes, remaining = best.routes[slot], best.remaining[slot]
                if config.local_search:
                    routes, change = improve(
                        problem,
                        routes,
                        coefficients,
                        self.shared,
                        available[slot, : problem.options],
                        remaining[: problem.size],
                    )
                    best.value[slot] += change
                    best.usage[slot, : problem.options] = np.bincount(
                        routes.option, minlength=problem.options
                    )
                builder.add(problem, routes)
            for w, o, offer in zip(which, option, offers, strict=True):
                spot_left[int(days[w])][offer] -= best.usage[w, o]
            score += float(best.value.sum())
        return score

    def _one_by_one(
        self,
        coefficients: Mapping[str, float],
        spot_left: dict[int, NDArray[np.int64]],
        builder: PlanBuilder,
    ) -> float:
        """Scheduled instances: the subproblems of each day in planning order, with the ledger.

        Subproblems that touch only centers without handling capacity or vehicle
        limits cannot affect one another, so (without daily spot limits) they
        run side by side first; the others run one by one.
        """
        ledger = Ledger(self.instance)
        coordinator = Coordinator(self.instance)
        score = 0.0
        count = index = 0
        for day in range(len(self.instance.days)):
            problems = coordinator.day(day)
            count += len(problems)
            free = [] if self.limited else [p for p in problems if self._free(p)]
            bound = [p for p in problems if not any(p is other for other in free)]
            groups = [*self._groups(free), *([problem] for problem in bound)]
            for group in groups:
                batch = Batch(group, self.shared)
                room = None if group[0] in free else ledger.room(group[0])
                available = batch.available.copy()
                for slot, problem in enumerate(group):
                    spot = problem.kind == SPOT
                    options = available[slot, : problem.options]
                    options[spot] = np.minimum(options[spot], spot_left[day][problem.source[spot]])
                best = self._colonies(batch, index, coefficients, available, room)
                index += 1
                for slot, problem in enumerate(group):
                    score += self._settle(
                        problem,
                        best,
                        slot,
                        coefficients,
                        available[slot, : problem.options],
                        room,
                        ledger,
                        coordinator,
                        builder,
                    )
                    used = np.bincount(best.routes[slot].option, minlength=problem.options)
                    spot = problem.kind == SPOT
                    spot_left[day][problem.source[spot]] -= used[spot]
        self.stats |= {"subproblems": count, "batches": index}
        return score

    def _free(self, problem: Subproblem) -> bool:
        """True if no center of ``problem`` has a handling capacity or a vehicle limit."""
        instance = self.instance
        places = np.concatenate([[problem.origin], problem.nodes])
        return bool(
            np.all(np.isinf(instance.handling_capacity[places]))
            and np.all(instance.vehicle_limit[places] == UNLIMITED)
        )

    def _settle(
        self,
        problem: Subproblem,
        best: _Outcome,
        slot: int,
        coefficients: Mapping[str, float],
        available: NDArray[np.int64],
        room: Room | None,
        ledger: Ledger,
        coordinator: Coordinator,
        builder: PlanBuilder,
    ) -> float:
        """Improve, charge and record the routes of one scheduled subproblem; return their value.

        The value leaves out demand handed to the next day, which is valued there.
        """
        shared = self.shared
        routes = best.routes[slot]
        if self.config.local_search:
            routes, _ = improve(
                problem,
                routes,
                coefficients,
                shared,
                available,
                best.remaining[slot][: problem.size],
                room,
                best.item_left[slot][: problem.size, : problem.item_row.shape[1]],
            )
        best.routes[slot] = routes
        remaining, item_left = problem.leftover(routes, self.instance.rules.split_demand)
        alone = Batch([problem], shared)
        final = Colony.single(problem, routes, remaining, item_left)
        value = float(scores(alone, final, coefficients, shared, room, waiting=False)[0, 0])
        ledger.charge(problem, timeline(alone, final, shared))
        coordinator.defer(problem, remaining, item_left)
        builder.add(problem, routes)
        return value

    def _colonies(
        self,
        batch: Batch,
        index: int,
        coefficients: Mapping[str, float],
        available: NDArray[np.int64],
        room: Room | None = None,
    ) -> _Outcome:
        """Run the colonies of a batch until they stop improving."""
        config, ants = self.config, self.config.ants
        count = batch.count
        rng = np.random.default_rng([self.seed, index])
        pheromone = Pheromone.filled(batch, 1.0)
        best_used = Pheromone.filled(batch, 0.0)
        best = np.full(count, np.inf)
        best_routes: list[Routes | None] = [None] * count
        best_remaining: list[NDArray[np.float64] | None] = [None] * count
        best_left: list[NDArray[np.bool_] | None] = [None] * count
        best_usage = np.zeros((count, batch.n), dtype=np.int64)
        stale = np.zeros(count, dtype=np.int64)
        active = np.ones(count, dtype=np.bool_)
        for _ in range(config.iterations):
            self.iterations += 1
            self.constructions += int(active.sum()) * ants
            colony = self._build(
                batch, pheromone, np.repeat(available, ants, axis=0), active, rng, room
            )
            values = scores(batch, colony, coefficients, self.shared, room)
            winner = values.argmin(axis=1)
            value = values[np.arange(count), winner]
            rows = np.arange(count) * ants + winner
            improved = active & _improves(value, best)
            usage = colony.usage(batch.n)
            for position in np.flatnonzero(improved):
                best_routes[position] = colony.routes(int(rows[position]))
                best_usage[position] = usage[rows[position]]
                best_remaining[position] = colony.remaining[rows[position]]
                best_left[position] = colony.item_left[rows[position]]
            best[improved] = value[improved]
            stale[improved] = 0
            stale[active & ~improved] += 1
            used = colony.used(np.where(active, rows, -1), batch)
            if config.update == "best_so_far":
                best_used.copy_from(used, improved)
                used = best_used
            pheromone.update(used, config.evaporation, config.tau_min, active)
            active &= stale < config.patience
            if not active.any():
                break
        routes = [route for route in best_routes if route is not None]
        remaining = [left for left in best_remaining if left is not None]
        item_left = [left for left in best_left if left is not None]
        assert len(routes) == len(remaining) == len(item_left) == count
        return _Outcome(best, routes, remaining, item_left, best_usage)

    def _exact(
        self,
        pieces: list[Subproblem],
        coefficients: Mapping[str, float],
        spot_left: dict[int, NDArray[np.int64]],
        builder: PlanBuilder,
    ) -> float:
        """Solve ``pieces`` exactly; return the summed objective, without constants."""
        from pirilib.solvers.exact import solve_subproblem

        score = 0.0
        for problem in pieces:
            available = problem.available.copy()
            spot = problem.kind == SPOT
            left = spot_left[problem.day]
            available[spot] = np.minimum(available[spot], left[problem.source[spot]])
            result = solve_subproblem(
                problem, coefficients, self.shared, available, self.config.exact_nodes
            )
            builder.add(problem, result.routes)
            used = np.bincount(result.routes.option, minlength=problem.options)
            left[problem.source[spot]] -= used[spot]
            score += result.value
        self.stats["exact"] = len(pieces)
        return score

    def general(self, goal: Objective, constraints: Sequence[Constraint]) -> PlanBuilder:
        """Each ant builds a whole plan; plans are compared with ``goal``.

        On scheduled instances every ant keeps its own ledger and builds the
        subproblems one by one; nothing is carried to the next day.
        """
        config, ants = self.config, self.config.ants
        if self.scheduled:
            self.waves = [Batch([problem], self.shared) for problem in Coordinator(self.instance)]
            self.stats |= {"subproblems": len(self.waves), "batches": len(self.waves)}
        else:
            self.waves = self._batches(self.pieces)
        pheromones = [Pheromone.filled(batch, 1.0) for batch in self.waves]
        best_used = [Pheromone.filled(batch, 0.0) for batch in self.waves]
        best_value = np.inf
        best_routes: list[list[Routes]] = []
        stale = iterations = 0
        days = len(self.instance.days)
        for iteration in range(config.iterations):
            iterations += 1
            spot_left = np.broadcast_to(self.limits, (days, ants, self.limits.size)).copy()
            ledger = Ledger(self.instance, ants) if self.scheduled else None
            colonies: list[Colony] = []
            for index, batch in enumerate(self.waves):
                available = np.repeat(batch.available, ants, axis=0)
                which, option, offers = self._spot_rows(batch)
                day = np.array([batch.problems[w].day for w in which], dtype=np.int64)
                rows = (which[:, None] * ants + np.arange(ants)[None, :]).ravel()
                columns = np.repeat(option, ants)
                left = spot_left[
                    np.repeat(day, ants),
                    np.tile(np.arange(ants), which.size),
                    np.repeat(offers, ants),
                ]
                available[rows, columns] = np.minimum(available[rows, columns], left)
                rng = np.random.default_rng([self.seed, iteration, index])
                active = np.ones(batch.count, dtype=np.bool_)
                room = None if ledger is None else ledger.room(batch.problems[0])
                colony = self._build(batch, pheromones[index], available, active, rng, room)
                if ledger is not None:
                    ledger.charge(batch.problems[0], timeline(batch, colony, self.shared))
                usage = colony.usage(batch.n)
                np.subtract.at(
                    spot_left,
                    (
                        np.repeat(day, ants),
                        np.tile(np.arange(ants), which.size),
                        np.repeat(offers, ants),
                    ),
                    usage[rows, columns],
                )
                colonies.append(colony)
            values = np.array(
                [self._value(goal, constraints, colonies, ant) for ant in range(ants)]
            )
            winner = int(values.argmin())
            used = [
                colony.used(np.arange(batch.count) * ants + winner, batch)
                for batch, colony in zip(self.waves, colonies, strict=True)
            ]
            improved = bool(_improves(values[winner : winner + 1], np.array([best_value]))[0])
            if improved:
                best_value = float(values[winner])
                best_routes = [
                    [colony.routes(row * ants + winner) for row in range(batch.count)]
                    for batch, colony in zip(self.waves, colonies, strict=True)
                ]
                stale = 0
            else:
                stale += 1
            every = [np.ones(batch.count, dtype=np.bool_) for batch in self.waves]
            for pheromone, mark, best_mark, everyone in zip(
                pheromones, used, best_used, every, strict=True
            ):
                if config.update == "best_so_far":
                    if improved:
                        best_mark.copy_from(mark, everyone)
                    mark = best_mark
                pheromone.update(mark, config.evaporation, config.tau_min, everyone)
            if stale >= config.patience:
                break
        builder = PlanBuilder(self.instance)
        for batch, routes in zip(self.waves, best_routes, strict=True):
            for problem, part in zip(batch.problems, routes, strict=True):
                builder.add(problem, part)
        self.stats |= {
            "mode": "general",
            "iterations": iterations,
            "constructions": iterations * ants,
        }
        return builder

    def _value(
        self, goal: Objective, constraints: Sequence[Constraint], colonies: list[Colony], ant: int
    ) -> float:
        builder = PlanBuilder(self.instance)
        for batch, colony in zip(self.waves, colonies, strict=True):
            for row, problem in enumerate(batch.problems):
                builder.add(problem, colony.routes(row * self.config.ants + ant))
        return float(goal(evaluate(self.instance, builder.build(), constraints)))
