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
from pirilib.plan import SPOT
from pirilib.solvers._colony import Batch, Build, Colony, Pheromone, Shared, scores
from pirilib.solvers.base import SolveResult, default_objective, linear_coefficients
from pirilib.solvers.decompose import UNBOUNDED, PlanBuilder, Routes, Subproblem, subproblems
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
        exact_nodes: Solve subproblems with at most this many destinations exactly (scipy)
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
        if form is not None and not constraints and not instance.scheduled:
            builder = search.linear(linear_coefficients(instance, form))
        else:
            builder = search.general(goal, constraints)
        plan = builder.build()
        evaluation = evaluate(instance, plan, constraints)
        return SolveResult(
            plan=plan,
            evaluation=evaluation,
            objective=float(goal(evaluation)),
            seconds=time.perf_counter() - started,
            stats=search.stats,
        )


def _improves(value: NDArray[np.float64], best: NDArray[np.float64]) -> NDArray[np.bool_]:
    """True where ``value`` is better than ``best`` by more than rounding."""
    finite = np.isfinite(best)
    margin = np.where(finite, 1e-12 * np.abs(np.where(finite, best, 0.0)), 0.0)
    return np.asarray(value < best - margin)


class _Search:
    """Runs the colonies of one solve; shared by both search modes."""

    def __init__(self, instance: LineHaulInstance, config: ACOConfig, seed: int) -> None:
        self.instance, self.config, self.seed = instance, config, seed
        self.shared = Shared.of(instance)
        self.pieces = subproblems(instance)
        self.limits = np.where(
            instance.spot_available == UNLIMITED, UNBOUNDED, instance.spot_available
        ).astype(np.int64)
        self.limited = bool(np.any(instance.spot_available != UNLIMITED))
        self.stats: dict[str, float | int | str] = {"subproblems": len(self.pieces)}

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
        config, ants = self.config, self.config.ants
        builder = PlanBuilder(self.instance)
        spot_left = {day: self.limits.copy() for day in range(len(self.instance.days))}
        constructions = iterations = 0
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
            count = batch.count
            available = batch.available.copy()
            which, option, offers = self._spot_rows(batch)
            days = np.array([problem.day for problem in batch.problems])
            left = np.array(
                [spot_left[day][offer] for day, offer in zip(days[which], offers, strict=True)]
            )
            available[which, option] = np.minimum(available[which, option], left)
            rng = np.random.default_rng([self.seed, index])
            pheromone = Pheromone.filled(batch, 1.0)
            best_used = Pheromone.filled(batch, 0.0)
            best = np.full(count, np.inf)
            best_routes: list[Routes | None] = [None] * count
            best_remaining: list[NDArray[np.float64] | None] = [None] * count
            best_usage = np.zeros((count, batch.n), dtype=np.int64)
            stale = np.zeros(count, dtype=np.int64)
            active = np.ones(count, dtype=np.bool_)
            for _ in range(config.iterations):
                iterations += 1
                constructions += int(active.sum()) * ants
                colony = self._build(
                    batch, pheromone, np.repeat(available, ants, axis=0), active, rng
                )
                values = scores(batch, colony, coefficients, self.shared)
                winner = values.argmin(axis=1)
                value = values[np.arange(count), winner]
                rows = np.arange(count) * ants + winner
                improved = active & _improves(value, best)
                usage = colony.usage(batch.n)
                for position in np.flatnonzero(improved):
                    best_routes[position] = colony.routes(int(rows[position]))
                    best_usage[position] = usage[rows[position]]
                    best_remaining[position] = colony.remaining[rows[position]]
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
            for slot, problem in enumerate(batch.problems):
                routes, remaining = best_routes[slot], best_remaining[slot]
                assert routes is not None
                assert remaining is not None
                if config.local_search:
                    routes, change = improve(
                        problem,
                        routes,
                        coefficients,
                        self.shared,
                        available[slot, : problem.options],
                        remaining[: problem.size],
                    )
                    best[slot] += change
                    best_usage[slot, : problem.options] = np.bincount(
                        routes.option, minlength=problem.options
                    )
                builder.add(problem, routes)
            for w, o, offer in zip(which, option, offers, strict=True):
                spot_left[int(days[w])][offer] -= best_usage[w, o]
            score += float(best.sum())
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
            "iterations": iterations,
            "constructions": constructions,
            "score": score + coefficients.get("owned_fixed", 0.0) * owned_fixed,
        }
        return builder

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
        """Each ant builds a whole plan; plans are compared with ``goal``."""
        config, ants = self.config, self.config.ants
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
                colony = self._build(batch, pheromones[index], available, active, rng)
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
