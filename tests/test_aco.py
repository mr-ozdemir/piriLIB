"""Tests of the ant colony solver, :class:`pirilib.ACOSolver`, and of :class:`pirilib.ACOConfig`.

The expected values come from the solver's documentation (``pirilib.solvers.aco``): every
built-in rule except minimum fill and coverage holds by construction, a linear objective
without constraints of your own is optimized subproblem by subproblem (``mode`` ``"linear"``),
anything else over whole plans (``mode`` ``"general"``), and the result carries the evaluation
of its plan and the objective's value on it. :func:`pirilib.check_plan` is the independent
oracle for cost and violations. Every solve has a fixed seed.
"""

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import pirilib
from pirilib import ACOConfig, ACOSolver, SolveResult, check_plan, evaluate
from pirilib import objective as obj
from pirilib.constraints import CATALOG, Constraint, MaxDispatchesPerDay
from pirilib.plan import SPOT
from pirilib.solvers._colony import Batch, Build, Pheromone, Shared
from pirilib.solvers.decompose import subproblems
from pirilib.spec import LineHaulInstance, parse_spec
from pirilib.spec.instance import UNLIMITED

type RandomDocument = Callable[[int], dict[str, Any]]
type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

SMALL = ACOConfig(ants=6, iterations=8, patience=4)
"""A colony small enough that a random problem solves in a fraction of a second."""

VIOLATION_PENALTY = 1e6
"""Weight of a unit of violation in :data:`GOAL`: more than any test plan costs, so a plan
that breaks a rule never beats one that keeps it."""

GOAL = obj.total_cost() + VIOLATION_PENALTY * obj.violations()
"""The linear objective of most tests: total cost, with violations priced out."""

NO_CONSTRUCTION_VIOLATIONS = dict.fromkeys(sorted(set(CATALOG) - {"min_fill", "coverage"}), 0.0)
"""Every built-in rule but minimum fill and coverage, with no violation: the documented rules
that hold by construction."""

SCHEDULED_RULES = dict.fromkeys(
    ("route", "load_route", "ready", "deferral", "ban", "owned_line", "capacity"), 0.0
)
"""The timing, route and capacity rules a scheduled plan keeps, with no violation."""

RANDOM_SEEDS = range(40)
"""Seeds of the random problems (``random_document``) that the solver properties run on."""

UNLIMITED_SPOT_SEEDS = (
    *(0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 16, 17, 18, 19, 20, 22, 23, 24, 26),
    *(28, 29, 30, 31, 32, 33, 34, 35, 38, 39),
)
"""The seeds in :data:`RANDOM_SEEDS` whose problems have no daily spot limit
(``random_document`` limits the ``small`` spot offer with probability 0.3)."""

NO_SPLIT_SEED = 2
"""The first seed whose random problem forbids splitting demand (18 items)."""

OPENING_SEEDS = range(20)
"""Seeds of the random problems the kept opening weights are checked on."""

OPENING_ANTS = 4
"""Ants per subproblem in the opening weight tests: several rows per subproblem."""

EXAMPLE_SEED = 1
"""The seed of the bundled example's solve; any fixed seed would do."""

REPEATED_SEED = 3
"""A seed other than the default 0, so that the reproducibility test passes ``seed``."""

SAME_SUM = 1e-12
"""Relative difference allowed between two sums of the same costs in another order."""

REGROUPED_SUM = 1e-9
"""Relative difference allowed between sums grouped differently (per subproblem against the
whole plan)."""

VIOLATION_ATOL = 1e-6
"""Absolute difference allowed between violation amounts near zero; amounts count desi,
kilometres or vehicles, far above this."""

SPOT_PENALTY = 1e5
"""Price of a spot dispatch in the steering tests: far above the cost of any route there."""

DEAR_OWNED_PER_KM = 10.0
"""Cost per km of the owned ``big`` line in the steering tests: over the 111 km from ``A`` to
``B`` it costs 1110, against 300 + 2 per km (522) for a spot ``big`` vehicle."""

DAILY_DISPATCH_LIMIT = 5
"""A limit of your own above the at most three dispatches a day of the fixture problem; the
tests are about how such a constraint is handled, not about a binding limit."""

SPEED_KMH = 60.0
"""Speed of both vehicle types in the scheduled problem."""


def solve(
    instance: LineHaulInstance,
    objective: obj.Objective,
    constraints: Sequence[Constraint] = (),
    *,
    config: ACOConfig = SMALL,
    seed: int = 0,
) -> SolveResult:
    """The result of a colony with ``config`` for ``objective``."""
    return ACOSolver(config).solve(instance, objective, constraints, seed=seed)


@functools.cache
def solved_random(
    seed: int, make: RandomDocument, local_search: bool = True
) -> tuple[LineHaulInstance, SolveResult]:
    """The random problem of ``seed`` and the small colony's result for :data:`GOAL`.

    Solved once per arguments: the solve is deterministic and no test changes the result.
    """
    instance = parse_spec(make(seed)).compile()
    config = replace(SMALL, local_search=local_search)
    return instance, solve(instance, GOAL, config=config, seed=seed)


@functools.cache
def solved_example(path: Path) -> tuple[LineHaulInstance, SolveResult]:
    """The bundled example and the default colony's result for :data:`GOAL`, solved once."""
    instance = pirilib.load_spec(path).compile()
    return instance, solve(instance, GOAL, config=ACOConfig(), seed=EXAMPLE_SEED)


def spot_dispatches(result: SolveResult) -> int:
    """The number of spot dispatches of the plan of ``result``."""
    return int(np.count_nonzero(result.plan.kind == SPOT))


def plain_goal(evaluation: pirilib.Evaluation) -> float:
    """:data:`GOAL` written as a plain function, which has no linear form."""
    return evaluation.total_cost + VIOLATION_PENALTY * sum(evaluation.violations.values())


def total_cost(evaluation: pirilib.Evaluation) -> float:
    """Total cost alone, as a plain function."""
    return evaluation.total_cost


def dear_owned_line(document: dict[str, Any]) -> dict[str, Any]:
    """The fixture document with an owned line dearer than a spot vehicle on its route."""
    document["fleet"]["owned"][0]["cost"]["per_km"] = DEAR_OWNED_PER_KM
    return document


def scheduled(document: dict[str, Any]) -> dict[str, Any]:
    """The fixture document made scheduled (schema 0.2).

    Both vehicle types drive :data:`SPEED_KMH`, the first item is ready at 09:00 and every
    item is due 6 hours after it is ready; lateness costs 1 per desi and started hour.
    """
    document["schema"] = "pirilib/linehaul@0.2"
    document["vehicle_types"][0]["speed_kmh"] = SPEED_KMH
    document["vehicle_types"][1]["speed_kmh"] = SPEED_KMH
    document["demand"][0]["ready"] = "09:00"
    document["rules"] = {"service": {"due_hours": 6, "penalty": 1.0}}
    return document


class OpeningSpy(Build):
    """A colony construction that records its opening weights before every start step.

    ``kept`` holds the weights the build keeps up to date step by step, ``fresh`` the ones
    ``_opening`` computes from scratch for the same rows. Recording changes nothing the
    build does (see :func:`test_the_opening_spy_builds_what_the_colony_builds`).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kept: list[NDArray[np.float64]] = []
        self.fresh: list[NDArray[np.float64]] = []

    def _start(self, rows: NDArray[np.intp]) -> None:
        if self.pending:
            self._reopen()
        self.kept.append(self.opening[rows])
        self.fresh.append(self._opening(rows, self.of[rows]))
        super()._start(rows)


def colony_build[B: Build](kind: type[B], instance: LineHaulInstance, seed: int) -> B:
    """A construction of ``kind`` over every subproblem of ``instance``, without a room.

    :data:`OPENING_ANTS` ants per subproblem, uniform pheromone, the default weights of
    :class:`ACOConfig` and a generator seeded with ``seed``.
    """
    defaults = ACOConfig()
    shared = Shared.of(instance)
    batch = Batch(subproblems(instance), shared)
    return kind(
        batch,
        Pheromone.filled(batch, 1.0),
        shared,
        ants=OPENING_ANTS,
        alpha=defaults.alpha,
        beta=defaults.beta,
        exploitation=defaults.exploitation,
        rng=np.random.default_rng(seed),
        available=np.repeat(batch.available, OPENING_ANTS, axis=0),
        active=np.ones(batch.count, dtype=np.bool_),
    )


def test_the_example_plan_is_feasible(example_path: Path) -> None:
    _, result = solved_example(example_path)

    feasible = result.evaluation.feasible

    assert feasible


def test_a_linear_objective_is_optimized_subproblem_by_subproblem(example_path: Path) -> None:
    _, result = solved_example(example_path)

    mode = result.stats["mode"]

    assert mode == "linear"


def test_the_objective_of_a_feasible_plan_is_its_cost(example_path: Path) -> None:
    _, result = solved_example(example_path)

    objective = result.objective

    assert objective == pytest.approx(result.evaluation.total_cost, rel=SAME_SUM)


def test_the_example_score_is_the_objective(example_path: Path) -> None:
    _, result = solved_example(example_path)

    score = result.stats["score"]

    assert score == pytest.approx(result.objective, rel=REGROUPED_SUM)


def test_the_plan_checker_accepts_the_example_plan(example_path: Path) -> None:
    instance, result = solved_example(example_path)

    report = check_plan(instance, result.plan)

    assert report.feasible


def test_the_plan_checker_finds_the_example_cost(example_path: Path) -> None:
    instance, result = solved_example(example_path)

    report = check_plan(instance, result.plan)

    assert report.total_cost == pytest.approx(result.evaluation.total_cost, rel=SAME_SUM)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_rules_that_hold_by_construction_are_never_broken(
    seed: int, random_document: RandomDocument
) -> None:
    _, result = solved_random(seed, random_document)

    violations = result.evaluation.violations

    assert violations.items() >= NO_CONSTRUCTION_VIOLATIONS.items()


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_the_objective_is_the_goal_of_the_evaluated_plan(
    seed: int, random_document: RandomDocument
) -> None:
    _, result = solved_random(seed, random_document)

    objective = result.objective

    assert objective == GOAL(result.evaluation)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_the_colony_score_is_the_objective(seed: int, random_document: RandomDocument) -> None:
    _, result = solved_random(seed, random_document)

    score = result.stats["score"]

    assert score == pytest.approx(result.objective, rel=REGROUPED_SUM)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_the_plan_checker_finds_the_evaluated_cost(
    seed: int, random_document: RandomDocument
) -> None:
    instance, result = solved_random(seed, random_document)

    report = check_plan(instance, result.plan)

    assert report.total_cost == pytest.approx(result.evaluation.total_cost, rel=SAME_SUM)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_the_plan_checker_finds_the_evaluated_violations(
    seed: int, random_document: RandomDocument
) -> None:
    instance, result = solved_random(seed, random_document)

    report = check_plan(instance, result.plan)

    assert dict(result.evaluation.violations) == pytest.approx(
        report.totals(), rel=REGROUPED_SUM, abs=VIOLATION_ATOL
    )


@pytest.mark.parametrize("seed", UNLIMITED_SPOT_SEEDS)
def test_the_colony_alone_carries_all_demand_when_spot_is_unlimited(
    seed: int, random_document: RandomDocument
) -> None:
    instance, result = solved_random(seed, random_document, local_search=False)
    np.testing.assert_array_equal(instance.spot_available, UNLIMITED)

    coverage = result.evaluation.violations["coverage"]

    assert coverage == 0.0


@pytest.mark.parametrize("seed", UNLIMITED_SPOT_SEEDS)
def test_local_search_never_worsens_the_objective(
    seed: int, random_document: RandomDocument
) -> None:
    instance, plain = solved_random(seed, random_document, local_search=False)
    np.testing.assert_array_equal(instance.spot_available, UNLIMITED)

    _, improved = solved_random(seed, random_document)

    assert improved.objective <= plain.objective * (1 + SAME_SUM)


def test_whole_items_when_demand_may_not_be_split(random_document: RandomDocument) -> None:
    document = random_document(NO_SPLIT_SEED)
    assert document["rules"]["split_demand"] is False
    instance = parse_spec(document).compile()

    plan = solve(instance, GOAL).plan

    np.testing.assert_array_equal(
        plan.load_quantity, instance.demand_quantity[plan.load_demand], strict=True
    )
    assert np.unique(plan.load_demand).size == plan.load_demand.size


def test_the_same_seed_gives_the_same_plan(example_path: Path) -> None:
    instance = pirilib.load_spec(example_path).compile()

    first = solve(instance, GOAL, seed=REPEATED_SEED)
    second = solve(instance, GOAL, seed=REPEATED_SEED)

    np.testing.assert_equal(asdict(first.plan), asdict(second.plan))


def test_a_linear_spot_penalty_moves_demand_to_owned_vehicles(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Without the penalty all three items go by spot, as the owned line is dear; with it the
    900 desi from ``A`` to ``B`` take the owned line, and the two items the line does not
    serve still need two spot vehicles."""
    instance = compile_document(dear_owned_line(document))

    cheap = solve(instance, GOAL)
    few_spot = solve(instance, GOAL + SPOT_PENALTY * obj.dispatches("spot"))

    assert (spot_dispatches(cheap), spot_dispatches(few_spot)) == (3, 2)


def test_a_linear_spot_penalty_keeps_the_plan_feasible(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(dear_owned_line(document))

    result = solve(instance, GOAL + SPOT_PENALTY * obj.dispatches("spot"))

    assert result.evaluation.feasible


def test_a_plain_function_objective_is_optimized_over_whole_plans(
    instance: LineHaulInstance,
) -> None:
    result = solve(instance, plain_goal)

    assert result.stats["mode"] == "general"


def test_a_plain_function_objective_gives_a_feasible_plan(instance: LineHaulInstance) -> None:
    result = solve(instance, plain_goal)

    assert result.evaluation.feasible


def test_the_objective_is_the_plain_functions_value_on_the_plan(
    instance: LineHaulInstance,
) -> None:
    result = solve(instance, plain_goal)

    assert result.objective == plain_goal(result.evaluation)


def test_constraints_of_your_own_are_checked_on_whole_plans(instance: LineHaulInstance) -> None:
    constraint = MaxDispatchesPerDay(DAILY_DISPATCH_LIMIT)

    result = solve(instance, GOAL, [constraint])

    assert result.stats["mode"] == "general"


def test_a_constraint_of_your_own_is_reported_with_the_built_in_ones(
    instance: LineHaulInstance,
) -> None:
    constraint = MaxDispatchesPerDay(DAILY_DISPATCH_LIMIT)

    result = solve(instance, GOAL, [constraint])

    assert "max_dispatches_per_day" in result.evaluation.violations


def test_a_plan_with_a_constraint_of_your_own_is_feasible(instance: LineHaulInstance) -> None:
    constraint = MaxDispatchesPerDay(DAILY_DISPATCH_LIMIT)

    result = solve(instance, GOAL, [constraint])

    assert result.evaluation.feasible


@pytest.mark.parametrize("objective", [GOAL, total_cost], ids=["linear", "general"])
def test_spot_limits_are_shared_by_the_origins_of_a_day(
    objective: obj.Objective, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Three origins with 300 desi each on one day and two ``small`` spot vehicles for that
    day, no owned ones: two items travel, one (300 desi) cannot, the limit is reached once."""
    document["fleet"] = {"spot": [{"type": "small", "cost": {"fixed": 1.0}, "available": 2}]}
    document["demand"] = [
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 300.0}},
        {"origin": "B", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 300.0}},
        {"origin": "C", "destination": "A", "date": "2026-01-05", "quantity": {"desi": 300.0}},
    ]
    instance = compile_document(document)

    result = solve(instance, objective)

    violations = result.evaluation.violations
    outcome = (violations["spot_availability"], violations["coverage"])
    assert (*outcome, result.stats["spot_limits_reached"]) == (0.0, 300.0, 1)


def test_unknown_linear_features_are_rejected(instance: LineHaulInstance) -> None:
    odd = obj.Term("odd", lambda evaluation: 0.0, linear={"tolls": 1.0})

    with pytest.raises(ValueError, match=re.escape("unknown features ['tolls']")):
        solve(instance, obj.total_cost() + odd)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param({"ants": 0}, "ants must be at least 1", id="ants-0"),
        pytest.param({"iterations": 0}, "iterations must be at least 1", id="iterations-0"),
        pytest.param({"patience": 0}, "patience must be at least 1", id="patience-0"),
        pytest.param({"alpha": -0.5}, "alpha must not be negative", id="alpha-negative"),
        pytest.param({"beta": -0.5}, "beta must not be negative", id="beta-negative"),
        pytest.param({"evaporation": 0.0}, "evaporation must be in (0, 1]", id="evaporation-0"),
        pytest.param(
            {"evaporation": 1.5}, "evaporation must be in (0, 1]", id="evaporation-above-1"
        ),
        pytest.param(
            {"exploitation": -0.5}, "exploitation must be in [0, 1]", id="exploitation-negative"
        ),
        pytest.param(
            {"exploitation": 1.5}, "exploitation must be in [0, 1]", id="exploitation-above-1"
        ),
        pytest.param({"tau_min": 0.0}, "tau_min must be in (0, 1]", id="tau_min-0"),
        pytest.param({"tau_min": 2.0}, "tau_min must be in (0, 1]", id="tau_min-above-1"),
        pytest.param(
            {"exact_nodes": -1}, "exact_nodes must not be negative", id="exact_nodes-negative"
        ),
        pytest.param(
            {"update": "elitist"},
            "update must be 'iteration_best' or 'best_so_far'",
            id="update-unknown",
        ),
    ],
)
def test_an_invalid_config_is_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        ACOConfig(**change)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ants", 1),
        ("iterations", 1),
        ("patience", 1),
        ("alpha", 0.0),
        ("beta", 0.0),
        ("evaporation", 1.0),
        ("exploitation", 0.0),
        ("exploitation", 1.0),
        ("tau_min", 1.0),
        ("exact_nodes", 0),
        ("update", "best_so_far"),
    ],
)
def test_a_config_at_the_edge_of_its_range_is_accepted(field: str, value: Any) -> None:
    config = ACOConfig(**{field: value})

    assert getattr(config, field) == value


def test_the_solve_shortcut_returns_a_feasible_plan(instance: LineHaulInstance) -> None:
    result = pirilib.solve(instance, GOAL, solver=ACOSolver(SMALL))

    assert result.evaluation.feasible


def test_the_solve_shortcut_result_holds_the_evaluation_of_its_plan(
    instance: LineHaulInstance,
) -> None:
    result = pirilib.solve(instance, GOAL, solver=ACOSolver(SMALL))

    cost = evaluate(instance, result.plan).total_cost

    assert cost == pytest.approx(result.evaluation.total_cost, rel=SAME_SUM)


def test_a_solve_result_shows_its_objective_first(instance: LineHaulInstance) -> None:
    result = pirilib.solve(instance, GOAL, solver=ACOSolver(SMALL))

    text = repr(result)

    assert text.startswith("SolveResult(objective=")


@pytest.mark.parametrize(
    "entry", [pirilib.solve, ACOSolver(SMALL).solve], ids=["solve", "ACOSolver.solve"]
)
def test_there_is_no_default_objective(
    entry: Callable[..., SolveResult], instance: LineHaulInstance
) -> None:
    with pytest.raises(TypeError, match="'objective'"):
        entry(instance)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (pirilib.solve, "^solve needs an objective"),
        (ACOSolver(SMALL).solve, r"^ACOSolver\.solve needs an objective"),
    ],
    ids=["solve", "ACOSolver.solve"],
)
def test_a_none_objective_is_rejected(
    entry: Callable[..., SolveResult], message: str, instance: LineHaulInstance
) -> None:
    with pytest.raises(ValueError, match=message):
        entry(instance, None)


def test_the_scheduled_document_compiles_to_a_scheduled_instance(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(scheduled(document))

    assert instance.scheduled


def test_a_scheduled_problem_with_a_linear_objective_is_planned_piece_by_piece(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(scheduled(document))

    result = solve(instance, GOAL)

    assert result.stats["mode"] == "linear"


def test_a_scheduled_plan_keeps_the_timing_route_and_capacity_rules(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(scheduled(document))

    result = solve(instance, GOAL)

    assert result.evaluation.violations.items() >= SCHEDULED_RULES.items()


def test_the_scheduled_objective_is_the_goal_of_the_evaluated_plan(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(scheduled(document))

    result = solve(instance, GOAL)

    assert result.objective == GOAL(result.evaluation)


def test_the_scheduled_score_is_the_objective(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    instance = compile_document(scheduled(document))

    result = solve(instance, GOAL)

    assert result.stats["score"] == pytest.approx(result.objective, rel=REGROUPED_SUM)


@pytest.mark.parametrize("seed", OPENING_SEEDS)
def test_the_opening_spy_builds_what_the_colony_builds(
    seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(random_document(seed)).compile()

    spied = colony_build(OpeningSpy, instance, seed).run()
    plain = colony_build(Build, instance, seed).run()

    np.testing.assert_equal(asdict(spied), asdict(plain))


@pytest.mark.parametrize("seed", OPENING_SEEDS)
def test_kept_opening_weights_equal_fresh_ones(seed: int, random_document: RandomDocument) -> None:
    """Without a room, the kept ``opening`` always holds what ``_opening`` would compute.

    More than one start step is needed: the first one reads the weights computed when the
    build is set up, only later ones read weights kept up to date step by step.
    """
    instance = parse_spec(random_document(seed)).compile()
    build = colony_build(OpeningSpy, instance, seed)

    build.run()

    assert len(build.kept) > 1
    assert np.array_equal(np.concatenate(build.kept), np.concatenate(build.fresh))
