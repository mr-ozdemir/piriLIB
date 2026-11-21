"""Tests of the record of how every built-in solver deals with every rule.

The requirement is the documentation of :mod:`pirilib.solvers.support`: a rule that a solver is
said to keep is never broken by its plans, a rule it is said to refuse makes it raise
:class:`ValueError` when a problem uses the setting behind the rule, and the record covers
every rule of the catalog for every built-in solver. The record is only worth something while
it is true, so these tests run the real solvers on problems that put each rule to work, with
penalties so low that breaking a rule is almost free: a solver that only prices a rule would
break it here.

Oracles: :func:`~pirilib.evaluate` measures every rule independently of the solvers, and the
score a solver computes with its own model of the costs must be the objective that the
evaluator gives its plan.

Blind spots, left out on purpose: problems larger than the generated ones, the hybrid colony
(``ACOConfig.exact_nodes``), and proof that a rule recorded as penalized can in fact be
broken: that entry promises nothing.
"""

import importlib.util
import re
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from pirilib import ACOConfig, ACOSolver, Plan, SolveResult, evaluate
from pirilib.constraints import CATALOG, Constraint
from pirilib.objective import Objective
from pirilib.solvers.adapters import ORToolsSolver, PyVRPSolver
from pirilib.solvers.adapters.base import RoutingAdapter
from pirilib.solvers.base import Solver
from pirilib.solvers.direct import DirectShipping
from pirilib.solvers.exact import ExactSolver
from pirilib.solvers.support import SUPPORT, Support, rule_support
from pirilib.spec import LineHaulInstance, parse_spec
from pirilib.synthetic import generate

pytest.importorskip("scipy", reason="the exact solver needs scipy (pip install 'pirilib[exact]')")

type Change = Callable[[dict[str, Any]], None]

GUIDE = Path(__file__).parents[1] / "docs" / "guide.md"

CHEAP = {"cost": 1.0, "penalties": {"coverage": 1e7, "other": 1e-3}}
"""Leaving demand behind is dear and breaking any other rule is almost free."""

SCORE_TOLERANCE = 1e-9
"""Relative difference allowed between sums of the same costs taken in another order."""

HAS_PYVRP = importlib.util.find_spec("pyvrp") is not None
HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None


def no_min_fill(document: dict[str, Any]) -> None:
    document["rules"]["min_fill"] = {"spot": 0.0}


def high_min_fill(document: dict[str, Any]) -> None:
    document["rules"]["min_fill"] = {"spot": 0.4, "owned": 0.3}


def no_stopovers(document: dict[str, Any]) -> None:
    document["rules"]["stopover"]["enabled"] = False


def two_stops(document: dict[str, Any]) -> None:
    document["rules"]["stopover"]["max_stops"] = 2


def any_direction(document: dict[str, Any]) -> None:
    document["rules"]["stopover"] |= {"forward_only": False, "max_stretch": None}


def owned_without_stopovers(document: dict[str, Any]) -> None:
    for group in document["fleet"]["owned"]:
        group["stopovers"] = False


def spot_without_stopovers(document: dict[str, Any]) -> None:
    for offer in document["fleet"]["spot"]:
        offer["stopovers"] = False


def no_split(document: dict[str, Any]) -> None:
    document["rules"]["split_demand"] = False
    for row in document["demand"]:
        row["quantity"]["desi"] = min(row["quantity"]["desi"], 5000.0)


def spot_limits(document: dict[str, Any]) -> None:
    for offer in document["fleet"]["spot"]:
        offer["available"] = 2


def two_dimensions(document: dict[str, Any]) -> None:
    for vehicle in document["vehicle_types"]:
        vehicle["capacity"]["kg"] = vehicle["capacity"]["desi"] / 4
    for row in document["demand"]:
        row["quantity"]["kg"] = row["quantity"]["desi"] / 3


def ready_times(document: dict[str, Any]) -> None:
    for position, row in enumerate(document["demand"]):
        row["ready"] = f"{8 + position % 12:02d}:00"


def deferral(document: dict[str, Any]) -> None:
    document["rules"]["service"] = {"max_deferral_days": 1}


def handling_capacity(document: dict[str, Any]) -> None:
    for center in document["centers"]:
        center["handling_capacity"] = 30000


def vehicle_limits(document: dict[str, Any]) -> None:
    for center in document["centers"]:
        center["vehicle_limits"] = {"tir": 1, "kamyon": 1, "kamyonet": 2}


def late_and_tight(document: dict[str, Any]) -> None:
    """Loads ready in the evening, slow vehicles, lateness, deferral and tight center limits."""
    for position, row in enumerate(document["demand"]):
        row["ready"] = f"{18 + position % 6:02d}:{position * 7 % 60:02d}"
    for vehicle in document["vehicle_types"]:
        vehicle["speed_kmh"] = 50
    for offer in document["fleet"]["spot"]:
        offer["cost"]["per_hour"] = 100
        offer["available"] = 3
    document["rules"]["service"] = {"due_hours": 10, "penalty": 0.4, "max_deferral_days": 1}
    handling_capacity(document)
    vehicle_limits(document)


def handling_time(document: dict[str, Any]) -> None:
    late_and_tight(document)
    document["rules"]["handling"] = {"minutes_per_unit": 0.02}


SCENARIOS: dict[str, tuple[Change, ...]] = {
    "generated": (),
    "plain": (no_min_fill,),
    "high-min-fill": (high_min_fill,),
    "no-stopovers": (no_min_fill, no_stopovers),
    "two-stops": (no_min_fill, two_stops),
    "any-direction": (no_min_fill, any_direction),
    "owned-without-stopovers": (no_min_fill, owned_without_stopovers),
    "spot-without-stopovers": (no_min_fill, spot_without_stopovers),
    "no-split": (no_min_fill, no_split),
    "spot-limits": (no_min_fill, spot_limits),
    "two-dimensions": (no_min_fill, two_dimensions),
    "ready-times": (no_min_fill, ready_times),
    "deferral": (no_min_fill, spot_limits, deferral),
    "handling-capacity": (no_min_fill, handling_capacity),
    "vehicle-limits": (no_min_fill, vehicle_limits),
    "late-and-tight": (late_and_tight,),
    "handling-time": (handling_time,),
}
"""Problems that put the rules to work: the changes each makes to the generated document."""

SETTING = {
    "deferral": "deferral",
    "handling": "handling-capacity",
    "vehicle_limit": "vehicle-limits",
    "min_fill": "generated",
    "split": "no-split",
}
"""For every rule that a solver may refuse, the scenario that uses the setting behind it."""

SOLVERS: dict[str, Solver] = {
    "aco": ACOSolver(ACOConfig(ants=6, iterations=10, patience=5)),
    "exact": ExactSolver(),
    "direct": DirectShipping(),
    "pyvrp": PyVRPSolver(iterations=60),
    "ortools": ORToolsSolver(),
}

SKIPS = {
    "pyvrp": pytest.mark.skipif(not HAS_PYVRP, reason="pyvrp is not installed"),
    "ortools": pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools is not installed"),
}

EXACT_REFUSES = {"no-split", "handling-time"}
"""Scenarios the exact solver does not plan: unsplittable demand, and handling that takes time."""

EVERY_RUN = [
    pytest.param(name, scenario, marks=SKIPS.get(name, ()), id=f"{name}-{scenario}")
    for name in SOLVERS
    for scenario in SCENARIOS
]

SCORED_RUNS = [
    pytest.param(name, scenario, id=f"{name}-{scenario}")
    for name, refused in (("aco", set()), ("exact", EXACT_REFUSES))
    for scenario in SCENARIOS
    if scenario not in refused
]
"""The colony and the exact solver report the score of their own cost model."""

REFUSALS = [
    pytest.param(name, rule, marks=SKIPS.get(name, ()), id=f"{name}-{rule}")
    for name, solver in SOLVERS.items()
    for rule, way in rule_support(solver).items()
    if way is Support.REFUSED
]


class Idle:
    """A solver that sends nothing."""

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        plan = Plan.from_dispatches([], len(instance.dimensions))
        evaluation = evaluate(instance, plan)
        return SolveResult(plan, evaluation, float(objective(evaluation)), 0.0, {})


class Refusing:
    """A solver that supports nothing."""

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        raise ValueError("not supported")


class Colony(ACOSolver):
    """A solver derived from a built-in one."""


IDLE, REFUSING = Idle(), Refusing()


@cache
def problem(scenario: str) -> tuple[LineHaulInstance, Objective]:
    """The instance and objective of ``scenario``: 6 centers, 2 days, 12 demands a day."""
    document = generate(
        centers=6, days=2, demand_per_day=12, owned_lines=2, seed=3, objective=CHEAP
    )
    for change in SCENARIOS[scenario]:
        change(document)
    spec = parse_spec(document)
    return spec.compile(), spec.build_objective()


@cache
def outcome(solver: Solver, scenario: str) -> SolveResult | None:
    """What ``solver`` returns for ``scenario``, or None when it refuses the problem."""
    instance, goal = problem(scenario)
    try:
        return solver.solve(instance, goal, seed=0)
    except ValueError:
        return None


def broken_rules(solver: Solver, scenario: str) -> set[str]:
    """The rules the plan of ``solver`` breaks on ``scenario``; none when it refuses the problem."""
    result = outcome(solver, scenario)
    if result is None:
        return set()
    return set(evaluate(problem(scenario)[0], result.plan).violated())


def promised_rules(solver: Solver) -> set[str]:
    """The rules ``solver`` is said never to break: every one it does not record as penalized."""
    return {rule for rule, way in rule_support(solver).items() if way is not Support.PENALIZED}


def test_a_plan_that_carries_nothing_breaks_coverage() -> None:
    """The helper's own test: an empty plan leaves every demand behind and breaks nothing else."""
    broken = broken_rules(IDLE, "plain")

    assert broken == {"coverage"}


def test_a_refused_problem_breaks_no_rule() -> None:
    """The helper's own test: without a plan there is nothing to break."""
    broken = broken_rules(REFUSING, "plain")

    assert broken == set()


def test_the_promised_rules_are_those_not_penalized() -> None:
    """The helper's own test: the adapters may break coverage only, which leaves 18 of 19."""
    promised = promised_rules(SOLVERS["pyvrp"])

    assert promised == set(CATALOG) - {"coverage"}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_is_a_valid_problem(scenario: str) -> None:
    """The changes of a scenario leave a document that passes validation: 2 days are compiled."""
    instance, _ = problem(scenario)

    assert len(instance.days) == 2


@pytest.mark.parametrize("kind", SUPPORT, ids=lambda kind: kind.__name__)
def test_the_record_names_every_rule_of_the_catalog(kind: type) -> None:
    """In catalog order, so a rule that is added must be entered for every solver."""
    assert list(SUPPORT[kind]) == list(CATALOG)


def test_the_record_covers_the_built_in_solvers() -> None:
    assert set(SUPPORT) == {ACOSolver, ExactSolver, DirectShipping, RoutingAdapter}


@pytest.mark.parametrize(
    ("solver", "kind"),
    [
        (ACOSolver(), ACOSolver),
        (ACOSolver, ACOSolver),
        (Colony(), ACOSolver),
        (PyVRPSolver(), RoutingAdapter),
        (ORToolsSolver, RoutingAdapter),
    ],
    ids=["instance", "class", "subclass", "pyvrp", "ortools"],
)
def test_rule_support_finds_the_record_of_a_solver(solver: Solver | type, kind: type) -> None:
    """An instance, a class and a subclass of a built-in solver all have its record."""
    assert rule_support(solver) is SUPPORT[kind]


def test_rule_support_knows_nothing_about_other_solvers() -> None:
    with pytest.raises(ValueError, match="no rule support is recorded for Idle; recorded: ACO"):
        rule_support(IDLE)


@pytest.mark.parametrize(
    ("kind", "rule", "way"),
    [
        (ACOSolver, "capacity", Support.KEPT),
        (ACOSolver, "min_fill", Support.PENALIZED),
        (ExactSolver, "split", Support.REFUSED),
        (DirectShipping, "vehicle_limit", Support.PENALIZED),
        (RoutingAdapter, "min_fill", Support.REFUSED),
    ],
    ids=["aco-capacity", "aco-min-fill", "exact-split", "direct-limit", "adapter-min-fill"],
)
def test_the_record_says_how_a_solver_deals_with_a_rule(
    kind: type, rule: str, way: Support
) -> None:
    """One entry of each kind, and entries that differ between the solvers."""
    assert SUPPORT[kind][rule] is way


@pytest.mark.parametrize(
    "name", [pytest.param(name, marks=SKIPS.get(name, ())) for name in SOLVERS]
)
def test_every_solver_plans_the_plain_problem(name: str) -> None:
    """Without minimum fill, times or limits no solver refuses, so the other tests check plans."""
    result = outcome(SOLVERS[name], "plain")

    assert isinstance(result, SolveResult)


@pytest.mark.parametrize(("name", "scenario"), EVERY_RUN)
def test_a_solver_never_breaks_a_rule_it_is_said_to_keep(name: str, scenario: str) -> None:
    """Breaking a rule is almost free under the objective, and still no promised rule is broken."""
    solver = SOLVERS[name]

    broken = broken_rules(solver, scenario)

    assert broken & promised_rules(solver) == set()


@pytest.mark.parametrize(("name", "rule"), REFUSALS)
def test_a_solver_refuses_a_problem_that_uses_a_refused_rule(name: str, rule: str) -> None:
    """The scenario differs from the plain problem, which every solver plans, in that setting."""
    instance, goal = problem(SETTING[rule])

    with pytest.raises(ValueError, match="does not support"):
        SOLVERS[name].solve(instance, goal, seed=0)


@pytest.mark.parametrize(("name", "scenario"), SCORED_RUNS)
def test_the_score_of_a_solver_is_the_objective_of_its_plan(name: str, scenario: str) -> None:
    """The solver's own model of costs and penalties agrees with the evaluator on its plan."""
    result = outcome(SOLVERS[name], scenario)

    assert isinstance(result, SolveResult)
    assert result.stats["score"] == pytest.approx(result.objective, rel=SCORE_TOLERANCE)


def test_the_guide_shows_the_record() -> None:
    """Every row of the guide's table: the rule, then the colony, exact, direct and adapters."""
    section = GUIDE.read_text(encoding="utf-8").split("### Which solver keeps which rule")[1]

    rows = re.findall(r"^\| `(\w+)` \| (\w+) \| (\w+) \| (\w+) \| (\w+) \|$", section, re.MULTILINE)

    assert rows == [
        (
            rule,
            SUPPORT[ACOSolver][rule],
            SUPPORT[ExactSolver][rule],
            SUPPORT[DirectShipping][rule],
            SUPPORT[RoutingAdapter][rule],
        )
        for rule in CATALOG
    ]
