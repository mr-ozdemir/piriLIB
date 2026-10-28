"""Tests of the adapters to the external routing solvers PyVRP and OR-Tools.

The requirement is the documentation of :mod:`pirilib.solvers.adapters` and
:mod:`pirilib.solvers.adapters.base`: the reduction keeps a set of rules by construction, the
plans are evaluated like any other plan, and every problem outside the restricted class is
refused with a :class:`ValueError` that names what is not supported. These are integration
tests (``tests/conftest.py`` marks them so): they run the real external package, which is the
risk they reduce, namely a plan from an external solver that breaks a rule or costs other than
pirilib says. A test for a package that is not installed is skipped with the reason.

Oracles: :func:`~pirilib.check_plan` is an independent checker of every rule and of the cost,
and :class:`~pirilib.solvers.exact.ExactSolver` gives the optimum that no plan may beat.

Blind spots, left out on purpose: problems larger than the random ones of
``tests/conftest.py``, time limits and metaheuristics of OR-Tools (they depend on the machine),
and the quality of the routes beyond the exact bound.
"""

import importlib.util
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pirilib import check_plan
from pirilib import objective as obj
from pirilib.constraints import MaxRouteKm
from pirilib.objective import Objective
from pirilib.solvers.adapters import ORToolsSolver, PyVRPSolver, RoutingAdapter
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import UNLIMITED, LineHaulInstance, parse_spec

type RandomDocument = Callable[[int], dict[str, Any]]

VIOLATION_WEIGHT = 1e6
"""Objective weight of one unit of violation, far above the cost of any of these small plans."""

GOAL = obj.total_cost() + VIOLATION_WEIGHT * obj.violations()

SEEDS = range(12)
"""Seeds of the random problems; with seeds 7, 8 and 10 the ``small`` spot offer is limited."""

EXACT_SEEDS = range(4)
"""Seeds compared with the exact solver, fewer because it takes seconds, not milliseconds."""

PYVRP_ITERATIONS = 100
"""Enough for these small problems; the rules hold for any number of iterations."""

COST_TOLERANCE = 1e-12
"""Relative: the checker and the evaluation add the same costs, only in another order."""

VIOLATION_RELATIVE = 1e-9
VIOLATION_ABSOLUTE = 1e-6
"""Tolerances of violation amounts, which the checker adds up dispatch by dispatch."""

BOUND_TOLERANCE = 1e-9
"""Relative slack below the exact optimum, for the rounding of its linear program."""

ROUTE_LIMIT_KM = 100
"""Limit of a constraint of your own; any such constraint is refused, whatever its limit."""

HAS_PYVRP = importlib.util.find_spec("pyvrp") is not None
HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None

BY_CONSTRUCTION = frozenset(
    {
        "route",
        "load_route",
        "ban",
        "owned_line",
        "owned_availability",
        "spot_availability",
        "capacity",
        "stopover",
        "max_stops",
        "stretch",
        "forward_only",
        "split",
        "overdelivery",
    }
)
"""The rules the reduction keeps by construction (see :mod:`pirilib.solvers.adapters.base`)."""

ADAPTERS = [
    pytest.param(
        lambda: PyVRPSolver(iterations=PYVRP_ITERATIONS),
        id="pyvrp",
        marks=pytest.mark.skipif(not HAS_PYVRP, reason="pyvrp is not installed"),
    ),
    pytest.param(
        ORToolsSolver,
        id="ortools",
        marks=pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools is not installed"),
    ),
]


def routable(document: dict[str, Any]) -> dict[str, Any]:
    """Make ``document`` one the adapters accept: no minimum fill, demand may be split."""
    document["rules"]["min_fill"] = {"spot": 0.0}
    document["rules"]["split_demand"] = True
    return document


def unlimited(document: dict[str, Any]) -> dict[str, Any]:
    """:func:`routable`, and the ``small`` spot offer (the only one that may be limited) is not."""
    document = routable(document)
    document["fleet"]["spot"][1].pop("available", None)
    return document


@pytest.mark.parametrize("seed", SEEDS)
def test_unlimited_leaves_no_spot_offer_limited(seed: int, random_document: RandomDocument) -> None:
    document = unlimited(random_document(seed))

    instance = parse_spec(document).compile()

    np.testing.assert_array_equal(instance.spot_available, UNLIMITED)


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_plans_keep_the_rules_of_the_reduction(
    make: Callable[[], RoutingAdapter], seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(routable(random_document(seed))).compile()

    result = make().solve(instance, GOAL, seed=seed)

    assert result.evaluation.violated().keys() & BY_CONSTRUCTION == set()


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_plans_are_feasible_when_spot_vehicles_are_unlimited(
    make: Callable[[], RoutingAdapter], seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(unlimited(random_document(seed))).compile()

    result = make().solve(instance, GOAL, seed=seed)

    assert result.evaluation.violated() == {}


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_plans_cost_what_the_checker_computes(
    make: Callable[[], RoutingAdapter], seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(routable(random_document(seed))).compile()

    result = make().solve(instance, GOAL, seed=seed)

    report = check_plan(instance, result.plan)
    assert result.evaluation.total_cost == pytest.approx(report.total_cost, rel=COST_TOLERANCE)


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize("seed", SEEDS)
def test_plans_violate_what_the_checker_finds(
    make: Callable[[], RoutingAdapter], seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(routable(random_document(seed))).compile()

    result = make().solve(instance, GOAL, seed=seed)

    report = check_plan(instance, result.plan)
    assert dict(result.evaluation.violations) == pytest.approx(
        report.totals(), rel=VIOLATION_RELATIVE, abs=VIOLATION_ABSOLUTE
    )


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize("seed", EXACT_SEEDS)
def test_plans_are_not_better_than_the_exact_optimum(
    make: Callable[[], RoutingAdapter], seed: int, random_document: RandomDocument
) -> None:
    instance = parse_spec(unlimited(random_document(seed))).compile()
    exact = ExactSolver().solve(instance, GOAL)

    routed = make().solve(instance, GOAL, seed=seed)

    assert routed.objective >= exact.objective * (1 - BOUND_TOLERANCE)


UNSUPPORTED = [
    pytest.param(
        {"min_fill": {"spot": 0.1}, "split_demand": False},
        GOAL,
        (),
        r"split_demand = false; minimum fill",
        id="unsplit-demand-and-minimum-fill",
    ),
    pytest.param(
        {"service": {"max_deferral_days": 1}},
        GOAL,
        (),
        "times, deferral, handling capacity or vehicle limits",
        id="scheduling-rules",
    ),
    pytest.param(
        {},
        lambda evaluation: evaluation.total_cost,
        (),
        "objectives without a linear form",
        id="objective-without-linear-form",
    ),
    pytest.param(
        {},
        GOAL,
        (MaxRouteKm(ROUTE_LIMIT_KM),),
        "constraints of your own",
        id="constraints-of-your-own",
    ),
    pytest.param(
        {},
        obj.total_cost() + obj.unused_capacity("spot"),
        (),
        r"objective terms \['unused\[spot\]'\]",
        id="unused-capacity-term",
    ),
    pytest.param(
        {},
        obj.total_cost() + -1.0 * obj.distance_km(),
        (),
        r"negative weights of \['km\[owned\]', 'km\[spot\]'\]",
        id="negative-weight",
    ),
]
"""Rules, objective, constraints and message of each kind of problem the adapters refuse.

The first case breaks two rules at once, so the message must name both, joined in order.
"""


@pytest.mark.parametrize("make", ADAPTERS)
@pytest.mark.parametrize(("rules", "objective", "constraints", "message"), UNSUPPORTED)
def test_unsupported_problems_are_refused(
    make: Callable[[], RoutingAdapter],
    rules: dict[str, Any],
    objective: Objective,
    constraints: tuple[MaxRouteKm, ...],
    message: str,
    document: dict[str, Any],
) -> None:
    document["rules"] = rules
    instance = parse_spec(document).compile()
    solver = make()

    with pytest.raises(ValueError, match=message):
        solver.solve(instance, objective, constraints=constraints)


@pytest.mark.parametrize("make", ADAPTERS)
def test_supported_problem_is_solved_without_violations(
    make: Callable[[], RoutingAdapter], instance: LineHaulInstance
) -> None:
    solver = make()

    result = solver.solve(instance, GOAL)

    assert result.evaluation.violated() == {}


@pytest.mark.parametrize(
    ("make", "module", "message"),
    [
        pytest.param(
            PyVRPSolver,
            "pyvrp",
            r'PyVRPSolver needs the pyvrp package: pip install "pirilib\[pyvrp\]"',
            id="pyvrp",
        ),
        pytest.param(
            ORToolsSolver,
            "ortools.constraint_solver",
            r'ORToolsSolver needs the ortools package: pip install "pirilib\[ortools\]"',
            id="ortools",
        ),
    ],
)
def test_missing_package_is_explained(
    make: Callable[[], RoutingAdapter],
    module: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    instance: LineHaulInstance,
) -> None:
    """A ``None`` entry in ``sys.modules`` makes importing that module fail as if not installed.

    The adapters import their package only when they solve and take no import function, so
    this is the one way to show a missing package; ``monkeypatch`` restores the entry.
    """
    monkeypatch.setitem(sys.modules, module, None)
    solver = make()

    with pytest.raises(ImportError, match=message):
        solver.solve(instance, GOAL)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        pytest.param(
            lambda: PyVRPSolver(iterations=0),
            "iterations must be at least 1",
            id="pyvrp-no-iterations",
        ),
        pytest.param(
            lambda: PyVRPSolver(time_limit=0.0),
            "time_limit must be positive",
            id="pyvrp-zero-time-limit",
        ),
        pytest.param(
            lambda: ORToolsSolver(time_limit=0.0),
            "time_limit must be positive",
            id="ortools-zero-time-limit",
        ),
        pytest.param(
            lambda: ORToolsSolver(metaheuristic="GUIDED_LOCAL_SEARCH"),
            "metaheuristic GUIDED_LOCAL_SEARCH needs a time_limit",
            id="ortools-metaheuristic-without-time-limit",
        ),
    ],
)
def test_invalid_settings_are_refused(make: Callable[[], RoutingAdapter], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        make()


SHORTEST_TIME_LIMIT = 0.001
"""Seconds; a limit just above zero, the other side of the refused limit of zero."""


@pytest.mark.parametrize(
    ("make", "setting", "value"),
    [
        pytest.param(lambda: PyVRPSolver(iterations=1), "iterations", 1, id="pyvrp-one-iteration"),
        pytest.param(
            lambda: PyVRPSolver(time_limit=SHORTEST_TIME_LIMIT),
            "time_limit",
            SHORTEST_TIME_LIMIT,
            id="pyvrp-short-time-limit",
        ),
        pytest.param(
            lambda: ORToolsSolver(
                metaheuristic="GUIDED_LOCAL_SEARCH", time_limit=SHORTEST_TIME_LIMIT
            ),
            "time_limit",
            SHORTEST_TIME_LIMIT,
            id="ortools-metaheuristic-with-time-limit",
        ),
    ],
)
def test_settings_at_the_limit_are_kept(
    make: Callable[[], RoutingAdapter], setting: str, value: float
) -> None:
    solver = make()

    assert getattr(solver, setting) == value


@pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools is not installed")
def test_unknown_ortools_search_setting_is_refused(instance: LineHaulInstance) -> None:
    solver = ORToolsSolver(first_solution="NO_SUCH_STRATEGY")

    with pytest.raises(ValueError, match="unknown OR-Tools search setting 'NO_SUCH_STRATEGY'"):
        solver.solve(instance, GOAL)
