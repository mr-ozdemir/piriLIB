"""Direct shipping: every line served on its own by vehicles that go straight to the destination.

The expected behavior comes from the module docstring of :mod:`pirilib.solvers.direct`, the
"Baselines" list of ``docs/guide.md`` ("full vehicles first, then the cheapest vehicle that fits.
It keeps bans, owned lines and spot limits"). The tests
compile documents in memory and observe the plan, its evaluation and the independent
:func:`pirilib.check_plan` through public names only.

Costs in the ``document`` fixture (see its docstring), with road distance equal to the haversine
distance on a sphere of radius 6371 km: ``A`` to ``B`` is 111.2 km and ``A`` to ``C`` 222.4 km.

======================  =========================  =========================
vehicle                 ``A`` to ``B``             ``A`` to ``C``
======================  =========================  =========================
owned ``big`` (1000)    100 + 1 x 111.2 = 211.2    its line is ``A`` to ``B``
spot ``big`` (1000)     300 + 2 x 111.2 = 522.4    banned at ``C``
spot ``small`` (400)    150 + 1.5 x 111.2 = 316.8  150 + 1.5 x 222.4 = 483.6
======================  =========================  =========================
"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pirilib import check_plan
from pirilib import objective as obj
from pirilib.plan import OWNED, SPOT, Plan
from pirilib.solvers.direct import DirectShipping
from pirilib.spec import LineHaulInstance, parse_spec

type RandomDocument = Callable[[int], dict[str, Any]]
type CompileDocument = Callable[[dict[str, Any]], LineHaulInstance]

A, B, C = 0, 1, 2
"""Positions of the centers ``A``, ``B`` and ``C`` of the ``document`` fixture."""

OWNED_GROUP = 0
"""Position of the only owned group of the ``document`` fixture (``big``, ``A`` to ``B``)."""

BIG_OFFER, SMALL_OFFER = 0, 1
"""Positions of the spot offers ``big`` and ``small``, in the fixture and in the tests' fleets."""

PENALTY_WEIGHT = 1e6
"""Weight of the violations in :data:`GOAL`, so that any violation outweighs any cost.

Direct shipping compares vehicles by dispatch cost whatever the objective; the objective only
scores the plan, so its weights do not change any expected value here.
"""

GOAL = obj.total_cost() + PENALTY_WEIGHT * obj.violations()

RANDOM_SEEDS = range(30)
"""Seeds of the random problems (``random_document`` in ``conftest.py``) the general checks use.

Thirty documents vary the centers, bans, owned offers, dimensions, split rule and stopover rules
while the tests stay fast.
"""

WHOLE_DEMAND_SEEDS = (2, 3, 4, 7, 10, 13, 14, 21, 28, 33, 35, 36, 38, 40, 42, 45, 56, 57)
"""The seeds below 60 whose random document forbids splitting demand.

``test_whole_demand_seed_draws_a_document_that_forbids_splitting`` checks this list against the
generator, so a change of the generator cannot turn these into split-demand documents unseen.
"""

UNLIMITED_SPOT: list[dict[str, Any]] = [
    {"type": "big", "cost": {"fixed": 900.0, "per_km": 3.0}},
    {"type": "small", "cost": {"fixed": 500.0, "per_km": 2.0}},
]
"""The spot offers of ``random_document`` (its docstring) without the optional ``small`` limit."""

COST_TOLERANCE = 1e-12
"""Relative tolerance between two computations of the same total cost (summation order only)."""

DECIMALS = 6
"""Load quantities are compared after rounding to this many decimals.

A split share is a float product such as ``2300 x 1000 / 2300``, which may differ from the exact
quantity in the last bit; every expected quantity here is a whole number of desi.
"""

FIRST_DAY = "2026-01-05"
"""The first day of the fixture's calendar, on which its owned vehicle is free."""


def relaxed(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` without minimum fill and with :data:`UNLIMITED_SPOT` as its spot offers.

    Without those two rules a vehicle that fits is always left, so the direct plan carries
    everything (every random item fits one vehicle when demand may not be split).
    """
    document["rules"]["min_fill"] = {"spot": 0.0}
    document["fleet"]["spot"] = copy.deepcopy(UNLIMITED_SPOT)
    return document


def line(origin: str, destination: str, desi: float) -> dict[str, Any]:
    """One demand of ``desi`` from ``origin`` to ``destination`` on :data:`FIRST_DAY`."""
    return {
        "origin": origin,
        "destination": destination,
        "date": FIRST_DAY,
        "quantity": {"desi": desi},
    }


def trips(plan: Plan) -> list[tuple[int, int, int, int]]:
    """Per dispatch ``(kind, source, origin, first stop)``, sorted; plan order is not compared."""
    destination = plan.stop_center[plan.stop_start[:-1]]
    return sorted(
        zip(
            plan.kind.tolist(),
            plan.source.tolist(),
            plan.origin.tolist(),
            destination.tolist(),
            strict=True,
        )
    )


def carried(plan: Plan) -> list[tuple[int, int, float]]:
    """Per load ``(kind, source, desi)`` of the vehicle carrying it, desi rounded, sorted."""
    return sorted(
        zip(
            plan.kind[plan.load_dispatch].tolist(),
            plan.source[plan.load_dispatch].tolist(),
            np.round(plan.load_quantity[:, 0], DECIMALS).tolist(),
            strict=True,
        )
    )


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_plan_without_min_fill_and_spot_limits_violates_nothing(
    seed: int, random_document: RandomDocument
) -> None:
    """Minimum fill and coverage are the only rules a direct plan can break (module docstring).

    Without minimum fill and spot limits a fitting vehicle is always left, so nothing is broken.
    """
    instance = parse_spec(relaxed(random_document(seed))).compile()

    result = DirectShipping().solve(instance, GOAL)

    assert result.evaluation.violated() == {}


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_plan_without_min_fill_and_spot_limits_passes_the_independent_checker(
    seed: int, random_document: RandomDocument
) -> None:
    """:func:`pirilib.check_plan` recomputes every rule in plain Python and finds none broken."""
    instance = parse_spec(relaxed(random_document(seed))).compile()
    plan = DirectShipping().solve(instance, GOAL).plan

    report = check_plan(instance, plan)

    assert report.feasible, str(report)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_reported_total_cost_matches_the_independent_checker(
    seed: int, random_document: RandomDocument
) -> None:
    """The evaluation returned with the plan costs it as :func:`pirilib.check_plan` does."""
    instance = parse_spec(relaxed(random_document(seed))).compile()
    result = DirectShipping().solve(instance, GOAL)

    report = check_plan(instance, result.plan)

    assert result.evaluation.total_cost == pytest.approx(report.total_cost, rel=COST_TOLERANCE)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_every_dispatch_goes_straight_to_one_destination(
    seed: int, random_document: RandomDocument
) -> None:
    """Nothing is combined on multi-stop routes, even when the document enables stopovers."""
    instance = parse_spec(relaxed(random_document(seed))).compile()

    plan = DirectShipping().solve(instance, GOAL).plan

    assert set(np.diff(plan.stop_start).tolist()) == {1}


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_result_reports_the_direct_mode(seed: int, random_document: RandomDocument) -> None:
    """``stats["mode"]`` names the solver that made the plan (``docs/guide.md``)."""
    instance = parse_spec(relaxed(random_document(seed))).compile()

    result = DirectShipping().solve(instance, GOAL)

    assert result.stats["mode"] == "direct"


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_plan_breaks_only_min_fill_and_coverage(seed: int, random_document: RandomDocument) -> None:
    """With every rule of the random document in force, only these two may be violated.

    Module docstring: "minimum fill and coverage are the only rules a direct plan can break".
    """
    instance = parse_spec(random_document(seed)).compile()

    violated = DirectShipping().solve(instance, GOAL).evaluation.violated()

    assert set(violated) - {"min_fill", "coverage"} == set()


@pytest.mark.parametrize("seed", WHOLE_DEMAND_SEEDS)
def test_whole_demand_seed_draws_a_document_that_forbids_splitting(
    seed: int, random_document: RandomDocument
) -> None:
    """Guard for :data:`WHOLE_DEMAND_SEEDS`: the generator still draws ``split_demand: false``."""
    document = random_document(seed)

    assert document["rules"]["split_demand"] is False


@pytest.mark.parametrize("seed", WHOLE_DEMAND_SEEDS)
def test_each_load_carries_one_whole_demand(seed: int, random_document: RandomDocument) -> None:
    """Without split demand every load holds the full quantity of its demand, per dimension."""
    instance = parse_spec(relaxed(random_document(seed))).compile()

    plan = DirectShipping().solve(instance, GOAL).plan

    np.testing.assert_array_equal(plan.load_quantity, instance.demand_quantity[plan.load_demand])


@pytest.mark.parametrize("seed", WHOLE_DEMAND_SEEDS)
def test_each_demand_is_carried_by_exactly_one_load(
    seed: int, random_document: RandomDocument
) -> None:
    """Without split demand no demand is spread over two loads and, unlimited, none is left."""
    instance = parse_spec(relaxed(random_document(seed))).compile()

    plan = DirectShipping().solve(instance, GOAL).plan

    np.testing.assert_array_equal(
        np.sort(plan.load_demand), np.arange(len(instance.demand_quantity))
    )


@pytest.mark.parametrize("seed", WHOLE_DEMAND_SEEDS)
def test_whole_demand_plan_violates_nothing(seed: int, random_document: RandomDocument) -> None:
    """Every random item fits one vehicle (``random_document``), so a whole-demand plan is clean."""
    instance = parse_spec(relaxed(random_document(seed))).compile()

    result = DirectShipping().solve(instance, GOAL)

    assert result.evaluation.violated() == {}


@pytest.mark.parametrize("seed", WHOLE_DEMAND_SEEDS)
def test_whole_demand_plan_passes_the_independent_checker(
    seed: int, random_document: RandomDocument
) -> None:
    """:func:`pirilib.check_plan` agrees that the whole-demand plan breaks no rule."""
    instance = parse_spec(relaxed(random_document(seed))).compile()
    plan = DirectShipping().solve(instance, GOAL).plan

    report = check_plan(instance, plan)

    assert report.feasible, str(report)


def test_fixture_lines_get_the_cheapest_allowed_vehicle(instance: LineHaulInstance) -> None:
    """``A`` to ``B`` 900 desi: the owned ``big`` (211.2) on its own line.

    ``A`` to ``C`` 250 desi: ``small`` (``big`` is banned at ``C``). ``B`` to ``A`` 120 desi:
    ``small`` (316.8 against 522.4); the owned vehicle may not leave its line.
    """
    plan = DirectShipping().solve(instance, GOAL).plan

    assert trips(plan) == [
        (OWNED, OWNED_GROUP, A, B),
        (SPOT, SMALL_OFFER, A, C),
        (SPOT, SMALL_OFFER, B, A),
    ]


def test_fixture_plan_violates_nothing(instance: LineHaulInstance) -> None:
    """Every fixture line has an allowed vehicle with capacity left, and no fill rule is set."""
    result = DirectShipping().solve(instance, GOAL)

    assert result.evaluation.violated() == {}


def test_center_that_bans_a_type_is_served_by_another_type(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """600 desi to ``C``: one spot ``big`` would fit, but ``C`` bans it.

    So ``small`` carries a full 400 and then the 200 left.
    """
    document["demand"] = [line("A", "C", 600.0)]
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert carried(plan) == [(SPOT, SMALL_OFFER, 200.0), (SPOT, SMALL_OFFER, 400.0)]


def test_owned_group_serves_only_its_own_line(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """With the ban on ``C`` lifted, the owned ``big`` would be cheapest to ``C`` (322.4).

    Its line is ``A`` to ``B``, so 300 desi to ``C`` go by spot ``small`` (483.6, below 744.8).
    """
    document["centers"][C]["banned_vehicle_types"] = []
    document["demand"] = [line("A", "C", 300.0)]
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert trips(plan) == [(SPOT, SMALL_OFFER, A, C)]


@pytest.mark.parametrize(
    ("desi", "expected"),
    [
        pytest.param(
            2300.0,
            [(OWNED, OWNED_GROUP, 1000.0), (SPOT, BIG_OFFER, 1000.0), (SPOT, SMALL_OFFER, 300.0)],
            id="rest-below-small-capacity",
        ),
        pytest.param(
            2400.0,
            [(OWNED, OWNED_GROUP, 1000.0), (SPOT, BIG_OFFER, 1000.0), (SPOT, SMALL_OFFER, 400.0)],
            id="rest-equal-to-small-capacity",
        ),
        pytest.param(
            2401.0,
            [(OWNED, OWNED_GROUP, 1000.0), (SPOT, BIG_OFFER, 401.0), (SPOT, BIG_OFFER, 1000.0)],
            id="rest-above-small-capacity",
        ),
    ],
)
def test_split_line_uses_full_vehicles_then_the_cheapest_that_fits_the_rest(
    desi: float,
    expected: list[tuple[int, int, float]],
    document: dict[str, Any],
    compile_document: CompileDocument,
) -> None:
    """``A`` to ``B`` above two ``big`` loads: full vehicles at the lowest cost per carried share.

    The owned ``big`` (211.2 per 1000) comes first, then a spot ``big`` (522.4 per 1000) rather
    than ``small`` (316.8 per 400). The rest goes into the cheapest vehicle that fits it:
    ``small`` up to its 400 desi, ``big`` above.
    """
    document["demand"] = [line("A", "B", desi)]
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert carried(plan) == expected


@pytest.mark.parametrize("desi", [2300.0, 2400.0, 2401.0])
def test_split_line_plan_passes_the_independent_checker(
    desi: float, document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """The split plans of the previous test break no rule in :func:`pirilib.check_plan`."""
    document["demand"] = [line("A", "B", desi)]
    instance = compile_document(document)
    plan = DirectShipping().solve(instance, GOAL).plan

    report = check_plan(instance, plan)

    assert report.feasible, str(report)


def test_whole_demands_are_packed_first_fit_decreasing_then_shrunk_to_the_cheapest_fit(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Demands of 300, 300 and 600 desi from ``A`` to ``B`` that may not be split.

    Largest first: 600 opens the owned ``big`` (lowest cost per capacity) and one 300 joins it;
    the other 300 opens a spot ``big`` (522.4 per 1000 against 316.8 per 400), which is then
    replaced by the cheapest type that fits its load, ``small``. In the given order the two 300s
    would share the owned vehicle and 600 would need a spot ``big``.
    """
    document["rules"] = {"split_demand": False}
    document["demand"] = [line("A", "B", 300.0), line("A", "B", 300.0), line("A", "B", 600.0)]
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert carried(plan) == [
        (OWNED, OWNED_GROUP, 300.0),
        (OWNED, OWNED_GROUP, 600.0),
        (SPOT, SMALL_OFFER, 300.0),
    ]


def test_new_vehicle_for_a_whole_demand_has_the_lowest_cost_per_capacity(
    document: dict[str, Any], compile_document: CompileDocument
) -> None:
    """Demands of 300, 300, 300 and 600 desi from ``A`` to ``B`` that may not be split.

    600 and one 300 fill the owned ``big`` to 900. The next 300 opens a spot ``big`` (522.4 per
    1000) rather than the cheaper ``small`` (316.8 per 400), so the last 300 joins it; with 600
    desi on board no smaller type fits. Opening ``small`` would need two of them.
    """
    document["rules"] = {"split_demand": False}
    document["demand"] = [
        line("A", "B", 300.0),
        line("A", "B", 300.0),
        line("A", "B", 300.0),
        line("A", "B", 600.0),
    ]
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert carried(plan) == [
        (OWNED, OWNED_GROUP, 300.0),
        (OWNED, OWNED_GROUP, 600.0),
        (SPOT, BIG_OFFER, 300.0),
        (SPOT, BIG_OFFER, 300.0),
    ]


@pytest.mark.parametrize(
    ("desi", "expected"),
    [
        pytest.param(400.0, [(SPOT, SMALL_OFFER, 400.0)], id="equal-to-small-capacity"),
        pytest.param(401.0, [(SPOT, BIG_OFFER, 401.0)], id="above-small-capacity"),
    ],
)
def test_whole_demand_fits_a_vehicle_up_to_its_exact_capacity(
    desi: float,
    expected: list[tuple[int, int, float]],
    document: dict[str, Any],
    compile_document: CompileDocument,
) -> None:
    """One whole demand from ``B`` to ``A``, off the owned line, so only spot vehicles serve it.

    ``small`` (316.8) is cheaper than ``big`` (522.4) and holds 400 desi: a load equal to its
    capacity fits it, one desi more needs ``big``. (A whole demand that no vehicle type can
    carry is rejected when the document is validated, so it never reaches the solver.)
    """
    document["rules"] = {"split_demand": False}
    document["demand"] = [line("B", "A", desi)]
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert carried(plan) == expected


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        pytest.param(2, {"coverage": 300.0}, id="limit-below-lines"),
        pytest.param(3, {}, id="limit-equal-to-lines"),
    ],
)
def test_spot_limit_leaves_the_demand_beyond_it_uncarried(
    available: int,
    expected: dict[str, float],
    document: dict[str, Any],
    compile_document: CompileDocument,
) -> None:
    """Three 300 desi lines on one day and only ``available`` spot ``small`` vehicles.

    The limit is kept (no ``spot_availability`` violation); a line beyond it stays uncarried and
    shows as 300 desi of ``coverage``. The fixed cost of 1 is irrelevant: it is the only offer.
    """
    document["fleet"] = {
        "spot": [{"type": "small", "cost": {"fixed": 1.0}, "available": available}]
    }
    document["demand"] = [line("A", "C", 300.0), line("B", "C", 300.0), line("C", "A", 300.0)]
    instance = compile_document(document)

    violated = DirectShipping().solve(instance, GOAL).evaluation.violated()

    assert violated == expected


FILL_FLEET: dict[str, Any] = {
    "spot": [
        {"type": "big", "cost": {"fixed": 100.0}},
        {"type": "small", "cost": {"fixed": 150.0}},
    ]
}
"""Spot offers where ``big`` (1000 desi, 100) is cheaper than ``small`` (400 desi, 150)."""

FILL_DESI = 350.0
"""One ``A`` to ``B`` demand: 35 % of a ``big`` vehicle and 87.5 % of a ``small`` one."""


@pytest.mark.parametrize(
    ("min_fill", "expected_offer"),
    [
        pytest.param(0.35, BIG_OFFER, id="big-reaches-it-exactly"),
        pytest.param(0.36, SMALL_OFFER, id="big-just-below-it"),
        pytest.param(0.5, SMALL_OFFER, id="only-small-reaches-it"),
        pytest.param(0.95, BIG_OFFER, id="none-reaches-it"),
    ],
)
def test_prefers_the_cheapest_vehicle_that_reaches_its_minimum_fill(
    min_fill: float,
    expected_offer: int,
    document: dict[str, Any],
    compile_document: CompileDocument,
) -> None:
    """:data:`FILL_DESI` with spot minimum fill ``min_fill`` and :data:`FILL_FLEET`.

    ``big`` reaches the fill up to 0.35 and ``small`` up to 0.875. The cheapest vehicle that
    reaches it wins; when none does, the cheapest that fits (``big``).
    """
    document["fleet"] = copy.deepcopy(FILL_FLEET)
    document["demand"] = [line("A", "B", FILL_DESI)]
    document["rules"] = {"min_fill": {"spot": min_fill}}
    instance = compile_document(document)

    plan = DirectShipping().solve(instance, GOAL).plan

    assert plan.source.tolist() == [expected_offer]


@pytest.mark.parametrize(
    ("min_fill", "expected"),
    [
        pytest.param(0.35, {}, id="big-reaches-it-exactly"),
        pytest.param(0.5, {}, id="only-small-reaches-it"),
        pytest.param(0.95, {"min_fill": pytest.approx(600.0)}, id="none-reaches-it"),
    ],
)
def test_minimum_fill_shortfall_is_reported_only_when_no_vehicle_reaches_it(
    min_fill: float,
    expected: dict[str, Any],
    document: dict[str, Any],
    compile_document: CompileDocument,
) -> None:
    """The same problem as above; the demand is carried even below the fill.

    With 0.95 the ``big`` vehicle carries 350 of the 950 desi it needs: 600 desi short.
    """
    document["fleet"] = copy.deepcopy(FILL_FLEET)
    document["demand"] = [line("A", "B", FILL_DESI)]
    document["rules"] = {"min_fill": {"spot": min_fill}}
    instance = compile_document(document)

    violated = DirectShipping().solve(instance, GOAL).evaluation.violated()

    assert violated == expected
