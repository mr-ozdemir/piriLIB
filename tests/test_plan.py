"""Plans in array form and in dispatch form (``pirilib.plan``).

Level: unit. The tests use the public names :class:`~pirilib.Plan`,
:class:`~pirilib.Dispatch`, :class:`~pirilib.Load`, :class:`~pirilib.PlanError` and the kind
codes :data:`pirilib.plan.OWNED` and :data:`pirilib.plan.SPOT`; only the index checks also use
the compiled ``instance`` fixture (``tests/conftest.py``).

Risk this lowers: an engine or a hand-built plan that is laid out wrongly (stops, loads or
vehicle kinds shifted to another dispatch) is evaluated as a different plan without any error;
a malformed plan, or one whose indices do not exist in the instance, must be refused with every
problem named instead of failing later inside the evaluator.

Requirement sources: the docstrings of ``Plan`` (array layout, dtypes, read-only arrays,
``start`` NaN by default), ``Dispatch`` (``source`` depends on ``kind``, ``start`` may be more
than a day, ``None`` means as soon as ready), ``Plan.from_dispatches`` (every load has
``dimensions`` quantities), ``Plan.check`` and the kind codes. The problem lines are pirilib's
own wording, pinned in full because the command line prints them to the user: a header line
(``malformed plan`` or ``plan does not fit the instance``), then one ``- <problem>`` line each.
Gray-box knowledge, named where it is used: a problem lists at most five positions and counts
the rest (``pirilib/plan.py``).

Oracles: arrays, dispatches and messages written by hand from the inputs below; and the round
trip property that the dispatches read back from a plan are the dispatches it was built from.

Input classes: valid plans (several dispatches, one dispatch, no dispatch, a dispatch without
stops or loads); malformed arrays (wrong number of dimensions, wrong length, ``stop_start``
not rising from 0 to the number of stops, negative or non-finite quantities, unknown kind
codes, negative or infinite start minutes) with the accepted values next to each limit; bad
dispatches (unknown kind, too few or too many quantities); and indices just outside and just
inside every range of the instance.

Blind spots: interpreter and NumPy versions are not varied (the exact dtypes are checked, so a
changed conversion shows); there is no concurrency (plans are immutable); speed and memory of
large plans belong to the performance tests; floating point is compared exactly, which holds
because every quantity here is exactly representable and NaN is compared with
``numpy.testing.assert_array_equal``, which treats NaN as equal; reading plan files and
evaluating plans are tested in ``tests/test_planio.py`` and ``tests/test_evaluation.py``; and
the expected values follow the docstrings, so a misunderstood requirement there is not found.
"""

import dataclasses
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pirilib import Dispatch, Load, Plan, PlanError
from pirilib.plan import OWNED, SPOT
from pirilib.spec import LineHaulInstance

DIMENSIONS = 2
"""Quantity dimensions of :data:`DISPATCHES` (desi and kg), so that rows and columns differ."""

DISPATCHES = [
    Dispatch(
        day=0,
        kind="owned",
        source=0,
        origin=2,
        stops=(1,),
        loads=(Load(0, (900.0, 600.0)),),
    ),
    Dispatch(
        day=0,
        kind="spot",
        source=1,
        origin=0,
        stops=(1, 2),
        loads=(Load(1, (250.0, 80.0)), Load(2, (120.0, 40.0))),
        start=480.0,
    ),
    Dispatch(day=1, kind="spot", source=0, origin=1, stops=(), start=1500.0),
]
"""Three dispatches whose fields differ from array to array, so a mixed-up array shows.

An owned vehicle with one stop and one load that starts as soon as its load is ready; a spot
vehicle with two stops and two loads that starts at 08:00 (480 minutes); and a spot vehicle
without stops or loads on the second day that starts at 01:00 of the next day (1500 minutes,
more than a day, which ``Dispatch`` allows). The indices are not checked against an instance
here, and the quantities only need to be distinct.
"""

ARRAYS_OF_DISPATCHES = [
    ("day", [0, 0, 1]),
    ("kind", [0, 1, 1]),
    ("source", [0, 1, 0]),
    ("origin", [2, 0, 1]),
    ("stop_start", [0, 1, 3, 3]),
    ("stop_center", [1, 1, 2]),
    ("load_dispatch", [0, 1, 1]),
    ("load_demand", [0, 1, 2]),
    ("load_quantity", [[900.0, 600.0], [250.0, 80.0], [120.0, 40.0]]),
    ("start", [np.nan, 480.0, 1500.0]),
]
"""The array form of :data:`DISPATCHES`, written out from the layout in the ``Plan`` docstring."""

DTYPES = [
    ("day", np.int64),
    ("kind", np.int8),
    ("source", np.int64),
    ("origin", np.int64),
    ("stop_start", np.int64),
    ("stop_center", np.int64),
    ("load_dispatch", np.int64),
    ("load_demand", np.int64),
    ("load_quantity", np.float64),
    ("start", np.float64),
]
"""Every array of a plan with the dtype the ``Plan`` annotations give it."""

ARRAY_NAMES = [
    "day",
    "kind",
    "source",
    "origin",
    "stop_start",
    "stop_center",
    "load_dispatch",
    "load_demand",
    "load_quantity",
    "start",
]
"""The names of every array of a plan."""

ONE_DISPATCH: dict[str, Any] = {
    "day": [0],
    "kind": [1],
    "source": [0],
    "origin": [0],
    "stop_start": [0, 1],
    "stop_center": [2],
    "load_dispatch": [0],
    "load_demand": [1],
    "load_quantity": [[250.0]],
}
"""Valid arrays of one spot dispatch with one stop and one load, as plain Python lists.

``start`` is left out, so it takes its default. Each malformed case below changes only the
arrays it names, so the plan has exactly the one problem the case is about.
"""

TWO_DISPATCHES: dict[str, Any] = {"day": [0, 0], "kind": [1, 1], "source": [0, 0], "origin": [0, 0]}
"""The per-dispatch arrays of two spot dispatches, for cases that need a second dispatch."""

MALFORMED = "malformed plan"
"""First line of the message of a PlanError for a malformed plan."""

UNFIT = "plan does not fit the instance"
"""First line of the message of a PlanError raised by ``Plan.check``."""

NOT_RISING = "stop_start must rise from 0 to the number of stops (1) without decreasing"
"""Problem of a ``stop_start`` that does not run from 0 to the one stop of :data:`ONE_DISPATCH`."""

BAD_QUANTITY = "load quantities must be finite and not negative; see loads 0"
"""Problem of :data:`ONE_DISPATCH` when its one load has a bad quantity."""

BAD_KIND = "kind must be 0 (owned) or 1 (spot); see dispatches 0"
"""Problem of :data:`ONE_DISPATCH` when its kind code is neither OWNED nor SPOT."""

BAD_START = "start must be NaN or finite and not negative; see dispatches 0"
"""Problem of :data:`ONE_DISPATCH` when its start is negative or infinite."""

SHOWN = 5
"""Positions a problem lists before it counts the rest (gray box: ``pirilib/plan.py``)."""

MALFORMED_ARRAYS = [
    pytest.param({"day": [[0]]}, "day must be one-dimensional", id="day-2d"),
    pytest.param({"kind": [[1]]}, "kind must be one-dimensional", id="kind-2d"),
    pytest.param({"source": [[0]]}, "source must be one-dimensional", id="source-2d"),
    pytest.param({"origin": [[0]]}, "origin must be one-dimensional", id="origin-2d"),
    pytest.param({"stop_start": [[0, 1]]}, "stop_start must be one-dimensional", id="stops-2d"),
    pytest.param({"stop_center": [[2]]}, "stop_center must be one-dimensional", id="centers-2d"),
    pytest.param({"kind": [1, 1]}, "kind has 2 entries, day has 1", id="kind-length"),
    pytest.param({"source": []}, "source has 0 entries, day has 1", id="source-length"),
    pytest.param({"origin": [0, 0]}, "origin has 2 entries, day has 1", id="origin-length"),
    pytest.param({"start": [0.0, 0.0]}, "start has 2 entries, day has 1", id="start-length"),
    pytest.param(
        {"stop_start": [0, 1, 1]}, "stop_start has 3 entries, expected 2", id="stops-length"
    ),
    pytest.param({"stop_start": [1, 1]}, NOT_RISING, id="stops-not-from-zero"),
    pytest.param({"stop_start": [0, 0]}, NOT_RISING, id="stops-short-of-the-end"),
    pytest.param({"stop_start": [0, 2]}, NOT_RISING, id="stops-past-the-end"),
    pytest.param({**TWO_DISPATCHES, "stop_start": [0, 2, 1]}, NOT_RISING, id="stops-decreasing"),
    pytest.param(
        {"load_quantity": [250.0]},
        "load_quantity must be two-dimensional (loads, dimensions)",
        id="quantity-1d",
    ),
    pytest.param(
        {"load_dispatch": [0, 0]},
        "load_dispatch must have one entry per load (1)",
        id="load-dispatch-length",
    ),
    pytest.param(
        {"load_dispatch": [[0]]},
        "load_dispatch must have one entry per load (1)",
        id="load-dispatch-2d",
    ),
    pytest.param(
        {"load_demand": []}, "load_demand must have one entry per load (1)", id="demand-length"
    ),
    pytest.param({"load_quantity": [[-1.0]]}, BAD_QUANTITY, id="quantity-negative"),
    pytest.param({"load_quantity": [[np.nan]]}, BAD_QUANTITY, id="quantity-nan"),
    pytest.param({"load_quantity": [[np.inf]]}, BAD_QUANTITY, id="quantity-infinite"),
    pytest.param({"load_quantity": [[250.0, -1.0]]}, BAD_QUANTITY, id="quantity-second-bad"),
    pytest.param({"kind": [2]}, BAD_KIND, id="kind-above-spot"),
    pytest.param({"kind": [-1]}, BAD_KIND, id="kind-below-owned"),
    pytest.param({"start": [-1.0]}, BAD_START, id="start-negative"),
    pytest.param({"start": [np.inf]}, BAD_START, id="start-infinite"),
    pytest.param({"start": [-np.inf]}, BAD_START, id="start-minus-infinite"),
    pytest.param(
        {
            "load_dispatch": [0] * SHOWN,
            "load_demand": [1] * SHOWN,
            "load_quantity": [[-1.0]] * SHOWN,
        },
        "load quantities must be finite and not negative; see loads 0, 1, 2, 3, 4",
        id="five-bad-loads-all-listed",
    ),
    pytest.param(
        {
            "load_dispatch": [0] * (SHOWN + 1),
            "load_demand": [1] * (SHOWN + 1),
            "load_quantity": [[-1.0]] * (SHOWN + 1),
        },
        "load quantities must be finite and not negative; see loads 0, 1, 2, 3, 4 and 1 more",
        id="six-bad-loads-one-counted",
    ),
]
"""Arrays that change :data:`ONE_DISPATCH` in one way, each with the one problem it has."""

ACCEPTED_ARRAYS = [
    pytest.param("load_quantity", [[0.0]], id="quantity-zero"),
    pytest.param("kind", [OWNED], id="kind-owned"),
    pytest.param("start", [0.0], id="start-midnight"),
    pytest.param("start", [1500.0], id="start-after-a-day"),
    pytest.param("start", [np.nan], id="start-nan"),
]
"""Values at the accepted side of a limit of :data:`MALFORMED_ARRAYS`, one array at a time."""

FIRST_OWNED = Dispatch(
    day=0, kind="owned", source=0, origin=0, stops=(0,), loads=(Load(0, (1.0,)),)
)
"""An owned dispatch whose every index is the first one of its range in the ``instance``."""

FIRST_SPOT = dataclasses.replace(FIRST_OWNED, kind="spot")
"""A spot dispatch whose every index is the first one of its range in the ``instance``."""

LAST_OWNED = Dispatch(day=1, kind="owned", source=0, origin=2, stops=(2,), loads=(Load(2, (1.0,)),))
"""An owned dispatch whose every index is the last one of its range in the ``instance``.

The ``instance`` fixture has 2 days, 3 centers, 1 owned vehicle group, 2 spot offers, 3
demands and 1 quantity dimension (see the ``document`` fixture in ``tests/conftest.py``).
"""

LAST_SPOT = dataclasses.replace(LAST_OWNED, kind="spot", source=1)
"""A spot dispatch whose every index is the last one; its source is past the owned groups."""

OUTSIDE = [
    pytest.param(
        dataclasses.replace(LAST_OWNED, day=2),
        "day index outside 0..1; see dispatches 0",
        id="day-after-the-last",
    ),
    pytest.param(
        dataclasses.replace(FIRST_OWNED, day=-1),
        "day index outside 0..1; see dispatches 0",
        id="day-before-the-first",
    ),
    pytest.param(
        dataclasses.replace(LAST_OWNED, origin=3),
        "origin index outside 0..2; see dispatches 0",
        id="origin-after-the-last",
    ),
    pytest.param(
        dataclasses.replace(FIRST_OWNED, origin=-1),
        "origin index outside 0..2; see dispatches 0",
        id="origin-before-the-first",
    ),
    pytest.param(
        dataclasses.replace(LAST_OWNED, stops=(3,)),
        "stop index outside 0..2; see stops 0",
        id="stop-after-the-last",
    ),
    pytest.param(
        dataclasses.replace(FIRST_OWNED, stops=(-1,)),
        "stop index outside 0..2; see stops 0",
        id="stop-before-the-first",
    ),
    pytest.param(
        dataclasses.replace(LAST_OWNED, source=1),
        "owned group index outside 0..0; see dispatches 0",
        id="owned-after-the-last",
    ),
    pytest.param(
        dataclasses.replace(FIRST_OWNED, source=-1),
        "owned group index outside 0..0; see dispatches 0",
        id="owned-before-the-first",
    ),
    pytest.param(
        dataclasses.replace(LAST_SPOT, source=2),
        "spot offer index outside 0..1; see dispatches 0",
        id="spot-after-the-last",
    ),
    pytest.param(
        dataclasses.replace(FIRST_SPOT, source=-1),
        "spot offer index outside 0..1; see dispatches 0",
        id="spot-before-the-first",
    ),
    pytest.param(
        dataclasses.replace(LAST_OWNED, loads=(Load(3, (1.0,)),)),
        "load demand index outside 0..2; see loads 0",
        id="demand-after-the-last",
    ),
    pytest.param(
        dataclasses.replace(FIRST_OWNED, loads=(Load(-1, (1.0,)),)),
        "load demand index outside 0..2; see loads 0",
        id="demand-before-the-first",
    ),
]
"""A dispatch with one index just outside its range in the ``instance``, and the problem."""

INDEX = st.integers(min_value=0, max_value=50)
"""Any small index; ``from_dispatches`` does not compare indices with an instance."""

QUANTITY = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
"""A finite, non-negative quantity of one dimension."""

GENERATED_DISPATCH = st.builds(
    Dispatch,
    day=INDEX,
    kind=st.sampled_from(["owned", "spot"]),
    source=INDEX,
    origin=INDEX,
    stops=st.lists(INDEX, max_size=3).map(tuple),
    loads=st.lists(
        st.builds(Load, demand=INDEX, quantity=st.tuples(QUANTITY, QUANTITY)), max_size=3
    ).map(tuple),
    start=st.none() | st.floats(min_value=0.0, max_value=2880.0, allow_nan=False),
)
"""A valid dispatch with :data:`DIMENSIONS` quantities per load.

Up to 3 stops and 3 loads; the start is ``None`` or up to two days (2880 minutes).
"""


@pytest.fixture
def plan() -> Plan:
    """The plan built from :data:`DISPATCHES`."""
    return Plan.from_dispatches(DISPATCHES, dimensions=DIMENSIONS)


def test_kind_codes_are_zero_for_owned_and_one_for_spot() -> None:
    kinds = (OWNED, SPOT)

    assert kinds == (0, 1)


@pytest.mark.parametrize(("name", "expected"), ARRAYS_OF_DISPATCHES)
def test_from_dispatches_lays_out_each_array(plan: Plan, name: str, expected: list[Any]) -> None:
    array = getattr(plan, name)

    np.testing.assert_array_equal(array, expected)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        pytest.param(0, [1], id="one-stop"),
        pytest.param(1, [1, 2], id="two-stops"),
        pytest.param(2, [], id="no-stop"),
    ],
)
def test_stops_are_those_of_one_dispatch_in_visiting_order(
    plan: Plan, position: int, expected: list[int]
) -> None:
    stops = plan.stops(position)

    assert stops.tolist() == expected


def test_length_is_the_number_of_dispatches(plan: Plan) -> None:
    length = len(plan)

    assert length == 3


def test_dimensions_is_the_number_of_quantities_per_load(plan: Plan) -> None:
    dimensions = plan.dimensions

    assert dimensions == DIMENSIONS


def test_repr_counts_dispatches_stops_loads_and_dimensions(plan: Plan) -> None:
    text = repr(plan)

    assert text == "Plan(dispatches=3, stops=3, loads=3, dimensions=2)"


def test_dispatches_are_read_back_in_plan_order(plan: Plan) -> None:
    dispatches = list(plan.dispatches())

    assert dispatches == DISPATCHES


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        pytest.param(0, DISPATCHES[0], id="owned-one-load"),
        pytest.param(1, DISPATCHES[1], id="spot-two-loads"),
        pytest.param(2, DISPATCHES[2], id="spot-empty"),
    ],
)
def test_dispatch_is_read_back_at_its_position(
    plan: Plan, position: int, expected: Dispatch
) -> None:
    dispatch = plan.dispatch(position)

    assert dispatch == expected


def test_from_dispatches_accepts_any_iterable() -> None:
    dispatches = iter(DISPATCHES)

    plan = Plan.from_dispatches(dispatches, dimensions=DIMENSIONS)

    assert list(plan.dispatches()) == DISPATCHES


@given(dispatches=st.lists(GENERATED_DISPATCH, max_size=5))
def test_any_valid_dispatches_are_read_back_unchanged(dispatches: list[Dispatch]) -> None:
    plan = Plan.from_dispatches(dispatches, dimensions=DIMENSIONS)

    assert list(plan.dispatches()) == dispatches


def test_empty_plan_has_no_dispatches() -> None:
    plan = Plan.empty(dimensions=DIMENSIONS)

    assert list(plan.dispatches()) == []


def test_empty_plan_keeps_the_quantity_dimensions() -> None:
    plan = Plan.empty(dimensions=DIMENSIONS)

    assert plan.load_quantity.shape == (0, DIMENSIONS)


def test_empty_plan_has_one_stop_offset_of_zero() -> None:
    plan = Plan.empty(dimensions=DIMENSIONS)

    assert plan.stop_start.tolist() == [0]


@pytest.mark.parametrize(("name", "dtype"), DTYPES)
def test_arrays_are_converted_to_their_dtypes(name: str, dtype: type) -> None:
    arrays = {**ONE_DISPATCH, "start": [0]}

    plan = Plan(**arrays)

    assert getattr(plan, name).dtype == dtype


@pytest.mark.parametrize("name", ARRAY_NAMES)
def test_arrays_are_read_only(name: str) -> None:
    array = getattr(Plan(**ONE_DISPATCH), name)

    with pytest.raises(ValueError, match="read-only"):
        array[0] = 0


def test_start_defaults_to_nan_for_every_dispatch() -> None:
    arrays = {**ONE_DISPATCH, **TWO_DISPATCHES, "stop_start": [0, 1, 1]}

    plan = Plan(**arrays)

    np.testing.assert_array_equal(plan.start, [np.nan, np.nan])


def test_malformed_arrays_are_reported_together() -> None:
    with pytest.raises(PlanError) as info:
        Plan(
            day=np.array([0, 0]),
            kind=np.array([1, 3]),
            source=np.array([0]),
            origin=np.array([0, 0]),
            stop_start=np.array([0, 2, 1]),
            stop_center=np.array([1]),
            load_dispatch=np.array([0]),
            load_demand=np.array([0, 1]),
            load_quantity=np.array([[-1.0]]),
        )

    assert str(info.value).splitlines() == [
        MALFORMED,
        "- source has 1 entries, day has 2",
        "- stop_start must rise from 0 to the number of stops (1) without decreasing",
        "- load_demand must have one entry per load (1)",
        "- load quantities must be finite and not negative; see loads 0",
        "- kind must be 0 (owned) or 1 (spot); see dispatches 1",
    ]


@pytest.mark.parametrize(("changes", "problem"), MALFORMED_ARRAYS)
def test_each_malformed_array_is_reported(changes: dict[str, Any], problem: str) -> None:
    arrays = {**ONE_DISPATCH, **changes}

    with pytest.raises(PlanError) as info:
        Plan(**arrays)

    assert str(info.value).splitlines() == [MALFORMED, f"- {problem}"]


@pytest.mark.parametrize(("name", "value"), ACCEPTED_ARRAYS)
def test_values_at_the_accepted_side_of_a_limit_are_kept(name: str, value: list[Any]) -> None:
    arrays = {**ONE_DISPATCH, name: value}

    plan = Plan(**arrays)

    np.testing.assert_array_equal(getattr(plan, name), value)


def test_bad_dispatches_are_reported_together() -> None:
    dispatches = [
        Dispatch(day=0, kind="rented", source=0, origin=0, stops=(1,)),
        Dispatch(day=0, kind="spot", source=0, origin=0, stops=(1,), loads=(Load(0, (1.0, 2.0)),)),
    ]

    with pytest.raises(PlanError) as info:
        Plan.from_dispatches(dispatches, dimensions=1)

    assert str(info.value).splitlines() == [
        MALFORMED,
        "- dispatch 0: kind must be 'owned' or 'spot', not 'rented'",
        "- dispatch 1: load of demand 0 has 2 quantities, expected 1",
    ]


@pytest.mark.parametrize(
    ("quantity", "problem"),
    [
        pytest.param((), "dispatch 0: load of demand 0 has 0 quantities, expected 1", id="fewer"),
        pytest.param(
            (1.0, 2.0), "dispatch 0: load of demand 0 has 2 quantities, expected 1", id="more"
        ),
    ],
)
def test_loads_must_have_one_quantity_per_dimension(
    quantity: tuple[float, ...], problem: str
) -> None:
    load = Load(0, quantity)
    dispatch = Dispatch(day=0, kind="spot", source=0, origin=0, stops=(1,), loads=(load,))

    with pytest.raises(PlanError) as info:
        Plan.from_dispatches([dispatch], dimensions=1)

    assert str(info.value).splitlines() == [MALFORMED, f"- {problem}"]


@pytest.mark.parametrize(
    "dispatches",
    [
        pytest.param([FIRST_OWNED, FIRST_SPOT], id="first-indices"),
        pytest.param([LAST_OWNED, LAST_SPOT], id="last-indices"),
    ],
)
def test_indices_inside_the_instance_fit(
    instance: LineHaulInstance, dispatches: list[Dispatch]
) -> None:
    plan = Plan.from_dispatches(dispatches, dimensions=1)

    result = plan.check(instance)

    assert result is None


@pytest.mark.parametrize(("dispatch", "problem"), OUTSIDE)
def test_each_index_outside_the_instance_is_reported(
    instance: LineHaulInstance, dispatch: Dispatch, problem: str
) -> None:
    plan = Plan.from_dispatches([dispatch], dimensions=1)

    with pytest.raises(PlanError) as info:
        plan.check(instance)

    assert str(info.value).splitlines() == [UNFIT, f"- {problem}"]


def test_loads_of_a_dispatch_the_plan_lacks_are_reported(instance: LineHaulInstance) -> None:
    plan = Plan(**{**ONE_DISPATCH, "load_dispatch": [1]})

    with pytest.raises(PlanError) as info:
        plan.check(instance)

    assert str(info.value).splitlines() == [
        UNFIT,
        "- load dispatch index outside 0..0; see loads 0",
    ]


def test_indices_outside_the_instance_are_reported_together(instance: LineHaulInstance) -> None:
    dispatches = [
        Dispatch(day=5, kind="owned", source=1, origin=0, stops=(3,)),
        Dispatch(day=0, kind="spot", source=2, origin=-1, stops=(1,), loads=(Load(7, (1.0,)),)),
    ]
    plan = Plan.from_dispatches(dispatches, dimensions=1)

    with pytest.raises(PlanError) as info:
        plan.check(instance)

    assert str(info.value).splitlines() == [
        UNFIT,
        "- day index outside 0..1; see dispatches 0",
        "- origin index outside 0..2; see dispatches 1",
        "- stop index outside 0..2; see stops 0",
        "- owned group index outside 0..0; see dispatches 0",
        "- spot offer index outside 0..1; see dispatches 1",
        "- load demand index outside 0..2; see loads 0",
    ]


@pytest.mark.parametrize(
    ("dimensions", "problem"),
    [
        pytest.param(0, "loads have 0 quantity dimensions, the instance has 1", id="fewer"),
        pytest.param(2, "loads have 2 quantity dimensions, the instance has 1", id="more"),
    ],
)
def test_quantity_dimensions_must_match_the_instance(
    instance: LineHaulInstance, dimensions: int, problem: str
) -> None:
    plan = Plan.empty(dimensions=dimensions)

    with pytest.raises(PlanError) as info:
        plan.check(instance)

    assert str(info.value).splitlines() == [UNFIT, f"- {problem}"]
