"""Validation of problem documents by :func:`pirilib.spec.parse_spec` (``pirilib.spec.models``).

Every test starts from the ``document`` fixture (``tests/conftest.py``), changes the fields the
behavior needs and either reads the parsed :class:`~pirilib.spec.LineHaulSpec` or the problems
listed by the :class:`~pirilib.SpecError` it raises. Expected values come from the docstrings and
field descriptions of ``pirilib.spec.models`` and from the message format of
:class:`~pirilib.SpecError`: one ``<source>: invalid problem specification`` line, then one
``- <location>: <problem>`` line per problem. Problems found by pydantic field checks keep the
pydantic wording; the consistency problems are pirilib's own.
"""

import datetime as dt
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from pirilib import SpecError
from pirilib.spec import parse_spec

HEADER = "<data>: invalid problem specification"
"""First line of the message of a SpecError raised by ``parse_spec`` for inline data."""

NO_VEHICLE = (
    "no vehicle can serve {lane}: no owned vehicles on this line, and no spot vehicle type "
    "is offered and allowed at both centers"
)
"""Problem of a demand that no vehicle may carry, for the lane ``{lane}``."""

COMPOSED = "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}stanbul"
"""A center name in Unicode NFC form: one code point for the dotted capital I."""

DECOMPOSED = "I\N{COMBINING DOT ABOVE}stanbul"
"""The same name as :data:`COMPOSED`, not normalized: a plain I followed by a combining dot."""

LATER_START = dt.date(2027, 1, 1)
"""Any date other than the fixture's calendar start (2026-01-05)."""

UNKNOWN_ORIGIN_ROW = {
    "origin": "X",
    "destination": "B",
    "date": "2026-01-05",
    "quantity": {"desi": 1.0},
}
"""A demand row with exactly one problem: its origin ``X`` is not a center."""


def reported_problems(document: dict[str, Any]) -> list[str]:
    """Return the problem lines of the SpecError ``parse_spec`` raises for ``document``.

    The ``- `` bullet is removed from every line. The helper fails the test when no SpecError
    is raised or when the message does not start with :data:`HEADER`.
    """
    with pytest.raises(SpecError) as info:
        parse_spec(document)
    header, *lines = str(info.value).splitlines()
    assert header == HEADER
    return [line.removeprefix("- ") for line in lines]


def as_given(document: dict[str, Any]) -> dict[str, Any]:
    """Return ``document`` unchanged: one capacity dimension, ``desi``."""
    return document


def with_kg(document: dict[str, Any]) -> dict[str, Any]:
    """Give ``document`` a second dimension, ``kg``, after ``desi``.

    ``big`` holds 9000 kg and ``small`` 3500 kg; every demand row weighs 10 kg, which every
    vehicle type carries, so the document stays valid.
    """
    document["vehicle_types"][0]["capacity"]["kg"] = 9000.0
    document["vehicle_types"][1]["capacity"]["kg"] = 3500.0
    document["demand"][0]["quantity"]["kg"] = 10.0
    document["demand"][1]["quantity"]["kg"] = 10.0
    document["demand"][2]["quantity"]["kg"] = 10.0
    return document


def owned_stopover_fleet(document: dict[str, Any]) -> dict[str, Any]:
    """Leave only the owned ``A`` to ``B`` line, run by ``small`` vehicles that may stop over.

    The spot market is emptied, so the ``A`` to ``C`` demand can only ride an owned vehicle
    that stops over on its way from ``A``. The ``B`` to ``A`` demand is removed, because no
    vehicle at all leaves ``B`` any more.
    """
    document["fleet"]["spot"] = []
    document["fleet"]["owned"][0].update(type="small", stopovers=True)
    document["demand"].pop()
    return document


@pytest.mark.parametrize(
    ("build", "dimensions"),
    [(as_given, ("desi",)), (with_kg, ("desi", "kg"))],
    ids=["one_dimension", "two_dimensions"],
)
def test_dimensions_follow_the_first_vehicle_type(
    document: dict[str, Any],
    build: Callable[[dict[str, Any]], dict[str, Any]],
    dimensions: tuple[str, ...],
) -> None:
    spec = parse_spec(build(document))

    assert spec.dimensions == dimensions


@pytest.mark.parametrize("build", [as_given, with_kg], ids=["one_dimension", "two_dimensions"])
def test_the_primary_dimension_defaults_to_the_first_dimension(
    document: dict[str, Any], build: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    spec = parse_spec(build(document))

    assert spec.primary_dimension == "desi"


def test_a_named_primary_dimension_is_used(document: dict[str, Any]) -> None:
    document = with_kg(document)
    document["rules"] = {"primary_dimension": "kg"}

    spec = parse_spec(document)

    assert spec.primary_dimension == "kg"


def test_days_run_from_the_calendar_start_to_its_end(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    assert spec.days == (dt.date(2026, 1, 5), dt.date(2026, 1, 6))


def test_a_calendar_may_start_and_end_on_the_same_day(document: dict[str, Any]) -> None:
    """Both ends of the calendar are included, so a one-day calendar has one day."""
    document["calendar"] = {"start": "2026-01-05", "end": "2026-01-05"}
    document["demand"][2]["date"] = "2026-01-05"

    spec = parse_spec(document)

    assert spec.days == (dt.date(2026, 1, 5),)


def test_omitted_rules_take_their_defaults(document: dict[str, Any]) -> None:
    """The fixture has no ``rules`` section: demand may be split and no minimum fill applies."""
    spec = parse_spec(document)

    assert spec.rules.split_demand is True
    assert spec.rules.min_fill.spot == 0.0


@pytest.mark.parametrize("given_id", [None, ""], ids=["missing", "empty"])
def test_demand_rows_without_an_id_are_numbered_by_position(
    document: dict[str, Any], given_id: str | None
) -> None:
    document["demand"][1]["id"] = given_id

    spec = parse_spec(document)

    assert spec.demand_ids == ("D000001", "D000002", "D000003")


def test_given_demand_ids_are_kept(document: dict[str, Any]) -> None:
    """The third row is still numbered by its position, not by the rows without an id."""
    document["demand"][1]["id"] = "order-7"

    spec = parse_spec(document)

    assert spec.demand_ids == ("D000001", "order-7", "D000003")


def test_a_document_without_demand_is_valid(document: dict[str, Any]) -> None:
    document["demand"] = []

    spec = parse_spec(document)

    assert spec.demand_ids == ()


@pytest.mark.parametrize("schema", ["pirilib/linehaul@0.1", "pirilib/linehaul@0.2"])
def test_known_schema_versions_are_accepted(document: dict[str, Any], schema: str) -> None:
    document["schema"] = schema

    spec = parse_spec(document)

    assert spec.schema_ == schema


def test_an_unknown_schema_version_is_rejected(document: dict[str, Any]) -> None:
    document["schema"] = "pirilib/linehaul@9.9"

    found = reported_problems(document)

    assert found == ["schema: Input should be 'pirilib/linehaul@0.1' or 'pirilib/linehaul@0.2'"]


def test_a_parsed_spec_cannot_be_changed(document: dict[str, Any]) -> None:
    spec = parse_spec(document)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        spec.calendar.start = LATER_START


def test_field_problems_are_reported_together(document: dict[str, Any]) -> None:
    """Three independent field problems give three lines, in document order.

    Only the start of the date problem is checked: the rest of that pydantic message (why
    ``soon`` is not a date) is not part of the behavior.
    """
    document["calendar"]["end"] = "soon"
    document["vehicle_types"][0]["capacity"]["desi"] = -1
    document["demand"][1]["quantity"]["desi"] = -5

    found = reported_problems(document)

    assert len(found) == 3
    assert found[0].startswith("calendar.end: Input should be a valid date")
    assert found[1:] == [
        "vehicle_types[0].capacity.desi: Input should be greater than 0",
        "demand[1].quantity.desi: Input should be greater than or equal to 0",
    ]


def test_unknown_fields_are_rejected(document: dict[str, Any]) -> None:
    """``min_fil`` is a misspelling of ``min_fill``."""
    document["rules"] = {"min_fil": {"spot": 0.1}}

    found = reported_problems(document)

    assert found == ["rules.min_fil: Extra inputs are not permitted"]


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"]
)
def test_non_finite_numbers_are_rejected(document: dict[str, Any], value: float) -> None:
    document["fleet"]["spot"][0]["cost"]["per_km"] = value

    found = reported_problems(document)

    assert found == ["fleet.spot[0].cost.per_km: Input should be a finite number"]


@pytest.mark.parametrize("share", [0.0, 0.5, 0.99])
def test_min_fill_accepts_a_share_from_zero_to_below_one(
    document: dict[str, Any], share: float
) -> None:
    document["rules"] = {"min_fill": {"spot": share}}

    spec = parse_spec(document)

    assert spec.rules.min_fill.spot == share


@pytest.mark.parametrize(
    ("share", "problem"),
    [(-0.1, "Input should be greater than or equal to 0"), (1.0, "Input should be less than 1")],
)
def test_min_fill_rejects_a_share_outside_zero_to_below_one(
    document: dict[str, Any], share: float, problem: str
) -> None:
    document["rules"] = {"min_fill": {"spot": share}}

    found = reported_problems(document)

    assert found == [f"rules.min_fill.spot: {problem}"]


@pytest.mark.parametrize(("lat", "lon"), [(90.0, 180.0), (-90.0, -180.0)])
def test_coordinates_on_the_edge_of_the_globe_are_accepted(
    document: dict[str, Any], lat: float, lon: float
) -> None:
    document["centers"][1].update(lat=lat, lon=lon)

    spec = parse_spec(document)

    assert (spec.centers[1].lat, spec.centers[1].lon) == (lat, lon)


@pytest.mark.parametrize(
    ("lat", "lon", "problem"),
    [
        (90.5, 0.0, "lat: Input should be less than or equal to 90"),
        (-90.5, 0.0, "lat: Input should be greater than or equal to -90"),
        (0.0, 180.5, "lon: Input should be less than or equal to 180"),
        (0.0, -180.5, "lon: Input should be greater than or equal to -180"),
    ],
)
def test_coordinates_beyond_the_globe_are_rejected(
    document: dict[str, Any], lat: float, lon: float, problem: str
) -> None:
    document["centers"][1].update(lat=lat, lon=lon)

    found = reported_problems(document)

    assert found == [f"centers[1].{problem}"]


def test_a_capacity_must_be_positive(document: dict[str, Any]) -> None:
    document["vehicle_types"][0]["capacity"]["desi"] = 0.0

    found = reported_problems(document)

    assert found == ["vehicle_types[0].capacity.desi: Input should be greater than 0"]


@pytest.mark.parametrize(
    ("group", "index", "field"),
    [("owned", 0, "count"), ("spot", 1, "available")],
    ids=["owned-count", "spot-available"],
)
def test_a_billion_vehicles_are_accepted(
    document: dict[str, Any], group: str, index: int, field: str
) -> None:
    document["fleet"][group][index][field] = 1_000_000_000

    spec = parse_spec(document)

    assert getattr(getattr(spec.fleet, group)[index], field) == 1_000_000_000


@pytest.mark.parametrize(
    ("group", "index", "field"),
    [("owned", 0, "count"), ("spot", 1, "available")],
    ids=["owned-count", "spot-available"],
)
def test_more_than_a_billion_vehicles_are_rejected(
    document: dict[str, Any], group: str, index: int, field: str
) -> None:
    document["fleet"][group][index][field] = 1_000_000_001

    found = reported_problems(document)

    assert found == [
        f"fleet.{group}[{index}].{field}: Input should be less than or equal to 1000000000"
    ]


def test_a_vehicle_limit_above_a_billion_is_rejected(document: dict[str, Any]) -> None:
    document["centers"][0]["vehicle_limits"] = {"big": 1_000_000_001}

    found = reported_problems(document)

    assert found == [
        "centers[0].vehicle_limits.big: Input should be less than or equal to 1000000000"
    ]


def test_a_demand_quantity_may_be_zero(document: dict[str, Any]) -> None:
    document["demand"][1]["quantity"]["desi"] = 0.0

    spec = parse_spec(document)

    assert spec.demand[1].quantity == {"desi": 0.0}


def test_a_problem_needs_two_centers(document: dict[str, Any]) -> None:
    """Field problems are reported alone: the references to ``B`` and ``C`` are not checked."""
    document["centers"] = document["centers"][:1]

    found = reported_problems(document)

    assert found == ["centers: Tuple should have at least 2 items after validation, not 1"]


def test_blank_names_are_rejected(document: dict[str, Any]) -> None:
    document["centers"][1]["id"] = "   "

    found = reported_problems(document)

    assert found == ["centers[1].id: must not be blank"]


@pytest.mark.parametrize(
    "raw",
    [f"  {COMPOSED} ", DECOMPOSED, f"  {DECOMPOSED} "],
    ids=["padded", "decomposed", "padded_and_decomposed"],
)
def test_names_are_stripped_and_nfc_normalized(document: dict[str, Any], raw: str) -> None:
    """Center ``A`` is renamed to ``raw``; every reference to it uses the normalized name.

    The references only resolve if the center id is stripped and normalized as well.
    """
    document["centers"][0]["id"] = raw
    document["fleet"]["owned"][0]["origin"] = COMPOSED
    document["demand"][0]["origin"] = COMPOSED
    document["demand"][1]["origin"] = COMPOSED
    document["demand"][2]["destination"] = COMPOSED

    spec = parse_spec(document)

    assert spec.centers[0].id == COMPOSED


def test_calendar_must_not_end_before_it_starts(document: dict[str, Any]) -> None:
    document["calendar"] = {"start": "2026-01-06", "end": "2026-01-05"}

    found = reported_problems(document)

    assert found == ["calendar: end must not be before start"]


def test_consistency_problems_are_reported_together(document: dict[str, Any]) -> None:
    """Unknown names, a lane without a vehicle and a date outside the calendar, in table order.

    Renaming the ``small`` spot offer to ``huge`` leaves no vehicle for ``A`` to ``C``: ``big``
    is banned at ``C``.
    """
    document["centers"][2]["banned_vehicle_types"] = ["big", "tiny"]
    document["fleet"]["spot"][1]["type"] = "huge"
    document["demand"][0]["origin"] = "X"
    document["demand"][2]["date"] = "2026-02-01"

    found = reported_problems(document)

    assert found == [
        "centers[2].banned_vehicle_types: unknown vehicle type 'tiny'",
        "fleet.spot[1].type: unknown vehicle type 'huge'",
        "demand[0].origin: unknown center 'X'",
        "demand[1]: " + NO_VEHICLE.format(lane="A -> C"),
        "demand[2].date: 2026-02-01 is outside the calendar 2026-01-05..2026-01-06",
    ]


def test_duplicates_are_reported(document: dict[str, Any]) -> None:
    """Each table reports its second entry with a name already used; other fields are filler."""
    document["centers"].append({"id": "A", "lat": 1.0, "lon": 1.0})
    document["vehicle_types"].append({"name": "big", "capacity": {"desi": 5.0}})
    document["fleet"]["spot"].append({"type": "small", "cost": {}})
    document["demand"][2]["id"] = "D000001"

    found = reported_problems(document)

    assert found == [
        "centers[3]: duplicate center id 'A'",
        "vehicle_types[2]: duplicate vehicle type 'big'",
        "fleet.spot[2]: duplicate spot offer for 'small'",
        "demand[2]: duplicate demand id 'D000001'",
    ]


def test_the_first_vehicle_type_defines_the_dimensions(document: dict[str, Any]) -> None:
    """``kg`` on the second vehicle type is neither shared nor a valid primary dimension."""
    document["vehicle_types"][1]["capacity"] = {"desi": 400.0, "kg": 3000.0}
    document["rules"] = {"primary_dimension": "kg"}

    found = reported_problems(document)

    assert found == [
        "vehicle_types[1].capacity: has dimensions ['desi', 'kg'], but the first vehicle type "
        "has ['desi']; every vehicle type needs the same dimensions",
        "rules.primary_dimension: 'kg' is not one of ['desi']",
    ]


def test_demand_needs_exactly_the_capacity_dimensions(document: dict[str, Any]) -> None:
    document["demand"][0]["quantity"] = {"kg": 10.0}

    found = reported_problems(document)

    assert found == [
        "demand[0].quantity: missing dimension(s) ['desi']",
        "demand[0].quantity: unknown dimension(s) ['kg']",
    ]


def test_owned_vehicles_need_a_known_type_and_known_centers(document: dict[str, Any]) -> None:
    document["fleet"]["owned"][0].update(type="huge", destination="Z")

    found = reported_problems(document)

    assert found == [
        "fleet.owned[0].type: unknown vehicle type 'huge'",
        "fleet.owned[0].destination: unknown center 'Z'",
    ]


def test_owned_vehicles_must_be_allowed_on_their_line(document: dict[str, Any]) -> None:
    """A ``big`` line from ``C`` to ``C``: ``big`` is banned at ``C`` and the line is a loop."""
    document["fleet"]["owned"].append(
        {"type": "big", "origin": "C", "destination": "C", "count": 1, "cost": {}}
    )

    found = reported_problems(document)

    assert found == [
        "fleet.owned[1].origin: vehicle type 'big' is banned at 'C'",
        "fleet.owned[1].destination: vehicle type 'big' is banned at 'C'",
        "fleet.owned[1]: origin and destination are both 'C'",
    ]


def test_a_demand_must_go_to_another_center(document: dict[str, Any]) -> None:
    document["demand"][2]["destination"] = "B"

    found = reported_problems(document)

    assert found == ["demand[2]: origin and destination are both 'B'"]


def test_every_demand_needs_an_allowed_vehicle(document: dict[str, Any]) -> None:
    """Only ``big`` is offered on the spot market, and ``big`` is banned at ``C``."""
    document["fleet"]["spot"] = [{"type": "big", "cost": {}}]

    found = reported_problems(document)

    assert found == ["demand[1]: " + NO_VEHICLE.format(lane="A -> C")]


def test_owned_vehicles_that_stop_over_serve_lanes_beyond_their_line(
    document: dict[str, Any],
) -> None:
    spec = parse_spec(owned_stopover_fleet(document))

    assert spec.demand_ids == ("D000001", "D000002")


def test_owned_vehicles_do_not_stop_over_when_the_rules_forbid_it(
    document: dict[str, Any],
) -> None:
    document = owned_stopover_fleet(document)
    document["rules"] = {"stopover": {"enabled": False}}

    found = reported_problems(document)

    assert found == ["demand[1]: " + NO_VEHICLE.format(lane="A -> C")]


@pytest.mark.parametrize("quantity", [250.0, 400.0], ids=["below_capacity", "at_capacity"])
def test_unsplittable_demand_that_fits_one_vehicle_is_accepted(
    document: dict[str, Any], quantity: float
) -> None:
    """Only ``small`` (400 desi) may serve ``A`` to ``C``."""
    document["rules"] = {"split_demand": False}
    document["demand"][1]["quantity"]["desi"] = quantity

    spec = parse_spec(document)

    assert spec.demand[1].quantity == {"desi": quantity}


@pytest.mark.parametrize("quantity", [400.5, 450.0], ids=["just_above_capacity", "above_capacity"])
def test_unsplittable_demand_must_fit_one_vehicle(
    document: dict[str, Any], quantity: float
) -> None:
    """Only ``small`` (400 desi) may serve ``A`` to ``C``."""
    document["rules"] = {"split_demand": False}
    document["demand"][1]["quantity"]["desi"] = quantity

    found = reported_problems(document)

    assert found == [
        "demand[1]: no vehicle that can serve A -> C carries the whole quantity, and "
        "rules.split_demand is false"
    ]


@pytest.mark.parametrize("coordinate", ["lat", "lon"])
def test_haversine_needs_coordinates(document: dict[str, Any], coordinate: str) -> None:
    del document["centers"][1][coordinate]

    found = reported_problems(document)

    assert found == ["centers[1]: lat and lon are required, because matrix.source is 'haversine'"]


def test_explicit_matrix_problems_are_reported_together(document: dict[str, Any]) -> None:
    """A repeated arc, an unknown center and the pairs left without a distance.

    The matrix is symmetric, so ``A -> B`` also gives ``B -> A``; the arc to the unknown
    center ``Z`` covers no pair of known centers. Distances are filler.
    """
    document["matrix"] = {
        "source": "explicit",
        "symmetric": True,
        "arcs": [
            {"origin": "A", "destination": "B", "km": 10.0},
            {"origin": "A", "destination": "B", "km": 11.0},
            {"origin": "B", "destination": "Z", "km": 5.0},
        ],
    }

    found = reported_problems(document)

    assert found == [
        "matrix.arcs[1]: duplicate arc A -> B",
        "matrix.arcs[2].destination: unknown center 'Z'",
        "matrix.arcs: no distance for A -> C, B -> C, C -> A, C -> B",
    ]


@pytest.mark.parametrize(
    ("arcs", "problem"),
    [
        (
            [{"origin": "A", "destination": "B", "km": 1.0}],
            "matrix.arcs: no distance for A -> C, B -> A, B -> C, C -> A, C -> B",
        ),
        ([], "matrix.arcs: no distance for A -> B, A -> C, B -> A, B -> C, C -> A and 1 more"),
    ],
    ids=["five_missing", "six_missing"],
)
def test_missing_distances_are_listed_up_to_five(
    document: dict[str, Any], arcs: list[dict[str, Any]], problem: str
) -> None:
    """Three centers have six ordered pairs; past five missing pairs the rest are counted."""
    document["matrix"] = {"source": "explicit", "arcs": arcs}

    found = reported_problems(document)

    assert found == [problem]


def test_an_arc_must_go_to_another_center(document: dict[str, Any]) -> None:
    """The first three arcs cover every pair (symmetric matrix); the fourth is a loop."""
    document["matrix"] = {
        "source": "explicit",
        "symmetric": True,
        "arcs": [
            {"origin": "A", "destination": "B", "km": 1.0},
            {"origin": "A", "destination": "C", "km": 1.0},
            {"origin": "B", "destination": "C", "km": 1.0},
            {"origin": "A", "destination": "A", "km": 0.0},
        ],
    }

    found = reported_problems(document)

    assert found == ["matrix.arcs[3]: origin and destination are both 'A'"]


def test_matrix_field_problems_leave_out_the_union_tag(document: dict[str, Any]) -> None:
    """The location is ``matrix.arcs[0].km``, not ``matrix.explicit.arcs[0].km``."""
    document["matrix"] = {
        "source": "explicit",
        "arcs": [{"origin": "A", "destination": "B", "km": -1.0}],
    }

    found = reported_problems(document)

    assert found == ["matrix.arcs[0].km: Input should be greater than or equal to 0"]


def test_an_unknown_matrix_source_is_rejected(document: dict[str, Any]) -> None:
    document["matrix"] = {"source": "osrm"}

    found = reported_problems(document)

    assert found == [
        "matrix: Input tag 'osrm' found using 'source' does not match any of the expected "
        "tags: 'haversine', 'explicit'"
    ]


@pytest.mark.parametrize(
    ("rows", "lines", "last"),
    [
        (50, 50, "demand[49].origin: unknown center 'X'"),
        (51, 51, "... and 1 more"),
        (60, 51, "... and 10 more"),
    ],
    ids=["at_the_cap", "one_over", "ten_over"],
)
def test_long_reports_list_fifty_problems_and_count_the_rest(
    document: dict[str, Any], rows: int, lines: int, last: str
) -> None:
    """A message lists at most 50 problems; one more line counts the problems left out.

    Every row has one problem (:data:`UNKNOWN_ORIGIN_ROW`), so ``rows`` problems are found.
    """
    document["demand"] = [UNKNOWN_ORIGIN_ROW] * rows

    found = reported_problems(document)

    assert len(found) == lines
    assert found[0] == "demand[0].origin: unknown center 'X'"
    assert found[-1] == last
