import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from pirilib import SpecError
from pirilib.spec import parse_spec

NO_VEHICLE = (
    "no vehicle can serve {lane}: no owned vehicles on this line, and no spot vehicle type "
    "is offered and allowed at both centers"
)


def problems(document: dict[str, Any]) -> list[str]:
    """Return the problems listed by the SpecError raised for ``document``."""
    with pytest.raises(SpecError) as info:
        parse_spec(document)
    header, *lines = str(info.value).splitlines()
    assert header == "<data>: invalid problem specification"
    return [line.removeprefix("- ") for line in lines]


def test_valid_document(document: dict[str, Any]) -> None:
    spec = parse_spec(document)
    assert spec.dimensions == ("desi",)
    assert spec.primary_dimension == "desi"
    assert spec.days == (dt.date(2026, 1, 5), dt.date(2026, 1, 6))
    assert spec.demand_ids == ("D000001", "D000002", "D000003")
    assert spec.rules.split_demand
    assert spec.rules.min_fill.spot == 0.0


def test_given_demand_ids_are_kept(document: dict[str, Any]) -> None:
    document["demand"][1]["id"] = "order-7"
    assert parse_spec(document).demand_ids == ("D000001", "order-7", "D000003")


def test_specs_are_immutable(document: dict[str, Any]) -> None:
    spec = parse_spec(document)
    with pytest.raises(ValidationError):
        spec.calendar.start = dt.date(2027, 1, 1)


def test_field_problems_are_reported_together(document: dict[str, Any]) -> None:
    document["calendar"]["end"] = "soon"
    document["vehicle_types"][0]["capacity"]["desi"] = -1
    document["demand"][1]["quantity"]["desi"] = -5
    found = problems(document)
    assert len(found) == 3
    assert found[0].startswith("calendar.end: Input should be a valid date")
    assert found[1] == "vehicle_types[0].capacity.desi: Input should be greater than 0"
    assert found[2] == "demand[1].quantity.desi: Input should be greater than or equal to 0"


def test_unknown_fields_are_rejected(document: dict[str, Any]) -> None:
    document["rules"] = {"min_fil": {"spot": 0.1}}
    assert problems(document) == ["rules.min_fil: Extra inputs are not permitted"]


def test_schema_version_is_checked(document: dict[str, Any]) -> None:
    document["schema"] = "pirilib/linehaul@9.9"
    [problem] = problems(document)
    assert problem.startswith("schema: Input should be 'pirilib/linehaul@0.1'")


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_non_finite_numbers_are_rejected(document: dict[str, Any], value: float) -> None:
    document["fleet"]["spot"][0]["cost"]["per_km"] = value
    [problem] = problems(document)
    assert problem.startswith("fleet.spot[0].cost.per_km: ")


@pytest.mark.parametrize("share", [-0.1, 1.0])
def test_min_fill_is_a_share_below_one(document: dict[str, Any], share: float) -> None:
    document["rules"] = {"min_fill": {"spot": share}}
    [problem] = problems(document)
    assert problem.startswith("rules.min_fill.spot: Input should be")


def test_blank_names_are_rejected(document: dict[str, Any]) -> None:
    document["centers"][1]["id"] = "   "
    assert problems(document) == ["centers[1].id: must not be blank"]


def test_names_are_stripped_and_nfc_normalized(document: dict[str, Any]) -> None:
    composed = "\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}stanbul"
    document["centers"][0]["id"] = "  I\N{COMBINING DOT ABOVE}stanbul "
    document["fleet"]["owned"][0]["origin"] = composed
    document["demand"][0]["origin"] = composed
    document["demand"][1]["origin"] = composed
    document["demand"][2]["destination"] = composed
    spec = parse_spec(document)
    assert spec.centers[0].id == composed


def test_calendar_must_not_end_before_it_starts(document: dict[str, Any]) -> None:
    document["calendar"] = {"start": "2026-01-06", "end": "2026-01-05"}
    assert problems(document) == ["calendar: end must not be before start"]


def test_consistency_problems_are_reported_together(document: dict[str, Any]) -> None:
    document["centers"][2]["banned_vehicle_types"] = ["big", "tiny"]
    document["fleet"]["spot"][1]["type"] = "huge"
    document["demand"][0]["origin"] = "X"
    document["demand"][2]["date"] = "2026-02-01"
    assert problems(document) == [
        "centers[2].banned_vehicle_types: unknown vehicle type 'tiny'",
        "fleet.spot[1].type: unknown vehicle type 'huge'",
        "demand[0].origin: unknown center 'X'",
        "demand[1]: " + NO_VEHICLE.format(lane="A -> C"),
        "demand[2].date: 2026-02-01 is outside the calendar 2026-01-05..2026-01-06",
    ]


def test_duplicates_are_reported(document: dict[str, Any]) -> None:
    document["centers"].append({"id": "A", "lat": 1.0, "lon": 1.0})
    document["fleet"]["spot"].append({"type": "small", "cost": {}})
    document["demand"][2]["id"] = "D000001"
    assert problems(document) == [
        "centers[3]: duplicate center id 'A'",
        "fleet.spot[2]: duplicate spot offer for 'small'",
        "demand[2]: duplicate demand id 'D000001'",
    ]


def test_vehicle_types_share_dimensions(document: dict[str, Any]) -> None:
    document["vehicle_types"][1]["capacity"] = {"desi": 400.0, "kg": 3000.0}
    document["rules"] = {"primary_dimension": "kg"}
    assert problems(document) == [
        "vehicle_types[1].capacity: has dimensions ['desi', 'kg'], but the first vehicle type "
        "has ['desi']; every vehicle type needs the same dimensions",
        "rules.primary_dimension: 'kg' is not one of ['desi']",
    ]


def test_several_dimensions(document: dict[str, Any]) -> None:
    for vehicle, kg in zip(document["vehicle_types"], [9000.0, 3500.0], strict=True):
        vehicle["capacity"]["kg"] = kg
    for row in document["demand"]:
        row["quantity"]["kg"] = 10.0
    document["rules"] = {"primary_dimension": "kg"}
    spec = parse_spec(document)
    assert spec.dimensions == ("desi", "kg")
    assert spec.primary_dimension == "kg"


def test_demand_needs_exactly_the_capacity_dimensions(document: dict[str, Any]) -> None:
    document["demand"][0]["quantity"] = {"kg": 10.0}
    assert problems(document) == [
        "demand[0].quantity: missing dimension(s) ['desi']",
        "demand[0].quantity: unknown dimension(s) ['kg']",
    ]


def test_owned_vehicles_are_checked(document: dict[str, Any]) -> None:
    document["fleet"]["owned"].append(
        {"type": "big", "origin": "C", "destination": "C", "count": 1, "cost": {}}
    )
    assert problems(document) == [
        "fleet.owned[1].origin: vehicle type 'big' is banned at 'C'",
        "fleet.owned[1].destination: vehicle type 'big' is banned at 'C'",
        "fleet.owned[1]: origin and destination are both 'C'",
    ]


def test_every_demand_needs_an_allowed_vehicle(document: dict[str, Any]) -> None:
    document["fleet"]["spot"] = [{"type": "big", "cost": {}}]
    assert problems(document) == ["demand[1]: " + NO_VEHICLE.format(lane="A -> C")]


def test_stopovers_extend_owned_lines(document: dict[str, Any]) -> None:
    document["fleet"]["spot"] = []
    document["fleet"]["owned"][0].update(type="small", stopovers=True)
    document["demand"].pop()
    parse_spec(document)

    document["rules"] = {"stopover": {"enabled": False}}
    assert problems(document) == ["demand[1]: " + NO_VEHICLE.format(lane="A -> C")]


def test_unsplittable_demand_must_fit_one_vehicle(document: dict[str, Any]) -> None:
    document["rules"] = {"split_demand": False}
    parse_spec(document)

    document["demand"][1]["quantity"]["desi"] = 450.0
    assert problems(document) == [
        "demand[1]: no vehicle that can serve A -> C carries the whole quantity, and "
        "rules.split_demand is false"
    ]


def test_haversine_needs_coordinates(document: dict[str, Any]) -> None:
    del document["centers"][1]["lat"]
    assert problems(document) == [
        "centers[1]: lat and lon are required, because matrix.source is 'haversine'"
    ]


def test_explicit_matrix_must_cover_every_pair(document: dict[str, Any]) -> None:
    document["matrix"] = {
        "source": "explicit",
        "symmetric": True,
        "arcs": [
            {"origin": "A", "destination": "B", "km": 10.0},
            {"origin": "A", "destination": "B", "km": 11.0},
            {"origin": "B", "destination": "Z", "km": 5.0},
        ],
    }
    assert problems(document) == [
        "matrix.arcs[1]: duplicate arc A -> B",
        "matrix.arcs[2].destination: unknown center 'Z'",
        "matrix.arcs: no distance for A -> C, B -> C, C -> A, C -> B",
    ]


def test_matrix_field_problems_leave_out_the_union_tag(document: dict[str, Any]) -> None:
    document["matrix"] = {
        "source": "explicit",
        "arcs": [{"origin": "A", "destination": "B", "km": -1.0}],
    }
    [problem] = problems(document)
    assert problem == "matrix.arcs[0].km: Input should be greater than or equal to 0"


def test_unknown_matrix_source(document: dict[str, Any]) -> None:
    document["matrix"] = {"source": "osrm"}
    [problem] = problems(document)
    assert problem.startswith("matrix: Input tag 'osrm' found using 'source' does not match")


def test_long_reports_are_truncated(document: dict[str, Any]) -> None:
    row = {"origin": "X", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 1.0}}
    document["demand"] = [dict(row) for _ in range(60)]
    found = problems(document)
    assert len(found) == 51
    assert found[0] == "demand[0].origin: unknown center 'X'"
    assert found[-1] == "... and 10 more"
