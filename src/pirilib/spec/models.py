"""Validated problem specification for the line-haul problem pack.

A specification describes one planning problem: transfer centers, distances,
vehicle types, the fleet (owned vehicles and on-demand "spot" offers), the demand
to carry and the business rules. Business rules are explicit fields, so that no
solver hides them in its code.

Field values are checked first. Once every field is valid, the whole document is
checked for consistency: references, dimensions, dates and whether every demand
can be served at all.
"""

import datetime as dt
import unicodedata
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from pirilib.errors import SpecError
from pirilib.spec._format import Loc, SpecIssues

if TYPE_CHECKING:
    from pirilib.objective import Sum
    from pirilib.spec.instance import LineHaulInstance

SCHEMA_ID: Final = "pirilib/linehaul@0.2"
SchemaId = Literal["pirilib/linehaul@0.1", "pirilib/linehaul@0.2"]
"""Accepted schema ids; 0.1 documents are 0.2 documents without the time fields."""

type Issue = tuple[Loc, str]


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value.strip())
    if not value:
        raise ValueError("must not be blank")
    return value


Name = Annotated[str, StringConstraints(min_length=1), AfterValidator(_normalize_name)]
"""An identifier with surrounding whitespace removed and Unicode NFC-normalized."""


class _Model(BaseModel):
    """Base of the specification models: unknown fields and non-finite numbers are errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Center(_Model):
    """A transfer center: a depot, hub or city that loads leave from or arrive at."""

    id: Name
    name: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    banned_vehicle_types: tuple[Name, ...] = Field(
        default=(),
        description="Vehicle types that may not depart from, stop at or arrive at this center.",
    )
    handling_capacity: PositiveFloat | None = Field(
        default=None,
        description=(
            "Primary quantity the center can load and unload per day, both counted; "
            "null means no limit."
        ),
    )
    vehicle_limits: dict[Name, NonNegativeInt] = Field(
        default_factory=dict,
        description=(
            "Vehicle type -> most vehicles of that type that may leave from, stop at or "
            "arrive at the center per day."
        ),
    )


class VehicleType(_Model):
    """Physical properties of a vehicle type. Costs belong to fleet entries."""

    name: Name
    capacity: dict[Name, PositiveFloat] = Field(
        min_length=1, description="Capacity per quantity dimension, for example {desi: 22400}."
    )
    speed_kmh: PositiveFloat | None = Field(
        default=None,
        description="Average speed; needed for travel times (hourly costs, due times).",
    )


class Cost(_Model):
    """Cost of one vehicle dispatch: a fixed part, a part per km and a part per hour.

    Hours run from the start of loading at the origin to the end of unloading at
    the last stop.
    """

    fixed: NonNegativeFloat = 0.0
    per_km: NonNegativeFloat = 0.0
    per_hour: NonNegativeFloat = 0.0


class OwnedVehicles(_Model):
    """Identical owned or long-term rented vehicles bound to one line."""

    type: Name
    origin: Name
    destination: Name
    count: PositiveInt = Field(description="Vehicles available on the line every day.")
    cost: Cost
    mandatory: bool = Field(
        default=True, description="The fixed cost is paid every day, even when the vehicle is idle."
    )
    stopovers: bool = Field(
        default=False, description="The vehicles may also drop loads at centers on the way."
    )


class SpotOffer(_Model):
    """On-demand vehicles of one type, hired per dispatch."""

    type: Name
    cost: Cost
    available: PositiveInt | None = Field(
        default=None, description="Vehicles per day; null means unlimited."
    )


class Fleet(_Model):
    """The vehicles that can be dispatched."""

    owned: tuple[OwnedVehicles, ...] = ()
    spot: tuple[SpotOffer, ...] = ()


class Demand(_Model):
    """A quantity to move from one center to another, shipped on its date."""

    id: Name | None = Field(
        default=None,
        description="Generated from the row position (D000001 for the first row) when omitted.",
    )
    origin: Name
    destination: Name
    date: dt.date
    quantity: dict[Name, NonNegativeFloat] = Field(
        min_length=1, description="Quantity per dimension; every capacity dimension is required."
    )
    ready: dt.time | None = Field(
        default=None, description="Time of day the demand can be loaded; null means 00:00."
    )
    due: dt.datetime | None = Field(
        default=None,
        description=(
            "When it must be unloaded; null means rules.service.due_hours after it is "
            "ready, or no due time."
        ),
    )


class Calendar(_Model):
    """The planning horizon; both ends are included."""

    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class HaversineMatrix(_Model):
    """Distances along the great circle, computed from center coordinates."""

    source: Literal["haversine"]
    factor: PositiveFloat = Field(
        default=1.0,
        description="Multiplier applied to every distance (1.0 means as the crow flies).",
    )


class Arc(_Model):
    """The distance from one center to another."""

    origin: Name
    destination: Name
    km: NonNegativeFloat


class ExplicitMatrix(_Model):
    """Distances given arc by arc."""

    source: Literal["explicit"]
    arcs: tuple[Arc, ...]
    symmetric: bool = Field(
        default=False, description="Use the distance of A->B for B->A when B->A is not given."
    )
    factor: PositiveFloat = 1.0


DistanceMatrix = Annotated[HaversineMatrix | ExplicitMatrix, Field(discriminator="source")]


class MinFill(_Model):
    """Minimum share of capacity (primary dimension) a vehicle must carry when it departs."""

    spot: float = Field(default=0.0, ge=0.0, lt=1.0)
    owned: float = Field(default=0.0, ge=0.0, lt=1.0)


class Stopover(_Model):
    """Whether, and how, one vehicle may serve several destinations on one route."""

    enabled: bool = True
    max_stretch: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "Upper bound on route length divided by the direct distance to the last stop; "
            "null means no bound."
        ),
    )
    forward_only: bool = Field(
        default=True, description="Each stop is farther from the origin than the one before it."
    )
    max_stops: PositiveInt | None = Field(
        default=None, description="Destinations per vehicle; null means no limit."
    )


class Service(_Model):
    """Due times, the cost of lateness and how long demand may wait."""

    due_hours: PositiveFloat | None = Field(
        default=None,
        description="Due time of demand without one, in hours after it is ready; null: none.",
    )
    penalty: NonNegativeFloat = Field(
        default=0.0,
        description="Cost per primary unit for every started hour of lateness.",
    )
    max_deferral_days: NonNegativeInt = Field(
        default=0, description="Days after its date a demand may still leave."
    )


class Handling(_Model):
    """Loading and unloading at centers."""

    minutes_per_unit: NonNegativeFloat = Field(
        default=0.0,
        description="Minutes to load or unload one primary unit, for example 0.01 per desi.",
    )


class Rules(_Model):
    """Business rules every solver and the validator must respect."""

    primary_dimension: Name | None = Field(
        default=None,
        description="Quantity used by fill rules; defaults to the first capacity dimension.",
    )
    min_fill: MinFill = MinFill()
    stopover: Stopover = Stopover()
    split_demand: bool = Field(
        default=True, description="A demand may be carried by several vehicles."
    )
    service: Service = Service()
    handling: Handling = Handling()


class ObjectiveSpec(_Model):
    """The objective to minimize, built from the costs and rules of the document.

    The objective is ``cost`` times the total cost (dispatch costs, the fixed
    cost of mandatory owned vehicles and lateness), plus the optional extra
    terms, plus a weight per unit of violation of every built-in rule. Every
    rule a plan can break needs a weight: name it under ``penalties`` (the names
    are those of :data:`pirilib.constraints.CATALOG`), or give ``other`` for
    every rule that is not named. A weight of 0 lets a plan break the rule for
    free, so it has to be written down.
    """

    OTHER: ClassVar[str] = "other"
    """The ``penalties`` key that applies to every built-in rule not named."""

    cost: PositiveFloat = Field(
        description=(
            "Weight of the total cost: dispatches (fixed, per km and per hour), the fixed "
            "cost of mandatory owned vehicles and lateness."
        )
    )
    distance_km: NonNegativeFloat = Field(default=0.0, description="Weight per km of route.")
    dispatches: NonNegativeFloat = Field(default=0.0, description="Weight per dispatch.")
    unused_capacity: NonNegativeFloat = Field(
        default=0.0,
        description="Weight per primary unit of capacity left empty on departure.",
    )
    penalties: dict[str, NonNegativeFloat] = Field(
        description=(
            "Weight per unit of violation, by rule name; 'other' applies to every rule "
            "not named. Without 'other', every rule must be named."
        )
    )

    @field_validator("penalties")
    @classmethod
    def _check_penalties(cls, penalties: dict[str, float]) -> dict[str, float]:
        from pirilib.constraints import CATALOG

        known = [*CATALOG, cls.OTHER]
        issues: list[Issue] = [
            ((name,), f"unknown rule '{name}'; known: {', '.join(known)}")
            for name in penalties
            if name not in known
        ]
        missing = [name for name in CATALOG if name not in penalties]
        if cls.OTHER not in penalties and missing:
            issues.append(
                (
                    (),
                    f"no penalty for {', '.join(missing)}; every rule needs a weight: add "
                    f"them, or '{cls.OTHER}' for every rule not named",
                )
            )
        if issues:
            raise SpecIssues(issues)
        return penalties

    def penalty(self, name: str) -> float:
        """The weight per unit of violation of the built-in rule ``name``, named or ``other``."""
        from pirilib.constraints import CATALOG

        if name not in CATALOG:
            raise KeyError(f"no built-in rule named '{name}'; known: {', '.join(CATALOG)}")
        if name in self.penalties:
            return self.penalties[name]
        return self.penalties[self.OTHER]

    def build(self) -> "Sum":
        """The objective as a :class:`~pirilib.objective.Sum`, with a linear form.

        Extra terms with weight 0 are left out; every built-in rule gets its
        penalty term, also when the weight is 0.
        """
        from pirilib import objective as obj
        from pirilib.constraints import CATALOG

        extras = (
            (self.distance_km, obj.distance_km),
            (self.dispatches, obj.dispatches),
            (self.unused_capacity, obj.unused_capacity),
        )
        terms = [
            obj.total_cost() * self.cost,
            *(make() * weight for weight, make in extras if weight),
            *(obj.violation(name) * self.penalty(name) for name in CATALOG),
        ]
        return obj.Sum(tuple(terms))


_NO_OBJECTIVE: Final = (
    "the problem document has no objective: section; solving and tuning need one, "
    "for example\n\n"
    "objective:\n"
    "  cost: 1\n"
    "  penalties:\n"
    "    coverage: 10000000\n"
    "    other: 1000000\n\n"
    "Every rule a plan can break needs a penalty weight, by name or through 'other'."
)
"""Why :meth:`LineHaulSpec.build_objective` fails on a document without an objective."""


class LineHaulSpec(_Model):
    """A complete, validated line-haul planning problem."""

    schema_: SchemaId = Field(alias="schema")
    name: str | None = Field(default=None, description="A label used in reports.")
    calendar: Calendar
    centers: tuple[Center, ...] = Field(min_length=2)
    matrix: DistanceMatrix
    vehicle_types: tuple[VehicleType, ...] = Field(min_length=1)
    fleet: Fleet
    demand: tuple[Demand, ...]
    rules: Rules = Rules()
    objective: ObjectiveSpec | None = Field(
        default=None,
        description="The objective to minimize; solving and tuning need it.",
    )

    @model_validator(mode="before")
    @classmethod
    def _assign_demand_ids(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        rows = data.get("demand")
        if not isinstance(rows, list | tuple):
            return data
        numbered: list[Any] = []
        for position, row in enumerate(rows, start=1):
            generated = f"D{position:06d}"
            if isinstance(row, Mapping) and row.get("id") in (None, ""):
                row = {**row, "id": generated}
            elif isinstance(row, Demand) and row.id is None:
                row = row.model_copy(update={"id": generated})
            numbered.append(row)
        return {**data, "demand": numbered}

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        issues = _consistency_issues(self)
        if issues:
            raise SpecIssues(issues)
        return self

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Quantity dimensions, in the order of the first vehicle type's capacity."""
        return tuple(self.vehicle_types[0].capacity)

    @property
    def primary_dimension(self) -> str:
        """The dimension the rules call primary, else the first one."""
        return self.rules.primary_dimension or self.dimensions[0]

    @property
    def days(self) -> tuple[dt.date, ...]:
        """Every date from the calendar start to its end, inclusive."""
        span = (self.calendar.end - self.calendar.start).days + 1
        return tuple(self.calendar.start + dt.timedelta(days=offset) for offset in range(span))

    @property
    def demand_ids(self) -> tuple[str, ...]:
        """The id of each demand row (empty text when it has none)."""
        return tuple(row.id or "" for row in self.demand)

    def build_objective(self) -> "Sum":
        """The objective of the ``objective:`` section; :class:`~pirilib.SpecError` without it."""
        if self.objective is None:
            raise SpecError(_NO_OBJECTIVE)
        return self.objective.build()

    def compile(self) -> "LineHaulInstance":
        """Return the read-only array form used by the engines."""
        from pirilib.spec.instance import compile_spec

        return compile_spec(self)


def _consistency_issues(spec: LineHaulSpec) -> list[Issue]:
    """Return every cross-reference and feasibility problem of ``spec``."""
    return [
        *_duplicates(("centers",), "center id", [center.id for center in spec.centers]),
        *_duplicates(("vehicle_types",), "vehicle type", [v.name for v in spec.vehicle_types]),
        *_duplicates(
            ("fleet", "spot"), "spot offer for", [offer.type for offer in spec.fleet.spot]
        ),
        *_duplicates(("demand",), "demand id", [row.id for row in spec.demand]),
        *_dimension_issues(spec),
        *_fleet_issues(spec),
        *_demand_issues(spec),
        *_matrix_issues(spec),
        *_time_issues(spec),
    ]


def needs_time(spec: "LineHaulSpec") -> bool:
    """True if costs or due times depend on when vehicles run."""
    costs = [group.cost for group in spec.fleet.owned] + [offer.cost for offer in spec.fleet.spot]
    return (
        any(cost.per_hour > 0 for cost in costs)
        or spec.rules.service.due_hours is not None
        or any(row.due is not None for row in spec.demand)
    )


def _time_issues(spec: "LineHaulSpec") -> list[Issue]:
    types = {vehicle.name for vehicle in spec.vehicle_types}
    issues: list[Issue] = [
        (("centers", position, "vehicle_limits"), f"unknown vehicle type '{name}'")
        for position, center in enumerate(spec.centers)
        for name in center.vehicle_limits
        if name not in types
    ]
    if needs_time(spec):
        issues += [
            (
                ("vehicle_types", position, "speed_kmh"),
                "is required, because hourly costs or due times are used",
            )
            for position, vehicle in enumerate(spec.vehicle_types)
            if vehicle.speed_kmh is None
        ]
    for position, row in enumerate(spec.demand):
        if row.due is not None:
            ready = dt.datetime.combine(row.date, row.ready or dt.time())
            if row.due < ready:
                issues.append(((("demand", position, "due")), f"{row.due} is before it is ready"))
    return issues


def _duplicates(table: Loc, label: str, values: Sequence[str | None]) -> list[Issue]:
    seen: set[str] = set()
    issues: list[Issue] = []
    for position, value in enumerate(values):
        if value is None:
            continue
        if value in seen:
            issues.append(((*table, position), f"duplicate {label} '{value}'"))
        seen.add(value)
    return issues


def _banned(spec: LineHaulSpec) -> set[tuple[str, str]]:
    """Return the (center, vehicle type) pairs that are not allowed."""
    return {(center.id, name) for center in spec.centers for name in center.banned_vehicle_types}


def _dimension_issues(spec: LineHaulSpec) -> list[Issue]:
    dimensions = spec.dimensions
    issues: list[Issue] = [
        (
            ("vehicle_types", position, "capacity"),
            f"has dimensions {list(vehicle.capacity)}, but the first vehicle type has "
            f"{list(dimensions)}; every vehicle type needs the same dimensions",
        )
        for position, vehicle in enumerate(spec.vehicle_types)
        if set(vehicle.capacity) != set(dimensions)
    ]
    primary = spec.rules.primary_dimension
    if primary is not None and primary not in dimensions:
        issues.append(
            (("rules", "primary_dimension"), f"'{primary}' is not one of {list(dimensions)}")
        )
    return issues


def _fleet_issues(spec: LineHaulSpec) -> list[Issue]:
    types = {vehicle.name for vehicle in spec.vehicle_types}
    centers = {center.id for center in spec.centers}
    banned = _banned(spec)
    issues: list[Issue] = [
        (("centers", position, "banned_vehicle_types"), f"unknown vehicle type '{name}'")
        for position, center in enumerate(spec.centers)
        for name in center.banned_vehicle_types
        if name not in types
    ]
    for position, group in enumerate(spec.fleet.owned):
        loc: Loc = ("fleet", "owned", position)
        if group.type not in types:
            issues.append(((*loc, "type"), f"unknown vehicle type '{group.type}'"))
        for field, center in (("origin", group.origin), ("destination", group.destination)):
            if center not in centers:
                issues.append(((*loc, field), f"unknown center '{center}'"))
            elif (center, group.type) in banned:
                issues.append(
                    ((*loc, field), f"vehicle type '{group.type}' is banned at '{center}'")
                )
        if group.origin == group.destination:
            issues.append((loc, f"origin and destination are both '{group.origin}'"))
    issues += [
        (("fleet", "spot", position, "type"), f"unknown vehicle type '{offer.type}'")
        for position, offer in enumerate(spec.fleet.spot)
        if offer.type not in types
    ]
    return issues


def _demand_issues(spec: LineHaulSpec) -> list[Issue]:
    centers = {center.id for center in spec.centers}
    dimensions = spec.dimensions
    capacity = {vehicle.name: vehicle.capacity for vehicle in spec.vehicle_types}
    banned = _banned(spec)
    line_types: dict[tuple[str, str], set[str]] = {}
    stopover_types: dict[str, set[str]] = {}
    for group in spec.fleet.owned:
        if group.type in capacity:
            line_types.setdefault((group.origin, group.destination), set()).add(group.type)
            if group.stopovers and spec.rules.stopover.enabled:
                stopover_types.setdefault(group.origin, set()).add(group.type)
    spot_types = {offer.type for offer in spec.fleet.spot if offer.type in capacity}
    start, end = spec.calendar.start, spec.calendar.end

    issues: list[Issue] = []
    for position, row in enumerate(spec.demand):
        loc: Loc = ("demand", position)
        found = len(issues)
        for field, center in (("origin", row.origin), ("destination", row.destination)):
            if center not in centers:
                issues.append(((*loc, field), f"unknown center '{center}'"))
        if row.origin == row.destination:
            issues.append((loc, f"origin and destination are both '{row.origin}'"))
        if not start <= row.date <= end:
            issues.append(((*loc, "date"), f"{row.date} is outside the calendar {start}..{end}"))
        missing = [name for name in dimensions if name not in row.quantity]
        if missing:
            issues.append(((*loc, "quantity"), f"missing dimension(s) {missing}"))
        unknown = [name for name in row.quantity if name not in dimensions]
        if unknown:
            issues.append(((*loc, "quantity"), f"unknown dimension(s) {unknown}"))
        if len(issues) > found:
            continue

        allowed = (
            line_types.get((row.origin, row.destination), set())
            | {
                name
                for name in stopover_types.get(row.origin, ())
                if (row.destination, name) not in banned
            }
            | {
                name
                for name in spot_types
                if (row.origin, name) not in banned and (row.destination, name) not in banned
            }
        )
        lane = f"{row.origin} -> {row.destination}"
        if not allowed:
            issues.append(
                (
                    loc,
                    f"no vehicle can serve {lane}: no owned vehicles on this line, and no spot "
                    "vehicle type is offered and allowed at both centers",
                )
            )
        elif not spec.rules.split_demand and not any(
            all(row.quantity[name] <= capacity[vehicle][name] for name in dimensions)
            for vehicle in allowed
        ):
            issues.append(
                (
                    loc,
                    f"no vehicle that can serve {lane} carries the whole quantity, and "
                    "rules.split_demand is false",
                )
            )
    return issues


def _matrix_issues(spec: LineHaulSpec) -> list[Issue]:
    matrix = spec.matrix
    if isinstance(matrix, HaversineMatrix):
        return [
            (
                ("centers", position),
                "lat and lon are required, because matrix.source is 'haversine'",
            )
            for position, center in enumerate(spec.centers)
            if center.lat is None or center.lon is None
        ]

    centers = [center.id for center in spec.centers]
    known = set(centers)
    given: set[tuple[str, str]] = set()
    issues: list[Issue] = []
    for position, arc in enumerate(matrix.arcs):
        loc: Loc = ("matrix", "arcs", position)
        for field, center in (("origin", arc.origin), ("destination", arc.destination)):
            if center not in known:
                issues.append(((*loc, field), f"unknown center '{center}'"))
        if arc.origin == arc.destination:
            issues.append((loc, f"origin and destination are both '{arc.origin}'"))
        elif (arc.origin, arc.destination) in given:
            issues.append((loc, f"duplicate arc {arc.origin} -> {arc.destination}"))
        given.add((arc.origin, arc.destination))
    if matrix.symmetric:
        given |= {(b, a) for a, b in given}
    missing = [(a, b) for a in centers for b in centers if a != b and (a, b) not in given]
    if missing:
        sample = ", ".join(f"{a} -> {b}" for a, b in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        issues.append((("matrix", "arcs"), f"no distance for {sample}{more}"))
    return issues
