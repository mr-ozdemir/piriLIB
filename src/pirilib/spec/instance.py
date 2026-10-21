"""Read-only array form of a line-haul specification, used by the engines."""

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pirilib.geo import haversine_matrix
from pirilib.spec.models import Demand, ExplicitMatrix, LineHaulSpec, Rules, needs_time

UNLIMITED = -1
"""Value of :attr:`LineHaulInstance.spot_available` for offers without a daily limit."""


@dataclass(frozen=True, eq=False, slots=True)
class LineHaulInstance:
    """Index-based, read-only arrays describing one line-haul problem.

    Centers, vehicle types, days and quantity dimensions are referred to by their
    position in :attr:`centers`, :attr:`vehicle_types`, :attr:`days` and
    :attr:`dimensions`. Arrays prefixed ``spot_`` have one entry per spot offer,
    ``owned_`` one per owned vehicle group and ``demand_`` one per demand.

    Times are minutes from 00:00 of the first calendar day.

    Attributes:
        distance_km: (centers, centers)
        capacity: (vehicle types, dimensions)
        banned: (centers, vehicle types): True if the type may not use the center
        spot_available: Vehicles per day, or UNLIMITED
        demand_quantity: (demands, dimensions)
        speed_kmh: (vehicle types,), NaN where not given
        handling_capacity: (centers,) primary quantity per day, inf: no limit
        vehicle_limit: (centers, vehicle types) per day, or UNLIMITED
        demand_ready: Minutes
        demand_due: Minutes, inf where there is no due time
        timed: Costs or due times depend on when vehicles run
    """

    centers: tuple[str, ...]
    days: tuple[dt.date, ...]
    dimensions: tuple[str, ...]
    primary_dimension: int
    vehicle_types: tuple[str, ...]
    distance_km: NDArray[np.float64]
    capacity: NDArray[np.float64]
    banned: NDArray[np.bool_]
    spot_type: NDArray[np.int64]
    spot_fixed: NDArray[np.float64]
    spot_per_km: NDArray[np.float64]
    spot_available: NDArray[np.int64]
    owned_type: NDArray[np.int64]
    owned_origin: NDArray[np.int64]
    owned_destination: NDArray[np.int64]
    owned_count: NDArray[np.int64]
    owned_fixed: NDArray[np.float64]
    owned_per_km: NDArray[np.float64]
    owned_mandatory: NDArray[np.bool_]
    owned_stopovers: NDArray[np.bool_]
    demand_ids: tuple[str, ...]
    demand_origin: NDArray[np.int64]
    demand_destination: NDArray[np.int64]
    demand_day: NDArray[np.int64]
    demand_quantity: NDArray[np.float64]
    rules: Rules
    speed_kmh: NDArray[np.float64]
    spot_per_hour: NDArray[np.float64]
    owned_per_hour: NDArray[np.float64]
    handling_capacity: NDArray[np.float64]
    vehicle_limit: NDArray[np.int64]
    demand_ready: NDArray[np.float64]
    demand_due: NDArray[np.float64]
    timed: bool

    @property
    def scheduled(self) -> bool:
        """True if any rule depends on times, deferral or shared center limits."""
        return bool(
            self.timed
            or np.isfinite(self.handling_capacity).any()
            or (self.vehicle_limit != UNLIMITED).any()
            or self.rules.service.max_deferral_days > 0
            or np.isfinite(self.demand_due).any()
        )

    def __repr__(self) -> str:
        return (
            f"LineHaulInstance(centers={len(self.centers)}, days={len(self.days)}, "
            f"vehicle_types={len(self.vehicle_types)}, spot_offers={self.spot_type.size}, "
            f"owned_groups={self.owned_type.size}, demands={len(self.demand_ids)}, "
            f"dimensions={self.dimensions})"
        )


def compile_spec(spec: LineHaulSpec) -> LineHaulInstance:
    """Convert a validated specification into a :class:`LineHaulInstance`."""
    centers = tuple(center.id for center in spec.centers)
    center_index = {center: position for position, center in enumerate(centers)}
    vehicle_types = tuple(vehicle.name for vehicle in spec.vehicle_types)
    type_index = {name: position for position, name in enumerate(vehicle_types)}
    dimensions = spec.dimensions
    start = spec.calendar.start

    banned = np.zeros((len(centers), len(vehicle_types)), dtype=np.bool_)
    for center in spec.centers:
        for name in center.banned_vehicle_types:
            banned[center_index[center.id], type_index[name]] = True

    spot, owned, demand = spec.fleet.spot, spec.fleet.owned, spec.demand
    capacity = np.array(
        [[vehicle.capacity[d] for d in dimensions] for vehicle in spec.vehicle_types],
        dtype=np.float64,
    )
    quantity = np.array(
        [[row.quantity[d] for d in dimensions] for row in demand], dtype=np.float64
    ).reshape(len(demand), len(dimensions))
    limits = np.full((len(centers), len(vehicle_types)), UNLIMITED, dtype=np.int64)
    for center in spec.centers:
        for name, count in center.vehicle_limits.items():
            limits[center_index[center.id], type_index[name]] = count

    return LineHaulInstance(
        centers=centers,
        days=spec.days,
        dimensions=dimensions,
        primary_dimension=dimensions.index(spec.primary_dimension),
        vehicle_types=vehicle_types,
        distance_km=_read_only(_distance_matrix(spec, center_index)),
        capacity=_read_only(capacity),
        banned=_read_only(banned),
        spot_type=_ints(type_index[offer.type] for offer in spot),
        spot_fixed=_floats(offer.cost.fixed for offer in spot),
        spot_per_km=_floats(offer.cost.per_km for offer in spot),
        spot_available=_ints(
            UNLIMITED if offer.available is None else offer.available for offer in spot
        ),
        owned_type=_ints(type_index[group.type] for group in owned),
        owned_origin=_ints(center_index[group.origin] for group in owned),
        owned_destination=_ints(center_index[group.destination] for group in owned),
        owned_count=_ints(group.count for group in owned),
        owned_fixed=_floats(group.cost.fixed for group in owned),
        owned_per_km=_floats(group.cost.per_km for group in owned),
        owned_mandatory=_bools(group.mandatory for group in owned),
        owned_stopovers=_bools(group.stopovers for group in owned),
        demand_ids=spec.demand_ids,
        demand_origin=_ints(center_index[row.origin] for row in demand),
        demand_destination=_ints(center_index[row.destination] for row in demand),
        demand_day=_ints((row.date - start).days for row in demand),
        demand_quantity=_read_only(quantity),
        rules=spec.rules,
        speed_kmh=_floats(
            np.nan if vehicle.speed_kmh is None else vehicle.speed_kmh
            for vehicle in spec.vehicle_types
        ),
        spot_per_hour=_floats(offer.cost.per_hour for offer in spot),
        owned_per_hour=_floats(group.cost.per_hour for group in owned),
        handling_capacity=_floats(
            np.inf if center.handling_capacity is None else center.handling_capacity
            for center in spec.centers
        ),
        vehicle_limit=_read_only(limits),
        demand_ready=_floats(_minutes(start, row.date, row.ready) for row in demand),
        demand_due=_floats(_due(spec, row) for row in demand),
        timed=needs_time(spec),
    )


def _minutes(start: dt.date, date: dt.date, time: dt.time | None) -> float:
    clock = time or dt.time()
    return (date - start).days * 1440.0 + clock.hour * 60.0 + clock.minute + clock.second / 60.0


def _due(spec: LineHaulSpec, row: Demand) -> float:
    start = spec.calendar.start
    if row.due is not None:
        return (row.due - dt.datetime.combine(start, dt.time())).total_seconds() / 60.0
    hours = spec.rules.service.due_hours
    if hours is None:
        return float("inf")
    return _minutes(start, row.date, row.ready) + 60.0 * hours


def _distance_matrix(spec: LineHaulSpec, center_index: Mapping[str, int]) -> NDArray[np.float64]:
    matrix = spec.matrix
    if isinstance(matrix, ExplicitMatrix):
        size = len(center_index)
        km = np.full((size, size), np.nan, dtype=np.float64)
        for arc in matrix.arcs:
            km[center_index[arc.origin], center_index[arc.destination]] = arc.km
        if matrix.symmetric:
            km = np.asarray(np.where(np.isnan(km), km.T, km), dtype=np.float64)
        np.fill_diagonal(km, 0.0)
    else:
        km = haversine_matrix(
            [_coordinate(center.lat) for center in spec.centers],
            [_coordinate(center.lon) for center in spec.centers],
        )
    return np.asarray(km * matrix.factor, dtype=np.float64)


def _coordinate(value: float | None) -> float:
    if value is None:
        raise ValueError("missing coordinate in a validated specification")
    return value


def _read_only[T: np.generic](array: NDArray[T]) -> NDArray[T]:
    array.setflags(write=False)
    return array


def _ints(values: Iterable[int]) -> NDArray[np.int64]:
    return _read_only(np.fromiter(values, dtype=np.int64))


def _floats(values: Iterable[float]) -> NDArray[np.float64]:
    return _read_only(np.fromiter(values, dtype=np.float64))


def _bools(values: Iterable[bool]) -> NDArray[np.bool_]:
    return _read_only(np.fromiter(values, dtype=np.bool_))
