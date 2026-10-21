"""Evaluation of plans: costs, loads and constraint violations, computed with numpy.

:func:`evaluate` is the single place where the cost of a plan and the violation
of every rule are computed. Solvers, tuners, reports and objective functions all
read the :class:`Evaluation` it returns, so that they cannot disagree.
"""

import dataclasses
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import CATALOG, Constraint
from pirilib.plan import SPOT, Plan
from pirilib.spec.instance import UNLIMITED, LineHaulInstance

__all__ = [
    "TOLERANCE",
    "Evaluation",
    "evaluate",
]

TOLERANCE: Final = 1e-9
"""Relative tolerance of capacity, fill, stretch and coverage comparisons."""


@dataclass(frozen=True, eq=False, slots=True)
class Evaluation:
    """Everything known about a plan once it is evaluated.

    Arrays prefixed per dispatch have one entry per dispatch of :attr:`plan`;
    :attr:`served` and :attr:`unserved` have one row per demand. Quantities have
    one column per dimension of :attr:`instance`.

    Attributes:
        vehicle_type: Per dispatch
        route_km: Per dispatch: origin, then every stop in order
        direct_km: Per dispatch: origin to last stop
        load: Per dispatch and dimension
        capacity: Per dispatch and dimension
        fill: Per dispatch: load / capacity, primary dimension
        dispatch_cost: Per dispatch, including hourly cost
        start_minute: Per dispatch: loading starts (minutes, see the instance)
        end_minute: Per dispatch: unloading at the last stop ends
        load_finish: Per load: its unloading ends, NaN off the route
        load_late_hours: Per load: started hours after the due time
        served: Per demand and dimension
        unserved: Per demand and dimension, never negative
        owned_fixed_cost: Fixed cost of mandatory owned vehicles over the calendar
        service_cost: Lateness penalty
        violations: Built-in constraints first, then your own
    """

    instance: LineHaulInstance
    plan: Plan
    vehicle_type: NDArray[np.int64]
    route_km: NDArray[np.float64]
    direct_km: NDArray[np.float64]
    load: NDArray[np.float64]
    capacity: NDArray[np.float64]
    fill: NDArray[np.float64]
    dispatch_cost: NDArray[np.float64]
    start_minute: NDArray[np.float64]
    end_minute: NDArray[np.float64]
    load_finish: NDArray[np.float64]
    load_late_hours: NDArray[np.float64]
    served: NDArray[np.float64]
    unserved: NDArray[np.float64]
    owned_fixed_cost: float
    service_cost: float
    violations: Mapping[str, float]

    @property
    def total_cost(self) -> float:
        """Dispatches, the fixed cost of mandatory owned vehicles and lateness penalties."""
        return float(self.dispatch_cost.sum()) + self.owned_fixed_cost + self.service_cost

    @property
    def hours(self) -> NDArray[np.float64]:
        """Per dispatch, hours from the start of loading to the end of the last unloading."""
        return np.asarray((self.end_minute - self.start_minute) / 60.0, dtype=np.float64)

    @property
    def total_km(self) -> float:
        """The distance of all dispatches added up (km)."""
        return float(self.route_km.sum())

    @property
    def feasible(self) -> bool:
        """True when no constraint is violated."""
        return not any(self.violations.values())

    def violated(self) -> dict[str, float]:
        """The violated constraints and their amounts."""
        return {name: amount for name, amount in self.violations.items() if amount}

    def __repr__(self) -> str:
        state = "feasible" if self.feasible else f"violated={self.violated()}"
        return (
            f"Evaluation(total_cost={self.total_cost:.2f}, dispatches={len(self.plan)}, "
            f"total_km={self.total_km:.1f}, {state})"
        )


def evaluate(
    instance: LineHaulInstance, plan: Plan, constraints: Iterable[Constraint] = ()
) -> Evaluation:
    """Evaluate ``plan`` for ``instance``, with optional constraints of your own.

    Raises :class:`~pirilib.PlanError` if the plan refers to days, centers,
    vehicles or demands that the instance does not have.
    """
    plan.check(instance)
    context = _Context(instance, plan)
    evaluation = Evaluation(
        instance=instance,
        plan=plan,
        vehicle_type=_read_only(context.vehicle_type),
        route_km=_read_only(context.route_km),
        direct_km=_read_only(context.direct_km),
        load=_read_only(context.load),
        capacity=_read_only(context.capacity),
        fill=_read_only(context.load[:, instance.primary_dimension] / context.capacity_primary),
        dispatch_cost=_read_only(context.dispatch_cost()),
        start_minute=_read_only(context.start_minute),
        end_minute=_read_only(context.end_minute),
        load_finish=_read_only(context.load_finish),
        load_late_hours=_read_only(context.load_late_hours),
        served=_read_only(context.served),
        unserved=_read_only(np.clip(instance.demand_quantity - context.served, 0.0, None)),
        owned_fixed_cost=float(
            np.sum(instance.owned_count * instance.owned_fixed * instance.owned_mandatory)
            * len(instance.days)
        ),
        service_cost=float(
            instance.rules.service.penalty
            * np.sum(plan.load_quantity[:, instance.primary_dimension] * context.load_late_hours)
        ),
        violations=MappingProxyType(context.violations()),
    )
    own = list(constraints)
    if not own:
        return evaluation
    violations = dict(evaluation.violations)
    for item in own:
        if item.name in violations:
            raise ValueError(f"constraint name '{item.name}' is already in use")
        amount = float(item.violation(evaluation))
        if not (math.isfinite(amount) and amount >= 0):
            raise ValueError(f"constraint '{item.name}' returned {amount}, expected a number >= 0")
        violations[item.name] = amount
    return dataclasses.replace(evaluation, violations=MappingProxyType(violations))


class _Context:
    """Arrays shared by the computations of one evaluation."""

    def __init__(self, instance: LineHaulInstance, plan: Plan) -> None:
        self.instance, self.plan = instance, plan
        size = len(plan)
        self.counts = np.diff(plan.stop_start)
        self.has_stops = self.counts > 0
        self.stop_dispatch = np.repeat(np.arange(size, dtype=np.int64), self.counts)
        self.first_stop = plan.stop_start[:-1][self.has_stops]
        self.is_spot = plan.kind == SPOT
        self.owned_rows = np.flatnonzero(~self.is_spot)
        self.spot_rows = np.flatnonzero(self.is_spot)
        self.owned_source = plan.source[self.owned_rows]
        self.spot_source = plan.source[self.spot_rows]

        self.vehicle_type = self._per_kind(instance.owned_type, instance.spot_type, np.int64)
        km = instance.distance_km
        previous = np.empty_like(plan.stop_center)
        previous[1:] = plan.stop_center[:-1]
        previous[self.first_stop] = plan.origin[self.has_stops]
        legs = km[previous, plan.stop_center]
        self.route_km = np.bincount(self.stop_dispatch, weights=legs, minlength=size)
        self.last_stop = np.full(size, -1, dtype=np.int64)
        self.last_stop[self.has_stops] = plan.stop_center[plan.stop_start[1:][self.has_stops] - 1]
        self.direct_km = np.zeros(size)
        self.direct_km[self.has_stops] = km[
            plan.origin[self.has_stops], self.last_stop[self.has_stops]
        ]

        self.load = _sum_rows(plan.load_dispatch, plan.load_quantity, size)
        self.capacity = instance.capacity[self.vehicle_type]
        self.capacity_primary = self.capacity[:, instance.primary_dimension]
        self.served = _sum_rows(plan.load_demand, plan.load_quantity, len(instance.demand_ids))
        self._timeline(legs)

    def _timeline(self, legs: NDArray[np.float64]) -> None:
        """When every dispatch loads, travels and unloads; when every load arrives."""
        instance, plan = self.instance, self.plan
        size, primary = len(plan), instance.primary_dimension
        per_unit = instance.rules.handling.minutes_per_unit
        centers = len(instance.centers)
        stop_keys = self._stop_keys()
        order = np.argsort(stop_keys, kind="stable")
        load_keys = plan.load_dispatch * centers + instance.demand_destination[plan.load_demand]
        found = np.searchsorted(stop_keys[order], load_keys)
        inside = found < stop_keys.size
        hit = np.zeros(load_keys.size, dtype=np.bool_)
        hit[inside] = stop_keys[order][found[inside]] == load_keys[inside]
        self.load_stop = np.full(load_keys.size, -1, dtype=np.int64)
        self.load_stop[hit] = order[found[hit]]
        self.unload = np.bincount(
            self.load_stop[hit],
            weights=plan.load_quantity[hit, primary],
            minlength=plan.stop_center.size,
        )

        loading = _minutes(per_unit * self.load[:, primary])
        unloading = _minutes(per_unit * self.unload)
        speed = instance.speed_kmh[self.vehicle_type[self.stop_dispatch]]
        with np.errstate(invalid="ignore", divide="ignore"):
            travel = np.where(np.isfinite(speed), _minutes(legs / speed * 60.0), 0.0)
        ready = np.full(size, -np.inf)
        np.maximum.at(ready, plan.load_dispatch, instance.demand_ready[plan.load_demand])
        day_start = plan.day * 1440.0
        given = plan.start_minutes
        self.start_minute = np.where(
            np.isnan(given), np.maximum(day_start, ready), day_start + np.nan_to_num(given)
        )
        steps = np.cumsum(travel + unloading)
        before = np.concatenate([[0.0], steps])[plan.stop_start[:-1]]
        self.stop_finish = (
            self.start_minute[self.stop_dispatch]
            + loading[self.stop_dispatch]
            + steps
            - before[self.stop_dispatch]
        )
        self.stop_arrival = self.stop_finish - unloading
        self.end_minute = self.start_minute + loading
        last = plan.stop_start[1:][self.has_stops] - 1
        self.end_minute[self.has_stops] = self.stop_finish[last]
        self.load_finish = np.full(load_keys.size, np.nan)
        self.load_finish[hit] = self.stop_finish[self.load_stop[hit]]
        due = instance.demand_due[plan.load_demand]
        with np.errstate(invalid="ignore"):
            late = np.where(np.isfinite(due) & hit, self.load_finish - due, 0.0)
        self.load_late_hours = np.where(late > 1e-9, np.ceil(np.round(late / 60.0, 9)), 0.0)

    def _per_kind(
        self, owned: NDArray[Any], spot: NDArray[Any], dtype: type[np.generic]
    ) -> NDArray[Any]:
        """Per dispatch, the value of its owned group or of its spot offer."""
        values = np.empty(len(self.plan), dtype=dtype)
        values[self.owned_rows] = owned[self.owned_source]
        values[self.spot_rows] = spot[self.spot_source]
        return values

    def dispatch_cost(self) -> NDArray[np.float64]:
        instance = self.instance
        owned_fixed = np.where(instance.owned_mandatory, 0.0, instance.owned_fixed)
        fixed = self._per_kind(owned_fixed, instance.spot_fixed, np.float64)
        per_km = self._per_kind(instance.owned_per_km, instance.spot_per_km, np.float64)
        per_hour = self._per_kind(instance.owned_per_hour, instance.spot_per_hour, np.float64)
        hours = (self.end_minute - self.start_minute) / 60.0
        return np.asarray(fixed + per_km * self.route_km + per_hour * hours, dtype=np.float64)

    def violations(self) -> dict[str, float]:
        amounts = {
            "route": self._route(),
            "load_route": self._load_route(),
            "ready": self._ready(),
            "deferral": self._deferral(),
            "ban": self._ban(),
            "owned_line": self._owned_line(),
            "owned_availability": self._owned_availability(),
            "spot_availability": self._spot_availability(),
            "handling": self._handling(),
            "vehicle_limit": self._vehicle_limit(),
            "capacity": self._capacity(),
            "min_fill": self._min_fill(),
            "stopover": self._stopover(),
            "max_stops": self._max_stops(),
            "stretch": self._stretch(),
            "forward_only": self._forward_only(),
            "split": self._split(),
            **self._coverage(),
        }
        return {name: float(amounts[name]) for name in CATALOG}

    def _stop_keys(self) -> NDArray[np.int64]:
        return self.stop_dispatch * len(self.instance.centers) + self.plan.stop_center

    def _route(self) -> int:
        plan = self.plan
        at_origin = np.count_nonzero(plan.stop_center == plan.origin[self.stop_dispatch])
        keys = self._stop_keys()
        repeated = keys.size - np.unique(keys).size
        return int(np.count_nonzero(~self.has_stops) + at_origin + repeated)

    def _load_route(self) -> int:
        plan, instance = self.plan, self.instance
        dispatch, demand = plan.load_dispatch, plan.load_demand
        destination = dispatch * len(instance.centers) + instance.demand_destination[demand]
        wrong = (plan.origin[dispatch] != instance.demand_origin[demand]) | ~np.isin(
            destination, self._stop_keys()
        )
        return int(np.count_nonzero(wrong))

    def _ready(self) -> int:
        ready = self.instance.demand_ready[self.plan.load_demand]
        start = self.start_minute[self.plan.load_dispatch]
        return int(np.count_nonzero(ready > start + 1e-9))

    def _deferral(self) -> int:
        plan, instance = self.plan, self.instance
        delay = plan.day[plan.load_dispatch] - instance.demand_day[plan.load_demand]
        allowed = instance.rules.service.max_deferral_days
        return int(np.count_nonzero((delay < 0) | (delay > allowed)))

    def _touches(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Handling events: center, day, dispatch and primary quantity (loading, then unloading)."""
        plan = self.plan
        primary = self.instance.primary_dimension
        centers = np.concatenate([plan.origin, plan.stop_center])
        minutes = np.concatenate([self.start_minute, self.stop_arrival])
        days = np.floor(minutes / 1440.0).astype(np.int64)
        dispatches = np.concatenate([np.arange(len(plan), dtype=np.int64), self.stop_dispatch])
        quantity = np.concatenate([self.load[:, primary], self.unload])
        return centers, days, dispatches, quantity

    def _handling(self) -> float:
        capacity = self.instance.handling_capacity
        if not np.isfinite(capacity).any():
            return 0.0
        centers, days, _, quantity = self._touches()
        width = int(days.max(initial=0)) + 1
        handled = np.bincount(
            centers * width + days, weights=quantity, minlength=len(capacity) * width
        ).reshape(len(capacity), width)
        limit = capacity[:, None]
        excess = np.where(np.isfinite(limit), handled - limit, 0.0)
        return float(excess[excess > TOLERANCE * np.where(np.isfinite(limit), limit, 1.0)].sum())

    def _vehicle_limit(self) -> int:
        limits = self.instance.vehicle_limit
        if not (limits != UNLIMITED).any():
            return 0
        centers, days, dispatches, _ = self._touches()
        width = int(days.max(initial=0)) + 1
        types = limits.shape[1]
        touch = np.unique(np.column_stack([dispatches, centers, days]), axis=0)
        kind = self.vehicle_type[touch[:, 0]]
        used = np.bincount(
            (touch[:, 1] * width + touch[:, 2]) * types + kind,
            minlength=limits.shape[0] * width * types,
        ).reshape(limits.shape[0], width, types)
        limit = limits[:, None, :]
        excess = np.where(limit != UNLIMITED, np.clip(used - limit, 0, None), 0)
        return int(excess.sum())

    def _ban(self) -> int:
        banned = self.instance.banned
        at_origin = banned[self.plan.origin, self.vehicle_type]
        at_stops = banned[self.plan.stop_center, self.vehicle_type[self.stop_dispatch]]
        return int(np.count_nonzero(at_origin) + np.count_nonzero(at_stops))

    def _owned_line(self) -> int:
        instance, rows, source = self.instance, self.owned_rows, self.owned_source
        wrong_origin = self.plan.origin[rows] != instance.owned_origin[source]
        wrong_end = self.has_stops[rows] & (
            self.last_stop[rows] != instance.owned_destination[source]
        )
        return int(np.count_nonzero(wrong_origin) + np.count_nonzero(wrong_end))

    def _excess_use(
        self, rows: NDArray[np.intp], source: NDArray[np.int64], limit: NDArray[np.int64]
    ) -> int:
        days, offers = len(self.instance.days), limit.size
        keys = self.plan.day[rows] * offers + source
        used = np.bincount(keys, minlength=days * offers).reshape(days, offers)
        excess = np.clip(used - limit, 0, None)[:, limit != UNLIMITED]
        return int(excess.sum())

    def _owned_availability(self) -> int:
        return self._excess_use(self.owned_rows, self.owned_source, self.instance.owned_count)

    def _spot_availability(self) -> int:
        return self._excess_use(self.spot_rows, self.spot_source, self.instance.spot_available)

    def _capacity(self) -> float:
        excess = self.load - self.capacity
        return float(excess[excess > TOLERANCE * self.capacity].sum())

    def _min_fill(self) -> float:
        min_fill = self.instance.rules.min_fill
        share = np.where(self.is_spot, min_fill.spot, min_fill.owned)
        primary = self.load[:, self.instance.primary_dimension]
        shortfall = share * self.capacity_primary - primary
        return float(shortfall[shortfall > TOLERANCE * self.capacity_primary].sum())

    def _stopover(self) -> int:
        allowed = np.full(len(self.plan), self.instance.rules.stopover.enabled)
        allowed[self.owned_rows] &= self.instance.owned_stopovers[self.owned_source]
        return int(np.count_nonzero((self.counts > 1) & ~allowed))

    def _max_stops(self) -> int:
        limit = self.instance.rules.stopover.max_stops
        return 0 if limit is None else int(np.clip(self.counts - limit, 0, None).sum())

    def _stretch(self) -> float:
        limit = self.instance.rules.stopover.max_stretch
        if limit is None:
            return 0.0
        excess = self.route_km - limit * self.direct_km
        return float(excess[excess > TOLERANCE * self.route_km].sum())

    def _forward_only(self) -> int:
        if not self.instance.rules.stopover.forward_only:
            return 0
        plan = self.plan
        from_origin = self.instance.distance_km[plan.origin[self.stop_dispatch], plan.stop_center]
        previous = np.empty_like(from_origin)
        previous[1:] = from_origin[:-1]
        later = np.ones(from_origin.size, dtype=np.bool_)
        later[self.first_stop] = False
        return int(np.count_nonzero(later & (from_origin <= previous)))

    def _split(self) -> int:
        if self.instance.rules.split_demand:
            return 0
        loads = np.bincount(self.plan.load_demand, minlength=len(self.instance.demand_ids))
        return int(np.clip(loads - 1, 0, None).sum())

    def _coverage(self) -> dict[str, float]:
        quantity = self.instance.demand_quantity
        missing = quantity - self.served
        threshold = TOLERANCE * np.maximum(quantity, 1.0)
        return {
            "coverage": float(missing[missing > threshold].sum()),
            "overdelivery": float(-missing[-missing > threshold].sum()),
        }


def _sum_rows(
    index: NDArray[np.int64], values: NDArray[np.float64], size: int
) -> NDArray[np.float64]:
    """Sum the rows of ``values`` that share an ``index``; one output row per index."""
    totals = np.zeros((size, values.shape[1]))
    for column in range(values.shape[1]):
        totals[:, column] = np.bincount(index, weights=values[:, column], minlength=size)
    return totals


def _minutes(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Whole minutes, rounded up (ignoring float noise)."""
    return np.asarray(np.ceil(np.round(values, 9)), dtype=np.float64)


def _read_only[T: np.generic](array: NDArray[T]) -> NDArray[T]:
    array.setflags(write=False)
    return array
