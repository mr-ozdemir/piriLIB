"""An independent plan checker, written as plain Python loops.

:func:`check_plan` applies the rules that :func:`~pirilib.evaluate` measures,
without numpy, and explains every violation in words. Use it to audit a plan
from any source; the test suite uses it to cross-check the evaluator.
"""

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from pirilib.constraints import CATALOG
from pirilib.evaluation import TOLERANCE
from pirilib.plan import Dispatch, Plan
from pirilib.spec.instance import UNLIMITED, LineHaulInstance


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule: the constraint, the amount it adds and an explanation."""

    constraint: str
    amount: float
    message: str


@dataclass(frozen=True, slots=True)
class PlanReport:
    """The result of :func:`check_plan`."""

    violations: tuple[Violation, ...]
    total_cost: float

    @property
    def feasible(self) -> bool:
        return not self.violations

    def totals(self) -> dict[str, float]:
        """The violation amount of every built-in constraint, in catalog order."""
        totals = dict.fromkeys(CATALOG, 0.0)
        for item in self.violations:
            totals[item.constraint] += item.amount
        return totals

    def __str__(self) -> str:
        state = "feasible" if self.feasible else f"{len(self.violations)} violation(s)"
        lines = [f"total cost {self.total_cost:.2f}, {state}"]
        lines += [f"- {item.constraint}: {item.message}" for item in self.violations]
        return "\n".join(lines)


def check_plan(instance: LineHaulInstance, plan: Plan) -> PlanReport:
    """Check ``plan`` against every rule of ``instance`` and compute its cost."""
    plan.check(instance)
    return _Checker(instance).run(plan)


def _whole_minutes(value: float) -> float:
    return float(math.ceil(round(value, 9)))


def _day(minute: float) -> int:
    return math.floor(minute / 1440.0)


class _Checker:
    """Walks a plan dispatch by dispatch and collects the violations it finds."""

    def __init__(self, instance: LineHaulInstance) -> None:
        self.instance = instance
        self.km: list[list[float]] = instance.distance_km.tolist()
        self.capacity: list[list[float]] = instance.capacity.tolist()
        self.banned: list[list[bool]] = instance.banned.tolist()
        self.found: list[Violation] = []
        self.handled: defaultdict[tuple[int, int], float] = defaultdict(float)
        self.touches: Counter[tuple[int, int, int]] = Counter()
        self.service = 0.0

    def add(self, constraint: str, amount: float, message: str) -> None:
        self.found.append(Violation(constraint, float(amount), message))

    def name(self, center: int) -> str:
        return self.instance.centers[center]

    def run(self, plan: Plan) -> PlanReport:
        instance = self.instance
        cost = len(instance.days) * sum(
            float(count) * float(fixed)
            for count, fixed, mandatory in zip(
                instance.owned_count, instance.owned_fixed, instance.owned_mandatory, strict=True
            )
            if mandatory
        )
        used: Counter[tuple[str, int, int]] = Counter()
        dimensions = len(instance.dimensions)
        served = [[0.0] * dimensions for _ in instance.demand_ids]
        loads_per_demand = [0] * len(instance.demand_ids)

        for position, dispatch in enumerate(plan.dispatches()):
            cost += self.dispatch(position, dispatch)
            used[dispatch.kind, dispatch.source, dispatch.day] += 1
            for load in dispatch.loads:
                loads_per_demand[load.demand] += 1
                for column, quantity in enumerate(load.quantity):
                    served[load.demand][column] += quantity

        self.availability(used)
        self.handling()
        self.vehicle_limits()
        self.coverage(served, loads_per_demand)
        return PlanReport(tuple(self.found), cost + self.service)

    def dispatch(self, position: int, dispatch: Dispatch) -> float:
        """Check one dispatch and return its cost."""
        instance, rules, km = self.instance, self.instance.rules, self.km
        owned = dispatch.kind == "owned"
        source, origin, stops = dispatch.source, dispatch.origin, dispatch.stops
        if owned:
            vehicle = int(instance.owned_type[source])
            fixed = 0.0 if instance.owned_mandatory[source] else float(instance.owned_fixed[source])
            per_km = float(instance.owned_per_km[source])
            per_hour = float(instance.owned_per_hour[source])
        else:
            vehicle = int(instance.spot_type[source])
            fixed = float(instance.spot_fixed[source])
            per_km = float(instance.spot_per_km[source])
            per_hour = float(instance.spot_per_hour[source])
        type_name = instance.vehicle_types[vehicle]
        label = (
            f"dispatch {position} ({instance.days[dispatch.day]}, {dispatch.kind} {type_name} "
            f"from {self.name(origin)})"
        )

        if not stops:
            self.add("route", 1, f"{label} has no stops")
        seen: set[int] = set()
        for stop in stops:
            if stop == origin:
                self.add("route", 1, f"{label} stops at its origin")
            if stop in seen:
                self.add("route", 1, f"{label} visits {self.name(stop)} twice")
            seen.add(stop)

        length, here = 0.0, origin
        for stop in stops:
            length += km[here][stop]
            here = stop

        for center in (origin, *stops):
            if self.banned[center][vehicle]:
                self.add("ban", 1, f"{label}: {type_name} is banned at {self.name(center)}")

        if owned:
            line_origin = int(instance.owned_origin[source])
            line_end = int(instance.owned_destination[source])
            if origin != line_origin:
                self.add(
                    "owned_line", 1, f"{label}: the group's line starts at {self.name(line_origin)}"
                )
            if stops and stops[-1] != line_end:
                self.add(
                    "owned_line", 1, f"{label}: the group's line ends at {self.name(line_end)}"
                )

        if len(stops) > 1:
            if not rules.stopover.enabled:
                self.add(
                    "stopover", 1, f"{label} has {len(stops)} stops, but stopovers are disabled"
                )
            elif owned and not instance.owned_stopovers[source]:
                self.add(
                    "stopover",
                    1,
                    f"{label} has {len(stops)} stops; the group may not stop on the way",
                )
        limit = rules.stopover.max_stops
        if limit is not None and len(stops) > limit:
            self.add(
                "max_stops",
                len(stops) - limit,
                f"{label} has {len(stops)} stops, at most {limit} allowed",
            )
        stretch = rules.stopover.max_stretch
        if stretch is not None and stops:
            direct = km[origin][stops[-1]]
            excess = length - stretch * direct
            if excess > TOLERANCE * length:
                self.add(
                    "stretch",
                    excess,
                    f"{label}: route of {length:.1f} km is longer than {stretch} x {direct:.1f} km",
                )
        if rules.stopover.forward_only:
            for before, stop in itertools.pairwise(stops):
                if km[origin][stop] <= km[origin][before]:
                    self.add(
                        "forward_only",
                        1,
                        f"{label}: {self.name(stop)} is not farther from the origin than "
                        f"{self.name(before)}",
                    )

        self.loads(label, dispatch, vehicle)
        hours = self.timeline(label, dispatch, vehicle)
        return fixed + per_km * length + per_hour * hours

    def timeline(self, label: str, dispatch: Dispatch, vehicle: int) -> float:
        """Walk the dispatch through time: handling, lateness and readiness. Returns its hours."""
        instance = self.instance
        primary = instance.primary_dimension
        per_unit = instance.rules.handling.minutes_per_unit
        destination = [int(instance.demand_destination[item.demand]) for item in dispatch.loads]
        unload: dict[int, float] = {}
        for item, center in zip(dispatch.loads, destination, strict=True):
            if center in dispatch.stops:
                unload[center] = unload.get(center, 0.0) + item.quantity[primary]
        ready = max(
            (float(instance.demand_ready[item.demand]) for item in dispatch.loads), default=None
        )
        day_start = dispatch.day * 1440.0
        if dispatch.start is not None:
            start = day_start + dispatch.start
        else:
            start = day_start if ready is None else max(day_start, ready)
        total = sum(item.quantity[primary] for item in dispatch.loads)
        speed = float(instance.speed_kmh[vehicle])
        clock = start + _whole_minutes(per_unit * total)
        self.handled[dispatch.origin, _day(start)] += total
        touched = {(dispatch.origin, _day(start))}
        finish: dict[int, float] = {}
        here = dispatch.origin
        for stop in dispatch.stops:
            travel = 0.0 if speed != speed else _whole_minutes(self.km[here][stop] / speed * 60.0)
            arrival = clock + travel
            amount = unload.get(stop, 0.0) if stop not in finish else 0.0
            self.handled[stop, _day(arrival)] += amount
            touched.add((stop, _day(arrival)))
            clock = arrival + _whole_minutes(per_unit * amount)
            finish.setdefault(stop, clock)
            here = stop
        for center, day in touched:
            self.touches[center, day, vehicle] += 1

        penalty = instance.rules.service.penalty
        for item, center in zip(dispatch.loads, destination, strict=True):
            demand = instance.demand_ids[item.demand]
            if float(instance.demand_ready[item.demand]) > start + 1e-9:
                self.add("ready", 1, f"{label}: demand {demand} is not ready when loading starts")
            due = float(instance.demand_due[item.demand])
            if center in finish and due != float("inf"):
                late = finish[center] - due
                if late > 1e-9:
                    hours = math.ceil(round(late / 60.0, 9))
                    self.service += penalty * item.quantity[primary] * hours
        return (clock - start) / 60.0

    def handling(self) -> None:
        instance = self.instance
        for (center, day), quantity in sorted(self.handled.items()):
            capacity = float(instance.handling_capacity[center])
            excess = quantity - capacity
            if capacity != float("inf") and excess > TOLERANCE * capacity:
                self.add(
                    "handling",
                    excess,
                    f"{self.name(center)} handles {quantity:g} on day {day}, capacity {capacity:g}",
                )

    def vehicle_limits(self) -> None:
        instance = self.instance
        for (center, day, vehicle), count in sorted(self.touches.items()):
            limit = int(instance.vehicle_limit[center, vehicle])
            if limit != UNLIMITED and count > limit:
                self.add(
                    "vehicle_limit",
                    count - limit,
                    f"{count} {instance.vehicle_types[vehicle]} at {self.name(center)} on day "
                    f"{day}, at most {limit}",
                )

    def loads(self, label: str, dispatch: Dispatch, vehicle: int) -> None:
        instance = self.instance
        dimensions = instance.dimensions
        load = [0.0] * len(dimensions)
        for item in dispatch.loads:
            for column, quantity in enumerate(item.quantity):
                load[column] += quantity
            reasons = []
            if int(instance.demand_origin[item.demand]) != dispatch.origin:
                reasons.append("leaves from another center")
            delay = dispatch.day - int(instance.demand_day[item.demand])
            if not 0 <= delay <= instance.rules.service.max_deferral_days:
                self.add(
                    "deferral",
                    1,
                    f"{label}: demand {instance.demand_ids[item.demand]} leaves {delay} day(s) "
                    "after its date",
                )
            if int(instance.demand_destination[item.demand]) not in dispatch.stops:
                reasons.append("goes to a center that is not a stop")
            if reasons:
                demand = instance.demand_ids[item.demand]
                self.add("load_route", 1, f"{label}: demand {demand} " + " and ".join(reasons))

        capacity = self.capacity[vehicle]
        for column, name in enumerate(dimensions):
            excess = load[column] - capacity[column]
            if excess > TOLERANCE * capacity[column]:
                self.add(
                    "capacity",
                    excess,
                    f"{label} carries {load[column]:g} {name}, capacity is {capacity[column]:g}",
                )
        rules, primary = instance.rules, instance.primary_dimension
        share = rules.min_fill.owned if dispatch.kind == "owned" else rules.min_fill.spot
        shortfall = share * capacity[primary] - load[primary]
        if shortfall > TOLERANCE * capacity[primary]:
            self.add(
                "min_fill",
                shortfall,
                f"{label} carries {load[primary]:g} {dimensions[primary]}, the minimum fill is "
                f"{share:g} x {capacity[primary]:g}",
            )

    def availability(self, used: Counter[tuple[str, int, int]]) -> None:
        instance = self.instance
        for (kind, source, day), count in sorted(used.items()):
            if kind == "owned":
                limit, what = int(instance.owned_count[source]), f"owned group {source}"
            else:
                limit, what = int(instance.spot_available[source]), f"spot offer {source}"
            if limit != UNLIMITED and count > limit:
                self.add(
                    f"{kind}_availability",
                    count - limit,
                    f"{what} is used {count} times on {instance.days[day]}, {limit} available",
                )

    def coverage(self, served: list[list[float]], loads_per_demand: list[int]) -> None:
        instance = self.instance
        quantities: list[list[float]] = instance.demand_quantity.tolist()
        for row, (quantity, carried) in enumerate(zip(quantities, served, strict=True)):
            demand = instance.demand_ids[row]
            for column, name in enumerate(instance.dimensions):
                missing = quantity[column] - carried[column]
                threshold = TOLERANCE * max(quantity[column], 1.0)
                if missing > threshold:
                    self.add(
                        "coverage", missing, f"demand {demand}: {missing:g} {name} not carried"
                    )
                elif -missing > threshold:
                    self.add(
                        "overdelivery", -missing, f"demand {demand}: {-missing:g} {name} too many"
                    )
            if not instance.rules.split_demand and loads_per_demand[row] > 1:
                self.add(
                    "split",
                    loads_per_demand[row] - 1,
                    f"demand {demand} may not be split, but is carried in {loads_per_demand[row]} "
                    "loads",
                )
