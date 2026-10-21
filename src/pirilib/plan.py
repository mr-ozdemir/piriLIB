"""Plans: the vehicle dispatches that carry the demand.

A plan is a list of dispatches. A dispatch is one vehicle that leaves one center
on one day and visits one or more stops in order, carrying loads of demands to
them. Engines produce the array form, :class:`Plan`, and the evaluator reads it;
:class:`Dispatch` is the convenient form for building or inspecting plans by
hand.

Everything is index-based: days, centers, demands, owned vehicle groups and spot
offers are positions in the :class:`~pirilib.spec.LineHaulInstance`.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from pirilib.errors import PlanError
from pirilib.spec._format import bullets

__all__ = [
    "OWNED",
    "SPOT",
    "Dispatch",
    "Kind",
    "Load",
    "Plan",
]

if TYPE_CHECKING:
    from pirilib.spec.instance import LineHaulInstance

OWNED: Final = 0
"""Code of :attr:`Plan.kind` for a vehicle of an owned group."""
SPOT: Final = 1
"""Code of :attr:`Plan.kind` for a vehicle hired from a spot offer."""

Kind = Literal["owned", "spot"]
_CODES: Final[Mapping[str, int]] = {"owned": OWNED, "spot": SPOT}
_KINDS: Final[tuple[Kind, Kind]] = ("owned", "spot")
_SHOWN: Final = 5


@dataclass(frozen=True, slots=True)
class Load:
    """A quantity of one demand, per dimension, carried by a dispatch."""

    demand: int
    quantity: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Dispatch:
    """One vehicle leaving ``origin`` on ``day`` and visiting ``stops`` in order.

    ``source`` is the owned vehicle group or the spot offer, depending on ``kind``.
    ``start`` is when loading begins, in minutes after 00:00 of ``day`` (it may be
    more than a day); ``None`` means as soon as every load is ready.
    """

    day: int
    kind: Kind
    source: int
    origin: int
    stops: tuple[int, ...]
    loads: tuple[Load, ...] = ()
    start: float | None = None


@dataclass(frozen=True, eq=False, slots=True)
class Plan:
    """Dispatches in array form.

    Per dispatch: ``day``, ``kind`` (:data:`OWNED` or :data:`SPOT`), ``source``
    and ``origin``. The stops of dispatch ``i`` are
    ``stop_center[stop_start[i]:stop_start[i + 1]]``, in visiting order. Per load:
    the dispatch that carries it, the demand it belongs to and the quantity per
    dimension. ``start`` holds, per dispatch, the minutes after 00:00 of its day
    when loading begins, NaN for as soon as every load is ready (the default).
    Arrays are converted to their dtypes and made read-only.
    """

    day: NDArray[np.int64]
    kind: NDArray[np.int8]
    source: NDArray[np.int64]
    origin: NDArray[np.int64]
    stop_start: NDArray[np.int64]
    stop_center: NDArray[np.int64]
    load_dispatch: NDArray[np.int64]
    load_demand: NDArray[np.int64]
    load_quantity: NDArray[np.float64]
    start: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if self.start is None:
            object.__setattr__(self, "start", np.full(np.size(self.day), np.nan))
        for name, dtype in (
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
        ):
            array = np.array(getattr(self, name), dtype=dtype)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        problems = _shape_problems(self)
        if problems:
            raise PlanError("malformed plan\n" + bullets(problems))

    @classmethod
    def from_dispatches(cls, dispatches: Iterable[Dispatch], dimensions: int) -> "Plan":
        """Build a plan; every load quantity must have ``dimensions`` values."""
        items = list(dispatches)
        problems = [
            f"dispatch {position}: kind must be 'owned' or 'spot', not {item.kind!r}"
            for position, item in enumerate(items)
            if item.kind not in _CODES
        ]
        loads = [(position, load) for position, item in enumerate(items) for load in item.loads]
        problems += [
            f"dispatch {position}: load of demand {load.demand} has {len(load.quantity)} "
            f"quantities, expected {dimensions}"
            for position, load in loads
            if len(load.quantity) != dimensions
        ]
        if problems:
            raise PlanError("malformed plan\n" + bullets(problems))
        return cls(
            day=np.array([item.day for item in items], dtype=np.int64),
            kind=np.array([_CODES[item.kind] for item in items], dtype=np.int8),
            source=np.array([item.source for item in items], dtype=np.int64),
            origin=np.array([item.origin for item in items], dtype=np.int64),
            stop_start=np.cumsum([0, *(len(item.stops) for item in items)], dtype=np.int64),
            stop_center=np.array([stop for item in items for stop in item.stops], dtype=np.int64),
            load_dispatch=np.array([position for position, _ in loads], dtype=np.int64),
            load_demand=np.array([load.demand for _, load in loads], dtype=np.int64),
            load_quantity=np.array([load.quantity for _, load in loads], dtype=np.float64).reshape(
                len(loads), dimensions
            ),
            start=np.array(
                [np.nan if item.start is None else item.start for item in items], dtype=np.float64
            ),
        )

    @classmethod
    def empty(cls, dimensions: int) -> "Plan":
        """A plan without dispatches."""
        return cls.from_dispatches((), dimensions)

    def __len__(self) -> int:
        return int(self.day.size)

    @property
    def dimensions(self) -> int:
        """The number of quantity dimensions of the loads."""
        return int(self.load_quantity.shape[1])

    def stops(self, position: int) -> NDArray[np.int64]:
        """The stops of dispatch ``position``, in visiting order."""
        return self.stop_center[self.stop_start[position] : self.stop_start[position + 1]]

    def dispatch(self, position: int) -> Dispatch:
        """The dispatch at ``position`` with its stops and loads."""
        rows = np.flatnonzero(self.load_dispatch == position)
        return Dispatch(
            day=int(self.day[position]),
            kind=_KINDS[int(self.kind[position])],
            source=int(self.source[position]),
            origin=int(self.origin[position]),
            stops=tuple(int(stop) for stop in self.stops(position)),
            loads=tuple(
                Load(int(self.load_demand[row]), tuple(float(q) for q in self.load_quantity[row]))
                for row in rows
            ),
            start=None
            if np.isnan(self.start_minutes[position])
            else float(self.start_minutes[position]),
        )

    @property
    def start_minutes(self) -> NDArray[np.float64]:
        """``start``, typed as an array (it is never None after construction)."""
        assert self.start is not None
        return self.start

    def dispatches(self) -> Iterator[Dispatch]:
        """The dispatches in plan order."""
        return (self.dispatch(position) for position in range(len(self)))

    def check(self, instance: "LineHaulInstance") -> None:
        """Raise :class:`~pirilib.PlanError` if an index does not exist in ``instance``."""
        problems = _index_problems(self, instance)
        if problems:
            raise PlanError("plan does not fit the instance\n" + bullets(problems))

    def __repr__(self) -> str:
        return (
            f"Plan(dispatches={len(self)}, stops={self.stop_center.size}, "
            f"loads={self.load_demand.size}, dimensions={self.dimensions})"
        )


def _shape_problems(plan: Plan) -> list[str]:
    problems: list[str] = []
    size = plan.day.size
    for name in ("day", "kind", "source", "origin", "stop_start", "stop_center"):
        if getattr(plan, name).ndim != 1:
            problems.append(f"{name} must be one-dimensional")
    for name in ("kind", "source", "origin", "start"):
        if getattr(plan, name).size != size:
            problems.append(f"{name} has {getattr(plan, name).size} entries, day has {size}")
    start = plan.stop_start
    if start.ndim == 1:
        if start.size != size + 1:
            problems.append(f"stop_start has {start.size} entries, expected {size + 1}")
        elif start[0] != 0 or start[-1] != plan.stop_center.size or np.any(np.diff(start) < 0):
            problems.append(
                "stop_start must rise from 0 to the number of stops "
                f"({plan.stop_center.size}) without decreasing"
            )
    if plan.load_quantity.ndim != 2:
        problems.append("load_quantity must be two-dimensional (loads, dimensions)")
    else:
        loads = plan.load_quantity.shape[0]
        for name in ("load_dispatch", "load_demand"):
            array = getattr(plan, name)
            if array.ndim != 1 or array.size != loads:
                problems.append(f"{name} must have one entry per load ({loads})")
        bad = np.flatnonzero(
            ~np.all(np.isfinite(plan.load_quantity) & (plan.load_quantity >= 0), 1)
        )
        if bad.size:
            problems.append(
                f"load quantities must be finite and not negative; see loads {_some(bad)}"
            )
    begins = plan.start_minutes
    wrong = np.flatnonzero(~np.isnan(begins) & ~(np.isfinite(begins) & (begins >= 0)))
    if begins.ndim == 1 and wrong.size:
        problems.append(
            f"start must be NaN or finite and not negative; see dispatches {_some(wrong)}"
        )
    if plan.kind.ndim == 1:
        bad = np.flatnonzero((plan.kind != OWNED) & (plan.kind != SPOT))
        if bad.size:
            problems.append(
                f"kind must be {OWNED} (owned) or {SPOT} (spot); see dispatches {_some(bad)}"
            )
    return problems


def _index_problems(plan: Plan, instance: "LineHaulInstance") -> list[str]:
    is_spot = plan.kind == SPOT
    checks: list[tuple[str, NDArray[Any], NDArray[np.bool_] | None, int, str]] = [
        ("day", plan.day, None, len(instance.days), "dispatches"),
        ("origin", plan.origin, None, len(instance.centers), "dispatches"),
        ("stop", plan.stop_center, None, len(instance.centers), "stops"),
        ("owned group", plan.source, ~is_spot, instance.owned_type.size, "dispatches"),
        ("spot offer", plan.source, is_spot, instance.spot_type.size, "dispatches"),
        ("load dispatch", plan.load_dispatch, None, len(plan), "loads"),
        ("load demand", plan.load_demand, None, len(instance.demand_ids), "loads"),
    ]
    problems: list[str] = []
    for label, values, mask, limit, unit in checks:
        outside = (values < 0) | (values >= limit)
        bad = np.flatnonzero(outside if mask is None else outside & mask)
        if bad.size:
            problems.append(f"{label} index outside 0..{limit - 1}; see {unit} {_some(bad)}")
    if plan.dimensions != len(instance.dimensions):
        problems.append(
            f"loads have {plan.dimensions} quantity dimensions, the instance has "
            f"{len(instance.dimensions)}"
        )
    return problems


def _some(positions: NDArray[np.intp]) -> str:
    shown = ", ".join(str(int(position)) for position in positions[:_SHOWN])
    return shown + (f" and {positions.size - _SHOWN} more" if positions.size > _SHOWN else "")
