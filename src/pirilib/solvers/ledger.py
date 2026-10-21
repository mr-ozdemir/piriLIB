"""What the centers can still handle: the shared state of a scheduled search.

Subproblems are planned one after another (see
:class:`~pirilib.solvers.decompose.Coordinator`). Before a subproblem is
planned, the :class:`Ledger` gives it a :class:`Room`: the handling capacity and
vehicle slots left at its origin and its nodes. Once its routes are chosen, they
are timed and charged to the days their handling starts, so the next
subproblems see what is left. A ledger holds several rows: one in the linear
search, one per ant when every ant builds a whole plan.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pirilib.evaluation import TOLERANCE
from pirilib.solvers._timing import Timeline
from pirilib.solvers.decompose import UNBOUNDED, Subproblem
from pirilib.spec.instance import UNLIMITED, LineHaulInstance

EXTRA_DAYS: Final = 3
"""Days after the calendar kept in the ledger, for routes that arrive after it ends."""
MIN_ROOM: Final = 1e-6
"""Handling room, in primary units, below which a center takes nothing more."""


@dataclass(frozen=True, eq=False, slots=True)
class Room:
    """Handling capacity and vehicle slots left for one subproblem.

    Places are the origin (0) and the nodes of the subproblem (1 + node); days
    start at the subproblem's day. Routes are built against the first day;
    :meth:`excess` measures them against the days their handling really starts.

    Attributes:
        handling: (rows, 1 + k, days) primary quantity, inf where unlimited
        slots: (rows, 1 + k, days, vehicle types) vehicles, UNBOUNDED where unlimited
        limit: (1 + k,) handling capacity per day, inf where unlimited
        day: Day of the first column
    """

    handling: NDArray[np.float64]
    slots: NDArray[np.int64]
    limit: NDArray[np.float64]
    day: int

    @property
    def origin_handling(self) -> NDArray[np.float64]:
        """(rows,) on the subproblem's day."""
        return self.handling[:, 0, 0]

    @property
    def node_handling(self) -> NDArray[np.float64]:
        """(rows, k) on the subproblem's day."""
        return self.handling[:, 1:, 0]

    @property
    def origin_slots(self) -> NDArray[np.int64]:
        """(rows, vehicle types) on the subproblem's day."""
        return self.slots[:, 0, 0, :]

    @property
    def node_slots(self) -> NDArray[np.int64]:
        """(rows, k, vehicle types) on the subproblem's day."""
        return self.slots[:, 1:, 0, :]

    def excess(
        self, timeline: Timeline, rows: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(rows,) handling and vehicles beyond the room, for every row of a colony."""
        places, days = self.handling.shape[1], self.handling.shape[2]
        types = self.slots.shape[3]
        stop_row = timeline.route_row[timeline.stop_route]
        days_used = np.concatenate([timeline.start_day, timeline.arrival_day])
        offset = np.clip(days_used - self.day, 0, days - 1)
        row = np.concatenate([timeline.route_row, stop_row])
        origin = np.zeros(timeline.route_row.size, np.int64)
        place = np.concatenate([origin, 1 + timeline.stop_node])
        cell = (row * places + place) * days + offset
        handling = np.zeros(rows)
        vehicles = np.zeros(rows)
        if np.isfinite(self.limit).any():
            quantity = np.concatenate([timeline.load, timeline.unload])
            used = np.bincount(cell, weights=quantity, minlength=rows * places * days)
            room = np.broadcast_to(self.handling, (rows, places, days)).reshape(-1)
            with np.errstate(invalid="ignore"):
                over = used - np.maximum(room, 0.0)
            limit = np.where(np.isfinite(self.limit), self.limit, 1.0)
            scale = np.tile(np.repeat(limit, days), rows)
            handling = np.where(over > TOLERANCE * scale, over, 0.0).reshape(rows, -1).sum(axis=1)
        if (self.slots < UNBOUNDED // 2).any():
            kind = np.concatenate([timeline.route_type, timeline.route_type[timeline.stop_route]])
            count = np.bincount(cell * types + kind, minlength=rows * places * days * types)
            slots = np.broadcast_to(self.slots, (rows, places, days, types)).reshape(-1)
            excess = np.clip(count - np.maximum(slots, 0), 0, None)
            vehicles = excess.reshape(rows, -1).sum(axis=1).astype(np.float64)
        return handling, vehicles


class Ledger:
    """Handling capacity and vehicle slots left at every center and day.

    Attributes:
        handling: (rows, centers, days) primary quantity, inf where unlimited
        vehicles: (rows, centers, days, vehicle types), UNBOUNDED where unlimited
    """

    def __init__(self, instance: LineHaulInstance, rows: int = 1) -> None:
        self.instance = instance
        days = len(instance.days) + EXTRA_DAYS
        centers, types = instance.vehicle_limit.shape
        self.handling = np.broadcast_to(
            instance.handling_capacity[None, :, None], (rows, centers, days)
        ).copy()
        limits = np.where(instance.vehicle_limit == UNLIMITED, UNBOUNDED, instance.vehicle_limit)
        self.vehicles = np.broadcast_to(
            limits[None, :, None, :], (rows, centers, days, types)
        ).astype(np.int64)

    def room(self, problem: Subproblem, rows: NDArray[np.intp] | None = None) -> Room:
        """A snapshot of the room left for ``problem`` (all rows unless given)."""
        chosen = np.arange(self.handling.shape[0]) if rows is None else rows
        places = np.concatenate([[problem.origin], problem.nodes]).astype(np.intp)
        day = problem.day
        return Room(
            handling=self.handling[np.ix_(chosen, places)][:, :, day:].copy(),
            slots=self.vehicles[np.ix_(chosen, places)][:, :, day:, :].copy(),
            limit=self.instance.handling_capacity[places],
            day=day,
        )

    def charge(self, problem: Subproblem, timeline: Timeline) -> None:
        """Take the handling and vehicles of timed routes; colony rows are ledger rows."""
        last = self.handling.shape[2] - 1
        start = np.clip(timeline.start_day, 0, last)
        arrival = np.clip(timeline.arrival_day, 0, last)
        rows = timeline.route_row
        stop_rows = rows[timeline.stop_route]
        centers = problem.nodes[timeline.stop_node]
        origin = np.full(rows.size, problem.origin, dtype=np.int64)
        np.subtract.at(self.handling, (rows, origin, start), timeline.load)
        np.subtract.at(self.handling, (stop_rows, centers, arrival), timeline.unload)
        np.subtract.at(self.vehicles, (rows, origin, start, timeline.route_type), 1)
        stop_types = timeline.route_type[timeline.stop_route]
        np.subtract.at(self.vehicles, (stop_rows, centers, arrival, stop_types), 1)
