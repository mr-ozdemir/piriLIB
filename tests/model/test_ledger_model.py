"""Model-based test of :class:`~pirilib.solvers.ledger.Ledger` (decision K239).

The model is a plain table, written here from the problem document and not from the
compiled instance: what every center can still handle and how many vehicle slots it has
left, per ledger day. The same inputs go to the model and to the ledger, and after every
step their state and the room the ledger hands out for a subproblem must agree.

Model (table model, not a finite state machine; see K239):

- State: ``handling`` (centers, ledger days) primary quantity left, inf where unlimited,
  and ``slots`` (centers, ledger days, vehicle types) vehicles left, compared only on the
  limited cells.
- Initial state: the capacities of :data:`DOCUMENT` on every ledger day
  (:data:`HANDLING_CAPACITY`, :data:`SLOT_LIMIT`). Each run starts from a fresh ledger,
  so every sequence begins in the initial state (reset).
- Input: one timed single-stop route of the first subproblem (``A`` on day 0 to ``B`` or
  ``C``): stop node, vehicle type, start minute, travel minutes and load.
- Transition: the load leaves the origin's handling on the day loading starts and the
  stop's handling on the day the vehicle arrives; one slot of the vehicle type is taken at
  both places on the same days. A day after the last ledger column counts as that column.
- Output, checked after every step: ``Ledger.handling``, ``Ledger.vehicles`` on the
  limited cells and ``Ledger.room`` of the subproblem.
- No input is refused: a load beyond what is left (up to 1000 desi at ``C``, which handles
  800) makes the room negative, and :meth:`~pirilib.solvers.ledger.Room.excess` measures it.

Coverage target, in model terms: the strategies reach both stop nodes, both vehicle types,
every ledger day and minutes past the last ledger day (start up to day 6, arrival up to
day 8, while the last column is day 5).

``Timeline`` is the parameter type of the public :meth:`Ledger.charge`; it is imported
through :mod:`pirilib.solvers.ledger`, which takes it in its signature. No public function
builds a timeline from chosen minutes, so the inputs are built here.
"""

from typing import Any, Final

import numpy as np
import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from numpy.typing import NDArray

from pirilib.solvers.decompose import MINUTES_PER_DAY, Coordinator
from pirilib.solvers.ledger import EXTRA_DAYS, Ledger, Timeline
from pirilib.spec import parse_spec

DOCUMENT: dict[str, Any] = {
    "schema": "pirilib/linehaul@0.2",
    "calendar": {"start": "2026-01-05", "end": "2026-01-07"},
    "centers": [
        {"id": "A", "lat": 0.0, "lon": 0.0, "handling_capacity": 1500.0},
        {"id": "B", "lat": 0.0, "lon": 1.0, "vehicle_limits": {"big": 2}},
        {
            "id": "C",
            "lat": 0.0,
            "lon": 2.0,
            "handling_capacity": 800.0,
            "vehicle_limits": {"small": 1},
        },
    ],
    "matrix": {"source": "haversine"},
    "vehicle_types": [
        {"name": "big", "capacity": {"desi": 1000.0}, "speed_kmh": 60.0},
        {"name": "small", "capacity": {"desi": 400.0}, "speed_kmh": 60.0},
    ],
    "fleet": {"spot": [{"type": "big", "cost": {"fixed": 300.0, "per_km": 2.0}}]},
    "demand": [
        {"origin": "A", "destination": "B", "date": "2026-01-05", "quantity": {"desi": 900.0}},
        {"origin": "A", "destination": "C", "date": "2026-01-05", "quantity": {"desi": 250.0}},
    ],
}
"""Three centers ``A``, ``B``, ``C`` (indices 0, 1, 2) and two types ``big``, ``small`` (0, 1).

``A`` handles 1500 desi a day, ``B`` has no handling limit, ``C`` handles 800. ``B`` takes
at most 2 ``big`` vehicles a day and ``C`` at most 1 ``small``; every other slot is
unlimited. The calendar has three days; both demands leave ``A`` on the first day, so the
first subproblem is ``A`` on day 0 with the nodes ``B`` and ``C``.
"""

CALENDAR_DAYS: Final = 3
"""Days of the calendar of :data:`DOCUMENT` (2026-01-05 to 2026-01-07)."""
LEDGER_DAYS: Final = CALENDAR_DAYS + EXTRA_DAYS
"""Columns of the ledger: the calendar and the extra days after it (6)."""
LAST_DAY: Final = LEDGER_DAYS - 1
"""The last ledger column (day 5), where routes after the ledger are charged."""
ORIGIN: Final = 0
"""Center ``A``, the origin of the first subproblem."""
NODES: Final = (1, 2)
"""Centers ``B`` and ``C``, the nodes of the first subproblem, in node order."""
FIRST_DAY: Final = 0
"""Day of the first subproblem."""
PLACES: Final = np.array([ORIGIN, *NODES])
"""Places of the room of the first subproblem: the origin, then the nodes."""
HANDLING_CAPACITY: Final = np.array([1500.0, np.inf, 800.0])
"""Handling capacity per day of ``A``, ``B``, ``C``; inf where the document gives none."""
SLOT_LIMIT: Final = np.array([[0, 0], [2, 0], [0, 1]], dtype=np.int64)
"""Vehicles per day of (center, type); only the cells in :data:`LIMITED` are compared."""
LIMITED: Final = np.array([[False, False], [True, False], [False, True]])
"""(center, type) cells with a vehicle limit: ``B``/``big`` and ``C``/``small``."""
LIMITED_BY_DAY: Final = np.repeat(LIMITED[:, None, :], LEDGER_DAYS, axis=1)
"""(center, ledger day, type) cells with a vehicle limit."""
MAX_START: Final = 6 * MINUTES_PER_DAY
"""Latest start minute drawn: 00:00 of day 6, one day past the last ledger column."""
MAX_TRAVEL: Final = 2 * MINUTES_PER_DAY
"""Longest travel drawn: arrivals reach day 8, past the last ledger column."""
MAX_LOAD: Final = 1000.0
"""Largest load drawn: the capacity of ``big``, more than the 800 that ``C`` handles."""
FLOAT_TOLERANCE: Final = 1e-9
"""Relative and absolute tolerance on handling quantities (sums of up to 25 loads)."""
MAX_EXAMPLES: Final = 60
"""Runs of the state machine, each from a fresh ledger."""
STEPS: Final = 25
"""Charges per run at most: enough to drive the limited slots and ``C`` below zero."""


def model_day(minute: float) -> int:
    """The model's ledger column of ``minute``: its started day, at most :data:`LAST_DAY`."""
    return min(int(minute // MINUTES_PER_DAY), LAST_DAY)


def single_stop_timeline(
    node: int, kind: int, start: float, arrival: float, load: float
) -> Timeline:
    """One route of type ``kind`` loading ``load`` at ``start``, unloading it all at ``node``."""
    return Timeline(
        route_row=np.zeros(1, np.int64),
        route_type=np.array([kind], np.int64),
        start=np.array([start]),
        end=np.array([arrival]),
        load=np.array([load]),
        stop_route=np.zeros(1, np.int64),
        stop_node=np.array([node], np.int64),
        arrival=np.array([arrival]),
        finish=np.array([arrival]),
        unload=np.array([load]),
        service=np.zeros(1),
    )


def initial_handling() -> NDArray[np.float64]:
    """The model's handling table before any charge: the capacities on every ledger day."""
    return np.repeat(HANDLING_CAPACITY[:, None], LEDGER_DAYS, axis=1)


def initial_slots() -> NDArray[np.int64]:
    """The model's slot table before any charge: the vehicle limits on every ledger day."""
    return np.repeat(SLOT_LIMIT[:, None, :], LEDGER_DAYS, axis=1)


class LedgerModel(RuleBasedStateMachine):
    """Charges single-stop routes of the first subproblem to the ledger and to the model.

    Attributes:
        handling: (centers, ledger days) primary quantity left, inf where unlimited
        slots: (centers, ledger days, types) vehicles left, meaningful on the limited cells
    """

    @initialize()
    def start(self) -> None:
        self.instance = parse_spec(DOCUMENT).compile()
        self.problem = Coordinator(self.instance).day(FIRST_DAY)[0]
        self.ledger = Ledger(self.instance)
        self.handling = initial_handling()
        self.slots = initial_slots()

    @rule(
        node=st.integers(0, len(NODES) - 1),
        kind=st.integers(0, 1),
        start=st.floats(0.0, MAX_START),
        travel=st.floats(0.0, MAX_TRAVEL),
        load=st.floats(0.0, MAX_LOAD),
    )
    def charge(self, node: int, kind: int, start: float, travel: float, load: float) -> None:
        arrival = start + travel
        timeline = single_stop_timeline(node, kind, start, arrival, load)

        self.ledger.charge(self.problem, timeline)

        self.handling[ORIGIN, model_day(start)] -= load
        self.handling[NODES[node], model_day(arrival)] -= load
        self.slots[ORIGIN, model_day(start), kind] -= 1
        self.slots[NODES[node], model_day(arrival), kind] -= 1

    @invariant()
    def handling_matches_the_model(self) -> None:
        np.testing.assert_allclose(
            self.ledger.handling[0], self.handling, rtol=FLOAT_TOLERANCE, atol=FLOAT_TOLERANCE
        )

    @invariant()
    def limited_slots_match_the_model(self) -> None:
        np.testing.assert_array_equal(
            self.ledger.vehicles[0][LIMITED_BY_DAY], self.slots[LIMITED_BY_DAY]
        )

    @invariant()
    def room_handling_is_the_model_from_the_subproblem_day_on(self) -> None:
        room = self.ledger.room(self.problem)

        np.testing.assert_allclose(
            room.handling[0],
            self.handling[PLACES, FIRST_DAY:],
            rtol=FLOAT_TOLERANCE,
            atol=FLOAT_TOLERANCE,
        )

    @invariant()
    def room_limited_slots_are_the_model_from_the_subproblem_day_on(self) -> None:
        room = self.ledger.room(self.problem)
        limited = LIMITED_BY_DAY[PLACES, FIRST_DAY:]

        np.testing.assert_array_equal(
            room.slots[0][limited], self.slots[PLACES, FIRST_DAY:][limited]
        )

    @invariant()
    def room_starts_on_the_subproblem_day(self) -> None:
        room = self.ledger.room(self.problem)

        assert room.day == FIRST_DAY


TestLedgerAgreesWithModel = LedgerModel.TestCase
TestLedgerAgreesWithModel.settings = settings(
    max_examples=MAX_EXAMPLES, stateful_step_count=STEPS, deadline=None
)


def test_first_subproblem_is_a_on_day_zero_to_b_and_c() -> None:
    """The model's places (:data:`ORIGIN`, :data:`NODES`, :data:`FIRST_DAY`) are the input's."""
    coordinator = Coordinator(parse_spec(DOCUMENT).compile())

    problem = coordinator.day(FIRST_DAY)[0]

    assert (problem.origin, problem.nodes.tolist(), problem.day) == (0, [1, 2], 0)


def test_model_handling_starts_from_the_document_capacities_on_every_ledger_day() -> None:
    """``A`` 1500, ``B`` unlimited and ``C`` 800 desi on each of the 6 ledger days."""
    expected = [[1500.0] * 6, [np.inf] * 6, [800.0] * 6]

    handling = initial_handling()

    np.testing.assert_array_equal(handling, expected)


def test_model_limited_slots_start_from_the_document_limits_on_every_ledger_day() -> None:
    """``B`` 2 ``big`` slots, then ``C`` 1 ``small`` slot, on each of the 6 ledger days."""
    expected = [2] * 6 + [1] * 6

    slots = initial_slots()

    np.testing.assert_array_equal(slots[LIMITED_BY_DAY], expected)


@pytest.mark.parametrize(
    ("minute", "day"),
    [(0.0, 0), (1439.0, 0), (1440.0, 1), (3000.0, 2), (8639.0, 5)],
    ids=["midnight-day-0", "last-minute-day-0", "midnight-day-1", "inside-day-2", "end-day-5"],
)
def test_model_day_is_the_started_day(minute: float, day: int) -> None:
    """A minute belongs to the day whose 00:00 it has passed (1440 minutes a day)."""
    assert model_day(minute) == day


@pytest.mark.parametrize(
    "minute",
    [8640.0, 10081.0, 11520.0],
    ids=["midnight-day-6", "inside-day-7", "midnight-day-8"],
)
def test_model_day_past_the_ledger_is_the_last_column(minute: float) -> None:
    """A minute after day 5 (the 3 calendar and 3 extra days end at 8640) counts as day 5."""
    assert model_day(minute) == 5
