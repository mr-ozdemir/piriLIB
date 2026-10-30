"""Performance tests: wall-clock time of ``ACOSolver.solve`` on two fixed synthetic problems.

They run only with ``pytest --performance``.

KPI (service oriented, measured through the public API only): the response time of one
``ACOSolver().solve(instance, GOAL, seed=SOLVER_SEED)`` call, i.e. computation time. The
timed interval starts when ``solve`` is called and ends when it returns; generating,
parsing and compiling the problem and one warm-up call are outside it. Response time is
the metric because a user waits for a single solve. Time is wall-clock
(``timeit.default_timer``, i.e. ``time.perf_counter``); CPU time (``time.process_time``,
user + system) is recorded beside it as an efficiency indicator, not asserted.

Each case has a target and a threshold. The target is the time measured for the same
problem on Windows 10, Python 3.14.3, one core, and is only
recorded. The threshold, ``budget``, is an upper bound on the median of ``RUNS`` runs and
is what the test asserts. Budgets are tens of times the target so that a shared CI runner
does not fail them; they catch order-of-magnitude regressions, and the recorded numbers
catch smaller ones when compared across releases.

With ``--junitxml`` every case records, as test-suite properties prefixed with the case
name: the median, mean, minimum and maximum wall-clock seconds, every run (the tail), the
number of runs over the target, the median CPU seconds, the target, the budget and the
objective value of the plan (quality beside time). The environment (Python, platform,
CPU count, numpy and pirilib versions, commit) is recorded once.

If a median exceeds its budget: do not release; rerun on an idle machine to rule out noise
(the recorded minimum and maximum show it); profile the case with ``cProfile`` to name the
slow function; then fix it, or set a new budget from a new
measurement.
"""

import os
import platform
import statistics
import time
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter

import numpy as np
import pytest

import pirilib
from pirilib import ACOSolver, synthetic
from pirilib import objective as obj
from pirilib.spec import LineHaulInstance, parse_spec

VIOLATION_PENALTY = 1e6
"""Objective weight of one unit of rule violation, the weight ``synthetic.OBJECTIVE``
gives to every rule but coverage: large enough that any feasible plan is preferred."""

GOAL = obj.total_cost() + VIOLATION_PENALTY * obj.violations()

RUNS = 5
"""Timed runs per case after the warm-up; the median of an odd count is one real run."""

SOLVER_SEED = 0
"""Colony seed of every run, so that every run does the same work."""

PROBLEM = {"centers": 10, "days": 2, "demand_per_day": 40, "seed": 3}
"""``synthetic.generate`` arguments of the measured problem: 10 centers, 2 days, 80
demands, untimed objective 855,347."""

SPEED_KMH = 70.0
"""Speed given to every vehicle type in the scheduled problem."""

DUE_HOURS = 24
"""Hours after which a demand is late in the scheduled problem."""

LATENESS_PENALTY = 0.5
"""Cost per unit of lateness in the scheduled problem."""

HANDLING_MINUTES_PER_UNIT = 0.01
"""Handling time per desi in the scheduled problem: 100 desi per minute."""

HANDLING_CAPACITY = 250000.0
"""Daily handling capacity of center M001 (index 0) in the scheduled problem: the smallest
measured capacity with a feasible plan."""


@dataclass(frozen=True, slots=True)
class Case:
    """A timed problem, its target and its threshold.

    Attributes:
        name: Test id and prefix of the recorded properties
        build: Returns the compiled problem; not timed
        scheduled: Whether the problem is expected to take the scheduled solver path
        target: Seconds measured for the problem; recorded only
        budget: Seconds the median of the runs must not exceed (upper bound)
    """

    name: str
    build: Callable[[], LineHaulInstance]
    scheduled: bool
    target: float
    budget: float


def untimed() -> LineHaulInstance:
    """The ``PROBLEM`` document as generated: no times, no center limits."""
    return parse_spec(synthetic.generate(**PROBLEM)).compile()


def scheduled() -> LineHaulInstance:
    """``PROBLEM`` with speeds, due times, handling times and one handling capacity."""
    document = synthetic.generate(**PROBLEM)
    document["vehicle_types"] = [
        {**vehicle, "speed_kmh": SPEED_KMH} for vehicle in document["vehicle_types"]
    ]
    document["rules"]["service"] = {"due_hours": DUE_HOURS, "penalty": LATENESS_PENALTY}
    document["rules"]["handling"] = {"minutes_per_unit": HANDLING_MINUTES_PER_UNIT}
    document["centers"][0]["handling_capacity"] = HANDLING_CAPACITY
    return parse_spec(document).compile()


UNTIMED = Case("untimed-10-centers", untimed, scheduled=False, target=0.23, budget=10.0)
"""Target: 0.23 s measured for this exact problem on the fast path."""

SCHEDULED = Case("scheduled-10-centers", scheduled, scheduled=True, target=0.58, budget=20.0)
"""Target: 0.58 s measured with a handling capacity of 250,000 desi; that problem also has
hourly costs and ready hours, so the target is the closest measurement."""

CASES = [UNTIMED, SCHEDULED]


@pytest.fixture(scope="module")
def record_environment(record_testsuite_property: Callable[[str, object], None]) -> None:
    """Record the conditions the times depend on once, beside the measurements."""
    record_testsuite_property("python", platform.python_version())
    record_testsuite_property("platform", platform.platform())
    record_testsuite_property("cpu_count", os.cpu_count())
    record_testsuite_property("numpy", np.__version__)
    record_testsuite_property("pirilib", pirilib.__version__)
    record_testsuite_property("commit", os.environ.get("GITHUB_SHA", "unknown"))


def record_measurements(
    record: Callable[[str, object], None],
    case: Case,
    wall: list[float],
    cpu: list[float],
    objective: float,
) -> None:
    """Record one case's measurements as test-suite properties named ``<case>.<value>``."""
    record(f"{case.name}.median_seconds", statistics.median(wall))
    record(f"{case.name}.mean_seconds", statistics.mean(wall))
    record(f"{case.name}.min_seconds", min(wall))
    record(f"{case.name}.max_seconds", max(wall))
    record(f"{case.name}.run_seconds", wall)
    record(f"{case.name}.runs_over_target", int(np.count_nonzero(np.greater(wall, case.target))))
    record(f"{case.name}.median_cpu_seconds", statistics.median(cpu))
    record(f"{case.name}.target_seconds", case.target)
    record(f"{case.name}.budget_seconds", case.budget)
    record(f"{case.name}.objective", objective)


@pytest.mark.parametrize("case", CASES, ids=attrgetter("name"))
def test_only_the_scheduled_case_builds_a_scheduled_instance(case: Case) -> None:
    """The untimed case reaches the fast path and the scheduled one the scheduled path."""
    instance = case.build()

    path_is_scheduled = instance.scheduled

    assert path_is_scheduled is case.scheduled


def test_scheduled_problem_gives_every_vehicle_type_the_set_speed() -> None:
    """``scheduled()`` sets ``SPEED_KMH`` on each of the synthetic vehicle types."""
    expected = np.full(len(synthetic.VEHICLE_TYPES), SPEED_KMH)

    instance = scheduled()

    np.testing.assert_array_equal(instance.speed_kmh, expected)


@pytest.mark.usefixtures("record_environment")
@pytest.mark.parametrize("case", CASES, ids=attrgetter("name"))
def test_median_solve_time_stays_within_budget(
    case: Case, record_testsuite_property: Callable[[str, object], None]
) -> None:
    """The median wall-clock time of ``RUNS`` solves after a warm-up is at most the budget."""
    instance = case.build()
    solver = ACOSolver()
    warm_up = solver.solve(instance, GOAL, seed=SOLVER_SEED)

    wall = timeit.repeat(
        lambda: solver.solve(instance, GOAL, seed=SOLVER_SEED), repeat=RUNS, number=1
    )
    cpu = timeit.repeat(
        lambda: solver.solve(instance, GOAL, seed=SOLVER_SEED),
        repeat=RUNS,
        number=1,
        timer=time.process_time,
    )

    record_measurements(record_testsuite_property, case, wall, cpu, warm_up.objective)
    assert statistics.median(wall) <= case.budget
