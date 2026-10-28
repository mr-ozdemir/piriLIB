"""Synthetic problems: :func:`pirilib.synthetic.generate` and :func:`pirilib.synthetic.write`.

Unit tests (level marker ``unit``, decision K235, given by ``tests/conftest.py``) of the public
:mod:`pirilib.synthetic` module. The expected behavior comes from the docstrings of the module,
of :func:`~pirilib.synthetic.generate`, :data:`~pirilib.synthetic.OBJECTIVE` and
:func:`~pirilib.synthetic.write`, and from ``docs/guide.md`` (``generate`` writes ``cost: 1``
with penalties ``coverage: 10000000`` and ``other: 1000000``):

- a generated document passes validation as it is, with the requested number of centers,
  ``days`` times ``demand_per_day`` demand items and a calendar of ``days`` days from ``start``;
- the same seed gives the same document, another seed draws other centers and demand;
- the document carries :data:`~pirilib.synthetic.OBJECTIVE` unless another objective is given;
- fewer than 2 centers, 1 day or 1 demand item per day is rejected with ``ValueError``;
- :func:`~pirilib.synthetic.write` creates the folder, writes ``problem.yaml`` with
  ``centers.csv`` and ``demand.csv`` and returns the path of ``problem.yaml``, and that file
  loads as the same problem as the inline document.

Every test fixes its seed, so every run draws the same documents. The numbers drawn come from
NumPy's ``default_rng`` (PCG64), whose stream NumPy keeps the same across versions. The
``write`` tests use the file system, but only their own ``tmp_path``.
"""

import datetime as dt
import os
from pathlib import Path
from typing import Any

import pytest

from pirilib import load_spec
from pirilib.spec import parse_spec
from pirilib.synthetic import generate, write

SEEDS = range(10)
"""Ten seeds: the size of a generated problem must not depend on the seed."""

CENTERS = 8
"""Centers requested in the seed tests: more than the minimum of 2, few enough to stay fast."""

DAYS = 2
"""Days requested in the seed tests: more than one, so the calendar has a first and a last day."""

DEMAND_PER_DAY = 20
"""Demand items per day requested in the seed tests."""

DEMAND_ITEMS = 40
"""Demand items of a problem with ``DAYS`` (2) days of ``DEMAND_PER_DAY`` (20) items."""

CALENDAR = (dt.date(2026, 1, 5), dt.date(2026, 1, 6))
"""The ``DAYS`` days of the calendar: the default ``start`` of ``generate`` and the next day."""

DEFAULT_OBJECTIVE = {"cost": 1.0, "penalties": {"coverage": 10_000_000.0, "other": 1_000_000.0}}
"""The default ``objective:`` section, as ``docs/guide.md`` and ``OBJECTIVE`` describe it."""

ROUND_TRIP: dict[str, Any] = {"centers": 6, "days": 1, "demand_per_day": 15, "seed": 5}
"""Arguments of the problem written and loaded back: small, with two centers that ban ``tir``.

With seed 5 the second (``M002``) and the sixth (``M006``) center ban ``tir``, so the
``banned_vehicle_types`` column of ``centers.csv`` is not empty;
:func:`test_round_trip_problem_has_centers_that_ban_a_vehicle_type` checks this choice.
"""

WRITTEN_FILES = ["centers.csv", "demand.csv", "problem.yaml"]
"""The files ``write`` creates, in sorted order."""

ARGUMENT_ERROR = r"^need at least 2 centers, 1 day and 1 demand per day$"
"""The whole message of the ``ValueError`` for arguments below their minimum."""


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_problem_is_valid_with_the_requested_number_of_centers(seed: int) -> None:
    """For every seed the document passes validation and has exactly ``CENTERS`` centers."""
    document = generate(centers=CENTERS, days=DAYS, demand_per_day=DEMAND_PER_DAY, seed=seed)

    spec = parse_spec(document)

    assert len(spec.centers) == CENTERS


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_problem_has_demand_per_day_items_on_every_day(seed: int) -> None:
    """For every seed two days of 20 items give ``DEMAND_ITEMS`` (40) demand items."""
    document = generate(centers=CENTERS, days=DAYS, demand_per_day=DEMAND_PER_DAY, seed=seed)

    spec = parse_spec(document)

    assert len(spec.demand) == DEMAND_ITEMS


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_calendar_runs_the_requested_days_from_the_start(seed: int) -> None:
    """For every seed the calendar of two days is 2026-01-05 (the default start) and 2026-01-06."""
    document = generate(centers=CENTERS, days=DAYS, demand_per_day=DEMAND_PER_DAY, seed=seed)

    spec = parse_spec(document)

    assert spec.days == CALENDAR


def test_same_seed_gives_the_same_document() -> None:
    """Two calls with seed 3 return equal documents: the seed alone decides every draw."""
    first = generate(seed=3)

    second = generate(seed=3)

    assert second == first


@pytest.mark.parametrize("table", ["centers", "demand"])
def test_different_seeds_draw_different_tables(table: str) -> None:
    """Seeds 3 and 4 draw different centers and different demand, not only a different name."""
    third = generate(seed=3)

    fourth = generate(seed=4)

    assert fourth[table] != third[table]


def test_document_name_carries_the_seed() -> None:
    """The document of seed 5 is named ``synthetic-5``."""
    document = generate(seed=5)

    assert document["name"] == "synthetic-5"


def test_document_carries_the_default_objective() -> None:
    """Without an ``objective`` argument the document carries ``DEFAULT_OBJECTIVE``."""
    document = generate(seed=0)

    assert document["objective"] == DEFAULT_OBJECTIVE


def test_document_carries_the_given_objective() -> None:
    """An ``objective`` argument is written as the document's ``objective:`` section."""
    objective = {"cost": 2.0, "penalties": {"coverage": 5.0}}

    document = generate(seed=0, objective=objective)

    assert document["objective"] == {"cost": 2.0, "penalties": {"coverage": 5.0}}


@pytest.mark.parametrize(
    ("max_stretch", "stopover"),
    [
        pytest.param(1.3, {"max_stretch": 1.3}, id="default-1.3"),
        pytest.param(1.6, {"max_stretch": 1.6}, id="given-1.6"),
        pytest.param(None, {}, id="none-means-no-limit"),
    ],
)
def test_max_stretch_is_written_as_the_stopover_rule(
    max_stretch: float | None, stopover: dict[str, float]
) -> None:
    """``max_stretch`` becomes ``rules.stopover.max_stretch``; ``None`` leaves the rule out."""
    document = generate(seed=0, max_stretch=max_stretch)

    assert document["rules"]["stopover"] == stopover


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"centers": 1}, id="one-center"),
        pytest.param({"days": 0}, id="no-day"),
        pytest.param({"demand_per_day": 0}, id="no-demand-per-day"),
    ],
)
def test_arguments_below_their_minimum_are_rejected(arguments: dict[str, int]) -> None:
    """One center, no day or no demand per day (one below each minimum) raises ``ValueError``."""
    with pytest.raises(ValueError, match=ARGUMENT_ERROR):
        generate(**arguments)


@pytest.mark.parametrize(
    ("arguments", "sizes"),
    [
        pytest.param({"centers": 2}, (2, 180), id="two-centers"),
        pytest.param({"days": 1}, (12, 60), id="one-day"),
        pytest.param({"demand_per_day": 1}, (12, 3), id="one-demand-per-day"),
    ],
)
def test_arguments_at_their_minimum_give_a_valid_problem(
    arguments: dict[str, int], sizes: tuple[int, int]
) -> None:
    """Each minimum is allowed; ``sizes`` are (centers, demand items) of the valid problem.

    The other arguments keep their defaults (12 centers, 3 days, 60 items per day): 2 centers
    give 3 x 60 = 180 items, 1 day gives 60 items and 1 item per day gives 3 items.
    """
    document = generate(**arguments)

    spec = parse_spec(document)

    assert (len(spec.centers), len(spec.demand)) == sizes


def test_round_trip_problem_has_centers_that_ban_a_vehicle_type() -> None:
    """The ``ROUND_TRIP`` problem bans ``tir`` at ``M002`` and ``M006``, as its docstring says."""
    centers = generate(**ROUND_TRIP)["centers"]

    banned = (centers[1]["banned_vehicle_types"], centers[5]["banned_vehicle_types"])

    assert banned == (["tir"], ["tir"])


def test_write_returns_the_path_of_the_problem_file(tmp_path: Path) -> None:
    """``write`` returns ``<folder>/problem.yaml``."""
    document = generate(**ROUND_TRIP)
    folder = tmp_path / "problem"

    path = write(document, folder)

    assert path == folder / "problem.yaml"


def test_write_creates_the_folder_with_the_problem_file_and_two_tables(tmp_path: Path) -> None:
    """``write`` creates a folder whose parent is missing too and writes exactly three files."""
    document = generate(**ROUND_TRIP)
    folder = tmp_path / "missing" / "problem"

    write(document, folder)

    assert sorted(os.listdir(folder)) == WRITTEN_FILES


def test_written_problem_loads_as_the_same_spec(tmp_path: Path) -> None:
    """Loading the written files gives the spec of the inline document.

    The oracle is :func:`~pirilib.spec.parse_spec` of the inline document: it reads the same
    tables without the CSV files, so a value lost or changed by ``write`` makes them differ.
    """
    document = generate(**ROUND_TRIP)
    path = write(document, tmp_path / "problem")

    loaded = load_spec(path)

    assert loaded == parse_spec(document)
