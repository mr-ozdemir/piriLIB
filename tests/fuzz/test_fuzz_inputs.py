"""Fuzz tests of the readers that take outside input: plan tables and problem documents.

Each test generates many inputs, either from an alphabet with the separators and
special characters the formats use, or by mutating a valid input. The oracle is
the documented contract: a reader returns its result or raises the library's
own error type (:class:`~pirilib.PlanError`, :class:`~pirilib.SpecError`);
any other exception is a defect. The risk this lowers: a user's malformed file must end in
a message the command line can print, never in a traceback.

Every generator draws from a bounded space (the sizes are named below) and runs with the
budget of the active Hypothesis profile (``tests/conftest.py``): 100 fixed (derandomized)
examples in ``dev`` and ``ci``, 2000 random ones in the weekly ``fuzz`` run. The generated
inputs are text, because the readers take files a person writes. Inputs that broke the
contract are kept as explicit cases, marked ``xfail`` while the defect they found is open: a
plan file that is not UTF-8, a plan cell past the CSV field limit and YAML nested deeper than
the Python stack (outside the generated space), and a vehicle count past the 64-bit range
(drawn by the ``fuzz`` profile, which therefore fails until that defect is fixed).

The input categories of a fuzz run, and where each one is drawn:

- empty, short and long input: lists and texts from size 0, the boundary tables of
  :data:`BOUNDARY_TABLES`, the long YAML line, the oversized cell;
- integer limits, zero and negative numbers: :data:`MUTATIONS`, :data:`ODD_VALUES`;
- special values and characters: NaN, infinity, ``None``, NUL, tab, newline and carriage
  return, non-breaking space and Turkish letters in :data:`ALPHABET`;
- format string characters: ``%s%x%n`` in :data:`TOKENS` and :data:`YAML_KEYWORDS`;
- separators, slashes, quotes and unprintable characters: :data:`ALPHABET`,
  :data:`YAML_ALPHABET`;
- words of the formats themselves: plan sources and dates in :data:`TOKENS`, center and
  type names in :data:`DOCUMENT_TOKENS`, YAML tags and keywords in :data:`YAML_KEYWORDS`.
"""

import contextlib
import copy
import csv
import dataclasses
import functools
import math
import operator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from pirilib import PlanError, SpecError, evaluate
from pirilib.plan import Dispatch, Load
from pirilib.planio import read_plan, write_plan
from pirilib.spec import LineHaulInstance, load_spec, parse_spec

HEADER = "dispatch,date,start,kind,vehicle_type,source,origin,stops,demand,destination,desi"
"""The plan columns for the ``document`` fixture, which has one quantity dimension (desi)."""

COLUMNS = len(HEADER.split(","))
"""The number of plan columns (11)."""

CELL_SIZE = 12
"""Longest generated cell: longer than a date (10 characters); longer names come from TOKENS."""

ROW_SIZE = COLUMNS + 2
"""Most cells in a generated row: rows can be short, exactly as wide as the header or wider."""

TABLE_ROWS = 6
"""Most rows in a generated table: twice the seed plan, so dispatch numbers can repeat."""

ALPHABET = st.sampled_from(
    list("0123456789ABCabc,;|.:-_ []()'\"%/\\\t\n\r\x00\u0131\u011f\u015f\u00a0")
)
"""Plan characters: digits, center letters, the CSV and plan separators, brackets, quotes,
``%``, slashes, tab, newline, carriage return, NUL, Turkish letters and a non-breaking
space."""

TOKENS = st.sampled_from(
    ["fleet.spot[0]", "fleet.owned[0]", "D000001", "2026-01-05", "A|B", "%s%x%n"]
)
"""Whole plan values a random text rarely hits: sources, a demand id, a calendar day, a
stop list and a format string."""

CELLS = st.one_of(st.text(ALPHABET, min_size=0, max_size=CELL_SIZE), TOKENS)

MUTATIONS = st.sampled_from(
    ["-1", "1e309", "nan", "inf", "-0", "99:99", "2026-02-30", "fleet.spot[9]"]
)
"""Cell values at numeric, calendar and index limits: negative, overflowing to infinity, NaN,
infinity, negative zero, an impossible time, an impossible date and a spot offer the
``document`` fixture does not have."""

BOUNDARY_TABLES = [
    [],
    [[]],
    [["1"]],
    [["1"] * (ROW_SIZE - 1)] * (TABLE_ROWS - 1),
    [["1"] * ROW_SIZE] * TABLE_ROWS,
]
"""Generated tables at the size limits: no row, one empty row, one row of one cell,
``max - 1`` rows of ``max - 1`` cells and ``max`` rows of ``max`` cells."""

SHARED_FIXTURES = settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
"""Lets a Hypothesis test use function scoped fixtures, shared by all of its examples.

That is safe here because no example changes them: ``document`` is copied before it is
mutated, ``instance`` is read-only and every example rewrites its own file in ``tmp_path``.
"""

VALID_ROWS = [
    ["1", "2026-01-05", "09:00", "spot", "big", "fleet.spot[0]", "A", "B", "D000001", "B", "900"],
    ["2", "2026-01-05", "", "spot", "small", "fleet.spot[1]", "A", "C", "D000002", "C", "250"],
    ["3", "2026-01-06", "26:30", "owned", "big", "fleet.owned[0]", "A", "B", "", "", ""],
]
"""The seed plan for the ``document`` fixture: a spot ``big`` with a start time, a spot
``small`` without one (as soon as loaded) and an empty owned run starting past midnight."""

SEED_DISPATCHES = [
    Dispatch(
        day=0,
        kind="spot",
        source=0,
        origin=0,
        stops=(1,),
        loads=(Load(demand=0, quantity=(900.0,)),),
        start=540.0,
    ),
    Dispatch(
        day=0,
        kind="spot",
        source=1,
        origin=0,
        stops=(2,),
        loads=(Load(demand=1, quantity=(250.0,)),),
        start=None,
    ),
    Dispatch(day=1, kind="owned", source=0, origin=0, stops=(1,), loads=(), start=1590.0),
]
""":data:`VALID_ROWS` as dispatches, by hand from the plan format: days, sources, centers and
demands are positions (``2026-01-05`` is day 0, ``A`` is 0, ``B`` 1, ``C`` 2, ``D000001`` is
demand 0); ``09:00`` is minute 540 and ``26:30`` minute 1590 of the dispatch day."""

WRITTEN_DISPATCHES = [
    SEED_DISPATCHES[0],
    dataclasses.replace(SEED_DISPATCHES[1], start=0.0),
    SEED_DISPATCHES[2],
]
""":data:`SEED_DISPATCHES` as :func:`write_plan` writes them: with the start the evaluation
resolved. The second dispatch has no start time, so it starts as soon as its load is ready,
and a demand without a ``ready`` time is ready at 00:00 of its day (minute 0)."""

INT64_MIN = -(2**63)
"""The smallest 64-bit integer, the integer type of the compiled arrays."""

INT64_PAST_MAX = 2**63
"""One past the largest 64-bit integer: a number a compiled array cannot hold."""

DOCUMENT_TOKENS = st.sampled_from(
    ["A", "B", "big", "small", "haversine", "2026-01-05", "2026-01-07", "pirilib/linehaul@0.1"]
)
"""Words of the problem format: a value that is valid somewhere may be wrong where it lands
(a repeated center id, a day outside the calendar, a type name as a number)."""

ODD_VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(INT64_MIN, INT64_PAST_MAX),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(ALPHABET, min_size=0, max_size=8),
    DOCUMENT_TOKENS,
    st.lists(st.integers(-3, 3), min_size=0, max_size=3),
    st.dictionaries(st.text(ALPHABET, min_size=0, max_size=4), st.integers(-3, 3), max_size=2),
)
"""Replacement values for one field of a problem document: every JSON type, numbers at their
limits, short texts (up to 8 characters), format words, and small lists (up to 3 integers
from -3 to 3) and dicts (up to 2 keys of up to 4 characters) that have the wrong shape for
any scalar field."""

NO_DISTANCES = np.empty((0, 0))
"""What a rejected document leaves to check: no distances at all."""

YAML_ALPHABET = st.sampled_from(list("abc:-[]{}'\"#&*!|>%@ \n\t0.9"))
"""YAML characters: flow and block indicators, quotes, comments, anchors, aliases, tags,
directives, white space and a few letters and digits."""

YAML_SIZE = 80
"""Longest generated YAML text: a few lines."""

YAML_KEYWORDS = st.sampled_from(
    [
        "~",
        "null",
        "yes",
        "no",
        ".inf",
        ".nan",
        "%YAML 1.1\n---\n",
        "!!python/object/apply:builtins.len [[]]",
        "&a [*a]",
        "%s%x%n",
    ]
)
"""YAML values with a meaning of their own: null, booleans of YAML 1.1, infinity, NaN, a
directive, a tag that only an unsafe loader would run, a recursive alias and a format
string."""

LONG_LINE = 100_000
"""Length of the long YAML line: far past anything generated, still quick to read."""

DEEP_NESTING = 5000
"""Depth of the nested YAML list: deeper than Python's default recursion limit of 1000."""

OVERSIZED_CELL = csv.field_size_limit() + 1
"""A cell one character longer than the CSV reader accepts (131072 by default)."""


def plan_file(directory: Path, rows: list[list[str]]) -> Path:
    """Write ``rows`` under :data:`HEADER` as ``plan.csv`` in ``directory`` and return its path."""
    path = directory / "plan.csv"
    path.write_text(
        "\n".join([HEADER, *(",".join(row) for row in rows)]) + "\n", encoding="utf-8", newline=""
    )
    return path


def leaves(document: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """Paths to every value inside nested dicts and lists; an empty container has none."""
    if isinstance(document, dict):
        return [p for key, value in document.items() for p in leaves(value, (*path, key))]
    if isinstance(document, list):
        return [p for index, value in enumerate(document) for p in leaves(value, (*path, index))]
    return [path]


def replaced(document: dict[str, Any], path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    """A copy of ``document`` with the value at ``path`` replaced by ``value``."""
    result = copy.deepcopy(document)
    functools.reduce(operator.getitem, path[:-1], result)[path[-1]] = value
    return result


def test_plan_file_writes_the_header_then_one_line_per_row(tmp_path: Path) -> None:
    rows = [["1", "2026-01-05"], ["2"]]

    path = plan_file(tmp_path, rows)

    assert path.read_text(encoding="utf-8") == f"{HEADER}\n1,2026-01-05\n2\n"


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ({"a": 1, "b": [2, {"c": 3}]}, [("a",), ("b", 0), ("b", 1, "c")]),
        ({"a": [], "b": {}, "c": None}, [("c",)]),
        (7, [()]),
    ],
    ids=["nested", "empty-containers", "scalar"],
)
def test_leaves_gives_the_path_of_every_value(
    document: Any, expected: list[tuple[Any, ...]]
) -> None:
    paths = leaves(document)

    assert paths == expected


def test_replaced_puts_the_value_at_the_path() -> None:
    document = {"a": 1, "b": [2, {"c": 3}]}

    mutant = replaced(document, ("b", 1, "c"), None)

    assert mutant == {"a": 1, "b": [2, {"c": None}]}


def test_replaced_leaves_the_original_document_unchanged() -> None:
    document = {"a": 1, "b": [2, {"c": 3}]}

    replaced(document, ("b", 1, "c"), None)

    assert document == {"a": 1, "b": [2, {"c": 3}]}


@pytest.mark.parametrize("rows", [0, 1, 2, 3])
def test_seed_plan_prefixes_read_as_their_dispatches(
    rows: int, instance: LineHaulInstance, tmp_path: Path
) -> None:
    """The seed plan, and every prefix of it down to no row, reads as its dispatches.

    The full seed is where the mutations start; the prefixes are its row counts at the
    limits (0, 1, ``max - 1``, ``max``).
    """
    path = plan_file(tmp_path, VALID_ROWS[:rows])

    plan = read_plan(instance, path)

    assert list(plan.dispatches()) == SEED_DISPATCHES[:rows]


def test_the_seed_plan_reads_back_with_resolved_starts_after_evaluate_and_write(
    instance: LineHaulInstance, tmp_path: Path
) -> None:
    """The seed plan evaluates, and the table :func:`write_plan` makes of it reads back as the
    same dispatches, each with the start the evaluation resolved."""
    evaluation = evaluate(instance, read_plan(instance, plan_file(tmp_path, VALID_ROWS)))
    again = tmp_path / "again.csv"

    write_plan(evaluation, again)

    assert list(read_plan(instance, again).dispatches()) == WRITTEN_DISPATCHES


@SHARED_FIXTURES
@given(
    cells=st.lists(st.lists(CELLS, min_size=0, max_size=ROW_SIZE), min_size=0, max_size=TABLE_ROWS)
)
@example(cells=BOUNDARY_TABLES[0])
@example(cells=BOUNDARY_TABLES[1])
@example(cells=BOUNDARY_TABLES[2])
@example(cells=BOUNDARY_TABLES[3])
@example(cells=BOUNDARY_TABLES[4])
def test_generated_plan_tables_raise_only_plan_errors(
    cells: list[list[str]], instance: LineHaulInstance, tmp_path: Path
) -> None:
    """A generated table reads and evaluates, or one of the two raises PlanError."""
    path = plan_file(tmp_path, cells)

    with contextlib.suppress(PlanError):
        evaluate(instance, read_plan(instance, path))


@SHARED_FIXTURES
@given(
    row=st.integers(0, len(VALID_ROWS) - 1),
    column=st.integers(0, COLUMNS - 1),
    value=st.one_of(CELLS, MUTATIONS),
)
def test_mutated_plan_tables_raise_only_plan_errors(
    row: int, column: int, value: str, instance: LineHaulInstance, tmp_path: Path
) -> None:
    """The seed plan with one cell replaced reads and evaluates, or raises PlanError."""
    cells = copy.deepcopy(VALID_ROWS)
    cells[row][column] = value
    path = plan_file(tmp_path, cells)

    with contextlib.suppress(PlanError):
        evaluate(instance, read_plan(instance, path))


def test_the_seed_document_loads_from_a_yaml_file(document: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "problem.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8", newline="")

    instance = load_spec(path).compile()

    assert instance.centers == ("A", "B", "C")


def test_the_seed_document_compiles_to_finite_distances(document: dict[str, Any]) -> None:
    """The unmutated document passes the same check the mutants must pass."""
    distances = parse_spec(document).compile().distance_km

    np.testing.assert_array_equal(np.isfinite(distances), True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("field", ["lat", "lon"])
def test_a_non_finite_coordinate_is_rejected_with_spec_error(
    field: str, value: float, document: dict[str, Any]
) -> None:
    """The mutants most likely to break the distances, tried every run, not only when drawn."""
    mutant = replaced(document, ("centers", 1, field), value)

    with pytest.raises(SpecError):
        parse_spec(mutant)


@SHARED_FIXTURES
@given(data=st.data())
def test_mutated_documents_raise_only_spec_errors(
    data: st.DataObject, document: dict[str, Any]
) -> None:
    """The seed document with one value replaced compiles, or raises SpecError."""
    path = data.draw(st.sampled_from(leaves(document)), label="path")
    value = data.draw(ODD_VALUES, label="value")
    mutant = replaced(document, path, value)

    with contextlib.suppress(SpecError):
        parse_spec(mutant).compile()


@SHARED_FIXTURES
@given(data=st.data())
def test_mutated_documents_that_compile_have_finite_distances(
    data: st.DataObject, document: dict[str, Any]
) -> None:
    """A mutant that passes validation has finite distances; NaN or infinity never gets in.

    A rejected mutant leaves :data:`NO_DISTANCES`, which holds nothing to check.
    """
    path = data.draw(st.sampled_from(leaves(document)), label="path")
    value = data.draw(ODD_VALUES, label="value")
    mutant = replaced(document, path, value)
    distances = NO_DISTANCES

    with contextlib.suppress(SpecError):
        distances = parse_spec(mutant).compile().distance_km

    np.testing.assert_array_equal(np.isfinite(distances), True)


@SHARED_FIXTURES
@given(text=st.one_of(st.text(YAML_ALPHABET, min_size=0, max_size=YAML_SIZE), YAML_KEYWORDS))
@example(text="")
@example(text="x" * LONG_LINE)
def test_random_yaml_files_are_rejected_with_spec_error(text: str, tmp_path: Path) -> None:
    """No generated text is a valid problem, so every one raises SpecError.

    A problem needs the ``schema`` field, and the alphabet cannot spell it.
    """
    path = tmp_path / "problem.yaml"
    path.write_text(text, encoding="utf-8", newline="")

    with pytest.raises(SpecError):
        load_spec(path)


@pytest.mark.xfail(
    raises=RecursionError,
    reason="open defect: load_spec lets the RecursionError of deeply nested YAML escape",
)
def test_deeply_nested_yaml_is_rejected_with_spec_error(tmp_path: Path) -> None:
    path = tmp_path / "problem.yaml"
    path.write_text("[" * DEEP_NESTING, encoding="utf-8", newline="")

    with pytest.raises(SpecError):
        load_spec(path)


@pytest.mark.xfail(
    raises=csv.Error,
    reason="open defect: read_plan lets csv.Error escape for a cell past the CSV field limit",
)
def test_a_plan_cell_past_the_csv_field_limit_is_rejected_with_plan_error(
    instance: LineHaulInstance, tmp_path: Path
) -> None:
    path = plan_file(tmp_path, [["1" * OVERSIZED_CELL]])

    with pytest.raises(PlanError):
        read_plan(instance, path)


@pytest.mark.xfail(
    raises=UnicodeDecodeError,
    reason="open defect: read_plan lets UnicodeDecodeError escape for a file that is not UTF-8",
)
def test_a_plan_file_that_is_not_utf8_is_rejected_with_plan_error(
    instance: LineHaulInstance, tmp_path: Path
) -> None:
    """``0xff`` never occurs in UTF-8; :func:`load_spec` turns the same case into SpecError."""
    path = tmp_path / "plan.csv"
    path.write_bytes(HEADER.encode() + b"\n1,\xff\n")

    with pytest.raises(PlanError):
        read_plan(instance, path)


@pytest.mark.xfail(
    raises=OverflowError,
    reason="open defect: compile lets OverflowError escape for a vehicle count past int64",
)
@pytest.mark.parametrize(
    "path",
    [("fleet", "owned", 0, "count"), ("fleet", "spot", 1, "available")],
    ids=["owned-count", "spot-available"],
)
def test_a_vehicle_count_past_int64_is_rejected_with_spec_error(
    path: tuple[Any, ...], document: dict[str, Any]
) -> None:
    """Found by the ``fuzz`` profile in :func:`test_mutated_documents_raise_only_spec_errors`."""
    mutant = replaced(document, path, INT64_PAST_MAX)

    with pytest.raises(SpecError):
        parse_spec(mutant).compile()
