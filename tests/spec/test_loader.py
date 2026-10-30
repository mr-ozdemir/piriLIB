"""Loading a problem file: the YAML document, the CSV tables it refers to and the problems found.

Integration tests (level marker ``integration``): each test writes its files into
its own ``tmp_path`` and reads them back through the public :func:`pirilib.load_spec`. The
expected behavior and message formats come from ``docs/guide.md`` ("Tables from CSV" and
"Validation") and the docstrings of :func:`pirilib.load_spec` and
:class:`pirilib.spec.TableSource`. A failing load reports a header line
``<problem file>: invalid problem specification`` followed by one ``- <problem>`` line per
problem, a CSV row problem being named ``<file>:<line>``.
"""

import datetime as dt
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from pirilib import SpecError, load_spec
from pirilib.spec import parse_spec

DEMAND_CSV = """\
origin,destination,date,desi
A,B,2026-01-05,900
A,C,2026-01-05,250
B,A,2026-01-06,120
"""
"""The three demand rows of the ``document`` fixture as a CSV file with the default columns.

The header is line 1, so the rows ``A,B``, ``A,C`` and ``B,A`` are lines 2, 3 and 4.
"""


def write_problem(folder: Path, document: dict[str, Any]) -> Path:
    """Write ``document`` as ``folder/problem.yaml`` and return its path."""
    path = folder / "problem.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), "utf-8")
    return path


def write_table(folder: Path, name: str, text: str) -> None:
    """Write ``text`` as the UTF-8 file ``folder/name``."""
    (folder / name).write_text(text, encoding="utf-8")


def with_demand_file(
    folder: Path, document: dict[str, Any], text: str = DEMAND_CSV, **options: Any
) -> Path:
    """Move the demand of ``document`` to ``demand.csv`` holding ``text``; write the problem file.

    The reference keeps the default columns and maps the quantity ``desi`` to the column
    ``desi``; ``options`` are further fields of the reference, such as ``delimiter``.
    """
    write_table(folder, "demand.csv", text)
    document["demand"] = {"file": "demand.csv", "quantity": {"desi": "desi"}, **options}
    return write_problem(folder, document)


def test_bundled_example_reads_its_yaml_fields(example_path: Path) -> None:
    """The name and stopover stretch are the values written in the example's problem.yaml.

    ``1.3`` is compared exactly: it is read from the same decimal literal.
    """
    spec = load_spec(example_path)

    assert (spec.name, spec.rules.stopover.max_stretch) == ("Marmara mini", 1.3)


def test_bundled_example_reads_its_centers_file(example_path: Path) -> None:
    """The example's centers.csv lists six centers, İstanbul first; Tekirdağ (sixth) bans tir."""
    spec = load_spec(example_path)

    assert len(spec.centers) == 6
    assert (spec.centers[0].id, spec.centers[5].id) == ("İstanbul", "Tekirdağ")
    assert spec.centers[5].banned_vehicle_types == ("tir",)


def test_bundled_example_reads_its_turkish_formatted_demand_file(example_path: Path) -> None:
    """The example's demand.csv has 18 rows in Turkish spreadsheet format.

    Line 2 is ``İstanbul;Kocaeli;11.05.2026;18.450,0`` and line 13 (row 12, index 11) ends in
    ``6.200,5``: ``.`` groups thousands and ``,`` is the decimal mark. Both values are exact
    binary fractions, so they are compared exactly.
    """
    spec = load_spec(example_path)

    assert len(spec.demand) == 18
    first = spec.demand[0]
    assert (first.origin, first.destination, first.date) == (
        "İstanbul",
        "Kocaeli",
        dt.date(2026, 5, 11),
    )
    assert first.quantity == {"desi": 18450.0}
    assert spec.demand[11].quantity == {"desi": 6200.5}


def test_demand_rows_without_ids_are_numbered_in_file_order(example_path: Path) -> None:
    """The example's demand file has no id column, so the ids are generated: D000001, ..."""
    spec = load_spec(example_path)

    assert spec.demand_ids[:2] == ("D000001", "D000002")


def test_demand_file_gives_the_same_spec_as_inline_demand(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """``DEMAND_CSV`` holds the fixture's inline demand rows, so both forms must agree."""
    inline = parse_spec(document)
    path = with_demand_file(tmp_path, document)

    spec = load_spec(path)

    assert spec == inline


def test_missing_problem_file_is_reported_as_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"

    with pytest.raises(SpecError, match=f"^{re.escape(f'cannot read {path}: ')}"):
        load_spec(path)


def test_problem_file_that_is_not_utf8_is_reported_with_the_byte_offset(tmp_path: Path) -> None:
    """Byte 6 (after ``name: ``) is ``0xff``, which never occurs in UTF-8 text.

    Only the decoder's reason text between the colon and the offset is left open: it is
    Python's wording, not pirilib's.
    """
    path = tmp_path / "problem.yaml"
    path.write_bytes(b"name: \xff\n")

    with pytest.raises(
        SpecError, match=f"^{re.escape(f'{path}: not valid UTF-8 text: ')}.+ at byte 6$"
    ):
        load_spec(path)


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    """The flow sequence ``[A, B`` is never closed."""
    path = tmp_path / "problem.yaml"
    path.write_text("centers: [A, B\n", encoding="utf-8")

    with pytest.raises(SpecError, match=f"^{re.escape(f'{path}: invalid YAML: ')}"):
        load_spec(path)


@pytest.mark.parametrize(
    "text",
    ["- A\n- B\n", "just a sentence\n", ""],
    ids=["sequence", "scalar", "empty-file"],
)
def test_problem_file_whose_top_level_is_not_a_mapping_is_rejected(
    tmp_path: Path, text: str
) -> None:
    """A sequence, a plain scalar and an empty file (YAML null) are not a mapping of fields."""
    path = tmp_path / "problem.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value) == f"{path}: the top level must be a mapping of fields"


def test_cell_problems_name_file_and_line(tmp_path: Path, document: dict[str, Any]) -> None:
    """Line 3 has a quantity that is not a number and line 4 an empty date; both are reported."""
    text = "origin,destination,date,desi\nA,B,2026-01-05,900\nA,C,2026-01-05,abc\nB,A,,120\n"
    path = with_demand_file(tmp_path, document, text)

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value).splitlines() == [
        f"{path}: invalid problem specification",
        "- demand.csv:3: 'desi': 'abc' is not a number",
        "- demand.csv:4: 'date' is empty",
    ]


def test_line_numbers_count_blank_lines(tmp_path: Path, document: dict[str, Any]) -> None:
    """Line 3 is blank, so the row with the non-ISO date is reported as line 4, not 3."""
    text = "origin,destination,date,desi\nA,B,2026-01-05,900\n\nA,C,05.01.2026,250\n"
    path = with_demand_file(tmp_path, document, text)

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value).splitlines() == [
        f"{path}: invalid problem specification",
        "- demand.csv:4: 'date': '05.01.2026' is not an ISO date (YYYY-MM-DD); set 'date_format'",
    ]


@pytest.mark.parametrize(
    ("row", "changed", "problem"),
    [
        (
            "A,C,2026-01-05,250",
            "A,C,2026-01-05,-250",
            "- demand.csv:3: quantity.desi: Input should be greater than or equal to 0",
        ),
        (
            "B,A,2026-01-06,120",
            "B,X,2026-01-06,120",
            "- demand.csv:4: destination: unknown center 'X'",
        ),
    ],
    ids=["negative-quantity", "unknown-center"],
)
def test_model_problems_in_rows_name_file_and_line(
    tmp_path: Path, document: dict[str, Any], row: str, changed: str, problem: str
) -> None:
    """A row that reads as a table row but breaks the model is named by its file line.

    The rows ``A,C`` and ``B,A`` are lines 3 and 4 of ``DEMAND_CSV``; a negative quantity and a
    center that is not in the problem are model problems, found after the file is read.
    """
    path = with_demand_file(tmp_path, document, DEMAND_CSV.replace(row, changed))

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value).splitlines() == [f"{path}: invalid problem specification", problem]


def test_missing_columns_are_listed_with_the_columns_found(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """The header has ``from`` and ``to`` in place of the default ``origin`` and ``destination``.

    Every missing column is reported, then the columns the file does have.
    """
    path = with_demand_file(tmp_path, document, DEMAND_CSV.replace("origin,destination", "from,to"))

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value).splitlines() == [
        f"{path}: invalid problem specification",
        "- demand (demand.csv): column 'origin' (for origin) not found",
        "- demand (demand.csv): column 'destination' (for destination) not found",
        "- demand (demand.csv): the columns in the file are ['from', 'to', 'date', 'desi']",
    ]


def test_invalid_file_reference_options_are_all_reported(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """A two-character delimiter and an encoding Python does not know are both reported."""
    path = with_demand_file(tmp_path, document, delimiter=";;", encoding="klingon")

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value).splitlines() == [
        f"{path}: invalid problem specification",
        "- demand.delimiter: String should have at most 1 character",
        "- demand.encoding: unknown encoding 'klingon'",
    ]


def test_problems_in_several_files_are_reported_together(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """The centers file does not exist and line 2 of the demand file has a word for a number.

    The reason the operating system gives for the missing file is its own wording, so only the
    text before it is compared.
    """
    document["centers"] = {"file": "nowhere.csv"}
    path = with_demand_file(tmp_path, document, DEMAND_CSV.replace("900", "nine hundred"))

    with pytest.raises(SpecError) as info:
        load_spec(path)

    lines = str(info.value).splitlines()
    assert len(lines) == 3
    assert lines[0] == f"{path}: invalid problem specification"
    assert lines[1].startswith(f"- centers: cannot read {tmp_path / 'nowhere.csv'}: ")
    assert lines[2] == "- demand.csv:2: 'desi': 'nine hundred' is not a number"


def test_turkish_spreadsheet_export_reads_with_its_format_options(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """A cp1254 file with ``;`` columns, ``,`` decimals, day-first dates and Turkish headers.

    ``1.234,5`` is 1234.5 (``.`` groups thousands) and ``0,5`` is 0.5; these and 250.0 are
    exact binary fractions, so they are compared exactly.
    """
    text = (
        "Çıkış;Varış;Tarih;Desi\nA;B;05.01.2026;1.234,5\nA;C;05.01.2026;250\nB;A;06.01.2026;0,5\n"
    )
    (tmp_path / "talep.csv").write_bytes(text.encode("cp1254"))
    document["demand"] = {
        "file": "talep.csv",
        "delimiter": ";",
        "encoding": "cp1254",
        "decimal": ",",
        "date_format": "%d.%m.%Y",
        "columns": {"origin": "Çıkış", "destination": "Varış", "date": "Tarih"},
        "quantity": {"desi": "Desi"},
    }
    path = write_problem(tmp_path, document)

    spec = load_spec(path)

    assert (spec.demand[0].origin, spec.demand[0].destination) == ("A", "B")
    assert (spec.demand[0].date, spec.demand[2].date) == (dt.date(2026, 1, 5), dt.date(2026, 1, 6))
    assert (spec.demand[0].quantity, spec.demand[1].quantity, spec.demand[2].quantity) == (
        {"desi": 1234.5},
        {"desi": 250.0},
        {"desi": 0.5},
    )


def test_demand_file_in_the_wrong_encoding_is_explained(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """``Ş`` is byte 0xDE in cp1254, which is not valid UTF-8 where it stands.

    The default encoding is ``utf-8-sig``. The decoder's reason in the parentheses is Python's
    wording, so only the text around it is compared.
    """
    (tmp_path / "demand.csv").write_bytes(
        "origin,destination,date,desi\nŞ,B,2026-01-05,1\n".encode("cp1254")
    )
    document["demand"] = {"file": "demand.csv", "quantity": {"desi": "desi"}}
    path = write_problem(tmp_path, document)

    with pytest.raises(SpecError) as info:
        load_spec(path)

    lines = str(info.value).splitlines()
    assert len(lines) == 2
    assert lines[0] == f"{path}: invalid problem specification"
    assert lines[1].startswith("- demand: demand.csv is not valid utf-8-sig text (")
    assert lines[1].endswith("); set 'encoding', for example to cp1254")


def test_utf8_byte_order_mark_is_ignored(tmp_path: Path, document: dict[str, Any]) -> None:
    """With the mark read as text the first column would be ``\\ufefforigin``, not ``origin``."""
    inline = parse_spec(document)
    (tmp_path / "demand.csv").write_text(DEMAND_CSV, encoding="utf-8-sig")
    document["demand"] = {"file": "demand.csv", "quantity": {"desi": "desi"}}
    path = write_problem(tmp_path, document)

    spec = load_spec(path)

    assert spec == inline


def test_empty_cell_in_centers_file_leaves_the_field_unset(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """Center ``B`` has an empty ``name`` cell, so its name is ``None`` and not ``""``.

    The coordinates are the fixture's, so the rest of the problem stays valid.
    """
    write_table(tmp_path, "centers.csv", "id,name,lat,lon\nA,Depot A,0,0\nB,,0,1\nC,Depot C,0,2\n")
    document["centers"] = {"file": "centers.csv"}
    path = write_problem(tmp_path, document)

    spec = load_spec(path)

    assert (spec.centers[0].name, spec.centers[1].name, spec.centers[2].name) == (
        "Depot A",
        None,
        "Depot C",
    )


def test_banned_types_in_centers_file_are_split_at_the_bar(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """The cell `` big | small `` gives both types without the spaces; empty cells give none.

    ``C`` then bans every vehicle type, so the fixture's demand row to ``C`` (index 1) is
    dropped: no vehicle could serve it and the problem would be invalid.
    """
    write_table(
        tmp_path,
        "centers.csv",
        "id,lat,lon,banned\nA,0,0,\nB,0,1,\nC,0,2, big | small \n",
    )
    document["centers"] = {"file": "centers.csv", "columns": {"banned_vehicle_types": "banned"}}
    del document["demand"][1]
    path = write_problem(tmp_path, document)

    spec = load_spec(path)

    assert (
        spec.centers[0].banned_vehicle_types,
        spec.centers[1].banned_vehicle_types,
        spec.centers[2].banned_vehicle_types,
    ) == ((), (), ("big", "small"))


def test_symmetric_arcs_file_fills_both_directions_of_the_distance_matrix(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    """The file gives A-B 10, A-C 20 and B-C 15 km once; ``symmetric`` mirrors each arc."""
    write_table(tmp_path, "arcs.csv", "origin,destination,km\nA,B,10\nA,C,20\nB,C,15\n")
    document["matrix"] = {"source": "explicit", "symmetric": True, "arcs": {"file": "arcs.csv"}}
    path = write_problem(tmp_path, document)

    spec = load_spec(path)

    np.testing.assert_array_equal(
        spec.compile().distance_km, [[0, 10, 20], [10, 0, 15], [20, 15, 0]]
    )


def test_arc_problems_name_file_and_line(tmp_path: Path, document: dict[str, Any]) -> None:
    """Line 4 names the unknown center ``D``, so no distance is left between ``B`` and ``C``."""
    write_table(tmp_path, "arcs.csv", "origin,destination,km\nA,B,10\nA,C,20\nB,D,15\n")
    document["matrix"] = {"source": "explicit", "symmetric": True, "arcs": {"file": "arcs.csv"}}
    path = write_problem(tmp_path, document)

    with pytest.raises(SpecError) as info:
        load_spec(path)

    assert str(info.value).splitlines() == [
        f"{path}: invalid problem specification",
        "- arcs.csv:4: destination: unknown center 'D'",
        "- matrix.arcs: no distance for B -> C, C -> B",
    ]
