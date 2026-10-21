import datetime as dt
from pathlib import Path
from typing import Any

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


def write_problem(folder: Path, document: dict[str, Any], **files: str) -> Path:
    """Write ``document`` as problem.yaml and each keyword argument as a CSV file."""
    for name, text in files.items():
        (folder / f"{name}.csv").write_text(text, encoding="utf-8")
    path = folder / "problem.yaml"
    path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), "utf-8")
    return path


def with_demand_file(
    folder: Path, document: dict[str, Any], text: str = DEMAND_CSV, **reference: Any
) -> Path:
    document["demand"] = {"file": "demand.csv", "quantity": {"desi": "desi"}, **reference}
    return write_problem(folder, document, demand=text)


def load_problems(path: Path) -> list[str]:
    """Return the problems listed by the SpecError raised for ``path``."""
    with pytest.raises(SpecError) as info:
        load_spec(path)
    header, *lines = str(info.value).splitlines()
    assert header == f"{path}: invalid problem specification"
    return [line.removeprefix("- ") for line in lines]


def test_example_loads(example_path: Path) -> None:
    spec = load_spec(example_path)
    assert spec.name == "Marmara mini"
    assert len(spec.centers) == 6
    assert spec.centers[0].id == "İstanbul"
    assert spec.centers[5].banned_vehicle_types == ("tir",)
    assert len(spec.demand) == 18
    first = spec.demand[0]
    assert (first.origin, first.destination, first.date) == (
        "İstanbul",
        "Kocaeli",
        dt.date(2026, 5, 11),
    )
    assert first.quantity == {"desi": 18450.0}
    assert spec.demand[11].quantity == {"desi": 6200.5}
    assert spec.demand_ids[:2] == ("D000001", "D000002")
    assert spec.rules.stopover.max_stretch == 1.3


def test_file_and_inline_tables_give_the_same_spec(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    inline = parse_spec(document)
    assert load_spec(with_demand_file(tmp_path, document)) == inline


def test_missing_problem_file(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="cannot read"):
        load_spec(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "problem.yaml"
    path.write_text("centers: [A, B\n", encoding="utf-8")
    with pytest.raises(SpecError, match="invalid YAML"):
        load_spec(path)


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "problem.yaml"
    path.write_text("- A\n- B\n", encoding="utf-8")
    with pytest.raises(SpecError, match="top level must be a mapping"):
        load_spec(path)


def test_cell_problems_name_file_and_line(tmp_path: Path, document: dict[str, Any]) -> None:
    text = "origin,destination,date,desi\nA,B,2026-01-05,900\nA,C,2026-01-05,abc\nB,A,,120\n"
    assert load_problems(with_demand_file(tmp_path, document, text)) == [
        "demand.csv:3: 'desi': 'abc' is not a number",
        "demand.csv:4: 'date' is empty",
    ]


def test_line_numbers_count_skipped_lines(tmp_path: Path, document: dict[str, Any]) -> None:
    text = "origin,destination,date,desi\nA,B,2026-01-05,900\n\nA,C,05.01.2026,250\n"
    assert load_problems(with_demand_file(tmp_path, document, text)) == [
        "demand.csv:4: 'date': '05.01.2026' is not an ISO date (YYYY-MM-DD); set 'date_format'"
    ]


def test_model_problems_name_file_and_line(tmp_path: Path, document: dict[str, Any]) -> None:
    text = DEMAND_CSV.replace("A,C,2026-01-05,250", "A,C,2026-01-05,-250")
    assert load_problems(with_demand_file(tmp_path, document, text)) == [
        "demand.csv:3: quantity.desi: Input should be greater than or equal to 0"
    ]

    text = DEMAND_CSV.replace("B,A,", "B,X,")
    assert load_problems(with_demand_file(tmp_path, document, text)) == [
        "demand.csv:4: destination: unknown center 'X'"
    ]


def test_missing_columns_are_listed(tmp_path: Path, document: dict[str, Any]) -> None:
    text = DEMAND_CSV.replace("origin,destination", "from,to")
    assert load_problems(with_demand_file(tmp_path, document, text)) == [
        "demand (demand.csv): column 'origin' (for origin) not found",
        "demand (demand.csv): column 'destination' (for destination) not found",
        "demand (demand.csv): the columns in the file are ['from', 'to', 'date', 'desi']",
    ]


def test_invalid_file_reference(tmp_path: Path, document: dict[str, Any]) -> None:
    path = with_demand_file(tmp_path, document, delimiter=";;", encoding="klingon")
    assert load_problems(path) == [
        "demand.delimiter: String should have at most 1 character",
        "demand.encoding: unknown encoding 'klingon'",
    ]


def test_problems_in_several_files_are_reported_together(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    document["centers"] = {"file": "nowhere.csv"}
    text = DEMAND_CSV.replace("900", "nine hundred")
    found = load_problems(with_demand_file(tmp_path, document, text))
    assert len(found) == 2
    assert found[0].startswith("centers: cannot read ")
    assert found[1] == "demand.csv:2: 'desi': 'nine hundred' is not a number"


def test_turkish_spreadsheet_export(tmp_path: Path, document: dict[str, Any]) -> None:
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
        "columns": {
            "origin": "Çıkış",
            "destination": "Varış",
            "date": "Tarih",
        },
        "quantity": {"desi": "Desi"},
    }
    spec = load_spec(write_problem(tmp_path, document))
    assert [row.quantity["desi"] for row in spec.demand] == [1234.5, 250.0, 0.5]
    assert spec.demand[2].date == dt.date(2026, 1, 6)


def test_wrong_encoding_is_explained(tmp_path: Path, document: dict[str, Any]) -> None:
    (tmp_path / "demand.csv").write_bytes(
        "origin,destination,date,desi\nŞ,B,x,1\n".encode("cp1254")
    )
    document["demand"] = {"file": "demand.csv", "quantity": {"desi": "desi"}}
    [problem] = load_problems(write_problem(tmp_path, document))
    assert problem.startswith("demand: demand.csv is not valid utf-8-sig text")
    assert problem.endswith("set 'encoding', for example to cp1254")


def test_utf8_byte_order_mark_is_ignored(tmp_path: Path, document: dict[str, Any]) -> None:
    (tmp_path / "demand.csv").write_text(DEMAND_CSV, encoding="utf-8-sig")
    document["demand"] = {"file": "demand.csv", "quantity": {"desi": "desi"}}
    assert len(load_spec(write_problem(tmp_path, document)).demand) == 3


def test_centers_and_arcs_from_files(tmp_path: Path, document: dict[str, Any]) -> None:
    document["centers"] = {"file": "centers.csv", "columns": {"banned_vehicle_types": "banned"}}
    document["matrix"] = {"source": "explicit", "symmetric": True, "arcs": {"file": "arcs.csv"}}
    document["demand"].pop(1)
    path = write_problem(
        tmp_path,
        document,
        centers="id,name,banned\nA,Depot A,\nB,,\nC,Depot C, big | small \n",
        arcs="origin,destination,km\nA,B,10\nA,C,20\nB,C,15\n",
    )
    spec = load_spec(path)
    assert [center.name for center in spec.centers] == ["Depot A", None, "Depot C"]
    assert spec.centers[2].banned_vehicle_types == ("big", "small")
    assert spec.compile().distance_km.tolist() == [[0, 10, 20], [10, 0, 15], [20, 15, 0]]


def test_arc_problems_name_file_and_line(tmp_path: Path, document: dict[str, Any]) -> None:
    document["matrix"] = {"source": "explicit", "symmetric": True, "arcs": {"file": "arcs.csv"}}
    path = write_problem(tmp_path, document, arcs="origin,destination,km\nA,B,10\nA,C,20\nB,D,15\n")
    assert load_problems(path) == [
        "arcs.csv:4: destination: unknown center 'D'",
        "matrix.arcs: no distance for B -> C, C -> B",
    ]
