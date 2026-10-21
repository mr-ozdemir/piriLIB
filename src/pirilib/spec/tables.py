"""Read table-shaped fields (centers, demand, distance arcs) from CSV files.

In a problem document, ``centers``, ``demand`` and ``matrix.arcs`` may be a list
of rows or a reference to a CSV file. File references are replaced by their rows
before the document is validated, so the validated specification never depends
on files.
"""

import codecs
import csv
import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pirilib.errors import SpecError
from pirilib.spec._format import Loc, bullets, validation_problems

LIST_SEPARATOR: Final = "|"
"""Separator of list values inside one CSV cell, for example ``tir|kamyon``."""

_DIGIT_GROUP_SPACES: Final = str.maketrans("", "", " \N{NO-BREAK SPACE}\N{NARROW NO-BREAK SPACE}")


class TableSource(BaseModel):
    """A CSV file that supplies the rows of a table field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str = Field(min_length=1, description="CSV path, relative to the problem file.")
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    encoding: str = Field(
        default="utf-8-sig", description="Text encoding, for example utf-8-sig or cp1254."
    )
    decimal: Literal[".", ","] = Field(
        default=".",
        description="Decimal separator. With ',' a '.' is read as a thousands separator.",
    )
    date_format: str | None = Field(
        default=None,
        description="strptime format of dates, for example '%d.%m.%Y'; ISO dates need none.",
    )
    columns: dict[str, str] = Field(
        default_factory=dict,
        description="Field -> CSV column; other fields are read from the column of the same name.",
    )
    quantity: dict[str, str] = Field(
        default_factory=dict, description="Demand only: quantity dimension -> CSV column."
    )

    @field_validator("encoding")
    @classmethod
    def _check_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError:
            raise ValueError(f"unknown encoding '{value}'") from None
        return value


@dataclass(frozen=True)
class _Layout:
    """The fields of one table and how their cells are converted."""

    text: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    lists: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    has_quantity: bool = False

    @property
    def fields(self) -> tuple[str, ...]:
        return self.text + self.numbers + self.dates + self.lists


_LAYOUTS: Final[Mapping[Loc, _Layout]] = {
    ("centers",): _Layout(
        text=("id", "name"),
        numbers=("lat", "lon"),
        lists=("banned_vehicle_types",),
        required=("id",),
    ),
    ("demand",): _Layout(
        text=("id", "origin", "destination"),
        dates=("date",),
        required=("origin", "destination", "date"),
        has_quantity=True,
    ),
    ("matrix", "arcs"): _Layout(
        text=("origin", "destination"),
        numbers=("km",),
        required=("origin", "destination", "km"),
    ),
}


@dataclass
class _Table:
    """Rows read from a CSV file, their file:line labels and the problems found."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def materialize_tables(
    data: Mapping[str, Any], base_dir: Path
) -> tuple[dict[str, Any], dict[Loc, list[str]]]:
    """Replace the file-referenced tables of a document by their rows.

    Returns the new document and, for every table read from a file, the
    ``file:line`` label of each row, so that later problems can name the line.
    Raises :class:`~pirilib.SpecError` listing every problem in every file.
    """
    document = dict(data)
    matrix = document.get("matrix")
    if isinstance(matrix, Mapping):
        document["matrix"] = matrix = dict(matrix)
    origins: dict[Loc, list[str]] = {}
    problems: list[str] = []
    for table, parent, key in (
        (("centers",), document, "centers"),
        (("demand",), document, "demand"),
        (("matrix", "arcs"), matrix, "arcs"),
    ):
        if not isinstance(parent, dict) or not _is_reference(parent.get(key)):
            continue
        result = _read_table(parent[key], base_dir, table)
        parent[key] = result.rows
        origins[table] = result.labels
        problems += result.problems
    if problems:
        raise SpecError(bullets(problems))
    return document, origins


def read_demand_table(reference: Mapping[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    """Read rows shaped like the demand table (origin, destination, date, quantity) from CSV.

    ``reference`` has the fields of :class:`TableSource`. Raises
    :class:`~pirilib.SpecError` listing every problem, with file and line.
    """
    result = _read_table(reference, base_dir, ("demand",))
    if result.problems:
        raise SpecError(bullets(result.problems))
    return result.rows


def _is_reference(value: object) -> bool:
    return isinstance(value, Mapping) and "file" in value


def _read_table(reference: Mapping[str, Any], base_dir: Path, table: Loc) -> _Table:
    name = ".".join(map(str, table))
    result = _Table()
    try:
        source = TableSource.model_validate(reference)
    except ValidationError as exc:
        result.problems += validation_problems(exc, prefix=table)
        return result
    layout = _LAYOUTS[table]
    path = base_dir / source.file
    try:
        with path.open(newline="", encoding=source.encoding) as handle:
            reader = csv.DictReader(handle, delimiter=source.delimiter)
            header = [column.strip() for column in reader.fieldnames or ()]
            reader.fieldnames = header
            columns, problems = _resolve_columns(source, layout, header)
            if problems:
                result.problems += [f"{name} ({source.file}): {problem}" for problem in problems]
                return result
            for record in reader:
                label = f"{source.file}:{reader.line_num}"
                row, problems = _convert(record, columns, source, layout)
                result.rows.append(row)
                result.labels.append(label)
                result.problems += [f"{label}: {problem}" for problem in problems]
    except OSError as exc:
        result.problems.append(f"{name}: cannot read {path}: {exc.strerror or exc}")
    except UnicodeDecodeError as exc:
        result.problems.append(
            f"{name}: {source.file} is not valid {source.encoding} text ({exc.reason}); "
            "set 'encoding', for example to cp1254"
        )
    except csv.Error as exc:
        result.problems.append(f"{name}: {source.file} is not a valid CSV file: {exc}")
    return result


def _resolve_columns(
    source: TableSource, layout: _Layout, header: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Map target fields, and ``quantity.<dimension>`` entries, to CSV columns."""
    problems: list[str] = []
    repeated = sorted({column for column in header if header.count(column) > 1})
    if repeated:
        problems.append(f"repeated column(s) {repeated}")
    unknown = sorted(set(source.columns) - set(layout.fields))
    if unknown:
        problems.append(f"unknown field(s) {unknown} in 'columns'; expected {list(layout.fields)}")
    if source.quantity and not layout.has_quantity:
        problems.append("'quantity' is only allowed for the demand table")
    if layout.has_quantity and not source.quantity:
        problems.append("'quantity' must map every dimension to a column, for example {desi: Desi}")

    available = set(header)
    columns: dict[str, str] = {}
    for target in layout.fields:
        column = source.columns.get(target, target)
        if column in available:
            columns[target] = column
        elif target in layout.required or target in source.columns:
            problems.append(f"column '{column}' (for {target}) not found")
    for dimension, column in source.quantity.items():
        if column in available:
            columns[f"quantity.{dimension}"] = column
        else:
            problems.append(f"column '{column}' (for quantity {dimension}) not found")
    if problems:
        problems.append(f"the columns in the file are {header}")
    return columns, problems


def _convert(
    record: Mapping[str, str | None],
    columns: Mapping[str, str],
    source: TableSource,
    layout: _Layout,
) -> tuple[dict[str, Any], list[str]]:
    """Convert one CSV record into a row of the table; empty optional cells are left out."""
    row: dict[str, Any] = {}
    quantity: dict[str, float] = {}
    problems: list[str] = []
    for target, column in columns.items():
        text = (record.get(column) or "").strip()
        is_quantity = target.startswith("quantity.")
        if not text:
            if is_quantity or target in layout.required:
                problems.append(f"'{column}' is empty")
            continue
        try:
            if is_quantity:
                quantity[target.removeprefix("quantity.")] = _number(text, source.decimal)
            elif target in layout.numbers:
                row[target] = _number(text, source.decimal)
            elif target in layout.dates:
                row[target] = _date(text, source.date_format)
            elif target in layout.lists:
                row[target] = [item.strip() for item in text.split(LIST_SEPARATOR) if item.strip()]
            else:
                row[target] = text
        except ValueError as exc:
            problems.append(f"'{column}': {exc}")
    if layout.has_quantity:
        row["quantity"] = quantity
    return row, problems


def _number(text: str, decimal: str) -> float:
    cleaned = text.translate(_DIGIT_GROUP_SPACES)
    if decimal == ",":
        cleaned = cleaned.replace(".", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"'{text}' is not a finite number")
    return value


def _date(text: str, date_format: str | None) -> dt.date:
    if date_format is None:
        try:
            return dt.date.fromisoformat(text)
        except ValueError:
            raise ValueError(
                f"'{text}' is not an ISO date (YYYY-MM-DD); set 'date_format'"
            ) from None
    try:
        return dt.datetime.strptime(text, date_format).date()
    except ValueError:
        raise ValueError(f"'{text}' does not match the date format '{date_format}'") from None
