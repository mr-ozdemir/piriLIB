"""Demand history: daily quantities per line, as a dense matrix."""

import datetime as dt
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pirilib.spec.tables import read_demand_table


@dataclass(frozen=True, eq=False, slots=True)
class History:
    """Daily quantities of one dimension per line (origin, destination).

    ``values[s, t]`` is the quantity of line ``s`` on day ``start + t``, NaN
    where nothing was recorded. Several records of a line and day are added up.

    Attributes:
        values: (lines, days)
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    start: dt.date
    values: NDArray[np.float64]
    dimension: str = "desi"

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64)
        lines = len(self.origins)
        if values.ndim != 2 or values.shape[0] != lines or len(self.destinations) != lines:
            raise ValueError("values must be (lines, days), one row per origin and destination")
        if np.any(values[~np.isnan(values)] < 0):
            raise ValueError("quantities must not be negative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_rows(
        cls, rows: Iterable[tuple[str, str, dt.date, float]], dimension: str = "desi"
    ) -> "History":
        """Build a history from (origin, destination, date, quantity) records."""
        records = list(rows)
        if not records:
            raise ValueError("a history needs at least one record")
        lines = sorted({(origin, destination) for origin, destination, _, _ in records})
        index = {line: position for position, line in enumerate(lines)}
        start = min(date for _, _, date, _ in records)
        end = max(date for _, _, date, _ in records)
        values = np.full((len(lines), (end - start).days + 1), np.nan)
        for origin, destination, date, quantity in records:
            row, column = index[origin, destination], (date - start).days
            previous = values[row, column]
            values[row, column] = quantity if np.isnan(previous) else previous + quantity
        return cls(
            origins=tuple(origin for origin, _ in lines),
            destinations=tuple(destination for _, destination in lines),
            start=start,
            values=values,
            dimension=dimension,
        )

    @property
    def lines(self) -> int:
        """The number of lines (origin-destination pairs)."""
        return len(self.origins)

    @property
    def length(self) -> int:
        """The number of days in the history."""
        return int(self.values.shape[1])

    @property
    def end(self) -> dt.date:
        """The last day of the history."""
        return self.start + dt.timedelta(days=self.length - 1)

    def day(self, date: dt.date) -> int:
        """Position of ``date`` in the history (may be outside it)."""
        return (date - self.start).days

    def date(self, day: int) -> dt.date:
        """The date at position ``day`` in the history."""
        return self.start + dt.timedelta(days=day)

    def index(self, origin: str, destination: str) -> int:
        """The row of the line ``origin`` to ``destination``; :class:`KeyError` if there is none."""
        for position, line in enumerate(zip(self.origins, self.destinations, strict=True)):
            if line == (origin, destination):
                return position
        raise KeyError(f"no line {origin} -> {destination}")

    def __repr__(self) -> str:
        observed = int(np.count_nonzero(~np.isnan(self.values)))
        return (
            f"History(lines={self.lines}, days={self.length}, {self.start}..{self.end}, "
            f"observed={observed}, dimension={self.dimension!r})"
        )


def load_history(
    path: str | os.PathLike[str],
    *,
    dimension: str = "desi",
    column: str | None = None,
    **table: Any,
) -> History:
    """Read a history from a CSV file with origin, destination, date and quantity columns.

    The quantity is read from ``column`` (default: the dimension name). ``table``
    takes the other options of :class:`~pirilib.spec.TableSource` (delimiter,
    encoding, decimal, date_format, columns).
    """
    file = Path(path)
    reference: Mapping[str, Any] = {
        **table,
        "file": file.name,
        "quantity": {dimension: column or dimension},
    }
    rows = read_demand_table(reference, file.parent)
    return History.from_rows(
        (
            (row["origin"], row["destination"], row["date"], row["quantity"][dimension])
            for row in rows
        ),
        dimension=dimension,
    )
