"""Time waves: split daily forecasts into the times of day demand becomes ready.

Line-haul demand often arrives in waves, for example a morning wave ready at
09:00 and an evening wave ready at 17:00. A :class:`WaveSplitter` learns, per
line and weekday, which share of the daily quantity each wave carries, from a
:class:`WaveHistory` of wave-level records. The learned :class:`WaveShares`
split a :class:`~pirilib.forecast.Forecast` into a :class:`WaveForecast`, and
:func:`~pirilib.forecast.planning_document` turns its rows into demand with a
``ready`` time.

As with the forecasters, only days before the forecast start are used, so a
split can be backtested honestly.
"""

import datetime as dt
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pirilib.forecast.history import History
from pirilib.forecast.models import Forecast
from pirilib.forecast.prepare import Calendar
from pirilib.spec.tables import read_wave_table

LEVELS: Final = ("weekday", "line", "global")
"""Names of the share levels in :attr:`WaveShares.level`: 0, 1 and 2."""


def _readonly[T: np.generic](values: NDArray[T]) -> NDArray[T]:
    values.setflags(write=False)
    return values


@dataclass(frozen=True, eq=False, slots=True)
class WaveHistory:
    """Quantities of one dimension per line (origin, destination), day and wave.

    A day on which a line has a record of any wave counts the other waves as 0;
    a day without any record of the line is NaN in every wave. Several records
    of a line, day and wave are added up.

    Attributes:
        waves: The wave times, in increasing order
        values: (lines, days, waves), ``values[s, t, w]`` is the quantity of
            line ``s`` on day ``start + t`` in wave ``w``
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    start: dt.date
    waves: tuple[dt.time, ...]
    values: NDArray[np.float64]
    dimension: str = "desi"

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64)
        lines = len(self.origins)
        if (
            values.ndim != 3
            or values.shape[0] != lines
            or len(self.destinations) != lines
            or values.shape[2] != len(self.waves)
        ):
            raise ValueError(
                "values must be (lines, days, waves), one row per origin and destination"
            )
        if not self.waves or list(self.waves) != sorted(set(self.waves)):
            raise ValueError("waves must be distinct times in increasing order")
        if np.any(values[~np.isnan(values)] < 0):
            raise ValueError("quantities must not be negative")
        object.__setattr__(self, "values", _readonly(values))

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[str, str, dt.date, dt.time, float]],
        dimension: str = "desi",
    ) -> "WaveHistory":
        """Build a wave history from (origin, destination, date, wave, quantity) records."""
        records = list(rows)
        if not records:
            raise ValueError("a wave history needs at least one record")
        lines = sorted({(origin, destination) for origin, destination, *_ in records})
        waves = sorted({wave for _, _, _, wave, _ in records})
        line_index = {line: position for position, line in enumerate(lines)}
        wave_index = {wave: position for position, wave in enumerate(waves)}
        start = min(date for _, _, date, _, _ in records)
        end = max(date for _, _, date, _, _ in records)
        values = np.full((len(lines), (end - start).days + 1, len(waves)), np.nan)
        for origin, destination, date, wave, quantity in records:
            row, column = line_index[origin, destination], (date - start).days
            cell = values[row, column]
            cell[np.isnan(cell)] = 0.0
            cell[wave_index[wave]] += quantity
        return cls(
            origins=tuple(origin for origin, _ in lines),
            destinations=tuple(destination for _, destination in lines),
            start=start,
            waves=tuple(waves),
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

    def daily(self) -> History:
        """The daily history: the waves of each day added up."""
        recorded = ~np.isnan(self.values).all(axis=2)
        totals = np.where(recorded, np.nansum(self.values, axis=2), np.nan)
        return History(self.origins, self.destinations, self.start, totals, self.dimension)

    def __repr__(self) -> str:
        waves = ", ".join(wave.isoformat("minutes") for wave in self.waves)
        return (
            f"WaveHistory(lines={self.lines}, days={self.length}, {self.start}..{self.end}, "
            f"waves=[{waves}], dimension={self.dimension!r})"
        )


@dataclass(frozen=True, eq=False, slots=True)
class WaveForecast:
    """Forecast quantities per line, day and wave for ``days`` days from ``start``.

    Attributes:
        waves: The wave times, in increasing order
        values: (lines, days, waves), never negative; the waves of a day add
            up to the daily forecast
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    start: dt.date
    waves: tuple[dt.time, ...]
    values: NDArray[np.float64]
    dimension: str = "desi"

    @property
    def days(self) -> int:
        """The number of forecast days."""
        return int(self.values.shape[1])

    def dates(self) -> list[dt.date]:
        """The forecast dates, one per day."""
        return [self.start + dt.timedelta(days=offset) for offset in range(self.days)]

    def rows(self, minimum: float = 0.5) -> Iterator[tuple[str, str, dt.date, dt.time, float]]:
        """(origin, destination, date, wave, quantity) for every piece of at least ``minimum``."""
        dates = self.dates()
        for line, column, wave in zip(*np.nonzero(self.values >= minimum), strict=True):
            yield (
                self.origins[line],
                self.destinations[line],
                dates[column],
                self.waves[wave],
                float(self.values[line, column, wave]),
            )

    def daily(self) -> Forecast:
        """The daily forecast: the waves of each day added up."""
        values = _readonly(self.values.sum(axis=2))
        return Forecast(self.origins, self.destinations, self.start, values, self.dimension)

    def total(self) -> float:
        """The forecast quantity of all lines, days and waves added up."""
        return float(self.values.sum())

    def __repr__(self) -> str:
        waves = ", ".join(wave.isoformat("minutes") for wave in self.waves)
        return (
            f"WaveForecast(lines={len(self.origins)}, days={self.days}, start={self.start}, "
            f"waves=[{waves}], total={self.total():,.1f} {self.dimension})"
        )


@dataclass(frozen=True, slots=True)
class WaveStability:
    """How much the learned wave shares move from week to week.

    Only line-weekdays with at least one usable record before the start are
    counted. The spread of a line-weekday is the largest standard deviation,
    over its waves, of the daily wave share on the days it was learned from; it
    is known only where the share was learned from at least two such days.

    Attributes:
        cells: Line-weekdays with a usable record
        weekday: Of these, shares learned from the line-weekday's own records
        line: Shares that fell back to the line over all weekdays
        overall: Shares that fell back to the share of all lines
        spread_mean: Mean spread (NaN when no spread is known)
        spread_median: Median spread
        spread_p90: 90th percentile of the spread
    """

    cells: int
    weekday: int
    line: int
    overall: int
    spread_mean: float
    spread_median: float
    spread_p90: float

    def describe(self) -> str:
        """One line for reports, for example on the command line."""
        spread = (
            "no week-to-week spread known"
            if np.isnan(self.spread_mean)
            else (
                f"week-to-week spread of a share: mean {self.spread_mean:.3f}, "
                f"median {self.spread_median:.3f}, p90 {self.spread_p90:.3f}"
            )
        )
        return (
            f"wave shares of {self.cells} line-weekdays: {self.weekday} own, "
            f"{self.line} from the line, {self.overall} global; {spread}"
        )


@dataclass(frozen=True, eq=False, slots=True)
class WaveShares:
    """Wave shares per line and weekday, learned from the days before ``start``.

    Lines that are not in the wave history get the global shares.

    Attributes:
        waves: The wave times, in increasing order
        shares: (lines, 7, waves), ``shares[s, d]`` is the split of a daily
            quantity of line ``s`` on weekday ``d`` (Monday is 0); it adds up to 1
        level: (lines, 7), where each share came from: 0 the line-weekday's own
            records, 1 the line over all weekdays, 2 all lines (see :data:`LEVELS`)
        days: (lines, 7), usable records of the line-weekday that were used
        spread: (lines, 7, waves), standard deviation of the daily share over
            those days; NaN where fewer than two days were used or the share
            fell back to a wider level
        overall: (waves,), the share of all lines together
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    start: dt.date
    waves: tuple[dt.time, ...]
    shares: NDArray[np.float64]
    level: NDArray[np.int8]
    days: NDArray[np.int64]
    spread: NDArray[np.float64]
    overall: NDArray[np.float64]

    def index(self, origin: str, destination: str) -> int | None:
        """Position of the line, or None when the wave history did not have it."""
        for position, line in enumerate(zip(self.origins, self.destinations, strict=True)):
            if line == (origin, destination):
                return position
        return None

    def share(self, origin: str, destination: str, date: dt.date) -> NDArray[np.float64]:
        """(waves,) split of the line's daily quantity on ``date``."""
        line = self.index(origin, destination)
        return self.overall if line is None else self.shares[line, date.weekday()]

    def split(self, forecast: Forecast) -> WaveForecast:
        """Split every daily quantity of ``forecast`` into the waves.

        The forecast must not start before the shares were learned, or the
        split would see days that came after the forecast start.
        """
        if forecast.start < self.start:
            raise ValueError(
                f"the shares were learned before {self.start}; the forecast starts "
                f"{forecast.start}, earlier"
            )
        lookup = {
            line: position
            for position, line in enumerate(zip(self.origins, self.destinations, strict=True))
        }
        table = np.empty((len(forecast.origins), 7, len(self.waves)))
        for row, line in enumerate(zip(forecast.origins, forecast.destinations, strict=True)):
            known = lookup.get(line)
            table[row] = self.overall if known is None else self.shares[known]
        weekdays = [date.weekday() for date in forecast.dates()]
        values = forecast.values[:, :, None] * table[:, weekdays, :]
        return WaveForecast(
            forecast.origins,
            forecast.destinations,
            forecast.start,
            self.waves,
            _readonly(values),
            forecast.dimension,
        )

    def stability(self) -> WaveStability:
        """Counts of the share levels and the week-to-week spread of the shares."""
        seen = self.days > 0
        spread = self.spread.max(axis=2)
        known = spread[seen & ~np.isnan(spread)]
        if known.size:
            mean, median, p90 = (
                float(known.mean()),
                float(np.median(known)),
                float(np.quantile(known, 0.9)),
            )
        else:
            mean = median = p90 = float("nan")
        return WaveStability(
            cells=int(seen.sum()),
            weekday=int((seen & (self.level == 0)).sum()),
            line=int((seen & (self.level == 1)).sum()),
            overall=int((seen & (self.level == 2)).sum()),
            spread_mean=mean,
            spread_median=median,
            spread_p90=p90,
        )

    def __repr__(self) -> str:
        waves = ", ".join(
            f"{wave.isoformat('minutes')}: {share:.3f}"
            for wave, share in zip(self.waves, self.overall, strict=True)
        )
        return f"WaveShares(lines={len(self.origins)}, start={self.start}, overall=[{waves}])"


@dataclass(frozen=True, slots=True)
class WaveSplitter:
    """Learns wave shares per line and weekday, as the prototype's wave split.

    For every line and weekday, the last ``weeks`` usual days of that weekday
    before the start on which the line had a positive quantity are used; the
    share of a wave is its quantity on those days over their total. A
    line-weekday with fewer than ``min_days`` such days falls back to the line's
    share over all its used days, and a line with fewer than ``min_days`` used
    days to the share of all lines together. Unusual days of the calendar are
    not learned from.

    Attributes:
        weeks: Recent days of the same weekday to learn from (4 in the prototype)
        min_days: Days needed before a share is trusted (2 in the prototype)
    """

    weeks: int = 4
    min_days: int = 2

    def __post_init__(self) -> None:
        if self.weeks < 1 or self.min_days < 1:
            raise ValueError("weeks and min_days must be at least 1")

    def fit(
        self, history: WaveHistory, start: dt.date, calendar: Calendar | None = None
    ) -> WaveShares:
        """Learn the shares for a forecast from ``start``, from the days before it."""
        calendar = Calendar() if calendar is None else calendar
        cut = min(history.day(start), history.length)
        if cut < 1:
            raise ValueError(
                f"the forecast start {start} must come after the wave history start {history.start}"
            )
        values = np.nan_to_num(history.values[:, :cut, :], nan=0.0)
        totals = values.sum(axis=2)
        usual = np.array(
            [history.date(day) not in calendar.unusual for day in range(cut)], dtype=np.bool_
        )
        usable = (totals > 0) & usual[None, :]
        if not usable.any():
            raise ValueError(f"the wave history has no usable record before {start}")
        weekday = np.array([history.date(day).weekday() for day in range(cut)], dtype=np.int64)
        used = np.zeros_like(usable)
        for dow in range(7):
            columns = np.flatnonzero(weekday == dow)
            block = usable[:, columns]
            from_end = np.cumsum(block[:, ::-1], axis=1)[:, ::-1]
            used[:, columns] = block & (from_end <= self.weeks)
        daily = np.divide(
            values, totals[:, :, None], out=np.zeros_like(values), where=used[..., None]
        )
        weights = np.where(used[..., None], values, 0.0)
        one_hot = (weekday[:, None] == np.arange(7)[None, :]).astype(np.float64)
        own = np.einsum("ltw,td->ldw", weights, one_hot)
        days = np.einsum("lt,td->ld", used.astype(np.float64), one_hot).astype(np.int64)
        line = own.sum(axis=1)
        line_days = days.sum(axis=1)
        everything = line.sum(axis=0)
        overall = everything / everything.sum()

        own_ok = days >= self.min_days
        line_ok = line_days >= self.min_days
        line_share = np.divide(
            line, line.sum(axis=1, keepdims=True), out=np.zeros_like(line), where=line_ok[:, None]
        )
        own_share = np.divide(
            own, own.sum(axis=2, keepdims=True), out=np.zeros_like(own), where=own_ok[..., None]
        )
        fallback = np.where(line_ok[:, None], line_share, overall[None, :])
        shares = np.where(own_ok[..., None], own_share, fallback[:, None, :])
        level = np.where(own_ok, 0, np.where(line_ok[:, None], 1, 2)).astype(np.int8)

        spread = np.full(own.shape, np.nan)
        for dow in range(7):
            columns = weekday == dow
            mask = used[:, columns]
            count = mask.sum(axis=1)
            pieces = daily[:, columns, :]
            mean = pieces.sum(axis=1) / np.maximum(count, 1)[:, None]
            deviation = np.where(mask[..., None], pieces - mean[:, None, :], 0.0)
            variance = (deviation**2).sum(axis=1) / np.maximum(count, 1)[:, None]
            rows = (count >= 2) & own_ok[:, dow]
            spread[rows, dow] = np.sqrt(variance[rows])
        return WaveShares(
            origins=history.origins,
            destinations=history.destinations,
            start=start,
            waves=history.waves,
            shares=_readonly(shares),
            level=_readonly(level),
            days=_readonly(days),
            spread=_readonly(spread),
            overall=_readonly(overall),
        )


def load_wave_history(
    path: str | os.PathLike[str],
    *,
    dimension: str = "desi",
    column: str | None = None,
    wave: str = "wave",
    **table: Any,
) -> WaveHistory:
    """Read a wave history from a CSV file with origin, destination, date, wave and quantity.

    The wave is read from the column ``wave`` as a time of day (``09:00``,
    ``9:00`` or an hour such as ``9``) and the quantity from ``column``
    (default: the dimension name). ``table`` takes the other options of
    :class:`~pirilib.spec.TableSource` (delimiter, encoding, decimal,
    date_format, columns).
    """
    file = Path(path)
    columns: Mapping[str, str] = table.pop("columns", {})
    reference: Mapping[str, Any] = {
        **table,
        "file": file.name,
        "columns": {"wave": wave, **columns},
        "quantity": {dimension: column or dimension},
    }
    rows = read_wave_table(reference, file.parent)
    return WaveHistory.from_rows(
        (
            (
                row["origin"],
                row["destination"],
                row["date"],
                row["wave"],
                row["quantity"][dimension],
            )
            for row in rows
        ),
        dimension=dimension,
    )
