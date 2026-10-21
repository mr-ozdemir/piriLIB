"""Calendar, cleaning and features of a demand history, all in numpy.

Cleaning follows the prototype:

1. Days the calendar marks as *unusual* (holidays, month ends, outages), and
   days without a record, are *imputed*: they get the median of the line on
   that weekday, else the line median, else the overall median.
2. Values above ``median + z / 0.6745 * MAD`` of the line and weekday are cut
   down to that cap (winsorizing), so single spikes do not drive the models.

Unlike the prototype, the medians use only days before the forecast start, so
backtests do not peek at the future. Days before a line's first record stay
missing (NaN): the line did not exist yet.
"""

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from pirilib.forecast.history import History

LAGS: Final = (7, 14, 21, 28)
"""Lags, in days, of the lag features; the forecast horizon may not exceed the smallest."""

FEATURES: Final = (
    *(f"lag{lag}" for lag in LAGS),
    "roll7_mean",
    "roll28_mean",
    "roll28_std",
    "dow",
    "day",
    "month",
    "week",
    "weekend",
    "line",
    "origin",
    "destination",
    "days_since_holiday",
    "days_to_holiday",
    "holiday",
    "after_holiday",
    "before_holiday",
)
"""Names of the columns of :func:`features`, in order."""


@dataclass(frozen=True, slots=True)
class Calendar:
    """Special days. *Unusual* days are not learned from or scored; *holidays* are features.

    Holidays are usually unusual as well; list them in both.
    """

    unusual: frozenset[dt.date] = field(default_factory=frozenset)
    holidays: frozenset[dt.date] = field(default_factory=frozenset)

    @classmethod
    def of(cls, unusual: Iterable[dt.date] = (), holidays: Iterable[dt.date] = ()) -> "Calendar":
        return cls(frozenset(unusual), frozenset(holidays))


@dataclass(frozen=True, eq=False, slots=True)
class Cleaned:
    """A history up to a cut day, cleaned; columns are history days ``0 .. cut - 1``.

    Attributes:
        filled: Observed values, imputed where needed
        winsorized: Filled, with spikes cut
        imputed: True where the value was not observed or the day is unusual
        started: True from the line's first record on
    """

    filled: NDArray[np.float64]
    winsorized: NDArray[np.float64]
    imputed: NDArray[np.bool_]
    started: NDArray[np.bool_]


def unusual_days(history: History, calendar: Calendar, days: int) -> NDArray[np.bool_]:
    """(days,) True for history days ``0 .. days - 1`` that the calendar marks unusual."""
    return np.array([history.date(day) in calendar.unusual for day in range(days)], dtype=np.bool_)


def clean(history: History, calendar: Calendar, cut: int, z: float = 4.0) -> Cleaned:
    """Clean history days ``0 .. cut - 1`` using only those days."""
    values = history.values[:, :cut]
    observed = ~np.isnan(values)
    started = np.cumsum(observed, axis=1) > 0
    imputed = ~observed | unusual_days(history, calendar, cut)[None, :]
    reference = np.where(observed & ~imputed, values, np.nan)
    overall = _nanmedian(reference.ravel())
    overall = 0.0 if np.isnan(overall) else overall
    line_median = np.array([_nanmedian(row) for row in reference])
    weekday = np.array([history.date(day).weekday() for day in range(cut)], dtype=np.int64)

    filled = reference.copy()
    cap = np.empty_like(values)
    for dow in range(7):
        columns = weekday == dow
        if not columns.any():
            continue
        block = reference[:, columns]
        median = np.array([_nanmedian(row) for row in block])
        center = np.where(np.isnan(median), line_median, median)
        center = np.where(np.isnan(center), overall, center)
        mad = np.array([_nanmedian(np.abs(row - m)) for row, m in zip(block, median, strict=True)])
        part = np.where(np.isnan(block), center[:, None], block)
        spread = np.where((mad > 0)[:, None], mad[:, None], 0.1 * part + 1.0)
        filled[:, columns] = part
        cap[:, columns] = center[:, None] + (z / 0.6745) * spread
    filled = np.where(started, filled, np.nan)
    winsorized = np.where(started, np.minimum(filled, cap), np.nan)
    return Cleaned(filled=filled, winsorized=winsorized, imputed=imputed, started=started)


def _nanmedian(values: NDArray[np.float64]) -> float:
    finite = values[~np.isnan(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def features(
    history: History,
    calendar: Calendar,
    winsorized: NDArray[np.float64],
    rows: NDArray[np.int64],
    days: NDArray[np.int64],
) -> NDArray[np.float64]:
    """(cells, features) for the (line, day) cells; see :data:`FEATURES`.

    ``winsorized`` holds the cleaned values of the days before the cut; days at or
    after the cut must not be read, which holds while every target day is less
    than ``min(LAGS)`` days after the cut.
    """
    cut = winsorized.shape[1]

    def lagged(lag: int) -> NDArray[np.float64]:
        column = days - lag
        known = (column >= 0) & (column < cut)
        out = np.full(days.size, np.nan)
        out[known] = winsorized[rows[known], column[known]]
        return out

    present = np.where(np.isnan(winsorized), 0.0, winsorized)
    count = (~np.isnan(winsorized)).astype(np.float64)
    sums = np.concatenate([np.zeros((present.shape[0], 1)), np.cumsum(present, axis=1)], axis=1)
    squares = np.concatenate(
        [np.zeros((present.shape[0], 1)), np.cumsum(present**2, axis=1)], axis=1
    )
    counts = np.concatenate([np.zeros((count.shape[0], 1)), np.cumsum(count, axis=1)], axis=1)

    def window(width: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        end = np.clip(days - 7 + 1, 0, cut)
        begin = np.clip(days - 7 - width + 1, 0, cut)
        n = counts[rows, end] - counts[rows, begin]
        total = sums[rows, end] - sums[rows, begin]
        square = squares[rows, end] - squares[rows, begin]
        return n, total, square

    n7, total7, _ = window(7)
    n28, total28, square28 = window(28)
    with np.errstate(invalid="ignore", divide="ignore"):
        roll7 = np.where(n7 > 0, total7 / n7, np.nan)
        roll28 = np.where(n28 > 0, total28 / n28, np.nan)
        variance = (square28 - n28 * roll28**2) / (n28 - 1)
        std28 = np.where(n28 > 1, np.sqrt(np.clip(variance, 0.0, None)), np.nan)

    dates = [history.date(int(day)) for day in days]
    holidays = np.array(sorted(calendar.holidays), dtype="datetime64[D]")
    stamps = np.array(dates, dtype="datetime64[D]")
    if holidays.size:
        before = np.searchsorted(holidays, stamps, side="right") - 1
        after = np.searchsorted(holidays, stamps, side="left")
        since = np.where(
            before >= 0, (stamps - holidays[np.clip(before, 0, None)]).astype(np.int64), 999
        )
        until = np.where(
            after < holidays.size,
            (holidays[np.clip(after, None, holidays.size - 1)] - stamps).astype(np.int64),
            999,
        )
    else:
        since = until = np.full(days.size, 999)
    origin_codes = _codes(history.origins)
    destination_codes = _codes(history.destinations)
    columns = [
        *(lagged(lag) for lag in LAGS),
        roll7,
        roll28,
        std28,
        np.array([date.weekday() for date in dates]),
        np.array([date.day for date in dates]),
        np.array([date.month for date in dates]),
        np.array([date.isocalendar().week for date in dates]),
        np.array([date.weekday() >= 5 for date in dates]),
        rows,
        origin_codes[rows],
        destination_codes[rows],
        np.clip(since, 0, 30),
        np.clip(until, 0, 30),
        np.array([date in calendar.holidays for date in dates]),
        (since >= 1) & (since <= 2),
        until == 1,
    ]
    return np.column_stack([np.asarray(column, dtype=np.float64) for column in columns])


def _codes(names: tuple[str, ...]) -> NDArray[np.int64]:
    index = {name: code for code, name in enumerate(sorted(set(names)))}
    return np.array([index[name] for name in names], dtype=np.int64)
