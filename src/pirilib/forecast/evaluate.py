"""Forecast accuracy and walk-forward backtests.

A backtest repeats: cut the history at a fold start, forecast the next days
from what came before, and score the forecast on the observed, usual days.
Scores are pooled over folds, weighted by volume. :func:`compare` always runs
the seasonal baselines too, so a model's numbers can be judged.

WMAPE, the main score, is ``100 * sum |forecast - actual| / sum actual``.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pirilib.forecast.history import History
from pirilib.forecast.models import Forecaster, SeasonalMean, SeasonalNaive
from pirilib.forecast.prepare import Calendar, unusual_days


@dataclass(frozen=True, slots=True)
class Scores:
    """Accuracy of forecasts over a set of scored cells.

    Attributes:
        wmape: Percent of volume
        bias: Percent of volume; positive means forecasts are too high
    """

    wmape: float
    bias: float
    mae: float
    rmse: float
    cells: int
    volume: float

    @classmethod
    def of(cls, actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> "Scores":
        error = predicted - actual
        volume = float(actual.sum())
        scale = max(volume, 1e-12)
        return cls(
            wmape=100.0 * float(np.abs(error).sum()) / scale,
            bias=100.0 * float(error.sum()) / scale,
            mae=float(np.abs(error).mean()) if error.size else 0.0,
            rmse=float(np.sqrt((error**2).mean())) if error.size else 0.0,
            cells=int(error.size),
            volume=volume,
        )


@dataclass(frozen=True, eq=False, slots=True)
class Fold:
    """One forecast of a backtest, with the scored cells.

    Attributes:
        lines: Per scored cell
        offsets: Day offset from start, per scored cell
    """

    start: dt.date
    lines: NDArray[np.int64]
    offsets: NDArray[np.int64]
    actual: NDArray[np.float64]
    predicted: NDArray[np.float64]

    @property
    def scores(self) -> Scores:
        return Scores.of(self.actual, self.predicted)


@dataclass(frozen=True, eq=False, slots=True)
class Backtest:
    """The folds of one walk-forward backtest of a forecaster."""

    name: str
    folds: tuple[Fold, ...] = field(repr=False)

    @property
    def scores(self) -> Scores:
        """Scores of all folds pooled (volume-weighted)."""
        return Scores.of(
            np.concatenate([fold.actual for fold in self.folds]),
            np.concatenate([fold.predicted for fold in self.folds]),
        )

    def __repr__(self) -> str:
        s = self.scores
        return (
            f"Backtest({self.name!r}, folds={len(self.folds)}, "
            f"wmape={s.wmape:.2f}, bias={s.bias:+.2f})"
        )


def fold_starts(history: History, folds: int, days: int) -> list[dt.date]:
    """Starts of ``folds`` back-to-back windows of ``days`` that end with the history."""
    if folds < 1:
        raise ValueError("folds must be at least 1")
    first = history.length - folds * days
    if first < 1:
        raise ValueError(f"the history is too short for {folds} folds of {days} days")
    return [history.date(first + index * days) for index in range(folds)]


def _before(history: History, cut: int) -> History:
    """The history up to (not including) day ``cut``."""
    return History(
        history.origins,
        history.destinations,
        history.start,
        history.values[:, :cut],
        history.dimension,
    )


def backtest(
    history: History,
    forecaster: Forecaster,
    *,
    starts: Sequence[dt.date] | None = None,
    folds: int = 8,
    days: int = 7,
    calendar: Calendar | None = None,
    name: str | None = None,
) -> Backtest:
    """Walk-forward backtest of ``forecaster``; by default 8 weekly folds ending the history."""
    calendar = Calendar() if calendar is None else calendar
    chosen = list(starts) if starts is not None else fold_starts(history, folds, days)
    results = []
    unusual = unusual_days(history, calendar, history.length)
    for start in chosen:
        cut = history.day(start)
        if not 1 <= cut < history.length:
            raise ValueError(f"fold start {start} is not inside the history")
        forecast = forecaster.forecast(_before(history, cut), start, days, calendar)
        end = min(cut + days, history.length)
        actual = history.values[:, cut:end]
        scored = ~np.isnan(actual) & ~unusual[None, cut:end]
        lines, offsets = np.nonzero(scored)
        results.append(
            Fold(
                start=start,
                lines=lines.astype(np.int64),
                offsets=offsets.astype(np.int64),
                actual=actual[lines, offsets],
                predicted=forecast.values[lines, offsets],
            )
        )
    label = name if name is not None else getattr(forecaster, "name", type(forecaster).__name__)
    return Backtest(str(label), tuple(results))


def compare(
    history: History,
    forecasters: Mapping[str, Forecaster],
    *,
    baselines: bool = True,
    starts: Sequence[dt.date] | None = None,
    folds: int = 8,
    days: int = 7,
    calendar: Calendar | None = None,
) -> list[Backtest]:
    """Backtest every forecaster (and the seasonal baselines), best pooled WMAPE first."""
    every: dict[str, Forecaster] = dict(forecasters)
    if baselines:
        every.setdefault("seasonal_mean_4", SeasonalMean(4))
        every.setdefault("seasonal_naive", SeasonalNaive())
    results = [
        backtest(
            history, model, starts=starts, folds=folds, days=days, calendar=calendar, name=name
        )
        for name, model in every.items()
    ]
    return sorted(results, key=lambda result: result.scores.wmape)


def report(results: Sequence[Backtest], baseline: str = "seasonal_mean_4") -> str:
    """A text table of backtest scores; ``skill`` is the WMAPE gain over the baseline."""
    reference = next((r.scores.wmape for r in results if r.name == baseline), None)
    width = max([5, *(len(result.name) for result in results)])
    lines = [f"{'model':{width}s} {'WMAPE':>7s} {'bias':>7s} {'MAE':>10s} {'skill':>7s}"]
    for result in results:
        s = result.scores
        skill = "" if reference is None else f"{100 * (1 - s.wmape / reference):+6.1f}%"
        numbers = f"{s.wmape:7.2f} {s.bias:+7.2f} {s.mae:10.1f} {skill:>7s}"
        lines.append(f"{result.name:{width}s} {numbers}")
    return "\n".join(lines)
