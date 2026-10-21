"""Forecasters: from a history to daily quantities per line for the days ahead.

Every forecaster has one method, ``forecast(history, start, days, calendar)``.
It learns only from days before ``start`` and predicts ``start .. start +
days - 1``. A forecaster keeps no state between calls, so walk-forward
backtests are honest.

- :class:`SeasonalMean` (the prototype baseline) and :class:`SeasonalNaive`
  are the baselines every model must beat.
- :class:`Regression` turns any regressor with ``fit``/``predict`` (scikit-learn,
  LightGBM, XGBoost, ...) into a forecaster over the features of
  :mod:`pirilib.forecast.prepare`.
- :func:`lightgbm_tweedie` is the prototype's tuned champion (needs
  ``pirilib[forecast]``).

Lines are only forecast on weekdays when they had a record in the last
``presence_days`` days before the start (as in the prototype); elsewhere the
forecast is 0.
"""

import datetime as dt
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from pirilib.forecast.history import History
from pirilib.forecast.prepare import FEATURES, LAGS, Calendar, clean, features, unusual_days

PRESENCE_DAYS: Final = 35

PROTOTYPE_TWEEDIE: Final[Mapping[str, Any]] = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 47,
    "max_depth": 8,
    "min_child_samples": 150,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.4,
    "reg_alpha": 0.01,
    "reg_lambda": 0.001,
    "tweedie_variance_power": 1.3,
}
"""LightGBM settings the prototype tuned with ABC for daily desi per line."""


@dataclass(frozen=True, eq=False, slots=True)
class Forecast:
    """Daily quantities per line for ``days`` days from ``start``.

    Attributes:
        values: (lines, days), never negative
    """

    origins: tuple[str, ...]
    destinations: tuple[str, ...]
    start: dt.date
    values: NDArray[np.float64]
    dimension: str = "desi"

    @property
    def days(self) -> int:
        """The number of forecast days."""
        return int(self.values.shape[1])

    def dates(self) -> list[dt.date]:
        """The forecast dates, one per day."""
        return [self.start + dt.timedelta(days=offset) for offset in range(self.days)]

    def rows(self, minimum: float = 0.5) -> Iterator[tuple[str, str, dt.date, float]]:
        """(origin, destination, date, quantity) for every cell of at least ``minimum``."""
        dates = self.dates()
        for line, column in zip(*np.nonzero(self.values >= minimum), strict=True):
            yield (
                self.origins[line],
                self.destinations[line],
                dates[column],
                float(self.values[line, column]),
            )

    def total(self) -> float:
        """The forecast quantity of all lines and days added up."""
        return float(self.values.sum())

    def __repr__(self) -> str:
        return (
            f"Forecast(lines={len(self.origins)}, days={self.days}, start={self.start}, "
            f"total={self.total():,.1f} {self.dimension})"
        )


class Forecaster(Protocol):
    """Anything that forecasts daily quantities per line from a history."""

    def forecast(
        self, history: History, start: dt.date, days: int, calendar: Calendar | None = None
    ) -> Forecast:
        """The forecast of ``days`` days from ``start``, learned from the history before it."""
        ...


def _window(history: History, start: dt.date, days: int) -> int:
    """The cut (first unknown history day) for a forecast; checks the request."""
    if days < 1:
        raise ValueError("days must be at least 1")
    cut = history.day(start)
    if cut < 1:
        raise ValueError(
            f"the forecast start {start} must come after the history start {history.start}"
        )
    return min(cut, history.length)


def active(history: History, cut: int, start: dt.date, days: int, window: int) -> NDArray[np.bool_]:
    """(lines, days): the line had a record on that weekday in the ``window`` days to the cut."""
    begin = max(0, cut - window)
    recent = ~np.isnan(history.values[:, begin:cut])
    weekdays = np.array([history.date(day).weekday() for day in range(begin, cut)], dtype=np.int64)
    seen = np.zeros((history.lines, 7), dtype=np.bool_)
    for dow in range(7):
        seen[:, dow] = recent[:, weekdays == dow].any(axis=1)
    targets = [(start + dt.timedelta(days=offset)).weekday() for offset in range(days)]
    return seen[:, targets]


def _result(history: History, start: dt.date, values: NDArray[np.float64]) -> Forecast:
    values = np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
    values.setflags(write=False)
    return Forecast(history.origins, history.destinations, start, values, history.dimension)


@dataclass(frozen=True, slots=True)
class SeasonalMean:
    """Mean of the last ``weeks`` usual records of the same weekday (the prototype baseline)."""

    weeks: int = 4
    presence_days: int | None = PRESENCE_DAYS

    def forecast(
        self, history: History, start: dt.date, days: int, calendar: Calendar | None = None
    ) -> Forecast:
        """The mean of the last ``weeks`` usual days of the same weekday, per line."""
        calendar = Calendar() if calendar is None else calendar
        cut = _window(history, start, days)
        usual = ~unusual_days(history, calendar, cut)
        values = np.where(usual[None, :], history.values[:, :cut], np.nan)
        weekday = np.array([history.date(day).weekday() for day in range(cut)], dtype=np.int64)
        by_dow = np.full((history.lines, 7), np.nan)
        for dow in range(7):
            block = values[:, weekday == dow]
            for line in range(history.lines):
                recent = block[line][~np.isnan(block[line])][-self.weeks :]
                if recent.size:
                    by_dow[line, dow] = recent.mean()
        line_mean = np.array([_nanmean(row) for row in values])
        by_dow = np.where(np.isnan(by_dow), line_mean[:, None], by_dow)
        targets = [(start + dt.timedelta(days=offset)).weekday() for offset in range(days)]
        result = by_dow[:, targets]
        if self.presence_days is not None:
            result = np.where(active(history, cut, start, days, self.presence_days), result, 0.0)
        return _result(history, start, result)


def SeasonalNaive(presence_days: int | None = PRESENCE_DAYS) -> SeasonalMean:
    """The last usual record of the same weekday."""
    return SeasonalMean(weeks=1, presence_days=presence_days)


def _nanmean(values: NDArray[np.float64]) -> float:
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else float("nan")


Target = Literal["raw", "log1p"]


@dataclass(frozen=True, slots=True)
class Regression:
    """A regressor over the features of :mod:`pirilib.forecast.prepare`.

    ``make`` returns a fresh, unfitted regressor with ``fit(X, y)`` and
    ``predict(X)``. With ``target="log1p"`` the regressor learns ``log1p`` of the
    cleaned quantity and its predictions are turned back with Duan's smearing
    factor, which removes the downward bias of back-transformed logs. Losses that
    work on raw quantities (Tweedie, Poisson) use ``target="raw"``.

    With ``scale=True`` quantities and level features are divided by the line's
    recent level (mean of the 28 known days ending a week before the target), so
    the regressor learns shapes rather than levels and can follow a line whose
    volume moves beyond what it saw in training.

    Training uses the cleaned (imputed, winsorized) values of usual, observed
    days from ``train_from`` (a date) on.
    """

    make: Callable[[], Any]
    target: Target = "raw"
    train_from: dt.date | None = None
    winsor_z: float = 4.0
    presence_days: int | None = PRESENCE_DAYS
    scale: bool = False
    name: str = field(default="regression", compare=False)

    def forecast(
        self, history: History, start: dt.date, days: int, calendar: Calendar | None = None
    ) -> Forecast:
        """The regression forecast; :class:`ValueError` if ``days`` exceeds the shortest lag."""
        calendar = Calendar() if calendar is None else calendar
        if days > min(LAGS):
            raise ValueError(f"days must not exceed {min(LAGS)}, the shortest lag")
        cut = _window(history, start, days)
        cleaned = clean(history, calendar, cut, self.winsor_z)
        trainable = cleaned.started & ~cleaned.imputed
        if self.train_from is not None:
            trainable[:, : max(0, history.day(self.train_from))] = False
        rows, columns = np.nonzero(trainable)
        if rows.size == 0:
            raise ValueError("no usual, observed days to learn from before the forecast start")
        x = features(
            history, calendar, cleaned.winsorized, rows.astype(np.int64), columns.astype(np.int64)
        )
        y = cleaned.winsorized[rows, columns]
        if self.scale:
            known = ~np.isnan(x[:, _ROLL28]) | ~np.isnan(x[:, _ROLL7])
            x, y = x[known], y[known]
            level = _level(x)
            x = _scaled(x, level)
            y = y / level
        model = self.make()
        model.fit(x, np.log1p(y) if self.target == "log1p" else y)

        first = history.day(start)
        target_rows = np.repeat(np.arange(history.lines, dtype=np.int64), days)
        target_days = np.tile(np.arange(first, first + days, dtype=np.int64), history.lines)
        x_new = features(history, calendar, cleaned.winsorized, target_rows, target_days)
        new_level = _level(x_new) if self.scale else np.ones(target_rows.size)
        if self.scale:
            x_new = _scaled(x_new, new_level)
        predicted = np.asarray(model.predict(x_new), dtype=np.float64)
        if self.target == "log1p":
            residuals = np.log1p(y) - np.asarray(model.predict(x), dtype=np.float64)
            predicted = np.expm1(predicted) * float(np.mean(np.exp(residuals)))
        predicted = predicted * new_level
        values = predicted.reshape(history.lines, days)
        values = np.where(cleaned.started[:, -1:], values, 0.0)
        if self.presence_days is not None:
            values = np.where(active(history, cut, start, days, self.presence_days), values, 0.0)
        return _result(history, start, values)


_LEVEL_COLUMNS: Final = tuple(
    position for position, name in enumerate(FEATURES) if name.startswith(("lag", "roll"))
)
_ROLL28: Final = FEATURES.index("roll28_mean")
_ROLL7: Final = FEATURES.index("roll7_mean")


def _level(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Recent level of each row: the 28-day mean, else the 7-day mean, at least 1."""
    level = np.where(np.isnan(x[:, _ROLL28]), x[:, _ROLL7], x[:, _ROLL28])
    return np.maximum(np.nan_to_num(level, nan=1.0), 1.0)


def _scaled(x: NDArray[np.float64], level: NDArray[np.float64]) -> NDArray[np.float64]:
    scaled = x.copy()
    scaled[:, _LEVEL_COLUMNS] = x[:, _LEVEL_COLUMNS] / level[:, None]
    return scaled


def lightgbm_tweedie(train_from: dt.date | None = None, **settings: Any) -> Regression:
    """The prototype's champion: LightGBM with a Tweedie loss on cleaned daily desi.

    ``settings`` override :data:`PROTOTYPE_TWEEDIE`. Needs ``pip install "pirilib[forecast]"``.
    """
    params = {**PROTOTYPE_TWEEDIE, **settings}

    def make() -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError(
                'LightGBM forecasts need lightgbm: pip install "pirilib[forecast]"'
            ) from None
        return LGBMRegressor(objective="tweedie", random_state=42, verbose=-1, n_jobs=1, **params)

    return Regression(make=make, target="raw", train_from=train_from, name="lightgbm_tweedie")
