"""Tuning forecasters with the bee colony, scored by walk-forward WMAPE."""

import datetime as dt
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pirilib.forecast.evaluate import backtest
from pirilib.forecast.history import History
from pirilib.forecast.models import TWEEDIE_SETTINGS, Forecaster
from pirilib.forecast.prepare import Calendar
from pirilib.tuning.bee_colony import BeeColony, SearchResult
from pirilib.tuning.space import Choice, SearchSpace
from pirilib.tuning.tune import Tuner


def lightgbm_space() -> SearchSpace:
    """The LightGBM search space, with the Tweedie variance power."""
    return SearchSpace(
        {
            "n_estimators": Choice((200, 300, 400, 600, 800, 1000, 1200, 1500)),
            "learning_rate": Choice((0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)),
            "num_leaves": Choice((15, 31, 47, 63, 95, 127, 255)),
            "max_depth": Choice((-1, 4, 6, 8, 10, 12, 14)),
            "min_child_samples": Choice((5, 10, 20, 30, 50, 70, 100, 150)),
            "subsample": Choice((0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            "colsample_bytree": Choice((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
            "reg_alpha": Choice((0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10)),
            "reg_lambda": Choice((0, 0.001, 0.01, 0.1, 0.5, 1, 5, 10)),
            "tweedie_variance_power": Choice((1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9)),
        }
    )


@dataclass(frozen=True, slots=True)
class ForecastTuning:
    """The best forecaster settings found and how they scored.

    Attributes:
        wmape: Pooled over the validation folds
    """

    settings: dict[str, Any]
    wmape: float
    base_wmape: float
    seconds: float
    search: SearchResult = field(repr=False)


def tune_forecaster(
    history: History,
    make: Callable[[Mapping[str, Any]], Forecaster],
    space: SearchSpace,
    *,
    base: Mapping[str, Any],
    starts: Sequence[dt.date] | None = None,
    folds: int = 3,
    days: int = 7,
    calendar: Calendar | None = None,
    budget: int = 30,
    tuner: Tuner | None = None,
    seed: int = 0,
) -> ForecastTuning:
    """Search ``space`` for the settings whose forecaster ``make(settings)`` has the lowest WMAPE.

    The score is the pooled WMAPE of a walk-forward backtest on the validation
    folds (by default the last 3 weeks). ``base`` is tried first; keep folds that
    are later used for testing out of the validation.
    """
    started = time.perf_counter()

    def wmape(settings: Mapping[str, Any]) -> float:
        result = backtest(
            history, make(settings), starts=starts, folds=folds, days=days, calendar=calendar
        )
        return result.scores.wmape

    def score(point: NDArray[np.float64]) -> float:
        return wmape({**base, **space.decode(point)})

    first = space.encode({name: base[name] for name in space.names})
    search = (BeeColony() if tuner is None else tuner).minimize(
        score, space.size, budget, seed=seed, start=[first]
    )
    settings = {**base, **space.decode(search.best_point)}
    return ForecastTuning(
        settings=settings,
        wmape=search.best_value,
        base_wmape=search.trace[0].value,
        seconds=time.perf_counter() - started,
        search=search,
    )


def tune_lightgbm(
    history: History,
    *,
    train_from: dt.date | None = None,
    **options: Any,
) -> ForecastTuning:
    """Tune :func:`~pirilib.forecast.lightgbm_tweedie` over :func:`lightgbm_space`."""
    from pirilib.forecast.models import lightgbm_tweedie

    base = {key: value for key, value in TWEEDIE_SETTINGS.items() if key != "subsample_freq"}
    return tune_forecaster(
        history,
        lambda settings: lightgbm_tweedie(train_from=train_from, **settings),
        lightgbm_space(),
        base=base,
        **options,
    )
