"""Demand forecasting: daily quantities (desi) per line, and the bridge to planning.

The core needs only numpy: history handling, cleaning, features, the seasonal
baselines, backtests, tuning and the split into time waves.
:func:`lightgbm_tweedie` and other regressors need ``pip install "pirilib[forecast]"``.
"""

from pirilib.forecast.bridge import buffer_factors, demand_rows, planning_document
from pirilib.forecast.evaluate import (
    Backtest,
    Backtester,
    Fold,
    Scores,
    backtest,
    compare,
    fold_starts,
    report,
)
from pirilib.forecast.history import History, load_history
from pirilib.forecast.models import (
    TWEEDIE_SETTINGS,
    Forecast,
    Forecaster,
    Regression,
    SeasonalMean,
    SeasonalNaive,
    lightgbm_tweedie,
)
from pirilib.forecast.prepare import FEATURES, Calendar, clean
from pirilib.forecast.tune import ForecastTuning, lightgbm_space, tune_forecaster, tune_lightgbm
from pirilib.forecast.waves import (
    WaveForecast,
    WaveHistory,
    WaveShares,
    WaveSplitter,
    WaveStability,
    load_wave_history,
)

__all__ = [
    "FEATURES",
    "TWEEDIE_SETTINGS",
    "Backtest",
    "Backtester",
    "Calendar",
    "Fold",
    "Forecast",
    "ForecastTuning",
    "Forecaster",
    "History",
    "Regression",
    "Scores",
    "SeasonalMean",
    "SeasonalNaive",
    "WaveForecast",
    "WaveHistory",
    "WaveShares",
    "WaveSplitter",
    "WaveStability",
    "backtest",
    "buffer_factors",
    "clean",
    "compare",
    "demand_rows",
    "fold_starts",
    "lightgbm_space",
    "lightgbm_tweedie",
    "load_history",
    "load_wave_history",
    "planning_document",
    "report",
    "tune_forecaster",
    "tune_lightgbm",
]
