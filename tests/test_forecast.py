import datetime as dt
import re
from operator import attrgetter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pirilib import synthetic
from pirilib.forecast import (
    FEATURES,
    Backtester,
    Calendar,
    History,
    Regression,
    SeasonalMean,
    SeasonalNaive,
    backtest,
    buffer_factors,
    clean,
    compare,
    demand_rows,
    fold_starts,
    lightgbm_tweedie,
    load_history,
    planning_document,
    report,
    tune_forecaster,
)
from pirilib.forecast.prepare import features
from pirilib.spec import parse_spec
from pirilib.tuning import Int, SearchSpace

MONDAY = dt.date(2026, 1, 5)
FOLDS = 3
"""Backtest folds of one week each, ending with the history."""


def weekly(values: list[list[float]], start: dt.date = MONDAY) -> History:
    """A history from rows of daily values (NaN for missing)."""
    return History(
        origins=tuple(f"O{i}" for i in range(len(values))),
        destinations=tuple(f"D{i}" for i in range(len(values))),
        start=start,
        values=np.array(values, dtype=np.float64),
    )


def network_for(history: History) -> dict[str, Any]:
    """A spot-only network with a center for every origin and destination of ``history``."""
    centers = sorted(set(history.origins) | set(history.destinations))
    return {
        "schema": "pirilib/linehaul@0.1",
        "centers": [
            {"id": name, "lat": 39.0 + i * 0.1, "lon": 30.0 + i * 0.2}
            for i, name in enumerate(centers)
        ],
        "matrix": {"source": "haversine"},
        "vehicle_types": [{"name": "tir", "capacity": {"desi": 22000}}],
        "fleet": {"spot": [{"type": "tir", "cost": {"fixed": 1000, "per_km": 2}}]},
    }


@pytest.fixture(scope="module")
def sample() -> tuple[History, Calendar]:
    return synthetic.history(lines=12, days=112, seed=3)


@pytest.fixture(scope="module")
def buffered(sample: tuple[History, Calendar]) -> Any:
    """A 4-fold backtest of the seasonal mean, the base of the buffer tests."""
    history, calendar = sample
    return backtest(history, SeasonalMean(), folds=4, calendar=calendar)


class MeanModel:
    """A regressor that predicts the training mean."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanModel":
        self.mean = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean)


class Spy:
    """A forecaster that records the length of every history it is given."""

    def __init__(self) -> None:
        self.seen: list[int] = []
        self.ends: list[dt.date] = []
        self.starts: list[dt.date] = []

    def forecast(self, past: History, start: dt.date, days: int, calendar: Any = None) -> Any:
        self.seen.append(past.length)
        self.ends.append(past.end)
        self.starts.append(start)
        return SeasonalMean().forecast(past, start, days, calendar)


def test_history_from_rows() -> None:
    history = History.from_rows(
        [
            ("A", "B", MONDAY, 10.0),
            ("A", "B", MONDAY, 5.0),
            ("A", "C", MONDAY + dt.timedelta(days=2), 7.0),
        ]
    )

    assert history.lines == 2
    assert history.length == 3
    assert history.values[0, 0] == 15.0
    assert np.isnan(history.values[0, 1])
    assert history.index("A", "C") == 1
    assert history.end == MONDAY + dt.timedelta(days=2)


def test_a_history_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        weekly([[1.0, -2.0]])


def test_load_history_from_a_turkish_spreadsheet(tmp_path: Path) -> None:
    path = tmp_path / "talep.csv"
    path.write_text(
        "Çıkış;Varış;Tarih;Desi\nA;B;05.01.2026;1.234,5\nA;B;06.01.2026;10\n", encoding="utf-8"
    )

    history = load_history(
        path,
        delimiter=";",
        decimal=",",
        date_format="%d.%m.%Y",
        columns={"origin": "Çıkış", "destination": "Varış", "date": "Tarih"},
        column="Desi",
    )

    assert history.values.tolist() == [[1234.5, 10.0]]


def test_clean_imputes_and_winsorizes() -> None:
    week = [100.0] * 7
    history = weekly([week * 3 + [100.0, np.nan, 100.0, 100.0, 5000.0, 100.0, 100.0]])

    cleaned = clean(history, Calendar(), cut=28)

    assert cleaned.imputed[0, 22]
    assert cleaned.filled[0, 22] == 100.0
    assert cleaned.filled[0, 25] == 5000.0
    assert cleaned.winsorized[0, 25] < 5000.0


def test_unusual_days_are_imputed() -> None:
    history = weekly([[100.0] * 14 + [10.0]])
    calendar = Calendar.of(unusual=[MONDAY + dt.timedelta(days=14)])

    cleaned = clean(history, calendar, cut=15)

    assert cleaned.imputed[0, 14]
    assert cleaned.filled[0, 14] == 100.0


def test_days_before_a_line_starts_stay_missing() -> None:
    history = weekly([[np.nan, np.nan, 50.0, 50.0]])

    cleaned = clean(history, Calendar(), cut=4)

    np.testing.assert_array_equal(cleaned.winsorized[0], [np.nan, np.nan, 50.0, 50.0])


def test_cleaning_and_features_do_not_look_ahead(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    cut = 84
    changed_values = history.values.copy()
    changed_values[:, cut:] = 1e6
    changed = History(history.origins, history.destinations, history.start, changed_values)
    rows = np.repeat(np.arange(history.lines), 7)
    days = np.tile(np.arange(cut, cut + 7), history.lines)

    first, second = clean(history, calendar, cut), clean(changed, calendar, cut)
    a = features(history, calendar, first.winsorized, rows, days)
    b = features(changed, calendar, second.winsorized, rows, days)

    np.testing.assert_array_equal(first.winsorized, second.winsorized)
    assert a.shape == (rows.size, len(FEATURES))
    np.testing.assert_array_equal(a, b)


def test_lag_and_rolling_features() -> None:
    """Day ``d`` has value ``d``, so a lag of k on day 35 is 35 - k."""
    history = weekly([np.arange(35.0).tolist()])
    cleaned = clean(history, Calendar(), cut=35)

    x = features(history, Calendar(), cleaned.winsorized, np.array([0]), np.array([35]))

    columns = dict(zip(FEATURES, x[0], strict=True))
    assert columns["lag7"] == 28.0
    assert columns["lag28"] == 7.0
    assert columns["roll7_mean"] == 25.0
    assert columns["dow"] == 0.0


SIX_WEEKS = weekly([(10 * np.arange(6)[:, None] + np.arange(7)).ravel().tolist()])
"""Week w, weekday d has value 10 w + d."""


def test_seasonal_mean_averages_the_last_weeks_per_weekday() -> None:
    forecast = SeasonalMean(weeks=4).forecast(SIX_WEEKS, MONDAY + dt.timedelta(days=42), 7)

    assert forecast.values.tolist() == [[35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0]]


def test_seasonal_naive_repeats_the_last_week() -> None:
    forecast = SeasonalNaive().forecast(SIX_WEEKS, MONDAY + dt.timedelta(days=42), 7)

    assert forecast.values.tolist() == [[50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0]]


OLD_AND_NEW = weekly([[100.0] * 7 + [np.nan] * 42, [100.0] * 49])
"""A line seen only in its first week, and a line seen every day."""


def test_presence_filter_drops_lines_that_stopped() -> None:
    forecast = SeasonalMean().forecast(OLD_AND_NEW, MONDAY + dt.timedelta(days=49), 7)

    assert forecast.values.tolist() == [[0.0] * 7, [100.0] * 7]


def test_without_presence_filter_old_lines_are_kept() -> None:
    kept = SeasonalMean(presence_days=None).forecast(OLD_AND_NEW, MONDAY + dt.timedelta(days=49), 7)

    assert kept.values[0].tolist() == [100.0] * 7


def test_regression_adapter_forecasts_every_line(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    raw = Regression(MeanModel).forecast(history, history.date(98), 7, calendar)

    assert raw.values.shape == (history.lines, 7)
    np.testing.assert_array_equal(raw.values >= 0, True)


def test_regression_adapter_with_a_log_target(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    logged = Regression(MeanModel, target="log1p").forecast(history, history.date(98), 7, calendar)

    assert logged.total() > 0


def test_regression_adapter_forecasts_at_most_a_week(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    with pytest.raises(ValueError, match="must not exceed 7"):
        Regression(MeanModel).forecast(history, history.date(98), 8, calendar)


def test_lightgbm_tweedie(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    forecast = lightgbm_tweedie().forecast(history, history.date(98), 7, calendar)

    assert forecast.values.shape == (history.lines, 7)
    assert repr(forecast).startswith("Forecast(lines=12, days=7")


def test_backtest_gives_each_fold_only_the_past(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    spy = Spy()

    backtest(history, spy, folds=FOLDS, calendar=calendar, name="spy")

    assert spy.seen == [91, 98, 105]
    assert list(map(history.day, fold_starts(history, FOLDS, 7))) == spy.seen
    assert np.less(spy.ends, spy.starts).tolist() == [True, True, True]


@pytest.mark.parametrize("fold", range(FOLDS))
def test_backtest_scores_only_observed_usual_days(
    sample: tuple[History, Calendar], fold: int
) -> None:
    history, calendar = sample

    scored = backtest(history, SeasonalMean(), folds=FOLDS, calendar=calendar).folds[fold]

    days = history.day(scored.start) + scored.offsets
    np.testing.assert_array_equal(np.isnan(history.values[scored.lines, days]), False)
    assert set(map(history.date, days.tolist())).isdisjoint(calendar.unusual)


def test_backtest_scores_are_the_sum_of_the_folds(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    result = backtest(history, SeasonalMean(), folds=FOLDS, calendar=calendar)

    assert result.scores.cells == sum(map(lambda fold: fold.scores.cells, result.folds))


def test_compare_always_runs_the_baselines(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    results = compare(history, {"mean": Regression(MeanModel)}, folds=2, calendar=calendar)

    assert sorted(map(attrgetter("name"), results)) == ["mean", "seasonal_mean_4", "seasonal_naive"]


def test_compare_ranks_by_wmape(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    results = compare(history, {"mean": Regression(MeanModel)}, folds=2, calendar=calendar)

    wmape = list(map(lambda result: result.scores.wmape, results))
    assert wmape == sorted(wmape)


def test_the_report_gives_skill_against_the_seasonal_mean(
    sample: tuple[History, Calendar],
) -> None:
    history, calendar = sample
    results = compare(history, {"mean": Regression(MeanModel)}, folds=2, calendar=calendar)

    table = report(results)

    assert "skill" in table.splitlines()[0]
    assert re.search(r"(?m)^seasonal_mean_4.*\+0\.0%", table)


def test_backtester_matches_the_functions(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    tester = Backtester(folds=2, calendar=calendar)
    model = SeasonalMean(4)

    direct = backtest(history, model, folds=2, calendar=calendar)
    again = tester.run(history, model)

    assert tester.starts(history) == list(map(attrgetter("start"), direct.folds))
    assert again.scores == direct.scores


def test_the_backtester_compares_like_the_function(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    names = list(
        map(attrgetter("name"), Backtester(folds=2, calendar=calendar).compare(history, {}))
    )

    expected = list(map(attrgetter("name"), compare(history, {}, folds=2, calendar=calendar)))
    assert names == expected


def test_tune_forecaster(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample

    result = tune_forecaster(
        history,
        lambda settings: SeasonalMean(weeks=settings["weeks"]),
        SearchSpace({"weeks": Int(1, 8)}),
        base={"weeks": 4},
        folds=2,
        calendar=calendar,
        budget=6,
    )

    assert result.wmape <= result.base_wmape
    assert result.settings["weeks"] in set(range(1, 9))
    assert result.search.evaluations == 6


def test_a_higher_quantile_gives_larger_buffers(
    sample: tuple[History, Calendar], buffered: Any
) -> None:
    history, _ = sample

    low = buffer_factors(buffered, 0.2, history.lines)
    high = buffer_factors(buffered, 0.9, history.lines)

    np.testing.assert_array_equal(high >= low, True)


def test_demand_rows_carry_the_buffered_forecast(
    sample: tuple[History, Calendar], buffered: Any
) -> None:
    """Rows are rounded and small pieces dropped, so at least 99 % of the total remains."""
    history, calendar = sample
    high = buffer_factors(buffered, 0.9, history.lines)
    forecast = SeasonalMean().forecast(history, history.date(history.length), 7, calendar)

    rows = demand_rows(forecast, factors=high)

    total = sum(map(lambda row: row["quantity"]["desi"], rows))
    assert total >= 0.99 * (forecast.values * high[:, None]).sum()


def test_a_planning_document_has_a_demand_per_row(
    sample: tuple[History, Calendar], buffered: Any
) -> None:
    history, calendar = sample
    high = buffer_factors(buffered, 0.9, history.lines)
    forecast = SeasonalMean().forecast(history, history.date(history.length), 7, calendar)

    spec = parse_spec(planning_document(network_for(history), forecast, factors=high))

    assert spec.calendar.start == forecast.start
    assert len(spec.demand) == len(demand_rows(forecast, factors=high))


def test_the_buffer_quantile_must_be_below_one(
    sample: tuple[History, Calendar], buffered: Any
) -> None:
    history, _ = sample

    with pytest.raises(ValueError, match="quantile"):
        buffer_factors(buffered, 1.0, history.lines)


def test_scaled_regression(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    start = history.date(98)
    bigger = History(history.origins, history.destinations, history.start, history.values * 10)

    scaled = Regression(MeanModel, scale=True).forecast(history, start, 7, calendar)
    tenfold = Regression(MeanModel, scale=True).forecast(bigger, start, 7, calendar)

    assert scaled.values.shape == (history.lines, 7)
    assert scaled.total() > 0
    assert tenfold.total() == pytest.approx(10 * scaled.total(), rel=1e-6)


def failing_import(real: Any) -> Any:
    """An ``__import__`` stub that fails for lightgbm as a missing libomp does on macOS."""

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lightgbm":
            raise OSError("Library not loaded: @rpath/libomp.dylib")
        return real(name, *args, **kwargs)

    return broken


def test_the_import_stub_fails_only_for_lightgbm() -> None:
    import builtins

    broken = failing_import(builtins.__import__)

    assert broken("math").sqrt(4.0) == 2.0
    with pytest.raises(OSError, match="libomp"):
        broken("lightgbm")


def test_lightgbm_load_failure_explains_the_fix(
    sample: tuple[History, Calendar], monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    monkeypatch.setattr(builtins, "__import__", failing_import(builtins.__import__))
    history, calendar = sample

    with pytest.raises(ImportError, match="brew install libomp"):
        lightgbm_tweedie().forecast(history, history.date(98), 7, calendar)
