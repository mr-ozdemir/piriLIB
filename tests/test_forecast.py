import datetime as dt
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
from pirilib.tuning import Int

MONDAY = dt.date(2026, 1, 5)


def weekly(values: list[list[float]], start: dt.date = MONDAY) -> History:
    """A history from rows of daily values (NaN for missing)."""
    return History(
        origins=tuple(f"O{i}" for i in range(len(values))),
        destinations=tuple(f"D{i}" for i in range(len(values))),
        start=start,
        values=np.array(values, dtype=np.float64),
    )


@pytest.fixture(scope="module")
def sample() -> tuple[History, Calendar]:
    return synthetic.history(lines=12, days=112, seed=3)


class MeanModel:
    """A regressor that predicts the training mean."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanModel":
        self.mean = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.mean)


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
    assert history.values[0].tolist()[:1] == [15.0]
    assert np.isnan(history.values[0, 1])
    assert history.index("A", "C") == 1
    assert history.end == MONDAY + dt.timedelta(days=2)
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
    values = week * 3 + [100.0, np.nan, 100.0, 100.0, 5000.0, 100.0, 100.0]
    history = weekly([values])
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
    assert np.isnan(cleaned.winsorized[0, :2]).all()
    assert cleaned.winsorized[0, 2:].tolist() == [50.0, 50.0]


def test_cleaning_and_features_do_not_look_ahead(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    cut = 84
    changed_values = history.values.copy()
    changed_values[:, cut:] = 1e6
    changed = History(history.origins, history.destinations, history.start, changed_values)
    first, second = clean(history, calendar, cut), clean(changed, calendar, cut)
    assert np.array_equal(first.winsorized, second.winsorized, equal_nan=True)
    rows = np.repeat(np.arange(history.lines), 7)
    days = np.tile(np.arange(cut, cut + 7), history.lines)
    a = features(history, calendar, first.winsorized, rows, days)
    b = features(changed, calendar, second.winsorized, rows, days)
    assert a.shape == (rows.size, len(FEATURES))
    assert np.array_equal(a, b, equal_nan=True)


def test_lag_and_rolling_features() -> None:
    history = weekly([[float(day) for day in range(35)]])
    cleaned = clean(history, Calendar(), cut=35)
    x = features(history, Calendar(), cleaned.winsorized, np.array([0]), np.array([35]))
    columns = dict(zip(FEATURES, x[0], strict=True))
    assert columns["lag7"] == 28.0
    assert columns["lag28"] == 7.0
    assert columns["roll7_mean"] == np.mean(range(22, 29))
    assert columns["dow"] == 0.0


def test_seasonal_mean() -> None:
    rows = [[float(10 * week + day) for week in range(6) for day in range(7)]]
    history = weekly(rows)
    forecast = SeasonalMean(weeks=4).forecast(history, MONDAY + dt.timedelta(days=42), 7)
    assert forecast.values.tolist() == [[35.0 + day for day in range(7)]]
    naive = SeasonalNaive().forecast(history, MONDAY + dt.timedelta(days=42), 7)
    assert naive.values.tolist() == [[50.0 + day for day in range(7)]]


def test_presence_filter() -> None:
    old = [100.0] * 7 + [np.nan] * 42
    new = [100.0] * 49
    history = weekly([old, new])
    forecast = SeasonalMean().forecast(history, MONDAY + dt.timedelta(days=49), 7)
    assert forecast.values[0].sum() == 0.0
    assert forecast.values[1].tolist() == [100.0] * 7
    kept = SeasonalMean(presence_days=None).forecast(history, MONDAY + dt.timedelta(days=49), 7)
    assert kept.values[0].tolist() == [100.0] * 7


def test_regression_adapter(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    start = history.date(98)
    raw = Regression(MeanModel).forecast(history, start, 7, calendar)
    assert raw.values.shape == (history.lines, 7)
    assert np.all(raw.values >= 0)
    logged = Regression(MeanModel, target="log1p").forecast(history, start, 7, calendar)
    assert logged.total() > 0
    with pytest.raises(ValueError, match="must not exceed 7"):
        Regression(MeanModel).forecast(history, start, 8, calendar)


def test_lightgbm_tweedie(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    forecast = lightgbm_tweedie().forecast(history, history.date(98), 7, calendar)
    assert forecast.values.shape == (history.lines, 7)
    assert repr(forecast).startswith("Forecast(lines=12, days=7")


def test_backtest_gives_each_fold_only_the_past(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    seen: list[int] = []

    class Spy:
        """A forecaster that records which history it was given."""

        def forecast(self, past: History, start: dt.date, days: int, calendar: Any = None) -> Any:
            seen.append(past.length)
            assert past.end < start
            return SeasonalMean().forecast(past, start, days, calendar)

    result = backtest(history, Spy(), folds=3, calendar=calendar, name="spy")
    starts = fold_starts(history, 3, 7)
    assert [history.day(start) for start in starts] == seen == [91, 98, 105]
    for fold in result.folds:
        days = history.day(fold.start) + fold.offsets
        assert not np.isnan(history.values[fold.lines, days]).any()
        assert not any(history.date(int(day)) in calendar.unusual for day in days)
    assert result.scores.cells == sum(f.scores.cells for f in result.folds)


def test_compare_always_runs_the_baselines(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    results = compare(history, {"mean": Regression(MeanModel)}, folds=2, calendar=calendar)
    names = [result.name for result in results]
    assert sorted(names) == ["mean", "seasonal_mean_4", "seasonal_naive"]
    assert [r.scores.wmape for r in results] == sorted(r.scores.wmape for r in results)
    table = report(results)
    assert "skill" in table.splitlines()[0]
    assert "+0.0%" in next(
        line for line in table.splitlines() if line.startswith("seasonal_mean_4")
    )


def test_backtester_matches_the_functions(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    tester = Backtester(folds=2, calendar=calendar)
    model = SeasonalMean(4)
    direct = backtest(history, model, folds=2, calendar=calendar)
    again = tester.run(history, model)
    assert tester.starts(history) == [fold.start for fold in direct.folds]
    assert again.scores == direct.scores
    assert [r.name for r in tester.compare(history, {})] == [
        r.name for r in compare(history, {}, folds=2, calendar=calendar)
    ]


def test_tune_forecaster(sample: tuple[History, Calendar]) -> None:
    from pirilib.tuning import SearchSpace

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
    assert 1 <= result.settings["weeks"] <= 8
    assert result.search.evaluations == 6


def test_buffer_and_planning_document(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    result = backtest(history, SeasonalMean(), folds=4, calendar=calendar)
    low, high = (
        buffer_factors(result, 0.2, history.lines),
        buffer_factors(result, 0.9, history.lines),
    )
    assert np.all(high >= low)
    forecast = SeasonalMean().forecast(history, history.date(history.length), 7, calendar)
    rows = demand_rows(forecast, factors=high)
    assert (
        sum(row["quantity"]["desi"] for row in rows)
        >= 0.99 * (forecast.values * high[:, None]).sum()
    )

    centers = sorted(set(history.origins) | set(history.destinations))
    network = {
        "schema": "pirilib/linehaul@0.1",
        "centers": [
            {"id": name, "lat": 39.0 + i * 0.1, "lon": 30.0 + i * 0.2}
            for i, name in enumerate(centers)
        ],
        "matrix": {"source": "haversine"},
        "vehicle_types": [{"name": "tir", "capacity": {"desi": 22000}}],
        "fleet": {"spot": [{"type": "tir", "cost": {"fixed": 1000, "per_km": 2}}]},
    }
    spec = parse_spec(planning_document(network, forecast, factors=high))
    assert spec.calendar.start == forecast.start
    assert len(spec.demand) == len(rows)
    with pytest.raises(ValueError, match="quantile"):
        buffer_factors(result, 1.0, history.lines)


def test_scaled_regression(sample: tuple[History, Calendar]) -> None:
    history, calendar = sample
    start = history.date(98)
    scaled = Regression(MeanModel, scale=True).forecast(history, start, 7, calendar)
    assert scaled.values.shape == (history.lines, 7)
    assert scaled.total() > 0
    bigger = History(history.origins, history.destinations, history.start, history.values * 10)
    tenfold = Regression(MeanModel, scale=True).forecast(bigger, start, 7, calendar)
    assert tenfold.total() == pytest.approx(10 * scaled.total(), rel=1e-6)


def test_lightgbm_load_failure_explains_the_fix(
    sample: tuple[History, Calendar], monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real = builtins.__import__

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lightgbm":
            raise OSError("Library not loaded: @rpath/libomp.dylib")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    history, calendar = sample
    with pytest.raises(ImportError, match="brew install libomp"):
        lightgbm_tweedie().forecast(history, history.date(98), 7, calendar)
