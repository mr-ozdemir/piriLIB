import datetime as dt
from operator import attrgetter, itemgetter
from pathlib import Path

import numpy as np
import pytest

from pirilib import synthetic
from pirilib.errors import SpecError
from pirilib.forecast import (
    Calendar,
    Forecast,
    SeasonalMean,
    WaveHistory,
    WaveSplitter,
    demand_rows,
    load_wave_history,
    planning_document,
)
from pirilib.spec import parse_spec

MONDAY = dt.date(2026, 1, 5)
NINE, FIVE = dt.time(9), dt.time(17)
START = MONDAY + dt.timedelta(weeks=5)
MINIMUM = 0.5
"""Smallest piece kept when a forecast is split into rows."""

Record = tuple[str, str, dt.date, dt.time, float]


def day(week: int, weekday: int = 0) -> dt.date:
    return MONDAY + dt.timedelta(weeks=week, days=weekday)


RECORDS: list[Record] = [
    ("A", "B", day(0), NINE, 100.0),
    ("A", "B", day(0), FIVE, 0.0),
    ("A", "B", day(1), NINE, 30.0),
    ("A", "B", day(1), FIVE, 70.0),
    ("A", "B", day(2), NINE, 10.0),
    ("A", "B", day(2), FIVE, 90.0),
    ("A", "B", day(3), NINE, 50.0),
    ("A", "B", day(3), FIVE, 50.0),
    ("A", "B", day(4), NINE, 30.0),
    ("A", "B", day(4), FIVE, 70.0),
    ("A", "B", day(4, 1), NINE, 10.0),
    ("A", "B", day(4, 1), FIVE, 10.0),
    ("C", "D", day(2, 2), NINE, 40.0),
]
"""Five Mondays of A -> B (the first one is older than four weeks), one Tuesday of A -> B
and one Wednesday of C -> D."""

LATER: list[Record] = [
    ("A", "B", START, NINE, 1000.0),
    ("A", "B", START + dt.timedelta(days=1), FIVE, 500.0),
    ("C", "D", START + dt.timedelta(days=9), FIVE, 77.0),
    ("E", "F", START + dt.timedelta(days=3), NINE, 5.0),
]
"""Records on and after the forecast start, which the shares must not see."""


def forecast(lines: list[tuple[str, str]], values: list[list[float]], start: dt.date) -> Forecast:
    origins, destinations = zip(*lines, strict=True)
    return Forecast(
        origins=origins,
        destinations=destinations,
        start=start,
        values=np.array(values, dtype=np.float64),
    )


def test_wave_history_from_rows() -> None:
    history = WaveHistory.from_rows(RECORDS)
    line = history.origins.index("C")

    daily = history.daily()

    assert history.waves == (NINE, FIVE)
    assert history.start == MONDAY
    assert history.values[line, history.day(day(2, 2))].tolist() == [40.0, 0.0]
    np.testing.assert_array_equal(np.isnan(history.values[line, 0]), True)
    assert daily.values[0, 0] == 100.0
    assert np.isnan(daily.values[line, 0])


def test_a_wave_history_needs_a_record() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        WaveHistory.from_rows([])


def test_exact_shares_and_fallbacks() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    ab, cd = shares.index("A", "B"), shares.index("C", "D")

    assert (ab, cd) == (0, 1)
    assert shares.shares[ab, 0] == pytest.approx([120 / 400, 280 / 400])
    assert shares.level[ab, 0] == 0
    assert shares.days[ab, 0] == 4
    assert shares.spread[ab, 0] == pytest.approx([np.sqrt(0.02)] * 2)
    assert shares.shares[ab, 1] == pytest.approx([130 / 420, 290 / 420])
    assert shares.level[ab, 1] == 1
    np.testing.assert_array_equal(np.isnan(shares.spread[ab, 1]), True)
    assert shares.overall == pytest.approx([170 / 460, 290 / 460])
    assert shares.shares[cd, 2] == pytest.approx(shares.overall)
    assert shares.level[cd].tolist() == [2] * 7
    np.testing.assert_allclose(shares.shares.sum(axis=2), 1.0)
    assert shares.share("X", "Y", START) == pytest.approx(shares.overall)


def test_stability_counts_the_levels_used() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    stability = shares.stability()

    assert (stability.cells, stability.weekday, stability.line, stability.overall) == (3, 1, 1, 1)
    assert stability.spread_mean == pytest.approx(np.sqrt(0.02))
    assert "3 line-weekdays" in stability.describe()


def test_weeks_limit_the_history_used() -> None:
    two = WaveSplitter(weeks=2).fit(WaveHistory.from_rows(RECORDS), START)

    assert two.shares[0, 0] == pytest.approx([80 / 200, 120 / 200])


def test_unusual_days_are_skipped() -> None:
    calendar = Calendar.of(unusual=[day(2)])

    skipped = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START, calendar)

    assert skipped.shares[0, 0] == pytest.approx([210 / 400, 190 / 400])


def test_the_splitter_needs_at_least_one_week() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        WaveSplitter(weeks=0)


@pytest.mark.parametrize("line", [("A", "B"), ("C", "D")], ids=["A-B", "C-D"])
@pytest.mark.parametrize("field", ["shares", "level", "days", "spread"])
def test_shares_do_not_look_ahead(line: tuple[str, str], field: str) -> None:
    before = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    after = WaveSplitter().fit(WaveHistory.from_rows(RECORDS + LATER), START)

    np.testing.assert_array_equal(
        getattr(after, field)[after.index(*line)], getattr(before, field)[before.index(*line)]
    )


def test_overall_shares_do_not_look_ahead() -> None:
    before = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    after = WaveSplitter().fit(WaveHistory.from_rows(RECORDS + LATER), START)

    np.testing.assert_array_equal(after.overall, before.overall)


def test_shares_need_a_record_before_the_start() -> None:
    history = WaveHistory.from_rows([("A", "B", MONDAY, NINE, 0.0), *LATER])

    with pytest.raises(ValueError, match="no usable record"):
        WaveSplitter().fit(history, START)


def test_the_start_must_come_after_the_history() -> None:
    with pytest.raises(ValueError, match="must come after"):
        WaveSplitter().fit(WaveHistory.from_rows(RECORDS), MONDAY)


@pytest.fixture(scope="module")
def split_week() -> tuple[Forecast, Forecast, WaveSplitter, object, object]:
    """A synthetic daily forecast of 10 lines for 7 days, and its split into 2 waves."""
    waves, calendar = synthetic.wave_history(lines=10, days=84, seed=4)
    history = waves.daily()
    start = history.date(history.length)
    shares = WaveSplitter().fit(waves, start, calendar)
    daily = SeasonalMean().forecast(history, start, 7, calendar)
    return daily, shares.split(daily), shares, history, calendar


def test_split_preserves_daily_totals(split_week: tuple[object, ...]) -> None:
    daily, split, *_ = split_week

    assert split.values.shape == (10, 7, 2)
    np.testing.assert_allclose(split.values.sum(axis=2), daily.values)
    np.testing.assert_allclose(split.daily().values, daily.values)


def test_split_rows_keep_at_least_the_minimum(split_week: tuple[object, ...]) -> None:
    _, split, *_ = split_week

    quantities = list(map(itemgetter(4), split.rows(MINIMUM)))

    assert min(quantities) >= MINIMUM


def test_split_rows_lose_at_most_the_dropped_pieces(split_week: tuple[object, ...]) -> None:
    """Each of the 140 pieces drops less than the minimum, so the total falls by less."""
    daily, split, *_ = split_week

    kept = sum(map(itemgetter(4), split.rows(MINIMUM)))

    assert kept <= daily.total() + 1e-6
    assert kept >= daily.total() - MINIMUM * split.values.size


def test_synthetic_shares_are_mostly_stable(split_week: tuple[object, ...]) -> None:
    _, _, shares, *_ = split_week

    stability = shares.stability()

    assert stability.weekday > 0
    assert stability.spread_median > 0
    assert stability.spread_median < 0.2


def test_split_rejects_a_forecast_starting_earlier(split_week: tuple[object, ...]) -> None:
    daily, _, shares, history, calendar = split_week
    earlier = SeasonalMean().forecast(history, daily.start - dt.timedelta(days=1), 7, calendar)

    with pytest.raises(ValueError, match="earlier"):
        shares.split(earlier)


def test_split_drops_small_pieces() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)
    small = forecast([("A", "B")], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], START)

    rows = list(shares.split(small).rows(MINIMUM))

    assert rows == [("A", "B", START, FIVE, pytest.approx(0.7))]


def test_demand_rows_of_a_split_forecast_carry_the_wave_as_ready_time() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)
    small = forecast([("A", "B")], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], START)

    rows = demand_rows(small, minimum=MINIMUM, waves=shares)

    assert rows == [
        {
            "origin": "A",
            "destination": "B",
            "date": START.isoformat(),
            "ready": "17:00",
            "quantity": {"desi": 0.7},
        }
    ]


NETWORK = {
    "schema": "pirilib/linehaul@0.2",
    "centers": [
        {"id": "A", "lat": 39.0, "lon": 30.0},
        {"id": "B", "lat": 39.1, "lon": 30.2},
        {"id": "C", "lat": 39.2, "lon": 30.4},
        {"id": "D", "lat": 39.3, "lon": 30.6},
    ],
    "matrix": {"source": "haversine"},
    "vehicle_types": [{"name": "tir", "capacity": {"desi": 22000}, "speed_kmh": 70}],
    "fleet": {"spot": [{"type": "tir", "cost": {"fixed": 1000, "per_km": 2}}]},
    "rules": {"service": {"due_hours": 24}},
}
WEEK = forecast(
    [("A", "B"), ("C", "D"), ("A", "D")],
    [[100.0] * 7, [40.0] * 7, [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    START,
)
"""15 non-zero line-days: A -> B and C -> D every day, A -> D on Tuesday."""


def test_a_daily_planning_document_has_no_ready_times() -> None:
    daily = parse_spec(planning_document(NETWORK, WEEK))

    assert len(daily.demand) == 15
    assert set(map(attrgetter("ready"), daily.demand)) == {None}


def test_a_planning_document_with_waves_has_a_row_per_wave() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    spec = parse_spec(planning_document(NETWORK, WEEK, waves=shares))

    assert len(spec.demand) == 30
    assert set(map(attrgetter("ready"), spec.demand)) == {NINE, FIVE}


def test_a_planning_document_with_waves_keeps_the_total() -> None:
    """Each of the 30 rows is rounded to three decimals."""
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    spec = parse_spec(planning_document(NETWORK, WEEK, waves=shares))

    total = sum(map(lambda row: row.quantity["desi"], spec.demand))
    assert total == pytest.approx(WEEK.total(), abs=0.0005 * 30)


def test_a_planning_document_splits_monday_by_the_wave_shares() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(RECORDS), START)

    spec = parse_spec(planning_document(NETWORK, WEEK, waves=shares))

    rows = set(
        map(lambda r: (r.origin, r.destination, r.date, r.ready, r.quantity["desi"]), spec.demand)
    )
    assert {("A", "B", START, NINE, 30.0), ("A", "B", START, FIVE, 70.0)} <= rows


TURKISH = (
    "cikis;varis;tarih;saat;toplam_desi\n"
    "A;B;05.01.2026;9:00;1.234,5\n"
    "A;B;05.01.2026;17:00;100\n"
    "A;B;06.01.2026;09:00;10\n"
    "A;B;06.01.2026;9;5\n"
    "B;A;06.01.2026;17;7,5\n"
)


def test_load_wave_history_from_a_turkish_spreadsheet(tmp_path: Path) -> None:
    path = tmp_path / "waves.csv"
    path.write_text(TURKISH, encoding="utf-8")

    history = load_wave_history(
        path,
        column="toplam_desi",
        wave="saat",
        delimiter=";",
        decimal=",",
        date_format="%d.%m.%Y",
        columns={"origin": "cikis", "destination": "varis", "date": "tarih"},
    )

    assert history.waves == (NINE, FIVE)
    assert history.values[0].tolist() == [[1234.5, 100.0], [15.0, 0.0]]
    assert history.values[1, 1].tolist() == [0.0, 7.5]
    np.testing.assert_array_equal(np.isnan(history.values[1, 0]), True)


def test_bad_wave_times_are_reported_with_their_lines(tmp_path: Path) -> None:
    path = tmp_path / "waves.csv"
    path.write_text(
        "origin,destination,date,wave,desi\nA,B,2026-01-05,25:00,1\nA,B,2026-01-05,,1\n",
        encoding="utf-8",
    )

    with pytest.raises(SpecError) as error:
        load_wave_history(path)

    assert "waves.csv:2: 'wave': '25:00' is not a time of day" in str(error.value)
    assert "waves.csv:3: 'wave' is empty" in str(error.value)


def test_synthetic_waves_add_up_to_the_daily_history() -> None:
    """Both are generated from the same seed; waves are rounded to 0.1 desi each."""
    waves, _ = synthetic.wave_history(lines=6, days=28, seed=2)

    daily, _ = synthetic.history(lines=6, days=28, seed=2)

    np.testing.assert_allclose(waves.daily().values, daily.values, atol=0.11)


def test_synthetic_wave_history_round_trips(tmp_path: Path) -> None:
    waves, _ = synthetic.wave_history(lines=6, days=28, seed=2)
    path = synthetic.write_wave_history(waves, tmp_path / "waves.csv")

    loaded = load_wave_history(path)

    present = ~np.isnan(loaded.values)
    observed = np.nan_to_num(waves.values[:, : loaded.length], nan=0.0)
    assert loaded.waves == waves.waves
    np.testing.assert_allclose(loaded.values[present], observed[present])
