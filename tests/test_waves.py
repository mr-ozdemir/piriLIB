import datetime as dt
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

Record = tuple[str, str, dt.date, dt.time, float]


def day(week: int, weekday: int = 0) -> dt.date:
    return MONDAY + dt.timedelta(weeks=week, days=weekday)


def records() -> list[Record]:
    """Five Mondays of A -> B (the first one is older than four weeks), one Tuesday of
    A -> B and one Wednesday of C -> D."""
    mondays = [(100.0, 0.0), (30.0, 70.0), (10.0, 90.0), (50.0, 50.0), (30.0, 70.0)]
    rows: list[Record] = []
    for week, (morning, evening) in enumerate(mondays):
        rows += [("A", "B", day(week), NINE, morning), ("A", "B", day(week), FIVE, evening)]
    rows += [("A", "B", day(4, 1), NINE, 10.0), ("A", "B", day(4, 1), FIVE, 10.0)]
    rows.append(("C", "D", day(2, 2), NINE, 40.0))
    return rows


def forecast(lines: list[tuple[str, str]], values: list[list[float]], start: dt.date) -> Forecast:
    return Forecast(
        origins=tuple(origin for origin, _ in lines),
        destinations=tuple(destination for _, destination in lines),
        start=start,
        values=np.array(values, dtype=np.float64),
    )


def test_wave_history_from_rows() -> None:
    history = WaveHistory.from_rows(records())
    assert history.waves == (NINE, FIVE)
    assert history.start == MONDAY
    line = history.origins.index("C")
    assert history.values[line, history.day(day(2, 2))].tolist() == [40.0, 0.0]
    assert np.isnan(history.values[line, 0]).all()
    daily = history.daily()
    assert daily.values[0, 0] == 100.0
    assert np.isnan(daily.values[line, 0])
    with pytest.raises(ValueError, match="at least one record"):
        WaveHistory.from_rows([])


def test_exact_shares_and_fallbacks() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(records()), START)
    ab, cd = shares.index("A", "B"), shares.index("C", "D")
    assert ab is not None
    assert cd is not None
    assert shares.shares[ab, 0] == pytest.approx([120 / 400, 280 / 400])
    assert shares.level[ab, 0] == 0
    assert shares.days[ab, 0] == 4
    assert shares.spread[ab, 0] == pytest.approx([np.sqrt(0.02)] * 2)
    assert shares.shares[ab, 1] == pytest.approx([130 / 420, 290 / 420])
    assert shares.level[ab, 1] == 1
    assert np.isnan(shares.spread[ab, 1]).all()
    assert shares.overall == pytest.approx([170 / 460, 290 / 460])
    assert shares.shares[cd, 2] == pytest.approx(shares.overall)
    assert shares.level[cd].tolist() == [2] * 7
    assert np.allclose(shares.shares.sum(axis=2), 1.0)
    assert shares.share("X", "Y", START) == pytest.approx(shares.overall)

    stability = shares.stability()
    assert (stability.cells, stability.weekday, stability.line, stability.overall) == (3, 1, 1, 1)
    assert stability.spread_mean == pytest.approx(np.sqrt(0.02))
    assert "3 line-weekdays" in stability.describe()


def test_weeks_and_unusual_days() -> None:
    history = WaveHistory.from_rows(records())
    two = WaveSplitter(weeks=2).fit(history, START)
    assert two.shares[0, 0] == pytest.approx([80 / 200, 120 / 200])
    calendar = Calendar.of(unusual=[day(2)])
    skipped = WaveSplitter().fit(history, START, calendar)
    assert skipped.shares[0, 0] == pytest.approx([210 / 400, 190 / 400])
    with pytest.raises(ValueError, match="at least 1"):
        WaveSplitter(weeks=0)


def test_shares_do_not_look_ahead() -> None:
    before = WaveSplitter().fit(WaveHistory.from_rows(records()), START)
    later: list[Record] = [
        ("A", "B", START, NINE, 1000.0),
        ("A", "B", START + dt.timedelta(days=1), FIVE, 500.0),
        ("C", "D", START + dt.timedelta(days=9), FIVE, 77.0),
        ("E", "F", START + dt.timedelta(days=3), NINE, 5.0),
    ]
    after = WaveSplitter().fit(WaveHistory.from_rows(records() + later), START)
    for line in range(len(before.origins)):
        moved = after.index(before.origins[line], before.destinations[line])
        assert moved is not None
        for name in ("shares", "level", "days", "spread"):
            assert np.array_equal(
                getattr(before, name)[line], getattr(after, name)[moved], equal_nan=True
            )
    assert np.array_equal(before.overall, after.overall)
    with pytest.raises(ValueError, match="no usable record"):
        WaveSplitter().fit(WaveHistory.from_rows([("A", "B", MONDAY, NINE, 0.0), *later]), START)
    with pytest.raises(ValueError, match="must come after"):
        WaveSplitter().fit(WaveHistory.from_rows(records()), MONDAY)


def test_split_preserves_daily_totals() -> None:
    waves, calendar = synthetic.wave_history(lines=10, days=84, seed=4)
    history = waves.daily()
    start = history.date(history.length)
    shares = WaveSplitter().fit(waves, start, calendar)
    daily = SeasonalMean().forecast(history, start, 7, calendar)
    split = shares.split(daily)
    assert split.values.shape == (10, 7, 2)
    assert np.allclose(split.values.sum(axis=2), daily.values)
    assert np.allclose(split.daily().values, daily.values)
    rows = list(split.rows(0.5))
    kept = sum(quantity for *_, quantity in rows)
    pieces = split.values.size
    assert daily.total() - 0.5 * pieces <= kept <= daily.total() + 1e-6
    assert all(quantity >= 0.5 for *_, quantity in rows)
    stability = shares.stability()
    assert stability.weekday > 0
    assert 0 < stability.spread_median < 0.2
    with pytest.raises(ValueError, match="earlier"):
        shares.split(SeasonalMean().forecast(history, start - dt.timedelta(days=1), 7, calendar))


def test_split_drops_small_pieces() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(records()), START)
    small = forecast([("A", "B")], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], START)
    rows = list(shares.split(small).rows(0.5))
    assert rows == [("A", "B", START, FIVE, pytest.approx(0.7))]
    assert demand_rows(small, minimum=0.5, waves=shares) == [
        {
            "origin": "A",
            "destination": "B",
            "date": START.isoformat(),
            "ready": "17:00",
            "quantity": {"desi": 0.7},
        }
    ]


def test_planning_document_with_ready_times() -> None:
    shares = WaveSplitter().fit(WaveHistory.from_rows(records()), START)
    week = forecast(
        [("A", "B"), ("C", "D"), ("A", "D")],
        [[100.0] * 7, [40.0] * 7, [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        START,
    )
    network = {
        "schema": "pirilib/linehaul@0.2",
        "centers": [
            {"id": name, "lat": 39.0 + i * 0.1, "lon": 30.0 + i * 0.2}
            for i, name in enumerate("ABCD")
        ],
        "matrix": {"source": "haversine"},
        "vehicle_types": [{"name": "tir", "capacity": {"desi": 22000}, "speed_kmh": 70}],
        "fleet": {"spot": [{"type": "tir", "cost": {"fixed": 1000, "per_km": 2}}]},
        "rules": {"service": {"due_hours": 24}},
    }
    daily = parse_spec(planning_document(network, week))
    assert all(row.ready is None for row in daily.demand)
    assert len(daily.demand) == 15
    spec = parse_spec(planning_document(network, week, waves=shares))
    assert len(spec.demand) == 30
    assert {row.ready for row in spec.demand} == {NINE, FIVE}
    total = sum(row.quantity["desi"] for row in spec.demand)
    assert total == pytest.approx(week.total(), abs=0.0005 * 30)
    monday = [row for row in spec.demand if row.origin == "A" and row.destination == "B"][:2]
    assert [(row.date, row.ready, row.quantity["desi"]) for row in monday] == [
        (START, NINE, pytest.approx(30.0)),
        (START, FIVE, pytest.approx(70.0)),
    ]


def test_load_wave_history_from_a_turkish_spreadsheet(tmp_path: Path) -> None:
    path = tmp_path / "waves.csv"
    path.write_text(
        "cikis;varis;tarih;saat;toplam_desi\n"
        "A;B;05.01.2026;9:00;1.234,5\n"
        "A;B;05.01.2026;17:00;100\n"
        "A;B;06.01.2026;09:00;10\n"
        "A;B;06.01.2026;9;5\n"
        "B;A;06.01.2026;17;7,5\n",
        encoding="utf-8",
    )
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
    assert np.isnan(history.values[1, 0]).all()

    path.write_text(
        "origin,destination,date,wave,desi\nA,B,2026-01-05,25:00,1\nA,B,2026-01-05,,1\n",
        encoding="utf-8",
    )
    with pytest.raises(SpecError) as error:
        load_wave_history(path)
    assert "waves.csv:2: 'wave': '25:00' is not a time of day" in str(error.value)
    assert "waves.csv:3: 'wave' is empty" in str(error.value)


def test_synthetic_wave_history_round_trips(tmp_path: Path) -> None:
    waves, _ = synthetic.wave_history(lines=6, days=28, seed=2)
    daily, _ = synthetic.history(lines=6, days=28, seed=2)
    assert np.allclose(waves.daily().values, daily.values, atol=0.11, equal_nan=True)
    path = synthetic.write_wave_history(waves, tmp_path / "waves.csv")
    loaded = load_wave_history(path)
    assert loaded.waves == waves.waves
    present = ~np.isnan(loaded.values)
    observed = np.nan_to_num(waves.values[:, : loaded.length], nan=0.0)
    assert np.allclose(loaded.values[present], observed[present])
