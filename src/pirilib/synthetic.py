"""Synthetic line-haul problems, for trying the library, tests and benchmarks.

Centers are spread over a rectangle of latitude and longitude, each with a
random size. Demand follows a gravity model: a pair of centers is chosen with
probability proportional to the product of their sizes, and quantities are
log-normal. Every figure is invented; nothing describes a real company.
"""

import copy
import csv
import datetime as dt
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from pirilib.spec.models import SCHEMA_ID
from pirilib.spec.tables import LIST_SEPARATOR

if TYPE_CHECKING:
    from pirilib.forecast.history import History
    from pirilib.forecast.prepare import Calendar

VEHICLE_TYPES: tuple[dict[str, Any], ...] = (
    {"name": "tir", "capacity": {"desi": 22000.0}},
    {"name": "kamyon", "capacity": {"desi": 12000.0}},
    {"name": "kamyonet", "capacity": {"desi": 5500.0}},
)
SPOT: tuple[dict[str, Any], ...] = (
    {"type": "tir", "cost": {"fixed": 12000.0, "per_km": 24.0}},
    {"type": "kamyon", "cost": {"fixed": 8000.0, "per_km": 20.0}},
    {"type": "kamyonet", "cost": {"fixed": 5000.0, "per_km": 17.0}},
)


def generate(
    *,
    centers: int = 12,
    days: int = 3,
    demand_per_day: int = 60,
    owned_lines: int = 2,
    mean_desi: float = 6000.0,
    min_fill: float = 0.1,
    max_stretch: float | None = 1.3,
    region: tuple[float, float, float, float] = (36.5, 41.5, 26.5, 44.0),
    start: dt.date = dt.date(2026, 1, 5),
    seed: int = 0,
) -> dict[str, Any]:
    """A problem document with inline tables; see the module documentation."""
    if centers < 2 or days < 1 or demand_per_day < 1:
        raise ValueError("need at least 2 centers, 1 day and 1 demand per day")
    rng = np.random.default_rng(seed)
    south, north, west, east = region
    names = [f"M{index + 1:03d}" for index in range(centers)]
    size = rng.lognormal(0.0, 0.8, centers)
    center_rows = [
        {
            "id": name,
            "lat": round(float(rng.uniform(south, north)), 4),
            "lon": round(float(rng.uniform(west, east)), 4),
        }
        for name in names
    ]
    for row in center_rows:
        if rng.random() < 0.1:
            row["banned_vehicle_types"] = ["tir"]

    weight = np.outer(size, size)
    np.fill_diagonal(weight, 0.0)
    pairs = np.flatnonzero(weight)
    chances = weight.ravel()[pairs] / weight.ravel()[pairs].sum()
    sigma = 0.9
    mu = np.log(mean_desi) - sigma**2 / 2
    demand = []
    for day in range(days):
        date = (start + dt.timedelta(days=day)).isoformat()
        for pair in rng.choice(pairs, size=demand_per_day, p=chances):
            origin, destination = divmod(int(pair), centers)
            demand.append(
                {
                    "origin": names[origin],
                    "destination": names[destination],
                    "date": date,
                    "quantity": {"desi": round(float(rng.lognormal(mu, sigma)), 1)},
                }
            )

    allowed = [row["id"] for row in center_rows if "banned_vehicle_types" not in row]
    owned = []
    for _ in range(min(owned_lines, len(allowed) // 2)):
        first, second = rng.choice(len(allowed), size=2, replace=False)
        origin, destination = allowed[first], allowed[second]
        owned.append(
            {
                "type": "tir",
                "origin": str(origin),
                "destination": str(destination),
                "count": int(rng.integers(1, 4)),
                "cost": {"fixed": 7500.0, "per_km": 12.0},
                "stopovers": True,
            }
        )

    rules: dict[str, Any] = {"min_fill": {"spot": min_fill}, "stopover": {}}
    if max_stretch is not None:
        rules["stopover"]["max_stretch"] = max_stretch
    return {
        "schema": SCHEMA_ID,
        "name": f"synthetic-{seed}",
        "calendar": {
            "start": start.isoformat(),
            "end": (start + dt.timedelta(days=days - 1)).isoformat(),
        },
        "centers": center_rows,
        "matrix": {"source": "haversine", "factor": 1.25},
        "vehicle_types": copy.deepcopy(list(VEHICLE_TYPES)),
        "fleet": {"owned": owned, "spot": copy.deepcopy(list(SPOT))},
        "demand": demand,
        "rules": rules,
    }


def write(document: dict[str, Any], folder: str | os.PathLike[str]) -> Path:
    """Write ``document`` as ``problem.yaml`` with ``centers.csv`` and ``demand.csv``."""
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    dimensions = list(document["vehicle_types"][0]["capacity"])
    with (target / "centers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "lat", "lon", "banned_vehicle_types"])
        for row in document["centers"]:
            banned = LIST_SEPARATOR.join(row.get("banned_vehicle_types", []))
            writer.writerow([row["id"], row["lat"], row["lon"], banned])
    with (target / "demand.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin", "destination", "date", *dimensions])
        for row in document["demand"]:
            quantities = [row["quantity"][name] for name in dimensions]
            writer.writerow([row["origin"], row["destination"], row["date"], *quantities])
    problem = {
        **document,
        "centers": {"file": "centers.csv"},
        "demand": {"file": "demand.csv", "quantity": {name: name for name in dimensions}},
    }
    path = target / "problem.yaml"
    path.write_text(yaml.safe_dump(problem, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def history(
    *,
    lines: int = 30,
    days: int = 168,
    start: dt.date = dt.date(2026, 1, 5),
    holidays: tuple[dt.date, ...] = (
        dt.date(2026, 3, 20),
        dt.date(2026, 4, 23),
        dt.date(2026, 5, 19),
    ),
    seed: int = 0,
) -> tuple["History", "Calendar"]:
    """A synthetic daily desi history with the patterns of real line-haul demand.

    Every line has its own level and weekly profile; some run only on weekdays.
    Demand grows slowly, drops on holidays and rebounds the day after, has
    occasional spikes, and a few records are missing. Returns the history and a
    calendar that marks the holidays (and the rebound days) unusual.
    """
    from pirilib.forecast.history import History
    from pirilib.forecast.prepare import Calendar

    rng = np.random.default_rng(seed)
    names = [f"M{index + 1:03d}" for index in range(max(4, int(np.sqrt(lines)) + 3))]
    pairs: set[tuple[str, str]] = set()
    while len(pairs) < lines:
        origin, destination = rng.choice(len(names), size=2, replace=False)
        pairs.add((names[origin], names[destination]))
    ordered = sorted(pairs)
    level = rng.lognormal(np.log(4000.0), 0.9, lines)
    profile = rng.uniform(0.6, 1.3, (lines, 7))
    profile[:, 6] *= 0.35
    weekdays_only = rng.random(lines) < 0.2
    growth = rng.normal(0.0008, 0.0006, lines)
    holiday_set = set(holidays)
    values = np.full((lines, days), np.nan)
    for day in range(days):
        date = start + dt.timedelta(days=day)
        dow = date.weekday()
        mean = level * profile[:, dow] * np.exp(growth * day)
        if date in holiday_set:
            mean = mean * 0.3
        elif date - dt.timedelta(days=1) in holiday_set:
            mean = mean * 1.5
        noise = rng.gamma(shape=12.0, scale=1 / 12.0, size=lines)
        spike = np.where(rng.random(lines) < 0.01, rng.uniform(2.0, 4.0, lines), 1.0)
        quantity = np.round(mean * noise * spike, 1)
        missing = rng.random(lines) < 0.03
        closed = weekdays_only & (dow >= 5)
        values[:, day] = np.where(missing | closed, np.nan, quantity)
    rebound = {day + dt.timedelta(days=1) for day in holiday_set}
    calendar = Calendar.of(unusual=holiday_set | rebound, holidays=holiday_set)
    result = History(
        origins=tuple(origin for origin, _ in ordered),
        destinations=tuple(destination for _, destination in ordered),
        start=start,
        values=values,
    )
    return result, calendar


def write_history(data: "History", path: str | os.PathLike[str]) -> Path:
    """Write a history as a CSV file with origin, destination, date and desi columns."""
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["origin", "destination", "date", data.dimension])
        for line in range(data.lines):
            for day in np.flatnonzero(~np.isnan(data.values[line])):
                writer.writerow(
                    [
                        data.origins[line],
                        data.destinations[line],
                        data.date(int(day)).isoformat(),
                        data.values[line, day],
                    ]
                )
    return target
