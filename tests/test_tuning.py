from typing import Any

import numpy as np
import pytest

from pirilib import ACOConfig
from pirilib import objective as obj
from pirilib.spec import LineHaulInstance
from pirilib.tuning import BeeColony, Choice, Float, Int, RandomSearch, SearchSpace, aco_space, tune

TARGET = np.array([0.3, 0.7, 0.2])


def sphere(point: np.ndarray) -> float:
    return float(np.sum((point - TARGET[: point.size]) ** 2))


def test_space_round_trip() -> None:
    space = aco_space()
    values = {
        "ants": 16,
        "alpha": 1.0,
        "beta": 2.0,
        "evaporation": 0.1,
        "exploitation": 0.0,
        "tau_min": 0.01,
        "update": "best_so_far",
    }
    decoded = space.decode(space.encode(values))
    assert decoded.keys() == values.keys()
    for name, value in values.items():
        if isinstance(value, float):
            assert decoded[name] == pytest.approx(value), name
        else:
            assert decoded[name] == value, name


def test_parameters() -> None:
    assert Float(1e-3, 1.0, log=True).decode(0.5) == pytest.approx(10**-1.5)
    assert [Int(1, 3).decode(u) for u in (0.0, 0.34, 0.67, 1.0)] == [1, 2, 3, 3]
    assert Choice(("a", "b")).decode(0.49) == "a"
    assert SearchSpace({"x": Float(0, 1)}).decode([7.0]) == {"x": 1.0}
    with pytest.raises(ValueError, match="low < high"):
        Float(1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        Float(0.0, 1.0, log=True)
    with pytest.raises(ValueError, match="two options"):
        Choice(("only",))


@pytest.mark.parametrize("tuner", [BeeColony(), RandomSearch()])
def test_budget_and_start(tuner: Any) -> None:
    start = np.array([0.3, 0.7, 0.2])
    result = tuner.minimize(sphere, 3, 37, seed=1, start=[start])
    assert result.evaluations == len(result.trace) == 37
    assert result.trace[0].point == pytest.approx(tuple(start))
    assert result.best_value == 0.0
    assert result.best_so_far()[-1] == 0.0
    again = tuner.minimize(sphere, 3, 37, seed=1, start=[start])
    assert [t.value for t in again.trace] == [t.value for t in result.trace]


def test_bee_colony_beats_random_search_on_a_sphere() -> None:
    abc = [BeeColony().minimize(sphere, 3, 150, seed=s).best_value for s in range(10)]
    random = [RandomSearch().minimize(sphere, 3, 150, seed=s).best_value for s in range(10)]
    assert np.median(abc) < np.median(random) / 3


def test_scouts_leave_exhausted_sources() -> None:
    result = BeeColony(sources=2, limit=2).minimize(lambda point: 1.0, 2, 40, seed=0)
    starts = {point for point, count in _counts(result).items() if count == 1}
    assert len(starts) > 2 + 20


def _counts(result: Any) -> dict[tuple[float, ...], int]:
    counts: dict[tuple[float, ...], int] = {}
    for item in result.trace:
        counts[item.point] = counts.get(item.point, 0) + 1
    return counts


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"sources": 1}, "sources must be at least 2"),
        ({"modification_rate": 0.0}, "modification_rate"),
        ({"limit": 0}, "limit must be at least 1"),
    ],
)
def test_bee_colony_settings(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BeeColony(**settings)


def test_tune_keeps_the_base_unless_it_is_beaten(instance: LineHaulInstance) -> None:
    base = ACOConfig(ants=4, iterations=3, patience=2)
    goal = obj.total_cost() + 1e6 * obj.violations()
    result = tune([instance], goal, budget=4, base=base, min_improvement=10.0)
    assert result.config == base
    assert result.score == 0.0
    assert result.search.evaluations == 4
    assert result.search.trace[0].value == 0.0


def test_tune_returns_a_valid_config(instance: LineHaulInstance) -> None:
    result = tune([instance], budget=3, base=ACOConfig(ants=4, iterations=3, patience=2))
    assert isinstance(result.config, ACOConfig)
    assert result.score <= 0.0
    with pytest.raises(ValueError, match="at least one training instance"):
        tune([], budget=3)
