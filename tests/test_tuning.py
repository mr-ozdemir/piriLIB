from collections import Counter
from operator import attrgetter
from typing import Any

import numpy as np
import pytest

from pirilib import ACOConfig
from pirilib import objective as obj
from pirilib.spec import LineHaulInstance, parse_spec
from pirilib.synthetic import generate
from pirilib.tuning import (
    BeeColony,
    Choice,
    Float,
    Int,
    RandomSearch,
    SearchSpace,
    aco_space,
    default_sources,
    tune,
)
from pirilib.tuning.bee_colony import latin_hypercube

TARGET = np.array([0.3, 0.7, 0.2])
"""Minimum of :func:`sphere`."""
GOAL = obj.total_cost() + 1e6 * obj.violations()
BUDGET = 37
"""An odd evaluation budget, so that it is not a multiple of the colony size."""
SEEDS = range(10)
"""Tuner seeds compared when one tuner must beat another."""


def sphere(point: np.ndarray) -> float:
    return float(np.sum((point - TARGET[: point.size]) ** 2))


def shifted_sphere(point: Any) -> float:
    """A sphere whose minimum (0.3, 0.3, 0.3) is narrow at small budgets."""
    shifted = 10 * (point - 0.3)
    return float(np.sum(shifted * shifted))


def best_values(tuner: Any, objective: Any, budget: int, seeds: range) -> list[float]:
    """The best value the tuner finds with each seed; input for comparisons only."""
    return [tuner.minimize(objective, 3, budget, seed=seed).best_value for seed in seeds]


ACO_VALUES = {
    "ants": 16,
    "alpha": 1.0,
    "beta": 2.0,
    "evaporation": 0.1,
    "exploitation": 0.0,
    "tau_min": 0.01,
    "update": "best_so_far",
}


def test_aco_space_decodes_what_it_encodes() -> None:
    space = aco_space()

    decoded = space.decode(space.encode(ACO_VALUES))

    assert decoded == pytest.approx(ACO_VALUES)


def test_a_log_float_decodes_the_middle_to_the_geometric_middle() -> None:
    assert Float(1e-3, 1.0, log=True).decode(0.5) == pytest.approx(10**-1.5)


@pytest.mark.parametrize(("unit", "value"), [(0.0, 1), (0.34, 2), (0.67, 3), (1.0, 3)])
def test_an_int_decodes_equal_slices_of_the_unit_interval(unit: float, value: int) -> None:
    assert Int(1, 3).decode(unit) == value


def test_a_choice_decodes_the_lower_half_to_the_first_option() -> None:
    assert Choice(("a", "b")).decode(0.49) == "a"


def test_a_search_space_clips_points_outside_the_unit_cube() -> None:
    assert SearchSpace({"x": Float(0, 1)}).decode([7.0]) == {"x": 1.0}


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Float(1.0, 1.0), "low < high"),
        (lambda: Float(0.0, 1.0, log=True), "positive"),
        (lambda: Choice(("only",)), "two options"),
    ],
    ids=["empty-range", "log-of-zero", "one-option"],
)
def test_invalid_parameters_are_rejected(build: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("tuner", [BeeColony(), RandomSearch()], ids=["colony", "random"])
def test_a_tuner_spends_exactly_its_budget(tuner: Any) -> None:
    result = tuner.minimize(sphere, 3, BUDGET, seed=1)

    assert (result.evaluations, len(result.trace)) == (BUDGET, BUDGET)


@pytest.mark.parametrize("tuner", [BeeColony(), RandomSearch()], ids=["colony", "random"])
def test_a_tuner_evaluates_the_start_point_first(tuner: Any) -> None:
    result = tuner.minimize(sphere, 3, BUDGET, seed=1, start=[TARGET])

    assert result.trace[0].point == pytest.approx(tuple(TARGET))
    assert result.best_value == 0.0
    assert result.best_so_far()[-1] == 0.0


@pytest.mark.parametrize("tuner", [BeeColony(), RandomSearch()], ids=["colony", "random"])
def test_a_tuner_repeats_its_trace_with_the_same_seed(tuner: Any) -> None:
    first = tuner.minimize(sphere, 3, BUDGET, seed=1)

    again = tuner.minimize(sphere, 3, BUDGET, seed=1)

    assert again.trace == first.trace


def test_bee_colony_beats_random_search_on_a_sphere() -> None:
    colony = best_values(BeeColony(), sphere, 150, SEEDS)
    random = best_values(RandomSearch(), sphere, 150, SEEDS)

    assert np.median(colony) < np.median(random) / 3


def test_scouts_leave_exhausted_sources() -> None:
    """On a flat objective every source is exhausted; scouts keep bringing new points."""
    result = BeeColony(sources=2, limit=2).minimize(lambda point: 1.0, 2, 40, seed=0)

    visits = Counter(map(attrgetter("point"), result.trace))

    assert list(visits.values()).count(1) > 2 + 20


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

    result = tune([instance], GOAL, budget=4, base=base, min_improvement=10.0)

    assert result.config == base
    assert result.score == 0.0
    assert result.search.evaluations == 4
    assert result.search.trace[0].value == 0.0


def test_tune_returns_a_config_at_least_as_good_as_the_base(instance: LineHaulInstance) -> None:
    result = tune([instance], GOAL, budget=3, base=ACOConfig(ants=4, iterations=3, patience=2))

    assert isinstance(result.config, ACOConfig)
    assert result.score <= 0.0


def test_tune_needs_a_training_instance() -> None:
    with pytest.raises(ValueError, match="at least one training instance"):
        tune([], GOAL, budget=3)


def test_tune_needs_an_objective_argument(instance: LineHaulInstance) -> None:
    with pytest.raises(TypeError):
        tune([instance], budget=3)


def test_tune_rejects_a_missing_objective(instance: LineHaulInstance) -> None:
    with pytest.raises(ValueError, match="tune needs an objective"):
        tune([instance], None, budget=3)


def test_latin_hypercube_points_lie_in_the_unit_cube() -> None:
    points = latin_hypercube(np.random.default_rng(0), 5, 3)

    assert points.shape == (5, 3)
    np.testing.assert_array_equal((points >= 0) & (points < 1), True)


@pytest.mark.parametrize("axis", range(3))
def test_latin_hypercube_fills_every_slice_of_an_axis(axis: int) -> None:
    points = latin_hypercube(np.random.default_rng(0), 5, 3)

    slices = np.sort((points[:, axis] * 5).astype(int))

    np.testing.assert_array_equal(slices, [0, 1, 2, 3, 4])


def test_default_colony_beats_random_search_at_small_budgets() -> None:
    colony = best_values(BeeColony(), shifted_sphere, 24, range(20))
    random = best_values(RandomSearch(), shifted_sphere, 24, range(20))

    assert np.mean(colony) < np.mean(random)


@pytest.mark.parametrize(
    ("budget", "sources"),
    [(1, 2), (23, 2), (24, 3), (40, 3), (60, 3), (119, 3), (120, 4), (500, 4)],
)
def test_default_sources_follow_the_budget(budget: int, sources: int) -> None:
    assert default_sources(budget) == sources


@pytest.mark.parametrize(("budget", "first"), [(1, 1), (24, 3), (120, 4)])
def test_the_colony_starts_from_a_latin_hypercube_of_its_sources(budget: int, first: int) -> None:
    expected = latin_hypercube(np.random.default_rng(2), default_sources(budget), 3)[:first]

    trace = BeeColony().minimize(sphere, 3, budget, seed=2).trace[:first]

    np.testing.assert_array_equal(np.array(list(map(attrgetter("point"), trace))), expected)


SMALL = ACOConfig(ants=2, iterations=2, patience=1, local_search=False)
"""A colony small enough that tuning on a synthetic instance takes a fraction of a second."""
TUNER_SEEDS = range(4)
"""Seeds of the validation tests."""


def synthetic(seed: int) -> LineHaulInstance:
    return parse_spec(generate(centers=8, days=1, demand_per_day=30, seed=seed)).compile()


def test_without_validation_there_are_no_candidates() -> None:
    plain = tune([synthetic(1)], GOAL, budget=8, base=SMALL)

    assert (plain.validation, plain.candidates) == (None, ())


def test_validation_on_the_training_instances_keeps_the_training_choice() -> None:
    training = synthetic(1)
    plain = tune([training], GOAL, budget=8, base=SMALL)

    checked = tune([training], GOAL, budget=8, base=SMALL, validation=[training])

    assert len(checked.candidates) in {1, 2, 3}
    assert checked.config == plain.config == checked.candidates[0].config
    assert checked.score == plain.score


def test_on_the_training_instances_a_candidate_validates_to_its_score() -> None:
    training = synthetic(1)

    checked = tune([training], GOAL, budget=8, base=SMALL, validation=[training])

    scores = list(map(attrgetter("score"), checked.candidates))
    assert list(map(attrgetter("validation"), checked.candidates)) == pytest.approx(scores)


@pytest.mark.parametrize("seed", TUNER_SEEDS)
def test_candidates_are_distinct_and_ordered_by_training_score(seed: int) -> None:
    result = tune([synthetic(1)], GOAL, budget=8, base=SMALL, seed=seed, validation=[synthetic(2)])

    scores = list(map(attrgetter("score"), result.candidates))
    configs = list(map(attrgetter("config"), result.candidates))
    assert scores == sorted(scores)
    assert len(set(configs)) == len(configs)


@pytest.mark.parametrize("seed", TUNER_SEEDS)
def test_the_first_candidate_is_the_training_choice(seed: int) -> None:
    training, held_out = synthetic(1), synthetic(2)
    plain = tune([training], GOAL, budget=8, base=SMALL, seed=seed)

    result = tune([training], GOAL, budget=8, base=SMALL, seed=seed, validation=[held_out])

    assert result.candidates[0].config == plain.config


@pytest.mark.parametrize("seed", TUNER_SEEDS)
def test_validation_chooses_the_best_candidate_on_new_instances(seed: int) -> None:
    result = tune([synthetic(1)], GOAL, budget=8, base=SMALL, seed=seed, validation=[synthetic(2)])

    best = min(result.candidates, key=attrgetter("validation"))
    assert (result.config, result.score, result.validation) == (
        best.config,
        best.score,
        best.validation,
    )


@pytest.mark.parametrize(("seed", "changed"), [(0, True), (1, True), (2, True), (3, False)])
def test_validation_changes_the_training_choice_for_these_seeds(seed: int, changed: bool) -> None:
    """Pinned: on the held-out instance validation picks another setting for seeds 0 to 2."""
    training, held_out = synthetic(1), synthetic(2)
    plain = tune([training], GOAL, budget=8, base=SMALL, seed=seed)

    result = tune([training], GOAL, budget=8, base=SMALL, seed=seed, validation=[held_out])

    assert (result.config != plain.config) is changed


def test_validation_keeps_the_base_unless_it_is_beaten_there() -> None:
    kept = tune(
        [synthetic(1)], GOAL, budget=4, base=SMALL, validation=[synthetic(2)], min_improvement=10.0
    )

    assert kept.config == SMALL
    assert (kept.score, kept.validation, kept.candidates) == (0.0, 0.0, ())


def test_validation_scores_as_many_candidates_as_asked() -> None:
    one = tune([synthetic(1)], GOAL, budget=8, base=SMALL, validation=[synthetic(2)], candidates=1)

    assert len(one.candidates) == 1


def test_validation_needs_at_least_one_candidate() -> None:
    with pytest.raises(ValueError, match="candidates must be at least 1"):
        tune([synthetic(1)], GOAL, budget=2, validation=[synthetic(2)], candidates=0)
