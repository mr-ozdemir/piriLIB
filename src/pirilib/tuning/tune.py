"""Tune the ant colony's settings on training instances.

The tuning objective of a setting is the mean relative difference between its
solver objective and that of the base setting, over every training instance and
seed (0 means as good as the base, -0.02 means 2 % better). Seeds are fixed, so
two settings are compared on the same random numbers. Tune on instances that
look like the ones you will plan, and check the result on others.

With validation instances, the best few settings of the search are scored on
them too, against the base setting, and the one that is best there is returned.
A setting that is better only on the training instances is then not chosen.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import Constraint
from pirilib.objective import Objective
from pirilib.solvers.aco import ACOConfig, ACOSolver
from pirilib.solvers.base import require_objective
from pirilib.spec.instance import LineHaulInstance
from pirilib.tuning.bee_colony import BeeColony, SearchResult, UnitObjective
from pirilib.tuning.space import SearchSpace, aco_space


class Tuner(Protocol):
    """A minimizer over the unit cube with a budget of objective evaluations."""

    def minimize(
        self,
        objective: UnitObjective,
        dimensions: int,
        budget: int,
        *,
        seed: int = 0,
        start: Sequence[NDArray[np.float64]] = (),
    ) -> SearchResult:
        """Minimize ``objective`` over the unit cube with at most ``budget`` evaluations."""
        ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """A setting that was also scored on the validation instances.

    Attributes:
        score: Mean relative difference to the base setting on the training instances
        validation: The same on the validation instances
    """

    config: ACOConfig
    score: float
    validation: float


@dataclass(frozen=True, slots=True)
class TuningResult:
    """The chosen setting, its tuning score and the search history.

    Attributes:
        score: Mean relative difference to the base setting on the training
            instances; lower is better. 0 when the base is kept.
        validation: The same on the validation instances; ``None`` without
            validation instances, 0 when the base is kept.
        candidates: The settings scored on the validation instances, best on the
            training instances first.
    """

    config: ACOConfig
    score: float
    seconds: float
    search: SearchResult = field(repr=False)
    validation: float | None = None
    candidates: tuple[Candidate, ...] = ()


def tune(
    instances: Sequence[LineHaulInstance],
    objective: Objective,
    constraints: Sequence[Constraint] = (),
    *,
    budget: int = 40,
    tuner: Tuner | None = None,
    space: SearchSpace | None = None,
    base: ACOConfig | None = None,
    seeds: Sequence[int] = (0,),
    seed: int = 0,
    min_improvement: float = 1e-3,
    validation: Sequence[LineHaulInstance] = (),
    candidates: int = 3,
) -> TuningResult:
    """Search ``space`` for the :class:`~pirilib.ACOConfig` that minimizes ``objective``.

    ``budget`` counts settings tried; each one is solved on every instance with
    every seed. The base setting is the first one tried. The best setting found is
    returned only if it beats the base by at least ``min_improvement`` (relative);
    otherwise the base is kept, so that noise is not mistaken for progress.

    With ``validation`` instances, up to ``candidates`` distinct settings that beat
    the base on the training instances are also solved on the validation
    instances with the same seeds. The one with the lowest validation score is
    returned if it beats the base there by ``min_improvement``; otherwise the base
    is kept. These solves do not count towards ``budget``.
    """
    if not instances:
        raise ValueError("tune needs at least one training instance")
    if candidates < 1:
        raise ValueError("candidates must be at least 1")
    goal = require_objective(objective, "tune")
    chosen_space = aco_space() if space is None else space
    start_config = ACOConfig() if base is None else base
    started = time.perf_counter()
    training = _Scorer(instances, goal, constraints, seeds, start_config)

    def config_of(point: NDArray[np.float64]) -> ACOConfig:
        return replace(start_config, **chosen_space.decode(point))

    def score(point: NDArray[np.float64]) -> float:
        return training.score(config_of(point))

    first = chosen_space.encode({name: getattr(start_config, name) for name in chosen_space.names})
    search = (BeeColony() if tuner is None else tuner).minimize(
        score, chosen_space.size, budget, seed=seed, start=[first]
    )
    if not validation:
        better = search.best_value <= -min_improvement
        return TuningResult(
            config=config_of(search.best_point) if better else start_config,
            score=search.best_value if better else 0.0,
            seconds=time.perf_counter() - started,
            search=search,
        )

    ranked: list[tuple[ACOConfig, float]] = []
    for item in sorted(search.trace, key=lambda trace: trace.value):
        if len(ranked) == candidates or item.value > -min_improvement:
            break
        config = config_of(np.array(item.point))
        if config != start_config and all(config != known for known, _ in ranked):
            ranked.append((config, item.value))
    checked: tuple[Candidate, ...] = ()
    if ranked:
        held_out = _Scorer(validation, goal, constraints, seeds, start_config)
        checked = tuple(
            Candidate(config, value, held_out.score(config)) for config, value in ranked
        )
    best = min(checked, key=lambda candidate: candidate.validation, default=None)
    if best is None or best.validation > -min_improvement:
        return TuningResult(
            config=start_config,
            score=0.0,
            seconds=time.perf_counter() - started,
            search=search,
            validation=0.0,
            candidates=checked,
        )
    return TuningResult(
        config=best.config,
        score=best.score,
        seconds=time.perf_counter() - started,
        search=search,
        validation=best.validation,
        candidates=checked,
    )


class _Scorer:
    """Scores settings by their mean relative difference to a base setting."""

    def __init__(
        self,
        instances: Sequence[LineHaulInstance],
        objective: Objective,
        constraints: Sequence[Constraint],
        seeds: Sequence[int],
        base: ACOConfig,
    ) -> None:
        self.instances, self.objective = instances, objective
        self.constraints, self.seeds = constraints, seeds
        self.reference = self.objectives(base)
        self.scale = np.maximum(np.abs(self.reference), 1e-12)

    def objectives(self, config: ACOConfig) -> NDArray[np.float64]:
        solver = ACOSolver(config)
        return np.array(
            [
                solver.solve(instance, self.objective, self.constraints, seed=run).objective
                for instance in self.instances
                for run in self.seeds
            ]
        )

    def score(self, config: ACOConfig) -> float:
        return float(np.mean((self.objectives(config) - self.reference) / self.scale))
