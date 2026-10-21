"""Tune the ant colony's settings on training instances.

The tuning objective of a setting is the mean relative difference between its
solver objective and that of the base setting, over every training instance and
seed (0 means as good as the base, -0.02 means 2 % better). Seeds are fixed, so
two settings are compared on the same random numbers. Tune on instances that
look like the ones you will plan, and check the result on others.
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
from pirilib.solvers.base import default_objective
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
class TuningResult:
    """The best setting found, its tuning score and the search history.

    Attributes:
        score: Mean relative difference to the base setting; lower is better
    """

    config: ACOConfig
    score: float
    seconds: float
    search: SearchResult = field(repr=False)


def tune(
    instances: Sequence[LineHaulInstance],
    objective: Objective | None = None,
    constraints: Sequence[Constraint] = (),
    *,
    budget: int = 40,
    tuner: Tuner | None = None,
    space: SearchSpace | None = None,
    base: ACOConfig | None = None,
    seeds: Sequence[int] = (0,),
    seed: int = 0,
    min_improvement: float = 1e-3,
) -> TuningResult:
    """Search ``space`` for the :class:`~pirilib.ACOConfig` that minimizes ``objective``.

    ``budget`` counts settings tried; each one is solved on every instance with
    every seed. The base setting is the first one tried. The best setting found is
    returned only if it beats the base by at least ``min_improvement`` (relative);
    otherwise the base is kept, so that noise is not mistaken for progress.
    """
    if not instances:
        raise ValueError("tune needs at least one training instance")
    goal = default_objective() if objective is None else objective
    chosen_space = aco_space() if space is None else space
    start_config = ACOConfig() if base is None else base
    started = time.perf_counter()

    def objectives(config: ACOConfig) -> NDArray[np.float64]:
        solver = ACOSolver(config)
        return np.array(
            [
                solver.solve(instance, goal, constraints, seed=run).objective
                for instance in instances
                for run in seeds
            ]
        )

    reference = objectives(start_config)
    scale = np.maximum(np.abs(reference), 1e-12)

    def score(point: NDArray[np.float64]) -> float:
        config = replace(start_config, **chosen_space.decode(point))
        return float(np.mean((objectives(config) - reference) / scale))

    first = chosen_space.encode({name: getattr(start_config, name) for name in chosen_space.names})
    search = (BeeColony() if tuner is None else tuner).minimize(
        score, chosen_space.size, budget, seed=seed, start=[first]
    )
    better = search.best_value <= -min_improvement
    best = (
        replace(start_config, **chosen_space.decode(search.best_point)) if better else start_config
    )
    return TuningResult(
        config=best,
        score=search.best_value if better else 0.0,
        seconds=time.perf_counter() - started,
        search=search,
    )
