"""Artificial bee colony (ABC) minimization over the unit cube.

Karaboga's algorithm, with the corrections the prototype needed:

- onlookers choose food sources by *rank*, not by ``1 / (1 + f)``, which made
  the choice almost uniform once objective values were large;
- a neighbour replaces a food source only if it is strictly better, so equal
  values do not reset the trial counter;
- a *modification rate* decides how many dimensions a neighbour changes
  (at least one), as in Akay and Karaboga (2012);
- the budget counts objective evaluations, so tuners can be compared fairly.

The search is minimization; every call of the objective counts towards the budget.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

type UnitObjective = Callable[[NDArray[np.float64]], float]


@dataclass(frozen=True, slots=True)
class Trace:
    """One objective evaluation of a search."""

    point: tuple[float, ...]
    value: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The best point a search found, its value and every evaluation made."""

    best_point: NDArray[np.float64]
    best_value: float
    evaluations: int
    trace: tuple[Trace, ...] = field(repr=False)

    def best_so_far(self) -> list[float]:
        """The best value after each evaluation."""
        best, values = np.inf, []
        for item in self.trace:
            best = min(best, item.value)
            values.append(best)
        return values


class _Budget:
    """Wraps the objective: counts evaluations and records them."""

    def __init__(self, objective: UnitObjective, budget: int) -> None:
        self.objective, self.budget = objective, budget
        self.trace: list[Trace] = []

    @property
    def left(self) -> int:
        return self.budget - len(self.trace)

    def __call__(self, point: NDArray[np.float64]) -> float:
        value = float(self.objective(point))
        self.trace.append(Trace(tuple(float(x) for x in point), value))
        return value

    def result(self) -> SearchResult:
        best = min(self.trace, key=lambda item: item.value)
        return SearchResult(np.array(best.point), best.value, len(self.trace), tuple(self.trace))


@dataclass(frozen=True, slots=True)
class BeeColony:
    """ABC settings: food sources, modification rate and abandonment limit.

    ``limit`` is the number of failed improvements before a source is abandoned;
    ``None`` means ``sources * dimensions``.
    """

    sources: int = 6
    modification_rate: float = 0.5
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.sources < 2:
            raise ValueError("sources must be at least 2")
        if not 0 < self.modification_rate <= 1:
            raise ValueError("modification_rate must be in (0, 1]")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1")

    def minimize(
        self,
        objective: UnitObjective,
        dimensions: int,
        budget: int,
        *,
        seed: int = 0,
        start: Sequence[NDArray[np.float64]] = (),
    ) -> SearchResult:
        """Minimize ``objective`` over ``[0, 1] ** dimensions`` with ``budget`` evaluations.

        Points in ``start`` (for example the current settings) are among the first
        food sources.
        """
        if budget < 1:
            raise ValueError("budget must be at least 1")
        rng = np.random.default_rng(seed)
        count = _Budget(objective, budget)
        limit = self.limit if self.limit is not None else self.sources * dimensions

        food = rng.random((self.sources, dimensions))
        for index, point in enumerate(list(start)[: self.sources]):
            food[index] = np.clip(point, 0.0, 1.0)
        values = np.full(self.sources, np.inf)
        for index in range(self.sources):
            if not count.left:
                return count.result()
            values[index] = count(food[index])
        trials = np.zeros(self.sources, dtype=np.int64)

        def try_neighbour(index: int) -> None:
            other = int(rng.choice([i for i in range(self.sources) if i != index]))
            change = rng.random(dimensions) < self.modification_rate
            change[rng.integers(dimensions)] = True
            phi = rng.uniform(-1.0, 1.0, dimensions)
            candidate = np.where(
                change, food[index] + phi * (food[index] - food[other]), food[index]
            )
            candidate = np.clip(candidate, 0.0, 1.0)
            value = count(candidate)
            if value < values[index]:
                food[index], values[index], trials[index] = candidate, value, 0
            else:
                trials[index] += 1

        while count.left:
            for index in range(self.sources):
                if not count.left:
                    break
                try_neighbour(index)
            ranks = np.empty(self.sources)
            ranks[np.argsort(values, kind="stable")] = np.arange(self.sources, 0, -1)
            chances = ranks / ranks.sum()
            for _ in range(self.sources):
                if not count.left:
                    break
                try_neighbour(int(rng.choice(self.sources, p=chances)))
            worst = int(np.argmax(trials))
            if trials[worst] > limit and count.left:
                food[worst] = rng.random(dimensions)
                values[worst] = count(food[worst])
                trials[worst] = 0
        return count.result()


@dataclass(frozen=True, slots=True)
class RandomSearch:
    """Uniform random points; the baseline every tuner should beat."""

    def minimize(
        self,
        objective: UnitObjective,
        dimensions: int,
        budget: int,
        *,
        seed: int = 0,
        start: Sequence[NDArray[np.float64]] = (),
    ) -> SearchResult:
        if budget < 1:
            raise ValueError("budget must be at least 1")
        rng = np.random.default_rng(seed)
        count = _Budget(objective, budget)
        for point in list(start)[:budget]:
            count(np.clip(point, 0.0, 1.0))
        while count.left:
            count(rng.random(dimensions))
        return count.result()
