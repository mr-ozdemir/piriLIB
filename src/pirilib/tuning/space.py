"""Search spaces: named parameters mapped to and from the unit cube.

Tuners search ``[0, 1] ** d``; a :class:`SearchSpace` turns a point of the cube
into parameter values and back. Floats may be searched on a log scale, integers
are rounded, and choices split the unit interval into equal bins.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Float:
    """A real-valued parameter between ``low`` and ``high``, optionally on a log scale."""

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"need low < high, got {self.low}, {self.high}")
        if self.log and self.low <= 0:
            raise ValueError("a log-scale range must be positive")

    def decode(self, unit: float) -> float:
        if self.log:
            return math.exp(math.log(self.low) + unit * (math.log(self.high) - math.log(self.low)))
        return self.low + unit * (self.high - self.low)

    def encode(self, value: Any) -> float:
        number = float(value)
        if self.log:
            return (math.log(number) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (number - self.low) / (self.high - self.low)


@dataclass(frozen=True, slots=True)
class Int:
    """An integer parameter between ``low`` and ``high``, both included."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"need low < high, got {self.low}, {self.high}")

    def decode(self, unit: float) -> int:
        return min(self.high, self.low + int(unit * (self.high - self.low + 1)))

    def encode(self, value: Any) -> float:
        return (int(value) - self.low + 0.5) / (self.high - self.low + 1)


@dataclass(frozen=True, slots=True)
class Choice:
    """A parameter that takes one of ``options``."""

    options: tuple[Any, ...]

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a choice needs at least two options")

    def decode(self, unit: float) -> Any:
        return self.options[min(len(self.options) - 1, int(unit * len(self.options)))]

    def encode(self, value: Any) -> float:
        return (self.options.index(value) + 0.5) / len(self.options)


type Parameter = Float | Int | Choice


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Named parameters; :meth:`decode` maps a point of the unit cube to their values."""

    parameters: Mapping[str, Parameter]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.parameters)

    @property
    def size(self) -> int:
        return len(self.parameters)

    def decode(self, point: Sequence[float] | NDArray[np.float64]) -> dict[str, Any]:
        clipped = np.clip(np.asarray(point, dtype=np.float64), 0.0, 1.0)
        return {
            name: parameter.decode(float(unit))
            for (name, parameter), unit in zip(self.parameters.items(), clipped, strict=True)
        }

    def encode(self, values: Mapping[str, Any]) -> NDArray[np.float64]:
        return np.array(
            [parameter.encode(values[name]) for name, parameter in self.parameters.items()]
        )


def aco_space() -> SearchSpace:
    """The default space of :class:`~pirilib.ACOConfig` parameters."""
    return SearchSpace(
        {
            "ants": Int(4, 40),
            "alpha": Float(0.0, 4.0),
            "beta": Float(0.0, 6.0),
            "evaporation": Float(0.01, 0.6, log=True),
            "exploitation": Float(0.0, 0.9),
            "tau_min": Float(1e-4, 0.2, log=True),
            "update": Choice(("iteration_best", "best_so_far")),
        }
    )
