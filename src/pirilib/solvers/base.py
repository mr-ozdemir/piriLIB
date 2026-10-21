"""What every solver accepts and returns."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from pirilib import objective as obj
from pirilib.constraints import Constraint
from pirilib.evaluation import Evaluation
from pirilib.objective import FEATURES, Objective
from pirilib.plan import Plan
from pirilib.spec.instance import LineHaulInstance

DEFAULT_PENALTY: Final = 1e6
"""Weight of every built-in violation in :func:`default_objective`."""


def default_objective() -> obj.Sum:
    """Total cost, plus :data:`DEFAULT_PENALTY` per unit of violation of any rule."""
    return obj.total_cost() + DEFAULT_PENALTY * obj.violations()


def linear_coefficients(instance: LineHaulInstance, form: Mapping[str, float]) -> dict[str, float]:
    """The non-zero coefficients of a linear form, checked against the known features."""
    known = set(FEATURES) | {f"unserved[{name}]" for name in instance.dimensions}
    unknown = sorted(set(form) - known)
    if unknown:
        raise ValueError(f"the objective uses unknown features {unknown}")
    return {feature: value for feature, value in form.items() if value}


@dataclass(frozen=True, slots=True)
class SolveResult:
    """The best plan a solver found, its evaluation and objective value."""

    plan: Plan
    evaluation: Evaluation
    objective: float
    seconds: float
    stats: Mapping[str, float | int | str]

    def __repr__(self) -> str:
        return (
            f"SolveResult(objective={self.objective:.2f}, seconds={self.seconds:.2f}, "
            f"{self.evaluation!r})"
        )


class Solver(Protocol):
    """Anything that turns an instance and an objective into a plan."""

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        """Return the best plan found; ``objective`` defaults to :func:`default_objective`."""
        ...
