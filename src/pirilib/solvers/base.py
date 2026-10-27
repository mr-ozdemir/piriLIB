"""What every solver accepts and returns."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pirilib.constraints import Constraint
from pirilib.evaluation import Evaluation
from pirilib.objective import FEATURES, Objective
from pirilib.plan import Plan
from pirilib.spec.instance import LineHaulInstance


def require_objective(objective: Objective | None, user: str) -> Objective:
    """Return ``objective``; :class:`ValueError` if it is None, as there is no default one.

    ``user`` names the caller in the message, for example ``"ACOSolver.solve"``.
    """
    if objective is None:
        raise ValueError(
            f"{user} needs an objective: build one with pirilib.objective, or from the "
            "objective: section of the problem document (LineHaulSpec.build_objective)"
        )
    return objective


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
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        """Return the best plan found for ``objective``; there is no default objective."""
        ...
