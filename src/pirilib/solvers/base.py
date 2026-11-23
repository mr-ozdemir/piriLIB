"""What every solver accepts and returns."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pirilib.constraints import Constraint
from pirilib.evaluation import Evaluation
from pirilib.objective import FEATURES, Objective
from pirilib.plan import SPOT, Plan
from pirilib.spec.instance import UNLIMITED, LineHaulInstance


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


def spot_limits_reached(instance: LineHaulInstance, plan: Plan) -> int:
    """How many (day, spot offer) pairs with a daily limit used every vehicle allowed.

    The origins of a day take the vehicles of a limited offer one after another.
    A count above zero means that this greedy sharing may have bound the plan:
    another split of the vehicles among the origins could be better.
    """
    used = np.zeros((len(instance.days), instance.spot_available.size), dtype=np.int64)
    spot = plan.kind == SPOT
    np.add.at(used, (plan.day[spot], plan.source[spot]), 1)
    limited = instance.spot_available != UNLIMITED
    return int(np.count_nonzero(limited[None, :] & (used >= instance.spot_available[None, :])))


def limit_stats(instance: LineHaulInstance, plan: Plan) -> dict[str, float | int | str]:
    """``spot_limits_reached`` for the stats of a result; empty when no offer has a limit."""
    if not np.any(instance.spot_available != UNLIMITED):
        return {}
    return {"spot_limits_reached": spot_limits_reached(instance, plan)}


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
