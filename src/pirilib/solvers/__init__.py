"""Solvers: from an instance and an objective to a plan.

:class:`ACOSolver` is the ant colony. Every solver follows the :class:`Solver`
protocol, so they can be swapped.
"""

from collections.abc import Sequence

from pirilib.constraints import Constraint
from pirilib.objective import Objective
from pirilib.solvers.aco import ACOConfig, ACOSolver
from pirilib.solvers.base import DEFAULT_PENALTY, Solver, SolveResult, default_objective
from pirilib.spec.instance import LineHaulInstance


def solve(
    instance: LineHaulInstance,
    objective: Objective | None = None,
    constraints: Sequence[Constraint] = (),
    *,
    solver: Solver | None = None,
    seed: int = 0,
) -> SolveResult:
    """Plan ``instance``; the solver defaults to :class:`ACOSolver` with default settings."""
    chosen = ACOSolver() if solver is None else solver
    return chosen.solve(instance, objective, constraints, seed=seed)


__all__ = [
    "DEFAULT_PENALTY",
    "ACOConfig",
    "ACOSolver",
    "SolveResult",
    "Solver",
    "default_objective",
    "solve",
]
