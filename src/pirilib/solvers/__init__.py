"""Solvers: from an instance and an objective to a plan.

:class:`ACOSolver` is the ant colony. Every solver follows the :class:`Solver`
protocol, so they can be swapped.
"""

from collections.abc import Sequence

from pirilib.constraints import Constraint
from pirilib.objective import Objective
from pirilib.solvers.aco import ACOConfig, ACOSolver
from pirilib.solvers.base import Solver, SolveResult, require_objective
from pirilib.spec.instance import LineHaulInstance


def solve(
    instance: LineHaulInstance,
    objective: Objective,
    constraints: Sequence[Constraint] = (),
    *,
    solver: Solver | None = None,
    seed: int = 0,
) -> SolveResult:
    """Plan ``instance`` for ``objective``; the solver defaults to :class:`ACOSolver`.

    There is no default objective: build one with :mod:`pirilib.objective`, or
    from the problem document with
    :meth:`~pirilib.spec.LineHaulSpec.build_objective`.
    """
    goal = require_objective(objective, "solve")
    chosen = ACOSolver() if solver is None else solver
    return chosen.solve(instance, goal, constraints, seed=seed)


__all__ = [
    "ACOConfig",
    "ACOSolver",
    "SolveResult",
    "Solver",
    "solve",
]
