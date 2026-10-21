"""Adapters to external vehicle routing solvers, as baselines for benchmarks.

:class:`PyVRPSolver` (``pip install "pirilib[pyvrp]"``) and
:class:`ORToolsSolver` (``pip install "pirilib[ortools]"``) follow the
:class:`~pirilib.solvers.Solver` protocol. Their packages are imported only
when they solve. They handle a restricted problem: demand that may be split
(over full direct trips only), no minimum fill, no times and a linear
objective; they raise :class:`ValueError` for anything else. See
:mod:`pirilib.solvers.adapters.base` for how a subproblem is reduced.
"""

from pirilib.solvers.adapters.base import RoutingAdapter, RoutingModel, RoutingSolution
from pirilib.solvers.adapters.ortools_solver import ORToolsSolver
from pirilib.solvers.adapters.pyvrp_solver import PyVRPSolver

__all__ = ["ORToolsSolver", "PyVRPSolver", "RoutingAdapter", "RoutingModel", "RoutingSolution"]
