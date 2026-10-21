"""pirilib: demand forecasting and ACO-ABC based distribution optimization.

Importing the package has no side effects beyond attaching a
``logging.NullHandler`` to the ``"pirilib"`` logger. Configuring log output is
left to the application.
"""

import logging

from pirilib._version import __version__
from pirilib.errors import PirilibError, PlanError, SpecError
from pirilib.evaluation import Evaluation, evaluate
from pirilib.plan import Dispatch, Load, Plan
from pirilib.solvers import ACOConfig, ACOSolver, SolveResult, solve
from pirilib.spec import load_spec
from pirilib.validate import check_plan

_logger = logging.getLogger(__name__)
if not any(isinstance(handler, logging.NullHandler) for handler in _logger.handlers):
    _logger.addHandler(logging.NullHandler())

__all__ = [
    "ACOConfig",
    "ACOSolver",
    "Dispatch",
    "Evaluation",
    "Load",
    "PirilibError",
    "Plan",
    "PlanError",
    "SolveResult",
    "SpecError",
    "__version__",
    "check_plan",
    "evaluate",
    "load_spec",
    "solve",
]
