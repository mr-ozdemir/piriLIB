"""Which built-in solver keeps which rule.

A solver deals with a rule of the constraint catalog in one of three ways: its
plans never break the rule, its plans may break it and the objective charges
the penalty, or it refuses problems that use the rule. :data:`SUPPORT` records
the way for every built-in solver and every rule:

```python
from pirilib import ACOSolver
from pirilib.solvers.support import Support, rule_support

support = rule_support(ACOSolver())
print(support["capacity"])
print([rule for rule, way in support.items() if way is Support.PENALIZED])
```

The record is checked by the tests: every solver plans problems that put each
rule to work, and a rule it is said to keep must come out unbroken. A new rule,
or a solver that treats a rule differently, has to be entered here.
"""

from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pirilib.constraints import CATALOG
from pirilib.solvers.aco import ACOSolver
from pirilib.solvers.adapters.base import RoutingAdapter
from pirilib.solvers.base import Solver
from pirilib.solvers.direct import DirectShipping
from pirilib.solvers.exact import ExactSolver

__all__ = ["SUPPORT", "Support", "rule_support"]


class Support(StrEnum):
    """How a solver deals with a rule.

    Attributes:
        KEPT: No plan of the solver breaks the rule, whatever the objective
        PENALIZED: A plan may break the rule. The evaluation reports the
            amount and the objective charges the rule's penalty for it; whether
            the solver weighs that penalty while it plans depends on the solver
        REFUSED: The solver raises :class:`ValueError` for a problem that uses
            the setting behind the rule, such as a minimum fill above 0. On the
            problems it accepts, its plans do not break the rule
    """

    KEPT = "kept"
    PENALIZED = "penalized"
    REFUSED = "refused"


def _record(*, penalized: Iterable[str], refused: Iterable[str] = ()) -> Mapping[str, Support]:
    ways = dict.fromkeys(CATALOG, Support.KEPT)
    ways |= dict.fromkeys(penalized, Support.PENALIZED)
    ways |= dict.fromkeys(refused, Support.REFUSED)
    return MappingProxyType(ways)


SUPPORT: Final[Mapping[type, Mapping[str, Support]]] = MappingProxyType(
    {
        ACOSolver: _record(penalized=("handling", "vehicle_limit", "min_fill", "coverage")),
        ExactSolver: _record(
            penalized=("handling", "vehicle_limit", "min_fill", "coverage"), refused=("split",)
        ),
        DirectShipping: _record(penalized=("handling", "vehicle_limit", "min_fill", "coverage")),
        RoutingAdapter: _record(
            penalized=("coverage",),
            refused=("deferral", "handling", "vehicle_limit", "min_fill", "split"),
        ),
    }
)
"""By solver class, the way it deals with each rule of the catalog, in catalog order.

:class:`~pirilib.solvers.adapters.base.RoutingAdapter` stands for the PyVRP and
OR-Tools adapters. They also refuse problems with hourly costs or due times,
which are costs and not rules of the catalog.

- The ant colony and the exact solver weigh ``min_fill`` and ``coverage``
  against the costs: a vehicle leaves below its minimum fill, or a demand stays
  behind, when that is cheaper under the objective.
- Direct shipping does not look at the penalties. It leaves demand behind only
  when no vehicle is left for it.
- ``handling`` and ``vehicle_limit`` are shared by all dispatches at a center
  on a day. The ant colony and the exact solver give each subproblem the room
  that is left, which keeps the limits in most plans but not in all; direct
  shipping does not look at them.
"""


def rule_support(solver: Solver | type) -> Mapping[str, Support]:
    """The record of ``solver``, an instance or a class, by rule name.

    A subclass of a built-in solver has the record of that solver.
    :class:`ValueError` for any other solver: nothing is known about its plans.
    """
    kind = solver if isinstance(solver, type) else type(solver)
    for base in kind.__mro__:
        if base in SUPPORT:
            return SUPPORT[base]
    known = ", ".join(sorted(item.__name__ for item in SUPPORT))
    raise ValueError(f"no rule support is recorded for {kind.__name__}; recorded: {known}")
