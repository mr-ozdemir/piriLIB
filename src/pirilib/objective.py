"""Building blocks for objective functions.

An objective is any callable that takes an :class:`~pirilib.evaluation.Evaluation`
and returns a number to minimize. The terms below can be weighted and added::

    from pirilib import objective as obj

    goal = obj.total_cost() + 1e6 * obj.violations() + 500 * obj.dispatches("spot")
    goal(evaluation)            # one number
    goal.breakdown(evaluation)  # the value of every term, for reports

A plain function works just as well::

    def goal(evaluation):
        return evaluation.total_cost + 2.0 * evaluation.unserved.sum()

Terms built here also know their *linear form*: their value as a weighted sum of
per-dispatch and per-demand features (see :data:`FEATURES`). Solvers use it to
optimize the problem piece by piece; objectives without a linear form, such as
plain functions, are optimized on whole plans, which is slower.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Union

import numpy as np
from numpy.typing import NDArray

from pirilib.constraints import CATALOG
from pirilib.plan import OWNED, SPOT, Kind

if TYPE_CHECKING:
    from pirilib.evaluation import Evaluation

type Objective = Callable[["Evaluation"], float]
"""Anything that maps an evaluation to a number to minimize."""

type LinearForm = Mapping[str, float]
"""Feature name -> coefficient."""

FEATURES: Final = (
    "cost[owned]",
    "cost[spot]",
    "km[owned]",
    "km[spot]",
    "dispatches[owned]",
    "dispatches[spot]",
    "unused[owned]",
    "unused[spot]",
    "unserved",
    "owned_fixed",
    "service",
    *(f"violation[{name}]" for name in CATALOG),
)
"""Features of linear forms, besides ``unserved[<dimension>]``.

``cost`` is the dispatch cost, ``unused`` the empty capacity of the primary
dimension, ``unserved`` the demand not carried (summed over dimensions) and
``owned_fixed`` the fixed cost of mandatory owned vehicles.
"""


@dataclass(frozen=True, slots=True)
class Term:
    """A named, weighted quantity read from an evaluation.

    Attributes:
        linear: Unweighted
    """

    name: str
    function: Callable[["Evaluation"], float]
    weight: float = 1.0
    linear: LinearForm | None = None

    def __call__(self, evaluation: "Evaluation") -> float:
        return self.weight * float(self.function(evaluation))

    def __mul__(self, factor: float) -> "Term":
        return replace(self, weight=self.weight * factor)

    __rmul__ = __mul__

    def __add__(self, other: Union["Term", "Sum"]) -> "Sum":
        return Sum((self,)) + other

    def breakdown(self, evaluation: "Evaluation") -> dict[str, float]:
        return {self.name: self(evaluation)}

    def linear_form(self) -> dict[str, float] | None:
        """The weighted linear form, or None if the term has none."""
        if self.linear is None:
            return None
        return {feature: self.weight * value for feature, value in self.linear.items()}


@dataclass(frozen=True, slots=True)
class Sum:
    """The sum of several terms."""

    terms: tuple[Term, ...]

    def __call__(self, evaluation: "Evaluation") -> float:
        return sum(term(evaluation) for term in self.terms)

    def __add__(self, other: Union[Term, "Sum"]) -> "Sum":
        if isinstance(other, Term):
            return Sum((*self.terms, other))
        if isinstance(other, Sum):
            return Sum((*self.terms, *other.terms))
        return NotImplemented

    def __mul__(self, factor: float) -> "Sum":
        return Sum(tuple(term * factor for term in self.terms))

    __rmul__ = __mul__

    def breakdown(self, evaluation: "Evaluation") -> dict[str, float]:
        """The weighted value of every term; terms with the same name are added."""
        values: dict[str, float] = {}
        for term in self.terms:
            values[term.name] = values.get(term.name, 0.0) + term(evaluation)
        return values

    def linear_form(self) -> dict[str, float] | None:
        """The summed linear forms of the terms, or None if a term has none."""
        total: dict[str, float] = {}
        for term in self.terms:
            form = term.linear_form()
            if form is None:
                return None
            for feature, value in form.items():
                total[feature] = total.get(feature, 0.0) + value
        return total


def linear_form(objective: Objective) -> dict[str, float] | None:
    """The linear form of ``objective``, or None if it has none (for example a function)."""
    if isinstance(objective, Term | Sum):
        return objective.linear_form()
    return None


def total_cost() -> Term:
    """Cost of all dispatches, the fixed cost of mandatory owned vehicles and lateness."""
    return Term(
        "total_cost",
        lambda evaluation: evaluation.total_cost,
        linear={"cost[owned]": 1.0, "cost[spot]": 1.0, "owned_fixed": 1.0, "service": 1.0},
    )


def dispatch_cost(kind: Kind | None = None) -> Term:
    """Cost of the dispatches, of one kind or of both."""
    return Term(
        _named("dispatch_cost", kind),
        lambda evaluation: float(evaluation.dispatch_cost[_of_kind(evaluation, kind)].sum()),
        linear=_per_kind("cost", kind),
    )


def service_cost() -> Term:
    """The lateness penalty (rules.service.penalty per unit and started hour late)."""
    return Term("service_cost", lambda evaluation: evaluation.service_cost, linear={"service": 1.0})


def owned_fixed_cost() -> Term:
    """Fixed cost of mandatory owned vehicles over the calendar."""
    return Term(
        "owned_fixed_cost",
        lambda evaluation: evaluation.owned_fixed_cost,
        linear={"owned_fixed": 1.0},
    )


def distance_km(kind: Kind | None = None) -> Term:
    """Length of all routes, of one kind of dispatch or of both."""
    return Term(
        _named("distance_km", kind),
        lambda evaluation: float(evaluation.route_km[_of_kind(evaluation, kind)].sum()),
        linear=_per_kind("km", kind),
    )


def dispatches(kind: Kind | None = None) -> Term:
    """Number of dispatches, of one kind or of both."""
    return Term(
        _named("dispatches", kind),
        lambda evaluation: float(np.count_nonzero(_of_kind(evaluation, kind))),
        linear=_per_kind("dispatches", kind),
    )


def unused_capacity(kind: Kind | None = None) -> Term:
    """Capacity left empty on departure, primary dimension."""

    def function(evaluation: "Evaluation") -> float:
        primary = evaluation.instance.primary_dimension
        empty = evaluation.capacity[:, primary] - evaluation.load[:, primary]
        return float(np.clip(empty, 0.0, None)[_of_kind(evaluation, kind)].sum())

    return Term(_named("unused_capacity", kind), function, linear=_per_kind("unused", kind))


def unserved(dimension: str | None = None) -> Term:
    """Demand quantity not carried, of one dimension or summed over all of them."""

    def function(evaluation: "Evaluation") -> float:
        if dimension is None:
            return float(evaluation.unserved.sum())
        return float(evaluation.unserved[:, _dimension(evaluation, dimension)].sum())

    name = "unserved" if dimension is None else f"unserved[{dimension}]"
    return Term(name, function, linear={name: 1.0})


def violation(name: str) -> Term:
    """The violation amount of one constraint, built-in or your own."""

    def function(evaluation: "Evaluation") -> float:
        try:
            return evaluation.violations[name]
        except KeyError:
            known = ", ".join(evaluation.violations)
            raise KeyError(f"no constraint named '{name}'; known: {known}") from None

    linear = {f"violation[{name}]": 1.0} if name in CATALOG else None
    return Term(f"violation[{name}]", function, linear=linear)


def violations(names: Iterable[str] | None = None) -> Term:
    """The summed violation amounts of the given constraints, or of all of them.

    Amounts have different units (counts, quantities, km); weight single
    constraints with :func:`violation` when that matters. Without ``names`` the
    linear form covers the built-in constraints only.
    """
    chosen = None if names is None else tuple(names)
    included = tuple(CATALOG) if chosen is None else chosen
    linear = (
        {f"violation[{name}]": 1.0 for name in included}
        if all(name in CATALOG for name in included)
        else None
    )

    def function(evaluation: "Evaluation") -> float:
        if chosen is None:
            return float(sum(evaluation.violations.values()))
        return sum(violation(name)(evaluation) for name in chosen)

    return Term("violations", function, linear=linear)


def _per_kind(feature: str, kind: Kind | None) -> dict[str, float]:
    kinds = ("owned", "spot") if kind is None else (kind,)
    return {f"{feature}[{one}]": 1.0 for one in kinds}


def _named(name: str, kind: Kind | None) -> str:
    return name if kind is None else f"{name}[{kind}]"


def _of_kind(evaluation: "Evaluation", kind: Kind | None) -> NDArray[np.bool_]:
    plan = evaluation.plan
    if kind is None:
        return np.ones(len(plan), dtype=np.bool_)
    if kind not in ("owned", "spot"):
        raise ValueError(f"kind must be 'owned', 'spot' or None, not {kind!r}")
    return np.asarray(plan.kind == (OWNED if kind == "owned" else SPOT))


def _dimension(evaluation: "Evaluation", dimension: str) -> int:
    dimensions = evaluation.instance.dimensions
    if dimension not in dimensions:
        raise KeyError(f"no quantity dimension '{dimension}'; known: {', '.join(dimensions)}")
    return dimensions.index(dimension)
