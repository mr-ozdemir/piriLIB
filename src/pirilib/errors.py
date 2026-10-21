"""Exception types raised by pirilib."""

__all__ = [
    "PirilibError",
    "PlanError",
    "SpecError",
]


class PirilibError(Exception):
    """Base class for every error raised by pirilib."""


class SpecError(PirilibError, ValueError):
    """A problem specification could not be read or is invalid.

    The message lists every problem that was found, so all of them can be fixed
    in one pass.
    """


class PlanError(PirilibError, ValueError):
    """A plan is malformed: inconsistent arrays, indices out of range or bad loads.

    Breaking a business rule (capacity, minimum fill, ...) is not an error; the
    evaluator measures it as a constraint violation.
    """
