"""Human-readable messages for specification problems."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pydantic import ValidationError

MAX_REPORTED: Final = 50
"""Problems listed in one message; the rest are only counted."""

type Loc = tuple[str | int, ...]
"""Location of a value in a problem document, for example ``("demand", 3, "origin")``."""

type Origins = Mapping[Loc, Sequence[str]]
"""``file:line`` label of every row of the tables read from files, keyed by table location."""

_UNION_TAGS: Final[Mapping[Loc, frozenset[str]]] = {
    ("matrix",): frozenset({"haversine", "explicit"}),
}


class SpecIssues(ValueError):
    """Problems found by whole-document checks, each with the location it refers to."""

    def __init__(self, issues: Iterable[tuple[Loc, str]]) -> None:
        self.issues = tuple(issues)
        super().__init__(bullets(locate(loc, message) for loc, message in self.issues))


def bullets(lines: Iterable[str]) -> str:
    """Return ``lines`` as a bullet list, truncated to :data:`MAX_REPORTED` entries."""
    items = list(lines)
    text = "\n".join(f"- {line}" for line in items[:MAX_REPORTED])
    if len(items) > MAX_REPORTED:
        text += f"\n- ... and {len(items) - MAX_REPORTED} more"
    return text


def validation_problems(
    exc: ValidationError, origins: Origins | None = None, prefix: Loc = ()
) -> list[str]:
    """Return one line per problem in ``exc``, each starting with its location."""
    lines: list[str] = []
    for error in exc.errors(include_url=False):
        cause = error.get("ctx", {}).get("error")
        if isinstance(cause, SpecIssues):
            lines += [locate((*prefix, *loc), message, origins) for loc, message in cause.issues]
        else:
            message = str(error["msg"]).removeprefix("Value error, ")
            lines.append(locate((*prefix, *error["loc"]), message, origins))
    return lines


def format_validation_errors(exc: ValidationError, origins: Origins | None = None) -> str:
    """Return the problems in ``exc`` as a bullet list."""
    return bullets(validation_problems(exc, origins))


def locate(loc: Loc, message: str, origins: Origins | None = None) -> str:
    where = render_loc(loc, origins)
    return f"{where}: {message}" if where else message


def render_loc(loc: Loc, origins: Origins | None = None) -> str:
    """Render ``loc`` as ``demand[3].origin``, or as ``demand.csv:5: origin`` for a file row."""
    loc = _without_union_tag(loc)
    for table, labels in (origins or {}).items():
        size = len(table)
        if loc[:size] == table and len(loc) > size:
            row = loc[size]
            if isinstance(row, int) and 0 <= row < len(labels):
                rest = _path(loc[size + 1 :])
                return f"{labels[row]}: {rest}" if rest else labels[row]
    return _path(loc)


def _without_union_tag(loc: Loc) -> Loc:
    for union, tags in _UNION_TAGS.items():
        size = len(union)
        if loc[:size] == union and len(loc) > size and loc[size] in tags:
            return (*union, *loc[size + 1 :])
    return loc


def _path(loc: Loc) -> str:
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text
