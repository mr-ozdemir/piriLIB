"""Griffe extension that prepares the pirilib docstrings for mkdocstrings.

The docstrings follow the Google layout but use reST markup, while mkdocstrings
renders Markdown. Once the package is loaded, every docstring is rewritten:

- ``:class:`~pirilib.Plan``` and the other Python roles become cross-references
  to the object that defines the name, looked up from the scope of the
  docstring as Sphinx does. Names that live in private modules, and names
  of the standard library, become code.
- A paragraph that ends in ``::`` ends in ``:``; its indented block stays code.
- Definition lists become Markdown definition lists (the ``def_list``
  extension).

Then the items of an ``Attributes:`` section become the docstrings of the
attributes they describe, so that every attribute is documented once, next to
its type and default, and can be linked to.
"""

import builtins
import re
import sys
from collections.abc import Iterator
from typing import Any

import griffe

ROLE = re.compile(r":(?:py:)?(?:class|func|meth|attr|data|const|mod|exc|obj):`(~?)([^`]+)`")
LITERAL = re.compile(r"(?m)(?<=\S)::$")
LIST_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s")
INDENT = "    "
MAX_ALIAS_HOPS = 8


class NameResolver:
    """Finds the object a name in a docstring refers to."""

    def __init__(self, collection: griffe.ModulesCollection) -> None:
        self._collection = collection

    def resolve(self, name: str, scope: griffe.Object) -> str | None:
        """Path of the object that defines ``name`` as seen from ``scope``, or ``None``.

        The scope, then every enclosing class and module, is searched for the
        first part of ``name``; a name found nowhere is taken as a full path.
        Re-exported names resolve to the object they import.
        """
        head = name.split(".", 1)[0]
        current: griffe.Object | None = scope
        while current is not None:
            if head in current.members:
                found = self._find(f"{current.path}.{name}", 0)
                if found is not None:
                    return found.path
            current = current.parent
        found = self._find(name, 0)
        return None if found is None else found.path

    def _find(self, path: str, hops: int) -> griffe.Object | None:
        first, *rest = path.split(".")
        current = self._target(self._collection.members.get(first), hops)
        for part in rest:
            if current is None:
                return None
            current = self._target(current.members.get(part), hops)
        return current

    def _target(
        self, member: griffe.Object | griffe.Alias | None, hops: int
    ) -> griffe.Object | None:
        if isinstance(member, griffe.Alias):
            return self._find(member.target_path, hops + 1) if hops < MAX_ALIAS_HOPS else None
        return member


class DocstringRewriter:
    """Rewrites the docstring of one object from reST markup to Markdown."""

    def __init__(self, resolver: NameResolver, scope: griffe.Object) -> None:
        self._resolver = resolver
        self._scope = scope

    def markdown(self, text: str) -> str:
        """Return ``text`` with roles, literal blocks and definition lists in Markdown."""
        text = self._definition_lists(text)
        text = ROLE.sub(self._reference, text)
        return LITERAL.sub(":", text)

    def _reference(self, match: re.Match[str]) -> str:
        short, name = match.group(1) == "~", match.group(2).strip()
        title = name.rsplit(".", 1)[-1] if short else name
        path = self._resolver.resolve(name, self._scope)
        if path is None:
            if _standard(name):
                return f"`{title}`"
            return f"[`{title}`][{name}]"
        if any(_private(part) for part in path.split(".")):
            return f"`{title}`"
        return f"[`{title}`][{path}]"

    def _definition_lists(self, text: str) -> str:
        lines = text.splitlines()
        result: list[str] = []
        after_term = in_definition = False
        for index, line in enumerate(lines):
            following = lines[index + 1] if index + 1 < len(lines) else ""
            if _is_term(line, following):
                if result and result[-1].strip():
                    result.append("")
                result.append(line)
                after_term = True
            elif after_term:
                result.append(":   " + line.removeprefix(INDENT))
                after_term, in_definition = False, True
            elif in_definition and line.startswith(INDENT):
                result.append(line)
            else:
                in_definition = False
                result.append(line)
        return "\n".join(result)


class DocstringMarkup(griffe.Extension):
    """Rewrites the docstrings of a package to Markdown and documents its attributes."""

    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any) -> None:
        """Rewrite the docstrings of ``pkg`` and its members once it is loaded."""
        resolver = NameResolver(loader.modules_collection)
        objects = list(_objects(pkg))
        for obj in objects:
            if obj.docstring is not None:
                obj.docstring.value = DocstringRewriter(resolver, obj).markdown(obj.docstring.value)
        for obj in objects:
            self._document_attributes(obj)

    def _document_attributes(self, obj: griffe.Object) -> None:
        if obj.docstring is None:
            return
        for section in obj.docstring.parse("google"):
            if section.kind is not griffe.DocstringSectionKind.attributes:
                continue
            for item in section.value:
                member = obj.members.get(item.name)
                if isinstance(member, griffe.Attribute) and member.docstring is None:
                    member.docstring = griffe.Docstring(
                        item.description, lineno=obj.docstring.lineno, parent=member
                    )


def _objects(obj: griffe.Object) -> Iterator[griffe.Object]:
    yield obj
    for member in obj.members.values():
        if not isinstance(member, griffe.Alias):
            yield from _objects(member)


def _standard(name: str) -> bool:
    """Whether ``name`` is a builtin or lives in a standard library module."""
    root = name.split(".", 1)[0]
    return hasattr(builtins, root) or root in sys.stdlib_module_names


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_term(line: str, following: str) -> bool:
    return (
        bool(line.strip())
        and not line[0].isspace()
        and not line.rstrip().endswith(":")
        and LIST_ITEM.match(line) is None
        and following.startswith(INDENT)
        and bool(following.strip())
    )
