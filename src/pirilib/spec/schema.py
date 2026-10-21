"""JSON Schema of line-haul problem documents.

Run ``python -m pirilib.spec.schema [PATH]`` to print the schema or write it to
``PATH``.
"""

import json
import sys
from pathlib import Path
from typing import Any

from pirilib.spec.models import SCHEMA_ID, LineHaulSpec
from pirilib.spec.tables import TableSource

_TABLE_REFERENCE = {"$ref": "#/$defs/TableSource"}


def document_json_schema() -> dict[str, Any]:
    """Return the JSON Schema of problem documents as they are written.

    It is the schema of :class:`~pirilib.spec.LineHaulSpec`, except that
    ``centers``, ``demand`` and ``matrix.arcs`` may also be a
    :class:`~pirilib.spec.TableSource` file reference.
    """
    schema = LineHaulSpec.model_json_schema(by_alias=True)
    definitions = schema.setdefault("$defs", {})
    definitions["TableSource"] = TableSource.model_json_schema()
    properties = schema["properties"]
    for name in ("centers", "demand"):
        properties[name] = {"anyOf": [properties[name], _TABLE_REFERENCE]}
    explicit = definitions["ExplicitMatrix"]["properties"]
    explicit["arcs"] = {"anyOf": [explicit["arcs"], _TABLE_REFERENCE]}
    document = {"$schema": "https://json-schema.org/draft/2020-12/schema"}
    document.update(schema)
    document["title"] = f"{SCHEMA_ID} problem document"
    return document


def render() -> str:
    """The JSON Schema of the problem document as text."""
    return json.dumps(document_json_schema(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the schema to the path in ``argv`` or to stdout; return the exit code."""
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        Path(arguments[0]).write_text(render(), encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
