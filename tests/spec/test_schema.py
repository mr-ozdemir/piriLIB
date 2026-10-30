"""Tests of :mod:`pirilib.spec.schema`, the JSON Schema of problem documents.

The expected behaviour comes from the docstring of
:func:`pirilib.spec.schema.document_json_schema`:
the schema is the one of :class:`pirilib.spec.LineHaulSpec`, ``centers``, ``demand`` and
``matrix.arcs`` may also be a :class:`pirilib.spec.TableSource` file reference, the keys of
``objective.penalties`` are the built-in rule names and ``other``, and the committed file in
``schemas/`` is the output of :func:`pirilib.spec.schema.render`.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

from pirilib.spec.schema import document_json_schema, render

SCHEMA_FILE = Path(__file__).parents[2] / "schemas" / "linehaul-0.2.schema.json"
"""The committed schema file."""

REGENERATE = "regenerate it: python -m pirilib.spec.schema schemas/linehaul-0.2.schema.json"
"""What to do when the committed schema file is out of date."""

BUILT_IN_RULES = (
    "route",
    "load_route",
    "ready",
    "deferral",
    "ban",
    "owned_line",
    "owned_availability",
    "spot_availability",
    "handling",
    "vehicle_limit",
    "capacity",
    "min_fill",
    "stopover",
    "max_stops",
    "stretch",
    "forward_only",
    "split",
    "coverage",
    "overdelivery",
)
"""The built-in rule names, written out from the rule table in ``docs/objective.md``."""

COST_WEIGHT = 1.0
"""A valid ``objective.cost`` (it must be above 0); its value plays no part in these tests."""

PENALTY_WEIGHT = 1.0
"""A valid penalty weight (it must be at least 0); only the penalty keys are under test."""

TABLE_REFERENCE = {"file": "table.csv"}
"""A file reference with only its required key; the schema never opens the file."""

RANDOM_SEEDS = range(10)
"""Seeds of the random documents that :func:`pirilib.spec.parse_spec` accepts."""


def violations(document: Any) -> list[tuple[str, str]]:
    """The JSON path and the keyword of every way ``document`` breaks the schema, sorted.

    Each top-level error is narrowed to its most specific cause with
    :func:`jsonschema.exceptions.best_match`, so that a wrong penalty name is reported at
    ``$.objective.penalties`` rather than as the ``anyOf`` of ``$.objective``. A document that
    matches the schema has no violations.
    """
    errors = Draft202012Validator(document_json_schema()).iter_errors(document)
    causes = [best_match([error]) for error in errors]
    return sorted((cause.json_path, cause.validator) for cause in causes)


def as_json(data: Any) -> Any:
    """The JSON form of loaded YAML: YAML dates become the ISO strings a JSON document holds."""
    return json.loads(json.dumps(data, default=str))


def test_committed_schema_file_is_the_rendered_schema() -> None:
    """Regression (golden file) test: ``schemas/`` holds the output of ``render()``."""
    committed = SCHEMA_FILE.read_text(encoding="utf-8")

    rendered = render()

    assert rendered == committed, REGENERATE


def test_schema_declares_draft_2020_12() -> None:
    """``$schema`` names draft 2020-12, so generic tools pick the matching validator."""
    schema = document_json_schema()

    declared = validator_for(schema)

    assert declared is Draft202012Validator


def test_schema_is_valid_against_the_draft_2020_12_meta_schema() -> None:
    """The schema itself has no error under the draft 2020-12 meta-schema and its formats."""
    meta = Draft202012Validator(
        Draft202012Validator.META_SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER
    )
    schema = document_json_schema()

    errors = list(meta.iter_errors(schema))

    assert errors == []


def test_bundled_example_matches_schema(example_path: Path) -> None:
    """The example problem file, with its ``centers`` and ``demand`` CSV references, is valid."""
    document = as_json(yaml.safe_load(example_path.read_text(encoding="utf-8")))

    found = violations(document)

    assert found == []


def test_inline_document_matches_schema(document: dict[str, Any]) -> None:
    """A document with inline tables that :func:`pirilib.spec.parse_spec` accepts is valid."""
    found = violations(document)

    assert found == []


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_random_valid_document_matches_schema(
    random_document: Callable[[int], dict[str, Any]], seed: int
) -> None:
    """The schema accepts the varied documents that the specification models accept."""
    document = random_document(seed)

    found = violations(document)

    assert found == []


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        pytest.param("rule", {}, [("$", "additionalProperties")], id="top-level"),
        pytest.param(
            "rules",
            {"min_fil": {"spot": 0.1}},
            [("$.rules", "additionalProperties")],
            id="misspelled-rule",
        ),
    ],
)
def test_schema_rejects_unknown_field(
    document: dict[str, Any], key: str, value: Any, expected: list[tuple[str, str]]
) -> None:
    """A field the models do not know (``rule``, ``rules.min_fil``) is reported where it is."""
    document[key] = value

    found = violations(document)

    assert found == expected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("centers", TABLE_REFERENCE, id="centers"),
        pytest.param("demand", TABLE_REFERENCE, id="demand"),
        pytest.param("matrix", {"source": "explicit", "arcs": TABLE_REFERENCE}, id="matrix-arcs"),
    ],
)
def test_table_field_may_be_a_file_reference(
    document: dict[str, Any], key: str, value: dict[str, Any]
) -> None:
    """``centers``, ``demand`` and ``matrix.arcs`` accept a file reference instead of rows."""
    document[key] = value

    found = violations(document)

    assert found == []


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        pytest.param("centers", {"delimiter": ";"}, [("$.centers", "required")], id="without-file"),
        pytest.param("centers", {"file": ""}, [("$.centers.file", "minLength")], id="empty-file"),
        pytest.param(
            "centers",
            {"file": "table.csv", "sep": ";"},
            [("$.centers", "additionalProperties")],
            id="unknown-key",
        ),
        pytest.param(
            "matrix",
            {"source": "explicit", "arcs": {"file": ""}},
            [("$.matrix.arcs.file", "minLength")],
            id="arcs-empty-file",
        ),
    ],
)
def test_schema_rejects_malformed_file_reference(
    document: dict[str, Any], key: str, value: dict[str, Any], expected: list[tuple[str, str]]
) -> None:
    """A file reference needs a non-empty ``file`` and only the keys of ``TableSource``."""
    document[key] = value

    found = violations(document)

    assert found == expected


@pytest.mark.parametrize("name", [*BUILT_IN_RULES, "other"])
def test_schema_accepts_built_in_penalty_name(document: dict[str, Any], name: str) -> None:
    """Every built-in rule name and ``other`` may key ``objective.penalties``."""
    document["objective"] = {"cost": COST_WEIGHT, "penalties": {name: PENALTY_WEIGHT}}

    found = violations(document)

    assert found == []


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("tolls", id="unknown"),
        pytest.param("Coverage", id="wrong-case"),
        pytest.param("", id="empty"),
    ],
)
def test_schema_rejects_unknown_penalty_name(document: dict[str, Any], name: str) -> None:
    """A penalty key that is no built-in rule name nor ``other`` is the only violation."""
    penalties = {"other": PENALTY_WEIGHT, name: PENALTY_WEIGHT}
    document["objective"] = {"cost": COST_WEIGHT, "penalties": penalties}

    found = violations(document)

    assert found == [("$.objective.penalties", "enum")]
