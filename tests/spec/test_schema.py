import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from pirilib.spec.schema import document_json_schema, render

SCHEMA_FILE = Path(__file__).parents[2] / "schemas" / "linehaul-0.2.schema.json"


def validator() -> Draft202012Validator:
    return Draft202012Validator(document_json_schema())


def test_committed_schema_is_current() -> None:
    assert SCHEMA_FILE.read_text(encoding="utf-8") == render(), (
        "regenerate it: python -m pirilib.spec.schema schemas/linehaul-0.2.schema.json"
    )


def test_schema_is_valid_json_schema() -> None:
    Draft202012Validator.check_schema(document_json_schema())


def test_example_matches_schema(example_path: Path) -> None:
    data = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    validator().validate(json.loads(json.dumps(data, default=str)))


def test_inline_document_matches_schema(document: dict[str, Any]) -> None:
    validator().validate(document)


def test_schema_rejects_unknown_fields(document: dict[str, Any]) -> None:
    document["rules"] = {"min_fil": {"spot": 0.1}}
    assert not validator().is_valid(document)


def test_schema_knows_the_penalty_names(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"coverage": 1e7, "other": 1e6}}
    assert validator().is_valid(document)
    document["objective"]["penalties"]["tolls"] = 1.0
    assert not validator().is_valid(document)
