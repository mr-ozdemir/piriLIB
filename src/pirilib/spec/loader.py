"""Load a line-haul problem specification from a YAML file."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pirilib.errors import SpecError
from pirilib.spec._format import Origins, format_validation_errors
from pirilib.spec.models import LineHaulSpec
from pirilib.spec.tables import materialize_tables

_INVALID = "invalid problem specification"


def load_spec(path: str | os.PathLike[str]) -> LineHaulSpec:
    """Read, validate and return the problem described by a YAML file.

    ``centers``, ``demand`` and ``matrix.arcs`` may refer to CSV files (see
    :class:`~pirilib.spec.TableSource`); their paths are relative to the YAML
    file, and problems in their rows are reported with file name and line.
    Every problem found is reported in one :class:`~pirilib.SpecError`.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read {file}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise SpecError(f"{file}: not valid UTF-8 text: {exc.reason} at byte {exc.start}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"{file}: invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SpecError(f"{file}: the top level must be a mapping of fields")
    try:
        document, origins = materialize_tables(data, base_dir=file.parent)
    except SpecError as exc:
        raise SpecError(f"{file}: {_INVALID}\n{exc}") from None
    return _validate(document, str(file), origins)


def parse_spec(data: Mapping[str, Any], *, source: str = "<data>") -> LineHaulSpec:
    """Validate a problem document whose tables are given inline."""
    return _validate(data, source, {})


def _validate(data: Mapping[str, Any], source: str, origins: Origins) -> LineHaulSpec:
    try:
        return LineHaulSpec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"{source}: {_INVALID}\n{format_validation_errors(exc, origins)}") from exc
