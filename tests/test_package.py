"""Tests of the package surface: what ``import pirilib`` gives an application and costs it.

Requirements: the ``pirilib`` package docstring and decision K11 (importing prints nothing,
and its only side effect is one ``logging.NullHandler`` on the ``"pirilib"`` logger),
decisions K48, K70 and K88 (the fifteen top level names), ``docs/stability.md`` (every
error caused by user input is a ``PirilibError``: ``SpecError`` and ``PlanError``, both also
``ValueError``), decision K224 (the package ships the ``py.typed`` marker) and the version
kept in ``src/pirilib/_version.py``, which hatch writes into the distribution metadata.

Level: system (``SuiteLevels`` in ``tests/conftest.py``); the silence test starts a fresh
interpreter, the others read the package the test process imported.

Risk reduced: an application that imports pirilib gets output, warnings, log handlers or
top level names it did not ask for, cannot catch input errors as documented, cannot type
check against the package, or reads a version that differs from the installed one.

Blind spots, left out on purpose: the file list of the built wheel (with ``PYTHONPATH=src``
the marker is looked up in the source tree; the wheel is checked when it is built, K224),
whether each exported name works (the tests of its own module) and the import time.
"""

import logging
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files

import pytest

import pirilib

PUBLIC_NAMES = [
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
"""The fifteen top level names of decisions K48, K70 and K88, sorted."""

IMPORT_TIMEOUT_SECONDS = 60
"""Upper bound for a fresh interpreter to import pirilib; the import takes about a second."""


def test_version_matches_installed_metadata() -> None:
    """``pirilib.__version__`` is the version the installed distribution reports."""
    installed = version("pirilib")

    reported = pirilib.__version__

    assert reported == installed


def test_import_prints_and_warns_nothing() -> None:
    """A fresh interpreter imports pirilib without output, even with warnings as errors.

    ``-W error`` turns a warning raised during the import into an exception, so the
    interpreter would exit with status 1 and print the traceback on stderr.
    """
    command = [sys.executable, "-W", "error", "-c", "import pirilib"]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=IMPORT_TIMEOUT_SECONDS,
    )

    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_import_leaves_one_null_handler_on_the_library_logger() -> None:
    """The ``"pirilib"`` logger has exactly one handler, a ``logging.NullHandler``.

    Configuring log output is left to the application (K11), and importing the package
    more than once must not stack handlers.
    """
    logger = logging.getLogger("pirilib")

    handler_types = list(map(type, logger.handlers))

    assert handler_types == [logging.NullHandler]


def test_package_ships_typing_marker() -> None:
    """The package contains the ``py.typed`` marker, so type checkers read its annotations."""
    marker = files("pirilib").joinpath("py.typed")

    found = marker.is_file()

    assert found is True


def test_top_level_exports_exactly_the_documented_names() -> None:
    """``pirilib.__all__`` lists the documented names, each once, and nothing else."""
    exported = sorted(pirilib.__all__)

    assert exported == PUBLIC_NAMES


def test_every_exported_name_is_defined() -> None:
    """Every name in ``pirilib.__all__`` is an attribute, so ``from pirilib import *`` works."""
    defined = set(dir(pirilib))

    missing = set(pirilib.__all__) - defined

    assert missing == set()


@pytest.mark.parametrize(
    "base", [pirilib.PirilibError, ValueError], ids=["PirilibError", "ValueError"]
)
@pytest.mark.parametrize(
    "error", [pirilib.SpecError, pirilib.PlanError], ids=["SpecError", "PlanError"]
)
def test_input_errors_derive_from_the_documented_bases(
    error: type[Exception], base: type[Exception]
) -> None:
    """``SpecError`` and ``PlanError`` are caught by ``except PirilibError`` and ``ValueError``."""
    derives = issubclass(error, base)

    assert derives is True
