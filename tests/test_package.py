import logging
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files

import pirilib


def test_version_matches_installed_metadata() -> None:
    assert pirilib.__version__ == version("pirilib")


def test_import_prints_nothing() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import pirilib"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (result.stdout, result.stderr) == ("", "")


def test_library_logger_only_has_null_handlers() -> None:
    handlers = logging.getLogger("pirilib").handlers
    assert handlers
    assert all(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_package_ships_typing_marker() -> None:
    assert files("pirilib").joinpath("py.typed").is_file()


def test_public_names() -> None:
    assert set(pirilib.__all__) == {
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
    }
    for error in (pirilib.SpecError, pirilib.PlanError):
        assert issubclass(error, pirilib.PirilibError)
        assert issubclass(error, ValueError)
