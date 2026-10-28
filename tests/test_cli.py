import inspect
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from pirilib.cli import _parser, main
from pirilib.tuning import tune


def run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_validate(capsys: pytest.CaptureFixture[str], example_path: Path) -> None:
    code, out, _ = run(capsys, "validate", str(example_path))
    assert code == 0
    assert "valid" in out
    assert "demands=18" in out


def test_invalid_problem(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "problem.yaml"
    path.write_text("schema: pirilib/linehaul@0.1\n", encoding="utf-8")
    code, _, err = run(capsys, "validate", str(path))
    assert code == 1
    assert err.startswith("error: ")
    assert "calendar: Field required" in err


def test_solve_and_check(
    capsys: pytest.CaptureFixture[str], example_path: Path, tmp_path: Path
) -> None:
    plan, numbers = tmp_path / "plan.csv", tmp_path / "summary.json"
    code, out, _ = run(
        capsys,
        "solve",
        str(example_path),
        "-o",
        str(plan),
        "--summary",
        str(numbers),
        "--seed",
        "1",
        "--delimiter",
        ";",
        "--decimal",
        ",",
    )
    assert code == 0
    assert "feasible" in out
    assert json.loads(numbers.read_text(encoding="utf-8"))["feasible"] is True
    code, out, _ = run(
        capsys, "check", str(example_path), str(plan), "--delimiter", ";", "--decimal", ","
    )
    assert code == 0
    assert "feasible" in out


def test_check_reports_broken_rules(
    capsys: pytest.CaptureFixture[str], example_path: Path, tmp_path: Path
) -> None:
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "dispatch,date,kind,vehicle_type,source,origin,stops,demand,destination,desi\n"
        "1,2026-05-11,spot,kamyonet,fleet.spot[2],İstanbul,Kocaeli,D000001,Kocaeli,18450\n",
        encoding="utf-8",
    )
    code, out, _ = run(capsys, "check", str(example_path), str(plan))
    assert code == 1
    assert "- capacity: dispatch 0" in out
    assert "- coverage: demand D000002" in out


def test_solve_with_settings(
    capsys: pytest.CaptureFixture[str], example_path: Path, tmp_path: Path
) -> None:
    settings = tmp_path / "config.json"
    settings.write_text(json.dumps({"ants": 4, "iterations": 3}), encoding="utf-8")
    code, _, _ = run(
        capsys, "solve", str(example_path), "--config", str(settings), "--no-local-search"
    )
    assert code == 0
    settings.write_text(json.dumps({"bees": 4}), encoding="utf-8")
    code, _, err = run(capsys, "solve", str(example_path), "--config", str(settings))
    assert code == 1
    assert "unknown ACO settings ['bees']" in err


def test_tune(capsys: pytest.CaptureFixture[str], example_path: Path, tmp_path: Path) -> None:
    base, output = tmp_path / "base.json", tmp_path / "tuned.json"
    base.write_text(json.dumps({"ants": 4, "iterations": 2, "patience": 1}), encoding="utf-8")
    code, out, _ = run(
        capsys, "tune", str(example_path), "--budget", "3", "--config", str(base), "-o", str(output)
    )
    assert code == 0
    assert "against the base settings" in out
    assert json.loads(output.read_text(encoding="utf-8"))["iterations"] == 2


def test_tune_budget_matches_the_library_default() -> None:
    budget = inspect.signature(tune).parameters["budget"].default
    assert budget == 40
    assert _parser().parse_args(["tune", "problem.yaml"]).budget == budget


def test_generate_then_solve(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, out, _ = run(
        capsys, "generate", str(tmp_path / "sample"), "--centers", "6", "--days", "1"
    )
    assert code == 0
    code, out, _ = run(capsys, "solve", str(tmp_path / "sample" / "problem.yaml"), "--ants", "4")
    assert code == 0
    assert "total cost" in out


@pytest.fixture
def example_copy(example_path: Path, tmp_path: Path) -> Path:
    """A copy of the example folder; returns the path of its problem file."""
    folder = shutil.copytree(example_path.parent, tmp_path / "example")
    return Path(folder) / example_path.name


def rewrite(path: Path, change: Callable[[dict[str, Any]], None]) -> Path:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), "utf-8")
    return path


def test_solving_needs_the_objective_section(
    capsys: pytest.CaptureFixture[str], example_copy: Path
) -> None:
    rewrite(example_copy, lambda document: document.pop("objective"))
    code, _, _ = run(capsys, "validate", str(example_copy))
    assert code == 0
    for command in ("solve", "tune"):
        code, out, err = run(capsys, command, str(example_copy))
        assert code == 2
        assert out == ""
        assert err.startswith(f"error: {example_copy}: the problem document has no objective:")
        assert "objective:\n  cost: 1\n  penalties:" in err


def test_tuning_needs_one_objective(
    capsys: pytest.CaptureFixture[str], example_path: Path, example_copy: Path
) -> None:
    rewrite(example_copy, lambda document: document["objective"].update(cost=2))
    code, _, err = run(capsys, "tune", str(example_path), str(example_copy), "--budget", "1")
    assert code == 2
    assert "define different objectives" in err


def test_the_penalty_option_is_gone(capsys: pytest.CaptureFixture[str], example_path: Path) -> None:
    for command in ("solve", "tune"):
        with pytest.raises(SystemExit) as info:
            main([command, str(example_path), "--penalty", "10"])
        assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        main(["bench", "tiny", "--penalty", "10"])
    assert info.value.code == 2


def test_schema(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    code, _, _ = run(capsys, "schema", str(path))
    assert code == 0
    assert json.loads(path.read_text(encoding="utf-8"))["title"].startswith("pirilib/linehaul@0.2")


def test_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["fly"])
    assert info.value.code == 2


@pytest.fixture
def history_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    import yaml

    from pirilib import synthetic

    history, calendar = synthetic.history(lines=8, days=84, seed=1)
    path = synthetic.write_history(history, tmp_path / "history.csv")
    holidays = tmp_path / "holidays.txt"
    holidays.write_text("\n".join(sorted(d.isoformat() for d in calendar.holidays)), "utf-8")
    names = sorted(set(history.origins) | set(history.destinations))
    network = synthetic.generate(centers=len(names), days=1, demand_per_day=1, seed=2)
    del network["demand"], network["calendar"]
    network_path = tmp_path / "network.yaml"
    network_path.write_text(yaml.safe_dump(network, sort_keys=False), "utf-8")
    return path, holidays, network_path


def test_backtest(
    capsys: pytest.CaptureFixture[str], history_files: tuple[Path, Path, Path]
) -> None:
    history, holidays, _ = history_files
    code, out, _ = run(
        capsys,
        "backtest",
        str(history),
        "--holidays",
        str(holidays),
        "--models",
        "seasonal,naive",
        "--folds",
        "3",
    )
    assert code == 0
    table = out.splitlines()
    assert table[1].split() == ["model", "WMAPE", "bias", "MAE", "skill"]
    assert {line.split()[0] for line in table[2:]} == {"seasonal", "naive"}


def test_forecast_to_problem(
    capsys: pytest.CaptureFixture[str], history_files: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    history, holidays, network = history_files
    forecast, problem = tmp_path / "forecast.csv", tmp_path / "problem.yaml"
    code, out, _ = run(
        capsys,
        "forecast",
        str(history),
        "--holidays",
        str(holidays),
        "--model",
        "seasonal",
        "-o",
        str(forecast),
        "--network",
        str(network),
        "--problem",
        str(problem),
        "--quantile",
        "0.8",
        "--buffer-folds",
        "3",
    )
    assert code == 0
    assert "buffered to quantile 0.8" in out
    assert forecast.read_text("utf-8").splitlines()[0] == "origin,destination,date,desi"
    code, out, _ = run(capsys, "solve", str(problem), "--ants", "4", "--iterations", "5")
    assert code == 0
    assert "total cost" in out


def test_forecast_without_buffer_warns(
    capsys: pytest.CaptureFixture[str], history_files: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    history, holidays, network = history_files
    problem = tmp_path / "problem.yaml"
    arguments = [
        "forecast",
        str(history),
        "--holidays",
        str(holidays),
        "--model",
        "seasonal",
        "--network",
        str(network),
        "--problem",
        str(problem),
    ]
    code, _, err = run(capsys, *arguments)
    assert code == 0
    assert problem.exists()
    assert "warning: the problem plans the expected forecast without a buffer" in err
    code, _, err = run(capsys, *arguments, "--quantile", "0.8", "--buffer-folds", "3")
    assert code == 0
    assert "warning" not in err


def test_forecast_with_unknown_centers(
    capsys: pytest.CaptureFixture[str], history_files: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    import yaml

    history, _, network = history_files
    document = yaml.safe_load(network.read_text("utf-8"))
    document["centers"] = document["centers"][:2]
    network.write_text(yaml.safe_dump(document, sort_keys=False), "utf-8")
    code, _, err = run(
        capsys,
        "forecast",
        str(history),
        "--model",
        "seasonal",
        "--network",
        str(network),
        "--problem",
        str(tmp_path / "problem.yaml"),
    )
    assert code == 1
    assert "unknown center" in err
    assert not (tmp_path / "problem.yaml").exists()


def test_forecast_split_into_waves(
    capsys: pytest.CaptureFixture[str], history_files: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    import yaml

    from pirilib import synthetic

    _, holidays, network = history_files
    waves, _ = synthetic.wave_history(lines=8, days=84, seed=1)
    path = synthetic.write_wave_history(waves, tmp_path / "waves.csv")
    forecast, problem = tmp_path / "forecast.csv", tmp_path / "problem.yaml"
    code, out, _ = run(
        capsys,
        "forecast",
        str(path),
        "--holidays",
        str(holidays),
        "--model",
        "seasonal",
        "--waves",
        str(path),
        "--wave-weeks",
        "3",
        "-o",
        str(forecast),
        "--network",
        str(network),
        "--problem",
        str(problem),
    )
    assert code == 0
    assert "wave shares of" in out
    lines = forecast.read_text("utf-8").splitlines()
    assert lines[0] == "origin,destination,date,wave,desi"
    assert {line.split(",")[3] for line in lines[1:]} == {"09:00", "17:00"}
    document = yaml.safe_load(problem.read_text("utf-8"))
    assert {row["ready"] for row in document["demand"]} == {"09:00", "17:00"}
    assert len(document["demand"]) == len(lines) - 1
    code, out, _ = run(capsys, "validate", str(problem))
    assert code == 0
    assert "valid" in out
