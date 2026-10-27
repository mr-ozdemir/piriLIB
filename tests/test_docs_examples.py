"""The objective page (docs/objective.md) and the scenarios in examples/objectives.

Conventions of the page that these tests rely on:

- A fenced block whose info string is exactly ``python`` runs. All of them run in
  page order, in one shared namespace, from the repository root, so later blocks
  use what earlier ones defined and paths such as
  ``examples/objectives/every_load.yaml`` resolve.
- A block that shows Python but is not meant to run is fenced as
  ``{.python .no-run}``: it is highlighted as Python and skipped here.
- A ``yaml`` block with ``title="<path>"`` shows that file of the repository and
  must equal it.
- Every other block is ``yaml``, ``text`` or ``bash`` and is not run.
- The table under "Rules and their units" lists exactly the names of
  :data:`pirilib.constraints.CATALOG`, in its order, with the unit that CATALOG
  gives in parentheses (``count`` when it gives none).
- The error messages under "Common errors" are the ones the code produces.
"""

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

import pirilib
from pirilib import ACOConfig, ACOSolver, Evaluation, Plan
from pirilib import objective as obj
from pirilib.cli import main
from pirilib.constraints import CATALOG
from pirilib.spec import LineHaulInstance, LineHaulSpec, parse_spec

ROOT = Path(__file__).parents[1]
PAGE = ROOT / "docs" / "objective.md"
SCENARIOS = ROOT / "examples" / "objectives"
FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
TITLED_YAML = re.compile(r'^yaml title="(?P<path>[^"]+)"$')
NOT_RUN = frozenset({"{.python .no-run}", "yaml", "text", "bash"})
RULE_ROW = re.compile(r"^\| `(?P<name>[a-z_]+)` \| (?P<unit>[a-z]+) \|", re.MULTILINE)
CATALOG_UNIT = re.compile(r"\((?P<unit>count|quantity|km)\)\.$")
SMALL = ACOConfig(ants=8, iterations=20, patience=5)


@dataclass(frozen=True, slots=True)
class Scenario:
    """What solving one example document must give.

    Attributes:
        violated: The rules the plan breaks and their amounts, as
            :meth:`pirilib.Evaluation.violated` gives them; empty when the plan is
            feasible
        dispatches: The number of dispatches of the plan
    """

    violated: Mapping[str, float]
    dispatches: int


EXPECTED = {
    "every_load.yaml": Scenario({"min_fill": 80.0}, 7),
    "hard_min_fill.yaml": Scenario({"coverage": 40.0}, 6),
    "distance_aware.yaml": Scenario({}, 2),
    "worked_cost.yaml": Scenario({}, 2),
}


@dataclass(frozen=True, slots=True)
class Block:
    """A fenced block of the page.

    Attributes:
        info: The info string after the opening fence
        body: The text between the fences
        line: The line of the opening fence, counted from 1
    """

    info: str
    body: str
    line: int


def blocks() -> list[Block]:
    text = PAGE.read_text(encoding="utf-8")
    return [
        Block(match["info"].strip(), match["body"], text.count("\n", 0, match.start()) + 1)
        for match in FENCE.finditer(text)
    ]


def section(heading: str) -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : None if end < 0 else end]


@pytest.fixture
def at_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from the repository root, where the page's paths resolve."""
    monkeypatch.chdir(ROOT)
    return ROOT


@pytest.fixture
def scenario_spec() -> LineHaulSpec:
    return pirilib.load_spec(SCENARIOS / "distance_aware.yaml")


@pytest.fixture
def scenario_document() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(
        (SCENARIOS / "distance_aware.yaml").read_text(encoding="utf-8")
    )
    return document


@pytest.fixture
def scenario_instance(scenario_spec: LineHaulSpec) -> LineHaulInstance:
    return scenario_spec.compile()


def test_every_scenario_has_an_expectation() -> None:
    assert sorted(path.name for path in SCENARIOS.glob("*.yaml")) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_scenarios_solve_as_described(name: str) -> None:
    spec = pirilib.load_spec(SCENARIOS / name)
    result = ACOSolver(SMALL).solve(spec.compile(), spec.build_objective(), seed=0)
    expected = EXPECTED[name]
    assert result.evaluation.violated() == pytest.approx(expected.violated)
    assert result.evaluation.feasible == (not expected.violated)
    assert len(result.plan) == expected.dispatches
    assert result.stats["mode"] == "linear"


def test_the_two_trade_off_scenarios_differ_only_in_their_objective() -> None:
    every, hard = (
        yaml.safe_load((SCENARIOS / name).read_text(encoding="utf-8"))
        for name in ("every_load.yaml", "hard_min_fill.yaml")
    )
    for document in (every, hard):
        document.pop("objective")
        document.pop("name")
    assert every == hard


def test_the_distance_weight_changes_the_plan(scenario_spec: LineHaulSpec) -> None:
    assert scenario_spec.objective is not None
    plain = scenario_spec.objective.model_copy(update={"distance_km": 0.0})
    instance = scenario_spec.compile()
    solver = ACOSolver(SMALL)
    detour = solver.solve(instance, plain.build(), seed=0)
    direct = solver.solve(instance, scenario_spec.build_objective(), seed=0)
    assert (len(detour.plan), detour.evaluation.total_km) == (1, 300.0)
    assert (len(direct.plan), direct.evaluation.total_km) == (2, 220.0)
    assert detour.evaluation.total_cost == pytest.approx(1300.0)
    assert direct.evaluation.total_cost == pytest.approx(2220.0)


def test_every_block_is_run_skipped_or_not_python() -> None:
    found = blocks()
    assert found
    for block in found:
        known = block.info == "python" or block.info in NOT_RUN
        assert known or TITLED_YAML.match(block.info), f"line {block.line}: {block.info!r}"


def test_python_blocks_run(at_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    namespace: dict[str, Any] = {"__name__": "objective_page"}
    runnable = [block for block in blocks() if block.info == "python"]
    assert len(runnable) >= 8
    for block in runnable:
        code = compile(block.body, f"{PAGE.name}, block at line {block.line}", "exec")
        exec(code, namespace)
    capsys.readouterr()


def test_titled_yaml_blocks_equal_their_files() -> None:
    titled = [(TITLED_YAML.match(block.info), block) for block in blocks()]
    shown = [(match["path"], block) for match, block in titled if match]
    assert shown
    for path, block in shown:
        assert block.body == (ROOT / path).read_text(encoding="utf-8"), path


def test_the_rule_table_lists_the_catalog() -> None:
    rows = RULE_ROW.findall(section("Rules and their units"))
    assert [name for name, _ in rows] == list(CATALOG)
    for name, unit in rows:
        given = CATALOG_UNIT.search(CATALOG[name])
        assert unit == (given["unit"] if given else "count"), name


def test_document_errors_are_current(scenario_document: dict[str, Any]) -> None:
    errors = section("Common errors")

    def message(change: Any) -> str:
        document = copy.deepcopy(scenario_document)
        change(document["objective"])
        with pytest.raises(pirilib.SpecError) as caught:
            parse_spec(document, source="problem.yaml")
        return str(caught.value)

    def weights(part: dict[str, Any]) -> None:
        part.update({"cost": 0, "distance": 3})
        part["penalties"]["min_fill"] = -1

    assert message(lambda part: part["penalties"].pop("other")) in errors
    assert message(lambda part: part["penalties"].update({"coverge": 5})) in errors
    assert message(weights) in errors


def test_command_line_errors_are_current(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario_document: dict[str, Any],
) -> None:
    errors = section("Common errors")
    monkeypatch.chdir(tmp_path)
    without = {key: value for key, value in scenario_document.items() if key != "objective"}
    Path("problem.yaml").write_text(yaml.safe_dump(without), encoding="utf-8")
    assert main(["solve", "problem.yaml"]) == 2
    assert capsys.readouterr().err.strip() in errors

    other = copy.deepcopy(scenario_document)
    other["objective"]["distance_km"] = 0
    Path("a.yaml").write_text(yaml.safe_dump(scenario_document), encoding="utf-8")
    Path("b.yaml").write_text(yaml.safe_dump(other), encoding="utf-8")
    assert main(["tune", "a.yaml", "b.yaml", "--budget", "2"]) == 2
    assert capsys.readouterr().err.strip() in errors


def test_python_errors_are_current(scenario_instance: LineHaulInstance) -> None:
    errors = section("Common errors")
    evaluation: Evaluation = pirilib.evaluate(scenario_instance, Plan.from_dispatches([], 1))
    calls: list[tuple[type[Exception], Any]] = [
        (TypeError, lambda: pirilib.solve(scenario_instance)),
        (ValueError, lambda: pirilib.solve(scenario_instance, None)),
        (KeyError, lambda: obj.violation("capacty")(evaluation)),
        (
            ValueError,
            lambda: pirilib.solve(scenario_instance, obj.total_cost() + obj.dispatches("rented")),
        ),
    ]
    for kind, call in calls:
        with pytest.raises(kind) as caught:
            call()
        assert f"{kind.__name__}: {caught.value}" in errors
