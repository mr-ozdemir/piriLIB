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
import doctest
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from operator import attrgetter, itemgetter
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
README = ROOT / "README.md"
SESSION = re.compile(r"^```pycon\n(?P<session>.*?)^```", re.MULTILINE | re.DOTALL)
"""The interactive session of the README's quick start."""
SCENARIOS = ROOT / "examples" / "objectives"
FENCE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
TITLED_YAML = re.compile(r'^yaml title="(?P<path>[^"]+)"$')
KNOWN_INFO = re.compile(r'^(python|\{\.python \.no-run\}|yaml|text|bash|yaml title="[^"]+")$')
"""Info strings a block may have: run, skipped Python, not Python, or a titled file."""
RULE_ROW = re.compile(r"^\| `(?P<name>[a-z_]+)` \| (?P<unit>[a-z]+) \|", re.MULTILINE)
OTHER_UNIT = re.compile(r"\((quantity|km)\)\.$")
"""A catalog description that ends with a unit other than ``count``."""
SMALL = ACOConfig(ants=8, iterations=20, patience=5)
MEASURED = {
    "handling": "quantity",
    "capacity": "quantity",
    "min_fill": "quantity",
    "stretch": "km",
    "coverage": "quantity",
    "overdelivery": "quantity",
}
"""Rules measured in a unit, which the catalog names in parentheses."""
COUNTED = (
    "route",
    "load_route",
    "ready",
    "deferral",
    "ban",
    "owned_line",
    "owned_availability",
    "spot_availability",
    "vehicle_limit",
    "stopover",
    "max_stops",
    "forward_only",
    "split",
)
"""Rules counted in breaches; the catalog says ``(count)`` or gives no unit."""
UNITS = {**dict.fromkeys(COUNTED, "count"), **MEASURED}


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


def blocks(text: str) -> list[Block]:
    return [
        Block(match["info"].strip(), match["body"], text.count("\n", 0, match.start()) + 1)
        for match in FENCE.finditer(text)
    ]


def python_only(text: str) -> str:
    """The page with every line blanked except the bodies of ``python`` blocks.

    Lines keep their numbers, so an error in a block names its line on the page.
    """
    lines = [""] * (text.count("\n") + 1)
    for block in blocks(text):
        if block.info == "python":
            body = block.body.splitlines()
            lines[block.line : block.line + len(body)] = body
    return "\n".join(lines)


def shown_files(found: list[Block]) -> list[tuple[str, Block]]:
    """The blocks that show a repository file, with the file's path."""
    titled = [(TITLED_YAML.match(block.info), block) for block in found]
    return [(match["path"], block) for match, block in titled if match]


def section(heading: str) -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : None if end < 0 else end]


def without(name: str, *keys: str) -> dict[str, Any]:
    """The scenario document ``name`` with ``keys`` removed."""
    document: dict[str, Any] = yaml.safe_load((SCENARIOS / name).read_text(encoding="utf-8"))
    for key in keys:
        document.pop(key)
    return document


BLOCKS = blocks(PAGE.read_text(encoding="utf-8"))
SHOWN = shown_files(BLOCKS)


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


def test_the_python_only_view_keeps_lines_and_python() -> None:
    page = "intro\n```python\nx = 1\n```\n```yaml\na: 1\n```\nend\n"

    view = python_only(page)

    assert view.splitlines() == ["", "", "x = 1", "", "", "", "", ""]


def test_shown_files_are_the_titled_yaml_blocks() -> None:
    found = [Block('yaml title="a.yaml"', "a: 1\n", 3), Block("yaml", "b: 2\n", 7)]

    assert shown_files(found) == [("a.yaml", found[0])]


def test_every_scenario_has_an_expectation() -> None:
    assert sorted(map(attrgetter("name"), SCENARIOS.glob("*.yaml"))) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_scenarios_solve_as_described(name: str) -> None:
    spec = pirilib.load_spec(SCENARIOS / name)

    result = ACOSolver(SMALL).solve(spec.compile(), spec.build_objective(), seed=0)

    expected = EXPECTED[name]
    assert result.evaluation.violated() == pytest.approx(expected.violated)
    assert result.evaluation.feasible == (expected.violated == {})
    assert len(result.plan) == expected.dispatches
    assert result.stats["mode"] == "linear"


def test_the_two_trade_off_scenarios_differ_only_in_their_objective() -> None:
    every = without("every_load.yaml", "objective", "name")

    hard = without("hard_min_fill.yaml", "objective", "name")

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


def test_the_page_has_blocks() -> None:
    assert len(BLOCKS) > 0


@pytest.mark.parametrize("block", BLOCKS, ids=lambda block: f"line-{block.line}")
def test_every_block_is_run_skipped_or_not_python(block: Block) -> None:
    assert KNOWN_INFO.match(block.info), f"line {block.line}: {block.info!r}"


def test_the_page_has_enough_python_to_show_the_workflow() -> None:
    assert list(map(attrgetter("info"), BLOCKS)).count("python") >= 8


def test_python_blocks_run(at_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    namespace: dict[str, Any] = {"__name__": "objective_page"}
    code = compile(python_only(PAGE.read_text(encoding="utf-8")), str(PAGE), "exec")

    exec(code, namespace)

    capsys.readouterr()
    assert "goal" in namespace


def test_the_readme_session_is_current(at_root: Path) -> None:
    session = SESSION.search(README.read_text(encoding="utf-8"))
    assert session is not None
    parser = doctest.DocTestParser()
    example = parser.get_doctest(session["session"], {}, "README", str(README), 0)

    results = doctest.DocTestRunner().run(example)

    assert results == doctest.TestResults(failed=0, attempted=4)


def test_the_page_shows_repository_files() -> None:
    assert len(SHOWN) > 0


@pytest.mark.parametrize(("path", "block"), SHOWN, ids=list(map(itemgetter(0), SHOWN)))
def test_titled_yaml_blocks_equal_their_files(path: str, block: Block) -> None:
    assert block.body == (ROOT / path).read_text(encoding="utf-8")


def test_the_rule_table_lists_the_catalog() -> None:
    rows = RULE_ROW.findall(section("Rules and their units"))

    assert dict(rows) == UNITS
    assert list(map(itemgetter(0), rows)) == list(CATALOG)


@pytest.mark.parametrize(("name", "unit"), list(MEASURED.items()))
def test_the_catalog_names_the_unit_of_measured_rules(name: str, unit: str) -> None:
    assert CATALOG[name].endswith(f"({unit}).")


@pytest.mark.parametrize("name", COUNTED)
def test_the_catalog_gives_no_other_unit_for_counted_rules(name: str) -> None:
    assert OTHER_UNIT.search(CATALOG[name]) is None


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


def test_the_missing_objective_error_is_current(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario_document: dict[str, Any],
) -> None:
    monkeypatch.chdir(tmp_path)
    document = copy.deepcopy(scenario_document)
    document.pop("objective")
    Path("problem.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")

    code = main(["solve", "problem.yaml"])

    assert code == 2
    assert capsys.readouterr().err.strip() in section("Common errors")


def test_the_different_objectives_error_is_current(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scenario_document: dict[str, Any],
) -> None:
    monkeypatch.chdir(tmp_path)
    other = copy.deepcopy(scenario_document)
    other["objective"]["distance_km"] = 0
    Path("a.yaml").write_text(yaml.safe_dump(scenario_document), encoding="utf-8")
    Path("b.yaml").write_text(yaml.safe_dump(other), encoding="utf-8")

    code = main(["tune", "a.yaml", "b.yaml", "--budget", "2"])

    assert code == 2
    assert capsys.readouterr().err.strip() in section("Common errors")


@pytest.mark.parametrize(
    ("kind", "call"),
    [
        (TypeError, lambda instance, evaluation: pirilib.solve(instance)),
        (ValueError, lambda instance, evaluation: pirilib.solve(instance, None)),
        (KeyError, lambda instance, evaluation: obj.violation("capacty")(evaluation)),
        (
            ValueError,
            lambda instance, evaluation: pirilib.solve(
                instance, obj.total_cost() + obj.dispatches("rented")
            ),
        ),
    ],
    ids=["no-objective", "none-objective", "unknown-rule", "unknown-kind"],
)
def test_python_errors_are_current(
    scenario_instance: LineHaulInstance,
    kind: type[Exception],
    call: Callable[[LineHaulInstance, Evaluation], Any],
) -> None:
    evaluation = pirilib.evaluate(scenario_instance, Plan.from_dispatches([], 1))

    with pytest.raises(kind) as caught:
        call(scenario_instance, evaluation)

    assert f"{kind.__name__}: {caught.value}" in section("Common errors")
