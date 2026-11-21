"""Tests of rules written as ``hard`` in the objective.

The requirement is the documentation of :class:`pirilib.spec.ObjectiveSpec`,
:attr:`pirilib.Evaluation.feasible` and "Hard rules" in ``docs/objective.md``: a rule written
as ``hard`` gets a weight far above the costs, a plan is feasible when no hard rule is broken
(when no rule is hard, when no rule at all is broken), a plan that breaks a hard rule is still
returned and says which rule it breaks, and ``pirilib solve`` then exits with status 3.

Oracles: the feasibility of the hand-made plans follows from the shared ``document`` fixture,
and the solver tests use ``examples/objectives/hard_min_fill.yaml``, whose data force a choice:
a 40-desi demand can only go in a vehicle that needs 120 desi for its minimum fill, so either
40 desi stay behind (``coverage``) or the vehicle leaves 80 desi short (``min_fill``).
"""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from pirilib import (
    ACOConfig,
    ACOSolver,
    Dispatch,
    Evaluation,
    Load,
    Plan,
    SpecError,
    check_plan,
    evaluate,
)
from pirilib import objective as obj
from pirilib.cli import main
from pirilib.constraints import MaxRouteKm
from pirilib.planio import write_plan
from pirilib.solvers.base import Solver
from pirilib.solvers.direct import DirectShipping
from pirilib.solvers.exact import ExactSolver
from pirilib.spec import LineHaulInstance, ObjectiveSpec, parse_spec
from pirilib.spec.schema import document_json_schema

pytest.importorskip("scipy", reason="the exact solver needs scipy (pip install 'pirilib[exact]')")

EXAMPLE = Path(__file__).parents[1] / "examples" / "objectives" / "hard_min_fill.yaml"
"""The example whose data do not allow both minimum fill and full coverage."""

PARTIAL = Plan.from_dispatches(
    [
        Dispatch(day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (900.0,)),)),
        Dispatch(day=0, kind="spot", source=1, origin=0, stops=(2,), loads=(Load(1, (250.0,)),)),
    ],
    dimensions=1,
)
"""The shared fixture's first two demands carried; the 120 desi of the third stay behind."""

PLANNERS: dict[str, Solver] = {
    "aco": ACOSolver(ACOConfig(ants=6, iterations=10, patience=5)),
    "exact": ExactSolver(),
}
"""The solvers that weigh the penalties while they plan."""


def choice(penalties: dict[str, Any]) -> dict[str, Any]:
    """The example document with ``penalties`` as its penalties."""
    document: dict[str, Any] = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    document["objective"]["penalties"] = penalties
    return document


def solved(name: str, penalties: dict[str, Any]) -> Evaluation:
    """The evaluation of the plan that solver ``name`` makes for :func:`choice`."""
    spec = parse_spec(choice(penalties))
    return PLANNERS[name].solve(spec.compile(), spec.build_objective(), seed=0).evaluation


def test_the_helper_replaces_the_penalties_of_the_example() -> None:
    """The helper's own test: the document keeps its seven demands and gets the new section."""
    document = choice({"other": 5})

    assert (len(document["demand"]), document["objective"]["penalties"]) == (7, {"other": 5})


def test_a_rule_can_be_written_as_hard(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"coverage": "hard", "other": 1000}}

    spec = parse_spec(document)

    assert spec.objective is not None
    assert spec.objective.hard_rules() == {"coverage"}


def test_other_can_make_every_unnamed_rule_hard(document: dict[str, Any]) -> None:
    """Two rules are named with weights; the other 17 of the 19 built-in rules are hard."""
    document["objective"] = {
        "cost": 1,
        "penalties": {"coverage": 1000, "min_fill": 50, "other": "hard"},
    }

    spec = parse_spec(document)

    assert spec.objective is not None
    hard = spec.objective.hard_rules()
    assert len(hard) == 17
    assert hard.isdisjoint({"coverage", "min_fill"})


def test_a_section_without_hard_has_no_hard_rules() -> None:
    section = ObjectiveSpec(cost=1, penalties={"coverage": 1e7, "other": 1e6})

    assert section.hard_rules() == frozenset()


@pytest.mark.parametrize(
    ("penalties", "weight"),
    [
        ({"coverage": "hard", "other": 1000.0}, 1e9),
        ({"coverage": "hard", "other": 1e8}, 1e10),
    ],
    ids=["least-weight", "above-the-largest"],
)
def test_a_hard_rule_weighs_far_more_than_everything_else(
    penalties: dict[str, Any], weight: float
) -> None:
    """A billion per unit, or 100 times the largest weight of the section when that is more."""
    section = ObjectiveSpec(cost=1, penalties=penalties)

    assert (section.hard_weight, section.penalty("coverage")) == (weight, weight)


def test_a_weighted_rule_keeps_its_weight_next_to_a_hard_one() -> None:
    section = ObjectiveSpec(cost=1, penalties={"coverage": "hard", "other": 1000.0})

    assert section.penalty("min_fill") == 1000.0


def test_a_word_other_than_hard_is_reported(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"coverage": "soft", "other": 1}}

    with pytest.raises(SpecError) as info:
        parse_spec(document)

    assert str(info.value).splitlines() == [
        "<data>: invalid problem specification",
        "- objective.penalties.coverage: Input should be 'hard'",
    ]


@pytest.mark.parametrize(
    ("penalty", "valid"), [("hard", True), (5, True), ("soft", False)], ids=["hard", "5", "soft"]
)
def test_the_json_schema_takes_hard_or_a_weight(
    document: dict[str, Any], penalty: object, valid: bool
) -> None:
    document["objective"] = {"cost": 1, "penalties": {"coverage": penalty, "other": 1}}

    accepted = Draft202012Validator(document_json_schema()).is_valid(document)

    assert accepted is valid


def test_the_built_objective_knows_its_hard_rules(document: dict[str, Any]) -> None:
    document["objective"] = {"cost": 1, "penalties": {"coverage": "hard", "other": 1000}}

    goal = parse_spec(document).build_objective()

    assert (goal.hard, obj.hard_rules(goal)) == ({"coverage"}, {"coverage"})


def test_sums_keep_their_hard_rules() -> None:
    """Adding a term, scaling, and adding two sums, which unites their hard rules."""
    first = obj.Sum((obj.total_cost(),), frozenset({"coverage"}))
    second = obj.Sum((obj.violation("capacity"),), frozenset({"capacity"}))

    assert (first + obj.distance_km()).hard == {"coverage"}
    assert (2.0 * first).hard == {"coverage"}
    assert (first + second).hard == {"coverage", "capacity"}


@pytest.mark.parametrize(
    "objective",
    [obj.total_cost(), obj.total_cost() + obj.violations(), lambda evaluation: 0.0],
    ids=["term", "sum", "function"],
)
def test_an_objective_of_your_own_declares_no_hard_rule(objective: obj.Objective) -> None:
    assert obj.hard_rules(objective) == frozenset()


def test_a_plan_that_breaks_a_hard_rule_is_not_feasible(instance: LineHaulInstance) -> None:
    """Coverage is hard and 120 desi stay behind."""
    evaluation = evaluate(instance, PARTIAL, hard={"coverage"})

    assert (evaluation.feasible, evaluation.broken()) == (False, {"coverage": 120.0})


def test_a_plan_that_breaks_only_weighted_rules_is_feasible(instance: LineHaulInstance) -> None:
    """Capacity is the hard rule; the coverage the plan misses is reported, not forbidden."""
    evaluation = evaluate(instance, PARTIAL, hard={"capacity"})

    assert (evaluation.feasible, evaluation.broken()) == (True, {})
    assert evaluation.violated() == {"coverage": 120.0}


def test_every_rule_counts_when_none_is_hard(instance: LineHaulInstance) -> None:
    """As before hard rules existed: any broken rule makes the plan infeasible."""
    evaluation = evaluate(instance, PARTIAL)

    assert (evaluation.feasible, evaluation.broken()) == (False, {"coverage": 120.0})


def test_a_constraint_of_your_own_can_be_hard(instance: LineHaulInstance) -> None:
    """The owned vehicle drives a degree of longitude, about 111 km, where 50 are allowed."""
    evaluation = evaluate(instance, PARTIAL, [MaxRouteKm(50.0)], hard={"max_route_km"})

    assert list(evaluation.broken()) == ["max_route_km"]


def test_a_hard_rule_that_does_not_exist_is_refused(instance: LineHaulInstance) -> None:
    with pytest.raises(ValueError, match=r"hard rules \['coverge'\] are not rules of this"):
        evaluate(instance, PARTIAL, hard={"coverge"})


def test_the_evaluation_shows_weighted_violations_of_a_feasible_plan(
    instance: LineHaulInstance,
) -> None:
    evaluation = evaluate(instance, PARTIAL, hard={"capacity"})

    assert repr(evaluation).endswith("feasible, violated={'coverage': 120.0})")


@pytest.mark.parametrize(
    ("hard", "feasible", "broken"),
    [({"coverage"}, False, 1), ({"capacity"}, True, 0), (set(), False, 1)],
    ids=["hard-broken", "hard-kept", "none-hard"],
)
def test_the_checker_judges_feasibility_by_the_hard_rules(
    instance: LineHaulInstance, hard: set[str], feasible: bool, broken: int
) -> None:
    """The same three cases as for the evaluator: one coverage violation in the plan."""
    report = check_plan(instance, PARTIAL, hard)

    assert (report.feasible, len(report.broken()), len(report.violations)) == (feasible, broken, 1)


def test_the_checker_report_says_feasible_and_lists_the_weighted_violation(
    instance: LineHaulInstance,
) -> None:
    report = check_plan(instance, PARTIAL, {"capacity"})

    assert str(report).splitlines() == [
        f"total cost {report.total_cost:.2f}, feasible, 1 violation(s)",
        "- coverage: demand D000003: 120 desi not carried",
    ]


@pytest.mark.parametrize("name", PLANNERS)
def test_a_hard_minimum_fill_leaves_the_small_demand_behind(name: str) -> None:
    """No vehicle leaves under its minimum fill; the 40 desi are the price, and it is allowed."""
    evaluation = solved(name, {"min_fill": "hard", "coverage": 1000, "other": 1000000})

    assert (evaluation.hard, evaluation.feasible) == ({"min_fill"}, True)
    assert evaluation.violated() == {"coverage": 40.0}


@pytest.mark.parametrize("name", PLANNERS)
def test_a_hard_coverage_sends_the_small_demand_in_an_underfilled_vehicle(name: str) -> None:
    """Every demand goes; the vehicle of the 40 desi is 80 short of its 120, which is allowed."""
    evaluation = solved(name, {"coverage": "hard", "min_fill": 1000, "other": 1000000})

    assert (evaluation.hard, evaluation.feasible) == ({"coverage"}, True)
    assert evaluation.violated() == {"min_fill": pytest.approx(80.0)}


@pytest.mark.parametrize("name", PLANNERS)
def test_hard_rules_that_cannot_all_hold_give_the_least_broken_plan(name: str) -> None:
    """Both rules are hard and weigh the same: leaving 40 desi breaks less than 80 desi short."""
    evaluation = solved(name, {"coverage": "hard", "min_fill": "hard", "other": 1000000})

    assert (evaluation.feasible, evaluation.broken()) == (False, {"coverage": 40.0})


def test_direct_shipping_is_judged_by_the_hard_rules_it_does_not_look_at() -> None:
    """It sends every load, so the hard minimum fill is broken and the plan is not feasible."""
    spec = parse_spec(choice({"min_fill": "hard", "coverage": 1000, "other": 1000000}))

    result = DirectShipping().solve(spec.compile(), spec.build_objective())

    assert (result.evaluation.feasible, list(result.evaluation.broken())) == (False, ["min_fill"])


def write_problem(folder: Path, penalties: dict[str, Any]) -> Path:
    """Write :func:`choice` as ``problem.yaml`` in ``folder`` and return its path."""
    path = folder / "problem.yaml"
    path.write_text(yaml.safe_dump(choice(penalties), sort_keys=False), encoding="utf-8")
    return path


def test_solve_exits_with_status_3_when_a_hard_rule_is_broken(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The plan is still written, and the message names the rule and the amount."""
    problem = write_problem(tmp_path, {"coverage": "hard", "min_fill": "hard", "other": 1000000})
    plan = tmp_path / "plan.csv"

    code = main(["solve", str(problem), "-o", str(plan)])

    assert code == 3
    assert "error: the plan breaks hard rules: coverage (40)" in capsys.readouterr().err
    assert plan.is_file()


def test_solve_succeeds_when_only_weighted_rules_are_broken(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    problem = write_problem(tmp_path, {"min_fill": "hard", "coverage": 1000, "other": 1000000})

    code = main(["solve", str(problem)])

    assert code == 0
    assert "feasible  violations {'coverage': 40.0}" in capsys.readouterr().out


def test_the_summary_file_lists_the_broken_hard_rules(tmp_path: Path) -> None:
    problem = write_problem(tmp_path, {"coverage": "hard", "min_fill": "hard", "other": 1000000})
    summary = tmp_path / "summary.json"

    main(["solve", str(problem), "--summary", str(summary)])

    numbers = json.loads(summary.read_text(encoding="utf-8"))
    assert (numbers["feasible"], numbers["hard_violations"]) == (False, {"coverage": 40.0})


@pytest.mark.parametrize(
    ("penalties", "code"),
    [
        ({"min_fill": "hard", "coverage": 1000, "other": 1000000}, 0),
        ({"coverage": "hard", "min_fill": 1000, "other": 1000000}, 1),
        ({"coverage": 1000, "min_fill": 1000, "other": 1000000}, 1),
    ],
    ids=["hard-kept", "hard-broken", "none-hard"],
)
def test_check_judges_a_plan_by_the_hard_rules_of_its_problem(
    tmp_path: Path, penalties: dict[str, Any], code: int
) -> None:
    """The plan leaves the 40 desi behind: fine unless coverage is hard or no rule is."""
    spec = parse_spec(choice({"min_fill": "hard", "coverage": 1000, "other": 1000000}))
    result = PLANNERS["exact"].solve(spec.compile(), spec.build_objective())
    plan = tmp_path / "plan.csv"
    write_plan(result.evaluation, plan)
    problem = write_problem(tmp_path, penalties)

    found = main(["check", str(problem), str(plan)])

    assert found == code
