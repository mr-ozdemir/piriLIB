import dataclasses
import datetime as dt
import json
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from operator import attrgetter
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import pirilib
from pirilib import ACOConfig, ACOSolver, Evaluation, SolveResult
from pirilib import objective as obj
from pirilib.bench import (
    COLUMNS,
    FAMILIES,
    SOLVERS,
    Benchmark,
    Change,
    Environment,
    Family,
    FamilyCatalog,
    InstanceInfo,
    Record,
    Report,
    ResultTable,
    SolverInfo,
    Spread,
    compare,
)
from pirilib.cli import main
from pirilib.constraints import Constraint, MaxRouteKm
from pirilib.objective import Objective
from pirilib.solvers.direct import DirectShipping
from pirilib.spec import LineHaulInstance, parse_spec

SMALL = ACOConfig(ants=4, iterations=5, patience=3)
GOAL = obj.total_cost() + 1e6 * obj.violations()
SEEDS = (0, 1)
"""Solver seeds of the benchmark run: 2 instances x 3 solvers x 2 seeds = 12 records."""


class Refusing:
    """A solver that supports nothing."""

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        raise ValueError("not supported")


def record(instance: str, solver: str, objective: float, **values: object) -> Record:
    fields: dict[str, object] = {
        "family": "f",
        "instance": instance,
        "index": 0,
        "instance_seed": 1,
        "solver": solver,
        "seed": 0,
        "status": "ok",
        "objective": objective,
        "total_cost": objective,
        "feasible": True,
        "violated": (),
        "seconds": 1.0,
        "dispatches": 3,
    }
    fields.update(values)
    return Record(**fields)


@pytest.fixture(scope="module")
def run() -> tuple[ResultTable, list[Record]]:
    """The tiny family's first two instances under a direct, a colony and a refusing solver."""
    benchmark = Benchmark(
        {"direct": DirectShipping(), "aco": ACOSolver(SMALL), "refusing": Refusing()},
        objective=GOAL,
        seeds=SEEDS,
    )
    seen: list[Record] = []
    table = benchmark.run(FAMILIES.get("tiny").instances(2), progress=seen.append)
    return table, seen


def test_an_instance_is_the_same_for_the_same_index() -> None:
    family = FAMILIES.get("tiny")

    first, again = family.instance(1), family.instance(1)

    assert first.document == again.document


def test_instances_differ_between_indices() -> None:
    family = FAMILIES.get("tiny")

    assert family.instance(1).document != family.instance(2).document


def test_an_instance_is_named_after_its_family_and_index() -> None:
    first = FAMILIES.get("tiny").instance(1)

    assert (first.name, first.document["name"]) == ("tiny/001", "tiny/001")


def test_an_instance_follows_its_settings() -> None:
    family = FAMILIES.get("tiny")
    first = family.instance(1)

    settings = first.settings

    assert settings["centers"] in set(range(family.centers[0], family.centers[1] + 1))
    assert len(first.instance.centers) == settings["centers"]
    assert len(first.instance.days) == settings["days"]


def test_a_family_lists_its_instances() -> None:
    family = FAMILIES.get("tiny")

    assert len(family.instances()) == family.size
    assert list(map(attrgetter("index"), family.instances(2, start=3))) == [3, 4]


def test_another_family_seed_gives_other_instances() -> None:
    reseeded = Family("tiny", "another seed", size=4, centers=(3, 5), days=(1, 1), seed=1)

    assert reseeded.instance(1).document != FAMILIES.get("tiny").instance(1).document


def test_instances_carry_an_objective() -> None:
    item = FAMILIES.get("tiny").instance(0)

    goal = parse_spec(item.document).build_objective()

    assert item.document["objective"] == {"cost": 1.0, "penalties": {"coverage": 1e7, "other": 1e6}}
    assert goal.linear_form() is not None


def test_benchmark_needs_an_objective_argument() -> None:
    with pytest.raises(TypeError):
        Benchmark({"direct": DirectShipping()})


def test_benchmark_rejects_a_missing_objective() -> None:
    with pytest.raises(ValueError, match="Benchmark needs an objective"):
        Benchmark({"direct": DirectShipping()}, objective=None)


def test_the_no_split_family_does_not_split_demand() -> None:
    no_split = FAMILIES.get("no-split").instance(0)

    assert not no_split.instance.rules.split_demand


def test_the_spot_limit_family_limits_every_offer() -> None:
    limited = FAMILIES.get("spot-limits").instance(0)

    limit = math.ceil(limited.settings["spot_limit"] * limited.settings["demand_per_day"])

    assert limited.instance.spot_available.tolist() == [limit] * 3


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Family("bad", "", centers=(1, 4)), "centers must be"),
        (lambda: Family("bad", "", split=()), "split needs at least one choice"),
        (lambda: FAMILIES.get("huge"), "unknown family 'huge'; known: tiny"),
        (lambda: FamilyCatalog((Family("a", ""), Family("a", ""))), "used twice"),
        (lambda: SOLVERS.make(["direct", "magic"]), "unknown solvers"),
    ],
    ids=["one-center", "no-split-choice", "unknown-family", "duplicate-family", "unknown-solver"],
)
def test_family_and_catalog_checks(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_a_catalog_family_can_be_replaced() -> None:
    custom = FAMILIES.with_family(Family("tiny", "replaced", size=1))

    assert custom.get("tiny").description == "replaced"
    assert custom.names().count("tiny") == 1


def test_benchmark_records_every_run(run: tuple[ResultTable, list[Record]]) -> None:
    table, seen = run

    assert (len(table), len(seen)) == (12, 12)
    assert table.solvers == ["direct", "aco", "refusing"]
    assert table.instances == ["tiny/000", "tiny/001"]


def test_a_refusing_solver_is_recorded_as_skipped(run: tuple[ResultTable, list[Record]]) -> None:
    table, _ = run

    refused = table.where(solver="refusing")

    assert set(map(attrgetter("status"), refused)) == {"skipped"}
    assert set(map(attrgetter("message"), refused)) == {"ValueError: not supported"}


def test_direct_runs_are_solved_and_feasible(run: tuple[ResultTable, list[Record]]) -> None:
    table, _ = run

    direct = table.where(solver="direct")

    assert set(map(attrgetter("solved"), direct)) == {True}
    assert set(map(attrgetter("feasible"), direct)) == {True}
    np.testing.assert_allclose(
        list(map(attrgetter("objective"), direct)), list(map(attrgetter("total_cost"), direct))
    )
    assert min(map(attrgetter("seconds"), direct)) >= 0


def test_the_best_known_covers_every_instance(run: tuple[ResultTable, list[Record]]) -> None:
    table, _ = run

    assert set(table.best_known()) == set(table.instances)


def test_the_summary_counts_runs_and_gaps(run: tuple[ResultTable, list[Record]]) -> None:
    table, _ = run

    summary = table.summary()

    direct, aco = summary.row("direct"), summary.row("aco")
    assert (direct.runs, direct.solved, direct.feasible_rate) == (4, 4, 1.0)
    assert min(direct.mean_gap, aco.mean_gap) >= 0
    assert max(direct.best_share, aco.best_share) > 0


def test_a_solver_with_no_solved_run_has_no_gap(run: tuple[ResultTable, list[Record]]) -> None:
    table, _ = run

    refusing = table.summary().row("refusing")

    assert (refusing.solved, refusing.best_share) == (0, 0.0)
    assert math.isnan(refusing.mean_gap)


def test_the_summary_prints_a_table(run: tuple[ResultTable, list[Record]]) -> None:
    table, _ = run

    text = str(table.summary())

    assert text.splitlines()[0].startswith("solver")
    assert "refusing" in text


def test_results_round_trip_through_csv(
    run: tuple[ResultTable, list[Record]], tmp_path: Path
) -> None:
    """Unsolved runs have no objective; NaN is written as an empty field and read back."""
    table, _ = run
    path = table.to_csv(tmp_path / "results.csv")

    again = ResultTable.from_csv(path)

    fields = attrgetter("instance", "violated", "status", "solved")
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)
    assert list(map(fields, again)) == list(map(fields, table))
    np.testing.assert_array_equal(
        list(map(attrgetter("objective"), again)), list(map(attrgetter("objective"), table))
    )


def test_the_summary_writes_csv(run: tuple[ResultTable, list[Record]], tmp_path: Path) -> None:
    table, _ = run

    path = table.summary().to_csv(tmp_path / "summary.csv")

    assert path.read_text(encoding="utf-8").startswith("solver,runs,solved,mean_gap")


SMALL_TABLE = ResultTable.of(
    [
        record("a", "one", 100.0),
        record("a", "two", 110.0, feasible=False, violated=("coverage",)),
        record("b", "one", 210.0, seconds=3.0),
        record("b", "two", 200.0),
        record("c", "one", math.nan, status="skipped"),
    ]
)
"""Two solvers on three instances: "one" is best on a, 5 % off on b and skipped on c."""


def test_summary_numbers() -> None:
    summary = SMALL_TABLE.summary()

    one, two = summary.row("one"), summary.row("two")
    assert one.mean_gap == pytest.approx(0.025)
    assert one.median_gap == pytest.approx(0.025)
    assert two.mean_gap == pytest.approx(0.05)
    assert (one.runs, one.solved, one.best_share) == (3, 2, pytest.approx(1 / 3))
    assert two.best_share == 0.5
    assert two.feasible_rate == 0.5
    assert one.mean_seconds == pytest.approx(2.0)


def test_a_given_best_known_lowers_the_best_share() -> None:
    lowered = SMALL_TABLE.summary(best_known={"a": 50.0})

    assert lowered.row("one").best_share == 0.0
    assert lowered.best_known["b"] == 200.0


def test_records_are_filtered_by_known_fields_only() -> None:
    with pytest.raises(ValueError, match="unknown record fields"):
        SMALL_TABLE.where(colour="red")


def test_exact_solver_is_skipped_where_it_cannot_solve() -> None:
    pytest.importorskip("scipy")
    instance = FAMILIES.get("no-split").instance(0)

    direct, exact = Benchmark(SOLVERS.make(["direct", "exact"]), objective=GOAL).run([instance])

    assert direct.solved
    assert exact.status == "skipped"
    assert "split_demand" in exact.message


def test_the_command_line_lists_families_and_solvers(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench"])

    listing = capsys.readouterr().out
    assert code == 0
    assert "tiny" in listing
    assert "direct" in listing


def test_the_command_line_runs_a_family(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    results, summary = tmp_path / "results.csv", tmp_path / "summary.csv"

    code = main(
        [
            "bench",
            "tiny",
            "--solvers",
            "direct",
            "--count",
            "2",
            "-o",
            str(results),
            "--summary",
            str(summary),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "tiny/001  direct" in out
    assert len(ResultTable.from_csv(results)) == 2
    assert summary.read_text(encoding="utf-8").splitlines()[1].startswith("direct,2,2,0.0,0.0,1.0")


def test_the_command_line_rejects_an_unknown_family(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench", "huge"])

    assert code == 1
    assert "unknown family" in capsys.readouterr().err


SPREAD_TABLE = ResultTable.of(
    [
        record("a", "one", 50.0, seed=0, seconds=5.0),
        record("a", "one", 10.0, seed=1, seconds=1.0),
        record("a", "one", 40.0, seed=2, seconds=4.0),
        record("a", "one", 20.0, seed=3, seconds=2.0),
        record("a", "one", 30.0, seed=4, seconds=3.0),
        record("a", "two", 8.0),
        record("a", "three", math.nan, status="skipped"),
        record("b", "three", math.nan, status="skipped"),
    ]
)
"""On instance a, "one" has five runs from 10 to 50 and "two" one run of 8; "three" is skipped."""


def test_the_spread_gives_the_quartiles_of_the_runs() -> None:
    """Five runs 10..50: quartiles 20, 30, 40; the median is (30 - 8) / 8 = 2.75 above the best."""
    spread = SPREAD_TABLE.spread()

    assert spread[0] == Spread(
        instance="a",
        solver="one",
        runs=5,
        solved=5,
        best=10.0,
        lower_quartile=20.0,
        median=30.0,
        upper_quartile=40.0,
        worst=50.0,
        gap=2.75,
        median_seconds=3.0,
    )


def test_one_run_is_its_own_spread() -> None:
    """With a single solved run, the best, the quartiles and the worst are that run."""
    spread = SPREAD_TABLE.spread()

    assert spread[1] == Spread(
        instance="a",
        solver="two",
        runs=1,
        solved=1,
        best=8.0,
        lower_quartile=8.0,
        median=8.0,
        upper_quartile=8.0,
        worst=8.0,
        gap=0.0,
        median_seconds=1.0,
    )


def test_quartiles_are_interpolated_between_runs() -> None:
    """Two runs 10 and 20: the quartiles lie a quarter, a half and three quarters of the way."""
    table = ResultTable.of([record("a", "one", 10.0), record("a", "one", 20.0, seed=1)])

    row = table.spread()[0]

    assert (row.lower_quartile, row.median, row.upper_quartile) == (12.5, 15.0, 17.5)


def test_a_solver_without_a_solved_run_has_no_spread() -> None:
    """The run is counted, and every number of the spread is NaN."""
    row = SPREAD_TABLE.spread()[2]

    assert (row.runs, row.solved) == (1, 0)
    assert np.isnan(
        [
            row.best,
            row.lower_quartile,
            row.median,
            row.upper_quartile,
            row.worst,
            row.gap,
            row.median_seconds,
        ]
    ).all()


def test_the_spread_has_one_row_per_instance_and_solver() -> None:
    """Rows follow the order in which the pairs first appear in the table."""
    spread = SPREAD_TABLE.spread()

    assert list(map(attrgetter("instance", "solver"), spread)) == [
        ("a", "one"),
        ("a", "two"),
        ("a", "three"),
        ("b", "three"),
    ]


def test_a_given_best_known_is_the_reference_of_the_spread_gap() -> None:
    """Against a best known objective of 5, the median 30 is (30 - 5) / 5 = 5 above it."""
    spread = SPREAD_TABLE.spread(best_known={"a": 5.0})

    assert spread[0].gap == 5.0


class Odd:
    """A solver whose settings are not all plain values, and one that is private."""

    def __init__(self) -> None:
        self.limit = math.inf
        self.steps = (1, 2)
        self.names = {"a": 0.5}
        self.share = Fraction(1, 3)
        self._cache = 7

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        raise ValueError("not supported")


class Bare:
    """A solver without attributes."""

    __slots__ = ()

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective,
        constraints: Sequence[Constraint] = (),
        *,
        seed: int = 0,
    ) -> SolveResult:
        raise ValueError("not supported")


def unbounded(evaluation: Evaluation) -> float:
    """An objective that is a plain function, and infinite for every plan."""
    return math.inf


@pytest.fixture(scope="module")
def report() -> Report:
    """The tiny family's first two instances under a direct, a colony and a refusing solver."""
    benchmark = Benchmark(
        {"direct": DirectShipping(), "aco": ACOSolver(SMALL), "refusing": Refusing()},
        objective=GOAL,
        constraints=[MaxRouteKm(5000.0)],
        seeds=SEEDS,
    )
    return benchmark.report(FAMILIES.get("tiny").instances(2))


def test_a_report_holds_one_record_per_run(report: Report) -> None:
    """2 instances x 3 solvers x 2 seeds."""
    assert len(report.table) == 12


def test_a_report_summarizes_its_records(report: Report) -> None:
    """The direct solver ran on 2 instances with 2 seeds and solved all four."""
    row = report.summary().row("direct")

    assert (row.runs, row.solved) == (4, 4)


def test_a_report_passes_every_record_to_progress() -> None:
    """One instance, one solver, one seed: progress sees the one record of the table."""
    seen: list[Record] = []
    benchmark = Benchmark({"direct": DirectShipping()}, objective=GOAL)

    made = benchmark.report(FAMILIES.get("tiny").instances(1), progress=seen.append)

    assert seen == list(made.table)


def test_a_report_lists_the_solvers_in_running_order(report: Report) -> None:
    assert list(report.solvers) == ["direct", "aco", "refusing"]


def test_a_report_keeps_the_settings_of_a_solver(report: Report) -> None:
    """The colony's configuration is written field by field."""
    assert report.solvers["aco"] == SolverInfo(
        type="pirilib.solvers.aco.ACOSolver",
        settings={
            "config": {
                "ants": 4,
                "iterations": 5,
                "patience": 3,
                "alpha": 1.0,
                "beta": 2.0,
                "evaporation": 0.1,
                "exploitation": 0.0,
                "tau_min": 0.01,
                "update": "iteration_best",
                "local_search": True,
                "exact_nodes": 0,
                "time_limit": None,
            }
        },
    )


def test_solver_settings_that_are_not_plain_values_are_written_as_text() -> None:
    """Infinity and a fraction become their ``repr``; a private attribute is left out."""
    made = Benchmark({"odd": Odd()}, objective=GOAL).report([])

    assert made.solvers["odd"].settings == {
        "limit": "inf",
        "steps": [1, 2],
        "names": {"a": 0.5},
        "share": "Fraction(1, 3)",
    }


def test_a_solver_without_attributes_has_no_settings() -> None:
    made = Benchmark({"bare": Bare()}, objective=GOAL).report([])

    assert made.solvers["bare"].settings == {}


def test_a_report_names_the_terms_of_the_objective(report: Report) -> None:
    assert report.objective == (("total_cost", 1.0), ("violations", 1e6))


def test_an_objective_of_one_term_is_reported_as_that_term() -> None:
    made = Benchmark({"direct": DirectShipping()}, objective=3.0 * obj.total_cost()).report([])

    assert made.objective == (("total_cost", 3.0),)


def test_an_objective_function_is_reported_by_its_name() -> None:
    made = Benchmark({"direct": DirectShipping()}, objective=unbounded).report([])

    assert made.objective == (("unbounded", 1.0),)


def test_a_report_names_the_constraints_and_seeds(report: Report) -> None:
    assert (report.constraints, report.seeds) == (("max_route_km",), SEEDS)


def test_a_report_says_where_an_instance_comes_from(report: Report) -> None:
    """The first tiny instance: its generator seed and settings, 5 centers, 1 day, 5 demand rows."""
    assert report.instances[0] == InstanceInfo(
        name="tiny/000",
        family="tiny",
        index=0,
        seed=461967004,
        settings={
            "centers": 5,
            "days": 1,
            "demand_per_day": 5,
            "owned_lines": 1,
            "min_fill": 0.0,
            "max_stretch": 1.3,
            "split": True,
            "spot_limit": None,
            "seed": 461967004,
        },
        centers=5,
        days=1,
        demands=5,
    )


def test_a_report_lists_the_instances_in_running_order(report: Report) -> None:
    assert list(map(attrgetter("name"), report.instances)) == ["tiny/000", "tiny/001"]


def test_a_report_records_the_library_and_package_versions(report: Report) -> None:
    environment = report.environment

    assert environment.pirilib == pirilib.__version__
    assert environment.packages["numpy"] == np.__version__


def test_packages_that_are_not_installed_are_left_out() -> None:
    environment = Environment.current(packages=("numpy", "no-such-package"))

    assert environment.packages == {"numpy": np.__version__}


def test_a_report_is_dated_in_utc(report: Report) -> None:
    assert report.created.utcoffset() == dt.timedelta(0)


def test_a_report_round_trips_through_json(report: Report, tmp_path: Path) -> None:
    """The conditions come back equal; records are compared as rows, since NaN is not equal."""
    path = report.to_json(tmp_path / "report.json")

    again = Report.from_json(path)

    conditions = attrgetter(
        "created", "environment", "solvers", "objective", "constraints", "seeds", "instances"
    )
    assert conditions(again) == conditions(report)
    assert list(map(Record.row, again.table)) == list(map(Record.row, report.table))


def test_a_report_file_names_its_format(report: Report, tmp_path: Path) -> None:
    path = report.to_json(tmp_path / "report.json")

    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["format"] == "pirilib/bench-report@1"


def test_a_report_file_holds_the_spread(report: Report, tmp_path: Path) -> None:
    """2 instances x 3 solvers; the refusing solver has no median, written as null."""
    path = report.to_json(tmp_path / "report.json")

    document = json.loads(path.read_text(encoding="utf-8"))

    assert len(document["spread"]) == 6
    assert document["spread"][2]["solver"] == "refusing"
    assert document["spread"][2]["median"] is None


def test_a_skipped_run_has_no_objective_in_the_report_file(report: Report, tmp_path: Path) -> None:
    """Records run instance by instance, solver by solver: the fifth is the refusing solver's."""
    path = report.to_json(tmp_path / "report.json")

    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["records"][4]["status"] == "skipped"
    assert document["records"][4]["objective"] is None


def test_an_infinite_objective_survives_the_report_file(tmp_path: Path) -> None:
    """JSON has no infinity; it is written as text and read back as the number."""
    benchmark = Benchmark({"direct": DirectShipping()}, objective=unbounded)
    path = benchmark.report(FAMILIES.get("tiny").instances(1)).to_json(tmp_path / "report.json")

    again = Report.from_json(path)

    assert again.table.records[0].objective == math.inf


@pytest.mark.parametrize("text", ['{"format": "pirilib/bench-report@2"}', "[]"], ids=["v2", "list"])
def test_a_file_of_another_format_is_not_read_as_a_report(tmp_path: Path, text: str) -> None:
    path = tmp_path / "other.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="not a benchmark report of format pirilib/bench-report@1"):
        Report.from_json(path)


def test_the_command_line_writes_a_report(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "report.json"

    code = main(["bench", "tiny", "--solvers", "direct", "--count", "1", "--report", str(path)])

    out = capsys.readouterr().out
    assert code == 0
    assert f"report written to {path}" in out
    assert len(Report.from_json(path).table) == 1


def report_of(*records: Record, seeds: tuple[int, ...] = (0,), version: str = "1.0") -> Report:
    """A report of ``records`` made under fixed conditions, so two of them compare alike."""
    environment = Environment(
        pirilib=version,
        python="3.14.0",
        platform="test",
        machine="x86_64",
        processor="test",
        cpus=4,
        packages=MappingProxyType({"numpy": "2.0"}),
    )
    solvers = {
        name: SolverInfo(type="tests.Solver", settings=MappingProxyType({"ants": 4}))
        for name in dict.fromkeys(item.solver for item in records)
    }
    return Report(
        created=dt.datetime(2026, 1, 5, tzinfo=dt.UTC),
        environment=environment,
        solvers=MappingProxyType(solvers),
        objective=(("total_cost", 1.0),),
        constraints=(),
        seeds=seeds,
        instances=(),
        table=ResultTable.of(records),
    )


BASELINE = report_of(
    record("i1", "aco", 100.0),
    record("i1", "direct", 150.0),
    record("i2", "aco", 200.0),
)
"""Three pairs: the colony on two instances, direct shipping on the first."""

NEW_RUN = report_of(
    record("i1", "aco", 90.0, seconds=2.0),
    record("i1", "direct", 150.0),
    record("i2", "aco", 210.0),
)
"""The same pairs later: one better by 10 %, one unchanged, one worse by 5 %."""


def test_a_comparison_has_one_change_per_pair_of_both_runs() -> None:
    comparison = compare(BASELINE, NEW_RUN)

    assert [(item.instance, item.solver) for item in comparison.changes] == [
        ("i1", "aco"),
        ("i1", "direct"),
        ("i2", "aco"),
    ]


def test_a_change_holds_both_medians_and_their_relative_difference() -> None:
    comparison = compare(BASELINE, NEW_RUN)

    assert comparison.changes[0] == Change(
        instance="i1",
        solver="aco",
        before=100.0,
        after=90.0,
        change=pytest.approx(-0.10),
        seconds_before=1.0,
        seconds_after=2.0,
    )


def test_a_pair_whose_median_rose_is_worse() -> None:
    comparison = compare(BASELINE, NEW_RUN)

    assert [(item.instance, item.solver) for item in comparison.worse()] == [("i2", "aco")]


def test_a_pair_whose_median_fell_is_better() -> None:
    comparison = compare(BASELINE, NEW_RUN)

    assert [(item.instance, item.solver) for item in comparison.better()] == [("i1", "aco")]


@pytest.mark.parametrize(
    ("tolerance", "worse"),
    [
        pytest.param(0.04, 1, id="below-the-rise"),
        pytest.param(0.05, 0, id="at-the-rise"),
        pytest.param(0.5, 0, id="above-the-rise"),
    ],
)
def test_a_rise_within_the_tolerance_is_not_worse(tolerance: float, worse: int) -> None:
    """The colony's median on ``i2`` rose from 200 to 210, by 5 %."""
    comparison = compare(BASELINE, NEW_RUN)

    assert len(comparison.worse(tolerance)) == worse


def test_the_median_over_the_seeds_is_what_is_compared() -> None:
    """Medians 100 and 110: one seed of the new run is far better, the median is worse."""
    before = report_of(
        *(record("i1", "aco", value, seed=seed) for seed, value in enumerate((90.0, 100.0, 120.0)))
    )
    after = report_of(
        *(record("i1", "aco", value, seed=seed) for seed, value in enumerate((10.0, 110.0, 130.0)))
    )

    comparison = compare(before, after)

    assert (comparison.changes[0].before, comparison.changes[0].after) == (100.0, 110.0)


def test_a_pair_the_new_run_did_not_solve_is_worse() -> None:
    after = report_of(record("i1", "aco", math.nan, status="skipped"))

    comparison = compare(report_of(record("i1", "aco", 100.0)), after)

    assert [(item.instance, item.solver) for item in comparison.worse()] == [("i1", "aco")]


def test_pairs_of_one_run_alone_are_listed_as_not_compared() -> None:
    before = report_of(record("i1", "aco", 100.0), record("i2", "aco", 200.0))
    after = report_of(record("i1", "aco", 100.0), record("i1", "direct", 150.0))

    comparison = compare(before, after)

    assert comparison.missing == (
        "i2 with aco is only in the baseline",
        "i1 with direct is only in the new run",
    )


def test_runs_under_the_same_conditions_have_no_differences() -> None:
    comparison = compare(BASELINE, NEW_RUN)

    assert comparison.differences == ()


def test_a_comparison_says_which_conditions_differ() -> None:
    after = report_of(record("i1", "aco", 100.0), seeds=(0, 1), version="1.1")

    comparison = compare(report_of(record("i1", "aco", 100.0)), after)

    assert comparison.differences == ("pirilib: 1.0 -> 1.1", "seeds: [0] -> [0, 1]")


def test_a_comparison_says_which_setting_of_a_solver_changed() -> None:
    before = report_of(record("i1", "aco", 100.0))
    settings = MappingProxyType({"ants": 8, "iterations": 5})
    after = dataclasses.replace(
        before, solvers=MappingProxyType({"aco": SolverInfo("tests.Solver", settings)})
    )

    comparison = compare(before, after)

    assert comparison.differences == (
        "solver aco ants: 4 -> 8",
        "solver aco iterations: absent -> 5",
    )


def test_a_comparison_looks_into_the_nested_settings_of_a_solver() -> None:
    """The colony keeps its settings in ``config``; only the one that changed is named."""
    before = report_of(record("i1", "aco", 100.0))
    old = MappingProxyType({"config": {"ants": 16, "iterations": 60}})
    new = MappingProxyType({"config": {"ants": 16, "iterations": 30}})
    first = dataclasses.replace(
        before, solvers=MappingProxyType({"aco": SolverInfo("tests.Solver", old)})
    )
    second = dataclasses.replace(
        before, solvers=MappingProxyType({"aco": SolverInfo("tests.Solver", new)})
    )

    comparison = compare(first, second)

    assert comparison.differences == ("solver aco config iterations: 60 -> 30",)


def test_a_comparison_describes_itself_in_words() -> None:
    comparison = compare(BASELINE, NEW_RUN)

    assert comparison.describe(0.01).splitlines() == [
        "3 pairs of instance and solver: 1 worse, 1 better, 1 within 1.00%",
        "worse:",
        "  i2  aco        200.00 -> 210.00  +5.00%",
        "better:",
        "  i1  aco        100.00 -> 90.00  -10.00%",
    ]


def test_two_runs_of_the_same_report_compare_as_unchanged(report: Report) -> None:
    comparison = compare(report, report)

    assert (comparison.worse(), comparison.better(), comparison.differences) == ((), (), ())


def test_the_command_line_passes_a_run_that_matches_its_baseline(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Direct shipping is deterministic: the second run is the first one again."""
    path = tmp_path / "baseline.json"
    main(["bench", "tiny", "--solvers", "direct", "--count", "2", "--report", str(path)])

    code = main(["bench", "tiny", "--solvers", "direct", "--count", "2", "--baseline", str(path)])

    out = capsys.readouterr().out
    assert (code, "2 pairs of instance and solver: 0 worse, 0 better" in out) == (0, True)


def test_the_command_line_fails_a_run_that_is_worse_than_its_baseline(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The baseline claims half the objective the solver reaches."""
    path = tmp_path / "baseline.json"
    main(["bench", "tiny", "--solvers", "direct", "--count", "1", "--report", str(path)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["records"][0]["objective"] /= 2
    path.write_text(json.dumps(document), encoding="utf-8")

    code = main(["bench", "tiny", "--solvers", "direct", "--count", "1", "--baseline", str(path)])

    out = capsys.readouterr().out
    assert (code, "1 worse" in out) == (1, True)


def test_the_command_line_lets_a_rise_within_the_tolerance_pass(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "baseline.json"
    main(["bench", "tiny", "--solvers", "direct", "--count", "1", "--report", str(path)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["records"][0]["objective"] /= 2
    path.write_text(json.dumps(document), encoding="utf-8")
    arguments = ["bench", "tiny", "--solvers", "direct", "--count", "1", "--baseline", str(path)]

    code = main([*arguments, "--tolerance", "1.5"])

    capsys.readouterr()
    assert code == 0


def test_the_command_line_rejects_a_negative_tolerance(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["bench", "tiny", "--solvers", "direct", "--tolerance", "-1"])

    err = capsys.readouterr().err
    assert (code, "--tolerance must not be negative" in err) == (1, True)
