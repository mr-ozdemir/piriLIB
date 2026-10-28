import math
from collections.abc import Callable, Sequence
from operator import attrgetter
from pathlib import Path

import numpy as np
import pytest

from pirilib import ACOConfig, ACOSolver, SolveResult
from pirilib import objective as obj
from pirilib.bench import (
    COLUMNS,
    FAMILIES,
    SOLVERS,
    Benchmark,
    Family,
    FamilyCatalog,
    Record,
    ResultTable,
)
from pirilib.cli import main
from pirilib.constraints import Constraint
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
