import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from pirilib import ACOConfig, ACOSolver, SolveResult
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
from pirilib.spec import LineHaulInstance

SMALL = ACOConfig(ants=4, iterations=5, patience=3)


class Refusing:
    """A solver that supports nothing."""

    def solve(
        self,
        instance: LineHaulInstance,
        objective: Objective | None = None,
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


def test_instances_are_reproducible() -> None:
    family = FAMILIES.get("tiny")
    first, again, other = family.instance(1), family.instance(1), family.instance(2)
    assert first.document == again.document
    assert first.document != other.document
    assert first.name == "tiny/001"
    assert first.document["name"] == "tiny/001"
    settings = first.settings
    assert family.centers[0] <= settings["centers"] <= family.centers[1]
    assert len(first.instance.centers) == settings["centers"]
    assert len(first.instance.days) == settings["days"]
    assert len(family.instances()) == family.size
    assert [item.index for item in family.instances(2, start=3)] == [3, 4]
    reseeded = Family("tiny", "another seed", size=4, centers=(3, 5), days=(1, 1), seed=1)
    assert reseeded.instance(1).document != first.document


def test_families_vary_split_and_spot_limits() -> None:
    no_split = FAMILIES.get("no-split").instance(0)
    assert not no_split.instance.rules.split_demand
    limited = FAMILIES.get("spot-limits").instance(0)
    limit = math.ceil(limited.settings["spot_limit"] * limited.settings["demand_per_day"])
    assert limited.instance.spot_available.tolist() == [limit] * 3


def test_family_and_catalog_checks() -> None:
    with pytest.raises(ValueError, match="centers must be"):
        Family("bad", "", centers=(1, 4))
    with pytest.raises(ValueError, match="split needs at least one choice"):
        Family("bad", "", split=())
    with pytest.raises(ValueError, match="unknown family 'huge'; known: tiny"):
        FAMILIES.get("huge")
    with pytest.raises(ValueError, match="used twice"):
        FamilyCatalog((Family("a", ""), Family("a", "")))
    custom = FAMILIES.with_family(Family("tiny", "replaced", size=1))
    assert custom.get("tiny").description == "replaced"
    assert custom.names().count("tiny") == 1
    with pytest.raises(ValueError, match="unknown solvers"):
        SOLVERS.make(["direct", "magic"])


def test_benchmark_records_every_run(tmp_path: Path) -> None:
    instances = FAMILIES.get("tiny").instances(2)
    benchmark = Benchmark(
        {"direct": DirectShipping(), "aco": ACOSolver(SMALL), "refusing": Refusing()},
        seeds=(0, 1),
    )
    seen: list[Record] = []
    table = benchmark.run(instances, progress=seen.append)
    assert len(table) == len(seen) == 2 * 3 * 2
    assert table.solvers == ["direct", "aco", "refusing"]
    assert table.instances == ["tiny/000", "tiny/001"]
    refused = table.where(solver="refusing")
    assert {item.status for item in refused} == {"skipped"}
    assert all(item.message == "ValueError: not supported" for item in refused)
    for item in table.where(solver="direct"):
        assert item.solved
        assert item.feasible
        assert item.objective == pytest.approx(item.total_cost)
        assert item.seconds >= 0
    best = table.best_known()
    assert set(best) == set(table.instances)
    summary = table.summary()
    direct, aco = summary.row("direct"), summary.row("aco")
    assert direct.runs == direct.solved == 4
    assert direct.feasible_rate == 1.0
    assert min(direct.mean_gap, aco.mean_gap) >= 0
    assert max(direct.best_share, aco.best_share) > 0
    refusing = summary.row("refusing")
    assert (refusing.solved, refusing.best_share) == (0, 0.0)
    assert math.isnan(refusing.mean_gap)
    text = str(summary)
    assert text.splitlines()[0].startswith("solver")
    assert "refusing" in text

    path = table.to_csv(tmp_path / "results.csv")
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)
    again = ResultTable.from_csv(path)
    assert len(again) == len(table)
    for before, after in zip(table, again, strict=True):
        assert after.instance == before.instance
        assert after.violated == before.violated
        assert after.status == before.status
        if before.solved:
            assert after.objective == before.objective
        else:
            assert math.isnan(after.objective)
    summary_path = summary.to_csv(tmp_path / "summary.csv")
    assert summary_path.read_text(encoding="utf-8").startswith("solver,runs,solved,mean_gap")


def test_summary_numbers() -> None:
    table = ResultTable.of(
        [
            record("a", "one", 100.0),
            record("a", "two", 110.0, feasible=False, violated=("coverage",)),
            record("b", "one", 210.0, seconds=3.0),
            record("b", "two", 200.0),
            record("c", "one", math.nan, status="skipped"),
        ]
    )
    summary = table.summary()
    one, two = summary.row("one"), summary.row("two")
    assert one.mean_gap == pytest.approx(0.025)
    assert one.median_gap == pytest.approx(0.025)
    assert two.mean_gap == pytest.approx(0.05)
    assert (one.runs, one.solved, one.best_share) == (3, 2, pytest.approx(1 / 3))
    assert two.best_share == 0.5
    assert two.feasible_rate == 0.5
    assert one.mean_seconds == pytest.approx(2.0)
    lowered = table.summary(best_known={"a": 50.0})
    assert lowered.row("one").best_share == 0.0
    assert lowered.best_known["b"] == 200.0
    with pytest.raises(ValueError, match="unknown record fields"):
        table.where(colour="red")


def test_exact_solver_is_skipped_where_it_cannot_solve() -> None:
    pytest.importorskip("scipy")
    instance = FAMILIES.get("no-split").instance(0)
    table = Benchmark(SOLVERS.make(["direct", "exact"])).run([instance])
    direct, exact = table.records
    assert direct.solved
    assert exact.status == "skipped"
    assert "split_demand" in exact.message


def test_command_line(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["bench"]) == 0
    listing = capsys.readouterr().out
    assert "tiny" in listing
    assert "direct" in listing
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
    assert main(["bench", "huge"]) == 1
    assert "unknown family" in capsys.readouterr().err
