"""Benchmark results: one record per run, a table of them and its summary per solver."""

import csv
import math
import os
import statistics
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal

Status = Literal["ok", "skipped"]

_RELATIVE: Final = 1e-9
_ABSOLUTE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class Record:
    """One run of one solver on one instance.

    Attributes:
        instance: Instance name, ``family/index``
        index: Position of the instance in its family
        instance_seed: Seed the instance was generated with
        seed: Seed of the run
        status: ``ok``, or ``skipped`` when the solver refused the instance
        objective: Objective of the plan, recomputed from its evaluation; NaN when skipped
        total_cost: NaN when skipped
        feasible: No constraint is violated
        violated: Names of the violated constraints
        seconds: Wall-clock time of the solver call
        dispatches: Vehicles in the plan
        message: Why the run was skipped
    """

    family: str
    instance: str
    index: int
    instance_seed: int
    solver: str
    seed: int
    status: Status
    objective: float
    total_cost: float
    feasible: bool
    violated: tuple[str, ...]
    seconds: float
    dispatches: int
    message: str = ""

    @property
    def solved(self) -> bool:
        """True when the solver returned a plan (status ``ok``)."""
        return self.status == "ok"

    def row(self) -> list[str]:
        """The record as CSV cells."""
        cells: list[str] = []
        for value in astuple(self):
            if isinstance(value, tuple):
                cells.append("|".join(value))
            elif isinstance(value, bool):
                cells.append("true" if value else "false")
            elif isinstance(value, float):
                cells.append("" if math.isnan(value) else repr(value))
            else:
                cells.append(str(value))
        return cells

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "Record":
        """Read a record back from :meth:`row` cells keyed by column."""

        def number(text: str) -> float:
            return float(text) if text else math.nan

        status = row["status"]
        if status not in ("ok", "skipped"):
            raise ValueError(f"unknown status {status!r}")
        return cls(
            family=row["family"],
            instance=row["instance"],
            index=int(row["index"]),
            instance_seed=int(row["instance_seed"]),
            solver=row["solver"],
            seed=int(row["seed"]),
            status="ok" if status == "ok" else "skipped",
            objective=number(row["objective"]),
            total_cost=number(row["total_cost"]),
            feasible=row["feasible"] == "true",
            violated=tuple(name for name in row["violated"].split("|") if name),
            seconds=number(row["seconds"]),
            dispatches=int(row["dispatches"]),
            message=row.get("message", ""),
        )


COLUMNS: Final = tuple(item.name for item in fields(Record))
"""Columns of result CSV files, in order."""


@dataclass(frozen=True, slots=True)
class SolverSummary:
    """The headline numbers of one solver over a result table.

    Gaps are relative to the best objective any run reached on the same
    instance: ``(objective - best) / max(|best|, 1)``.

    Attributes:
        runs: Runs, solved or skipped
        solved: Runs the solver did not refuse
        mean_gap: Over solved runs; NaN when none
        median_gap: Over solved runs; NaN when none
        best_share: Share of all runs that reached the best known objective
        feasible_rate: Share of solved runs without violations; NaN when none
        mean_seconds: Over solved runs; NaN when none
    """

    solver: str
    runs: int
    solved: int
    mean_gap: float
    median_gap: float
    best_share: float
    feasible_rate: float
    mean_seconds: float


SUMMARY_COLUMNS: Final = tuple(item.name for item in fields(SolverSummary))
"""Columns of summary CSV files, in order."""


@dataclass(frozen=True, slots=True)
class Summary:
    """One :class:`SolverSummary` per solver, in the order the solvers first appear.

    Attributes:
        best_known: The best objective per instance that the gaps refer to
    """

    rows: tuple[SolverSummary, ...]
    best_known: Mapping[str, float]

    def row(self, solver: str) -> SolverSummary:
        """The summary of ``solver``; :class:`KeyError` if it is not in the table."""
        for item in self.rows:
            if item.solver == solver:
                return item
        raise KeyError(solver)

    def to_csv(self, path: str | os.PathLike[str]) -> Path:
        """Write the summary as CSV to ``path`` (empty cells for NaN) and return the path."""
        target = Path(path)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_COLUMNS)
            for item in self.rows:
                writer.writerow(
                    [
                        "" if isinstance(value, float) and math.isnan(value) else value
                        for value in astuple(item)
                    ]
                )
        return target

    def __str__(self) -> str:
        header = ("solver", "runs", "solved", "mean gap", "median gap", "best", "feasible", "sec")
        lines = [
            (
                item.solver,
                str(item.runs),
                str(item.solved),
                _percent(item.mean_gap),
                _percent(item.median_gap),
                _percent(item.best_share),
                _percent(item.feasible_rate),
                "-" if math.isnan(item.mean_seconds) else f"{item.mean_seconds:.2f}",
            )
            for item in self.rows
        ]
        widths = [max(len(row[column]) for row in [header, *lines]) for column in range(8)]
        text = [
            "  ".join(
                cell.ljust(width) if column == 0 else cell.rjust(width)
                for column, (cell, width) in enumerate(zip(row, widths, strict=True))
            )
            for row in [header, *lines]
        ]
        return "\n".join(text)


@dataclass(frozen=True, slots=True)
class ResultTable:
    """The records of a benchmark run."""

    records: tuple[Record, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __add__(self, other: "ResultTable") -> "ResultTable":
        return ResultTable((*self.records, *other.records))

    @property
    def solvers(self) -> list[str]:
        """Solver names, in the order they first appear."""
        return list(dict.fromkeys(record.solver for record in self.records))

    @property
    def instances(self) -> list[str]:
        """Instance names, in the order they first appear."""
        return list(dict.fromkeys(record.instance for record in self.records))

    def best_known(self) -> dict[str, float]:
        """The lowest objective of any solved run, per instance with one."""
        best: dict[str, float] = {}
        for record in self.records:
            if record.solved and math.isfinite(record.objective):
                best[record.instance] = min(best.get(record.instance, math.inf), record.objective)
        return best

    def gap(self, record: Record, best: Mapping[str, float]) -> float:
        """Relative gap of ``record`` to the best objective of its instance; NaN if unknown."""
        if not record.solved or record.instance not in best:
            return math.nan
        reference = best[record.instance]
        return (record.objective - reference) / max(abs(reference), 1.0)

    def summary(self, best_known: Mapping[str, float] | None = None) -> Summary:
        """Per solver: gaps, best results, feasibility and time; see :class:`SolverSummary`.

        ``best_known`` overrides or adds best objectives (for example from
        earlier runs); the best of this table is used for the other instances,
        and an override is never above what this table reached.
        """
        best = self.best_known()
        for name, value in (best_known or {}).items():
            best[name] = min(value, best.get(name, math.inf))
        rows = []
        for solver in self.solvers:
            runs = [record for record in self.records if record.solver == solver]
            solved = [record for record in runs if record.solved]
            gaps = [self.gap(record, best) for record in solved]
            gaps = [value for value in gaps if not math.isnan(value)]
            at_best = [
                record
                for record in solved
                if record.instance in best
                and (
                    record.objective <= best[record.instance]
                    or math.isclose(
                        record.objective,
                        best[record.instance],
                        rel_tol=_RELATIVE,
                        abs_tol=_ABSOLUTE,
                    )
                )
            ]
            rows.append(
                SolverSummary(
                    solver=solver,
                    runs=len(runs),
                    solved=len(solved),
                    mean_gap=_mean(gaps),
                    median_gap=statistics.median(gaps) if gaps else math.nan,
                    best_share=len(at_best) / len(runs) if runs else math.nan,
                    feasible_rate=_mean([float(record.feasible) for record in solved]),
                    mean_seconds=_mean([record.seconds for record in solved]),
                )
            )
        return Summary(rows=tuple(rows), best_known=best)

    def to_csv(self, path: str | os.PathLike[str]) -> Path:
        """Write one row per record, with :data:`COLUMNS` as the header."""
        target = Path(path)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for record in self.records:
                writer.writerow(record.row())
        return target

    @classmethod
    def from_csv(cls, path: str | os.PathLike[str]) -> "ResultTable":
        """Read a table written by :meth:`to_csv`."""
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(set(COLUMNS) - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
            return cls(tuple(Record.from_row(row) for row in reader))

    @classmethod
    def of(cls, records: Iterable[Record]) -> "ResultTable":
        """The table of ``records``."""
        return cls(tuple(records))

    def where(self, **values: Any) -> "ResultTable":
        """The records whose fields equal ``values``, for example ``solver="aco"``."""
        unknown = sorted(set(values) - set(COLUMNS))
        if unknown:
            raise ValueError(f"unknown record fields {unknown}")
        return ResultTable(
            tuple(
                record
                for record in self.records
                if all(getattr(record, name) == value for name, value in values.items())
            )
        )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _percent(value: float) -> str:
    return "-" if math.isnan(value) else f"{100 * value:.2f}%"
