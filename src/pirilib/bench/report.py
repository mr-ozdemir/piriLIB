"""Benchmark reports: the results of a run together with the conditions they were made under.

A result is only comparable with another when both say what was run and where.
A :class:`Report` therefore keeps, next to the records, the instances and how
they were generated, the solvers and their settings, the objective, the seeds
and the machine and package versions. It is written as one JSON file.
"""

import datetime as dt
import json
import math
import os
import platform
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from pirilib._version import __version__
from pirilib.bench.families import BenchInstance
from pirilib.bench.results import Record, ResultTable, Spread, Summary
from pirilib.constraints import Constraint
from pirilib.objective import Objective, Sum, Term
from pirilib.solvers import Solver

FORMAT: Final = "pirilib/bench-report@1"
"""Identifier of the report file format."""

PACKAGES: Final = ("numpy", "scipy", "pyvrp", "ortools")
"""Packages whose installed versions a report records."""


@dataclass(frozen=True, slots=True)
class Environment:
    """The machine and the software a benchmark ran on.

    Attributes:
        pirilib: Version of this library
        python: Version of the interpreter
        platform: Operating system and its version
        machine: Processor architecture
        processor: Processor name, where the system gives one
        cpus: Logical processors; 0 when unknown
        packages: Installed version by package name
    """

    pirilib: str
    python: str
    platform: str
    machine: str
    processor: str
    cpus: int
    packages: Mapping[str, str]

    @classmethod
    def current(cls, packages: Iterable[str] = PACKAGES) -> "Environment":
        """The environment of this process; ``packages`` that are not installed are left out."""
        versions = {}
        for name in packages:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return cls(
            pirilib=__version__,
            python=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            cpus=os.cpu_count() or 0,
            packages=MappingProxyType(versions),
        )


@dataclass(frozen=True, slots=True)
class SolverInfo:
    """What a solver is and how it was set up.

    Attributes:
        type: Module and class name
        settings: The solver's public attributes. Settings classes are written
            as mappings; a value that is not a number, text, list or mapping is
            written as its ``repr``
    """

    type: str
    settings: Mapping[str, Any]

    @classmethod
    def of(cls, solver: Solver) -> "SolverInfo":
        """Describe ``solver`` from its class and its public attributes."""
        kind = type(solver)
        attributes = getattr(solver, "__dict__", {})
        return cls(
            type=f"{kind.__module__}.{kind.__qualname__}",
            settings=MappingProxyType(
                {
                    name: _plain(value)
                    for name, value in attributes.items()
                    if not name.startswith("_")
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class InstanceInfo:
    """Where a benchmark instance comes from and how large it is.

    Attributes:
        name: ``family/index``
        seed: Seed the instance was generated with
        settings: The generator settings drawn for the instance
        demands: Demand rows
    """

    name: str
    family: str
    index: int
    seed: int
    settings: Mapping[str, Any]
    centers: int
    days: int
    demands: int

    @classmethod
    def of(cls, item: BenchInstance) -> "InstanceInfo":
        """Describe ``item``."""
        return cls(
            name=item.name,
            family=item.family,
            index=item.index,
            seed=item.seed,
            settings=MappingProxyType({key: _plain(value) for key, value in item.settings.items()}),
            centers=len(item.instance.centers),
            days=len(item.instance.days),
            demands=len(item.instance.demand_day),
        )


@dataclass(frozen=True, slots=True)
class Report:
    """The records of a benchmark run and the conditions of the run.

    Attributes:
        created: When the run finished, in UTC
        solvers: By name, in the order they ran
        objective: Name and weight of every term. An objective that is a plain
            function is one term with the function's name and weight 1
        constraints: Names of the extra constraints
        seeds: Solver seeds; one run per seed, solver and instance
        instances: In the order they ran
        table: One record per run
    """

    created: dt.datetime
    environment: Environment
    solvers: Mapping[str, SolverInfo]
    objective: tuple[tuple[str, float], ...]
    constraints: tuple[str, ...]
    seeds: tuple[int, ...]
    instances: tuple[InstanceInfo, ...]
    table: ResultTable

    @classmethod
    def of(
        cls,
        table: ResultTable,
        *,
        solvers: Mapping[str, Solver],
        objective: Objective,
        constraints: Sequence[Constraint],
        seeds: Sequence[int],
        instances: Iterable[BenchInstance],
    ) -> "Report":
        """The report of ``table``, made now, on this machine, from the things that were run."""
        return cls(
            created=dt.datetime.now(dt.UTC).replace(microsecond=0),
            environment=Environment.current(),
            solvers=MappingProxyType(
                {name: SolverInfo.of(solver) for name, solver in solvers.items()}
            ),
            objective=_terms(objective),
            constraints=tuple(constraint.name for constraint in constraints),
            seeds=tuple(seeds),
            instances=tuple(InstanceInfo.of(item) for item in instances),
            table=table,
        )

    def summary(self) -> Summary:
        """Per solver: gaps, best results, feasibility and time."""
        return self.table.summary()

    def spread(self) -> tuple[Spread, ...]:
        """Per instance and solver, the distribution of the objective over the seeds."""
        return self.table.spread()

    def to_json(self, path: str | os.PathLike[str]) -> Path:
        """Write the report to ``path`` and return the path.

        The file holds the conditions, the spread per instance and solver and
        every record. A number that does not exist, such as the objective of a
        skipped run, is written as ``null``, and an infinite one as the text
        ``"inf"`` or ``"-inf"``.
        """
        document = {
            "format": FORMAT,
            "created": self.created.isoformat(),
            "environment": _plain(self.environment),
            "solvers": {name: _plain(info) for name, info in self.solvers.items()},
            "objective": [{"name": name, "weight": weight} for name, weight in self.objective],
            "constraints": list(self.constraints),
            "seeds": list(self.seeds),
            "instances": [_plain(item) for item in self.instances],
            "spread": [_numbers(asdict(item)) for item in self.spread()],
            "records": [_numbers(asdict(record)) for record in self.table],
        }
        target = Path(path)
        target.write_text(
            json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
        )
        return target

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "Report":
        """Read a report written by :meth:`to_json`.

        The spread is not read: it is computed from the records again.
        :class:`ValueError` if the file is not a report of format :data:`FORMAT`.
        """
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        found = document.get("format") if isinstance(document, dict) else None
        if found != FORMAT:
            raise ValueError(f"{path}: not a benchmark report of format {FORMAT} (found {found!r})")
        environment = document["environment"]
        return cls(
            created=dt.datetime.fromisoformat(document["created"]),
            environment=Environment(
                **{**environment, "packages": MappingProxyType(environment["packages"])}
            ),
            solvers=MappingProxyType(
                {
                    name: SolverInfo(type=info["type"], settings=MappingProxyType(info["settings"]))
                    for name, info in document["solvers"].items()
                }
            ),
            objective=tuple(
                (term["name"], float(term["weight"])) for term in document["objective"]
            ),
            constraints=tuple(document["constraints"]),
            seeds=tuple(document["seeds"]),
            instances=tuple(
                InstanceInfo(**{**item, "settings": MappingProxyType(item["settings"])})
                for item in document["instances"]
            ),
            table=ResultTable(tuple(_record(row) for row in document["records"])),
        )


def _terms(objective: Objective) -> tuple[tuple[str, float], ...]:
    if isinstance(objective, Term):
        return ((objective.name, objective.weight),)
    if isinstance(objective, Sum):
        return tuple((term.name, term.weight) for term in objective.terms)
    return ((getattr(objective, "__qualname__", type(objective).__qualname__), 1.0),)


def _plain(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return repr(value)


def _numbers(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _number(value) for name, value in row.items()}


def _number(value: object) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return str(value)
    return value


def _record(row: Mapping[str, Any]) -> Record:
    numbers = {
        name: math.nan if row[name] is None else float(row[name])
        for name in ("objective", "total_cost", "seconds")
    }
    return Record(**{**row, **numbers, "violated": tuple(row["violated"])})
