"""Benchmarks: reproducible instance families, solver runs and their summary.

```python
from pirilib.bench import FAMILIES, SOLVERS, Benchmark
from pirilib.spec import parse_spec

instances = FAMILIES.get("small").instances(10)
goal = parse_spec(instances[0].document).build_objective()
benchmark = Benchmark(SOLVERS.make(["direct", "aco", "exact"]), objective=goal)
report = benchmark.report(instances)
report.to_json("report.json")
report.table.to_csv("results.csv")
print(report.summary())
```

A later run is compared with a saved one by :func:`compare`.
"""

from pirilib.bench.compare import Change, Comparison, compare
from pirilib.bench.families import FAMILIES, BenchInstance, Family, FamilyCatalog
from pirilib.bench.report import Environment, InstanceInfo, Report, SolverInfo
from pirilib.bench.results import (
    COLUMNS,
    Record,
    ResultTable,
    SolverSummary,
    Spread,
    Summary,
)
from pirilib.bench.runner import SOLVERS, Benchmark, SolverCatalog

__all__ = [
    "COLUMNS",
    "FAMILIES",
    "SOLVERS",
    "BenchInstance",
    "Benchmark",
    "Change",
    "Comparison",
    "Environment",
    "Family",
    "FamilyCatalog",
    "InstanceInfo",
    "Record",
    "Report",
    "ResultTable",
    "SolverCatalog",
    "SolverInfo",
    "SolverSummary",
    "Spread",
    "Summary",
    "compare",
]
