"""Benchmarks: reproducible instance families, solver runs and their summary.

```python
from pirilib.bench import FAMILIES, SOLVERS, Benchmark
from pirilib.spec import parse_spec

instances = FAMILIES.get("small").instances(10)
goal = parse_spec(instances[0].document).build_objective()
benchmark = Benchmark(SOLVERS.make(["direct", "aco", "exact"]), objective=goal)
table = benchmark.run(instances)
table.to_csv("results.csv")
print(table.summary())
```
"""

from pirilib.bench.families import FAMILIES, BenchInstance, Family, FamilyCatalog
from pirilib.bench.results import COLUMNS, Record, ResultTable, SolverSummary, Summary
from pirilib.bench.runner import SOLVERS, Benchmark, SolverCatalog

__all__ = [
    "COLUMNS",
    "FAMILIES",
    "SOLVERS",
    "BenchInstance",
    "Benchmark",
    "Family",
    "FamilyCatalog",
    "Record",
    "ResultTable",
    "SolverCatalog",
    "SolverSummary",
    "Summary",
]
