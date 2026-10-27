# objectives

Small synthetic problems that show how the `objective:` section changes a
plan. They go with the page
[Building the objective](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/objective.md).

| File | Objective | What the plan does |
|---|---|---|
| `every_load.yaml` | `coverage: 10000000`, `min_fill: 1000`, `other: 1000000` | Carries every load. The 40-desi demand from A to D goes in a vehicle that needs 120 desi for its minimum fill, so minimum fill is broken by 80 desi. |
| `hard_min_fill.yaml` | `coverage: 1000`, `min_fill: 10000000`, `other: 1000000` | Keeps minimum fill. The same 40 desi stay behind (`coverage` 40). |
| `distance_aware.yaml` | `distance_km: 20`, `coverage: 10000000`, `other: 1000000` | Sends two vans direct (220 km, cost 2220) instead of one van on a detour (300 km, cost 1300), because the 80 km saved are worth 1600. Feasible. |
| `worked_cost.yaml` | `coverage: 1000`, `other: 100000` | The worked example of the page: a total cost of 1626 from fixed, per km and per hour costs, a mandatory owned vehicle and lateness. Feasible. |

`every_load.yaml` and `hard_min_fill.yaml` have the same network, fleet,
demand and rules; only their names and objectives differ. Their data allow no
feasible plan, so the objective decides which rule gives way.

Solve one from the command line, or all of them from Python:

```bash
pirilib solve examples/objectives/every_load.yaml -o plan.csv
```

```python
from pathlib import Path

import pirilib

for path in sorted(Path("examples/objectives").glob("*.yaml")):
    spec = pirilib.load_spec(path)
    result = pirilib.solve(spec.compile(), spec.build_objective(), seed=0)
    print(path.name, result.evaluation)
```

All quantities, distances and costs are invented for illustration. They do not
describe any company and were not derived from competition data.

The files in this folder are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
