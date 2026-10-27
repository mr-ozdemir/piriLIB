# linehaul_mini

A small synthetic line-haul problem, used in the documentation and the tests.

| File | Content |
|---|---|
| `problem.yaml` | Calendar, vehicle types, fleet, business rules, the objective, and references to the two tables |
| `centers.csv` | Six centers in the Marmara region with approximate city-center coordinates |
| `demand.csv` | Two days of demand in desi, in Turkish spreadsheet format (`;`, decimal comma, `dd.mm.yyyy`) |

All quantities and costs are invented for illustration. They do not describe
any company and were not derived from competition data.

```python
import pirilib

spec = pirilib.load_spec("examples/linehaul_mini/problem.yaml")
instance = spec.compile()
result = pirilib.solve(instance, spec.build_objective())
```

The files in this folder are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
