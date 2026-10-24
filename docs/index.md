# pirilib

Demand forecasting and ACO-ABC based distribution optimization for logistics
planning, driven by your own data, constraints and objective.

!!! warning "Status: pre-alpha (0.0.x)"

    Nothing here is ready for production use and the API will change without
    notice.

pirilib (short name *piri*) generalizes the solution built for the TEKNOFEST
2026 Hepsiburada intercity logistics challenge into a reusable library.

## What it does

- **Line-haul and fleet planning**: heterogeneous fleets of owned vehicles
  bound to lines and on-demand spot vehicles, multi-dimensional capacities,
  one-way multi-stop routes, split deliveries, bans, minimum fill and stopover
  rules.
- **Ant Colony Optimization** as the solver, vectorized with numpy, with local
  search and an optional exact solver for small subproblems; an **Artificial
  Bee Colony** tuner for its settings. Solvers and tuners are swappable.
- **Your objective, your rules**: building blocks for objectives and
  constraints, one evaluator shared by every solver, and an independent plan
  checker.
- **Demand forecasting** of daily desi per origin-destination line: cleaning,
  mandatory seasonal baselines, any scikit-learn-style regressor, the
  prototype's LightGBM-Tweedie, walk-forward backtests, ABC tuning, and a
  quantile-buffered bridge from the forecast to a planning problem.

## Quick start

Generate a synthetic problem, plan it (one row per load) and audit the plan
against the rules:

```bash
pip install pirilib
pirilib generate sample --centers 20 --days 3
pirilib solve sample/problem.yaml -o plan.csv
pirilib check sample/problem.yaml plan.csv
```

The same from Python: the specification is validated and immutable, and its
compiled form holds read-only numpy arrays for the engines.

```python
import pirilib
from pirilib import objective as obj

spec = pirilib.load_spec("sample/problem.yaml")
instance = spec.compile()
result = pirilib.solve(instance, objective=obj.total_cost() + 1e6 * obj.violations(), seed=1)
print(result)
print(pirilib.check_plan(instance, result.plan))
```

## Where to go next

- The [user guide](guide.md) covers the problem format, the rules, the
  objective building blocks, solving, tuning, forecasting and the command line.
- The [API reference](api/pirilib.md) is generated from the docstrings of the
  package.
- The source, the example problem in
  [`examples/linehaul_mini`](https://github.com/mr-ozdemir/piriLIB/tree/master/examples/linehaul_mini)
  and the JSON Schema of problem documents in
  [`schemas/`](https://github.com/mr-ozdemir/piriLIB/tree/master/schemas) are on
  [GitHub](https://github.com/mr-ozdemir/piriLIB).

pirilib is licensed under Apache-2.0.
