# pirilib

Fleet optimization with an ant colony solver and a bee colony tuner, driven by
your own data, constraints and objective.

!!! warning "Status: first alpha (0.1.0a1)"

    Everything may still change before 1.0, the public API and the internal
    structure alike. Fleet optimization is the default domain, and line-haul
    planning is the first supported problem type, not a fixed contract. See
    [Public API and stability](stability.md).

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
  LightGBM-Tweedie, walk-forward backtests, ABC tuning, and a
  quantile-buffered bridge from the forecast to a planning problem.

## Quick start

After [installing](guide.md#install), generate a synthetic problem, plan it
(one row per load) and audit the plan against the rules:

```bash
pirilib generate sample --centers 20 --days 3
pirilib solve sample/problem.yaml -o plan.csv
pirilib check sample/problem.yaml plan.csv
```

`solve` minimizes the objective that the problem file defines in its
`objective:` section; `generate` writes one, and there is no default.

The same from Python: the specification is validated and immutable, and its
compiled form holds read-only numpy arrays for the engines.

```python
import pirilib

spec = pirilib.load_spec("sample/problem.yaml")
instance = spec.compile()
result = pirilib.solve(instance, spec.build_objective(), seed=1)
print(result)
print(pirilib.check_plan(instance, result.plan))
```

## Your objective

The objective weighs the total cost of a plan against every rule it breaks,
and there is no default one. The problem file's `objective:` section gives a
weight for the cost and a penalty per unit of violation for every rule, by
name or through `other`; those penalties decide what gives way when not every
rule can hold. [Building the objective](objective.md) explains every field and
the unit of every rule, works through the total cost of a small plan, and
shows how to choose weights and write the same objective in Python.

## Where to go next

- The [user guide](guide.md) covers the problem format, the rules, the
  objective building blocks, solving, tuning, forecasting and the command line.
- [Building the objective](objective.md) explains the `objective:` section,
  penalty weights and objectives written in Python.
- The [API reference](api/pirilib.md) is generated from the docstrings of the
  package.
- The source, the example problem in
  [`examples/linehaul_mini`](https://github.com/mr-ozdemir/piriLIB/tree/master/examples/linehaul_mini)
  and the JSON Schema of problem documents in
  [`schemas/`](https://github.com/mr-ozdemir/piriLIB/tree/master/schemas) are on
  [GitHub](https://github.com/mr-ozdemir/piriLIB).

pirilib is licensed under Apache-2.0.
