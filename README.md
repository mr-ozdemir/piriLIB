# pirilib

Fleet optimization with an ant colony solver, a bee colony tuner and your own objective.

[![CI](https://github.com/mr-ozdemir/piriLIB/actions/workflows/ci.yml/badge.svg)](https://github.com/mr-ozdemir/piriLIB/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/mr-ozdemir/piriLIB/blob/master/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/mr-ozdemir/piriLIB/blob/master/LICENSE)

Describe the fleet, the demand and the rules in YAML, write what a good plan
means, and pirilib searches for the plan that serves the demand at the lowest
cost. One evaluator scores every plan the same way, whichever solver made it.

> **Alpha.** pirilib 0.1.0a1 is the first alpha; the API may change before 1.0.
> See [Public API and stability](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/stability.md).

## Installation

```bash
pip install "pirilib @ git+https://github.com/mr-ozdemir/piriLIB"
```

Optional extras: `exact` (scipy), `excel` (openpyxl), `forecast` (LightGBM,
scikit-learn), `pyvrp` and `ortools`.

## Quick start

```pycon
>>> import pirilib
>>> spec = pirilib.load_spec("examples/linehaul_mini/problem.yaml")
>>> result = pirilib.solve(spec.compile(), spec.build_objective(), seed=1)
>>> result.evaluation
Evaluation(total_cost=178394.63, dispatches=18, total_km=2891.0, feasible)
```

Or from the command line:

```bash
pirilib generate sample --centers 20 --days 3
pirilib solve sample/problem.yaml -o plan.csv
pirilib check sample/problem.yaml plan.csv
```

## Features

- **Line-haul planning** with owned and spot vehicles, multi-stop routes, split
  loads, times and deferral, handling capacity and vehicle limits.
- **Ant colony solver** in numpy with local search, and an exact solver for
  small problems.
- **Your objective**: cost, distance and a penalty for every rule, in YAML or
  Python, with an independent plan checker.
- **Bee colony tuner** for the solver settings, validated on held-out problems.
- **Demand forecasting** per line, with backtests and a bridge from the
  forecast to a planning problem.
- **Clear errors**: every problem in a document is reported at once, with file
  and line for table rows.

## Documentation

- [User guide](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/guide.md)
- [Building the objective](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/objective.md)
- [Public API and stability](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/stability.md)
- [Examples](https://github.com/mr-ozdemir/piriLIB/tree/master/examples)

## Contributing

See [CONTRIBUTING.md](https://github.com/mr-ozdemir/piriLIB/blob/master/CONTRIBUTING.md).

## License

Apache-2.0. Example data is synthetic; no third-party or competition data is
distributed.

## Türkçe

pirilib, filo optimizasyonu kütüphanesidir: problemi YAML ile tanımlarsınız,
ACO çözücü amaç fonksiyonunuza göre en iyi planı arar, ABC ayarlayıcı çözücü
ayarlarını bulur. CSV tabloları Türkçe biçimde (`;`, ondalık virgül, `gg.aa.yyyy`)
okunur ve yazılır, planlar Excel'e de yazılıp okunur.
