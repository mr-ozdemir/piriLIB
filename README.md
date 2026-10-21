# pirilib

Demand forecasting and ACO-ABC based distribution optimization for logistics
planning, driven by your own data, constraints and objective.

> **Status: pre-alpha (0.0.x).** Nothing here is ready for production use and
> the API will change without notice.

pirilib (short name *piri*) generalizes the solution built for the TEKNOFEST
2026 Hepsiburada intercity logistics challenge into a reusable library.

## What it does

- **Line-haul and fleet planning** (0.1): heterogeneous fleets of owned
  vehicles bound to lines and on-demand spot vehicles, multi-dimensional
  capacities, one-way multi-stop routes, split deliveries, bans, minimum fill
  and stopover rules.
- **Ant Colony Optimization** as the solver, vectorized with numpy, with local
  search and an optional exact solver for small subproblems; an **Artificial
  Bee Colony** tuner for its settings. Solvers and tuners are swappable.
- **Your objective, your rules**: building blocks for objectives and
  constraints, one evaluator shared by every solver, and an independent plan
  checker.
- **Demand forecasting** (0.2) of daily desi per origin-destination line:
  cleaning, mandatory seasonal baselines, any scikit-learn-style regressor,
  the prototype's LightGBM-Tweedie, walk-forward backtests, ABC tuning, and a
  quantile-buffered bridge from the forecast to a planning problem.

Read the [user guide](https://github.com/mr-ozdemir/piriLIB/blob/main/docs/guide.md) for the problem format, the rules, the
objective building blocks, solving, tuning and the command line.

## Command line

```bash
pip install pirilib
pirilib generate sample --centers 20 --days 3   # a synthetic problem to try
pirilib solve sample/problem.yaml -o plan.csv    # plan it; one row per load
pirilib check sample/problem.yaml plan.csv       # audit any plan against the rules
pirilib forecast history.csv --network network.yaml --problem week.yaml --quantile 0.8
```

## Python

A problem is described in YAML, and large tables can come from CSV files, as in
[`examples/linehaul_mini`](https://github.com/mr-ozdemir/piriLIB/tree/main/examples/linehaul_mini):

```python
import pirilib

spec = pirilib.load_spec("examples/linehaul_mini/problem.yaml")  # validated, immutable
instance = spec.compile()  # read-only numpy arrays for the engines
print(instance)
```

Every problem in a document is reported at once, with file and line for CSV
rows:

```text
problem.yaml: invalid problem specification
- demand.csv:8: destination: unknown center 'Bilecık'
- demand.csv:18: date: 2026-05-13 is outside the calendar 2026-05-11..2026-05-12
```

The JSON Schema of problem documents is in [`schemas/`](https://github.com/mr-ozdemir/piriLIB/tree/main/schemas). Editors with
YAML language support use it for completion and checks.

A plan is a list of dispatches. One evaluator computes its cost and how much it
violates every rule; your objective and your own constraints read the result:

```python
from pirilib import Dispatch, Load, Plan, check_plan, evaluate
from pirilib import objective as obj
from pirilib.constraints import MaxRouteKm

# One spot kamyon (offer 1) from İstanbul (center 0) to Kocaeli (center 1), day 0,
# carrying 12000 desi of the first demand.
dispatch = Dispatch(
    day=0, kind="spot", source=1, origin=0, stops=(1,), loads=(Load(0, (12000.0,)),)
)
plan = Plan.from_dispatches([dispatch], dimensions=1)

evaluation = evaluate(instance, plan, constraints=[MaxRouteKm(400)])
goal = obj.total_cost() + 10 * obj.unserved() + 1e6 * obj.violation("capacity")
print(evaluation.total_cost, evaluation.violated())
print(goal.breakdown(evaluation))
print(check_plan(instance, plan))  # the same rules, explained in words
```

The ant colony searches for the plan that minimizes your objective:

```python
from pirilib import ACOConfig, ACOSolver, solve

result = solve(instance, objective=obj.total_cost() + 1e6 * obj.violations(), seed=1)
print(result)  # objective, seconds, and the evaluation of the best plan

# Any function of an evaluation is an objective too (searched on whole plans, slower).
result = solve(
    instance,
    objective=lambda e: e.total_cost + 50 * len(e.plan),
    solver=ACOSolver(ACOConfig(ants=8)),
)
```

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . --group dev
pytest
ruff check . && ruff format --check . && mypy
```

pip 25.1 or newer is needed for `--group`. After changing the specification
models, regenerate the JSON Schema:

```bash
python -m pirilib.spec.schema schemas/linehaul-0.2.schema.json
```

### Documentation

The documentation site (the user guide and an API reference generated from the
docstrings) is built with MkDocs and Material for MkDocs from `mkdocs.yml` and
`docs/`. `mkdocs serve` shows a live preview at http://127.0.0.1:8000, and
`mkdocs build --strict` writes the static site to `site/`, failing on any
warning (as CI does):

```bash
python -m pip install --group docs
mkdocs serve
mkdocs build --strict
```

### Releasing

Pushing a version tag such as `v0.1.0` runs `.github/workflows/release.yml`.
The workflow builds the package and publishes it to TestPyPI and then to PyPI,
using trusted publishing, so no tokens are stored in the repository. Before
the first release:

1. Add the repository and the workflow as a trusted publisher on test.pypi.org
   and on pypi.org.
2. Create the GitHub environments `testpypi` and `pypi`.
3. Give `pypi` a required reviewer, so that a person approves each release.

## Roadmap

| Version | Scope |
|---|---|
| 0.1 | Problem specification, evaluator, constraint catalog, numpy ACO, local search, decomposition with an exact oracle, validator, ABC tuner, CLI |
| 0.2 | Forecast engine and the forecast-to-plan bridge |
| 0.3 | Time waves, service-level penalties, shared hub resources, deferral |
| 0.4 | Benchmark suite and baseline adapters (PyVRP, OR-Tools) |
| 1.0 | Stable API |

## License

Code is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`). Example data in
this repository is synthetic. Third-party or competition data is not
distributed.

## Türkçe

pirilib, TEKNOFEST 2026 Hepsiburada şehirlerarası lojistik çözümünü genel
amaçlı bir kütüphaneye dönüştürür: desi talep tahmini ve ACO çözücü ile ABC
ayarlayıcıya dayanan dağıtım/filo optimizasyonu. Amaç fonksiyonunu ve kısıtları
kullanıcı seçer; kütüphane bunları yazmayı kolaylaştıran yapı taşları sunar.
Problem YAML ile tanımlanır; büyük tablolar CSV'den okunabilir (noktalı virgül,
ondalık virgül, `gg.aa.yyyy` tarih ve Türkçe sütun adları desteklenir). Tüm
hatalar tek seferde, CSV satır numarasıyla raporlanır. Planlar `pirilib solve`
ile üretilir, `pirilib check` ile denetlenir; plan dosyaları Türkçe Excel
biçiminde (`--delimiter ";" --decimal ","`) yazılıp okunabilir. Proje henüz
erken geliştirme aşamasındadır.
