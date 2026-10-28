# pirilib

Fleet optimization with an ant colony solver and a bee colony tuner, driven by
your own data, constraints and objective.

> **Status: first alpha (0.1.0a1).** Everything may still change before 1.0,
> the public API and the internal structure alike. Fleet optimization is the
> default domain, and line-haul planning is the first supported problem type,
> not a fixed contract. See
> [Public API and stability](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/stability.md).

pirilib (short name *piri*) generalizes the solution built for the TEKNOFEST
2026 Hepsiburada intercity logistics challenge into a reusable library.

## What it does

- **Line-haul fleet planning**, the first problem type: heterogeneous fleets
  of owned vehicles bound to lines and on-demand spot vehicles,
  multi-dimensional capacities, one-way multi-stop routes, split deliveries,
  bans, minimum fill and stopover rules.
- **Ant Colony Optimization** as the solver, vectorized with numpy, with local
  search and an optional exact solver for small subproblems; an **Artificial
  Bee Colony** tuner for its settings. Solvers and tuners are swappable.
- **Your objective, your rules**: building blocks for objectives and
  constraints, one evaluator shared by every solver, and an independent plan
  checker.
- **Demand forecasting** of daily desi per origin-destination line:
  cleaning, mandatory seasonal baselines, any scikit-learn-style regressor,
  the prototype's LightGBM-Tweedie, walk-forward backtests, ABC tuning, and a
  quantile-buffered bridge from the forecast to a planning problem.

Read the [user guide](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/guide.md) for the problem format, the rules, the
objective building blocks, solving, tuning and the command line.

## Install

The alpha is published on TestPyPI only. Install it from there and take its
dependencies from PyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "pirilib==0.1.0a1"
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "pirilib[exact,excel]==0.1.0a1"
```

The extras are `exact` (scipy), `excel` (openpyxl), `forecast` (LightGBM and
scikit-learn), `pyvrp` and `ortools`. `pip install pirilib` from PyPI works from
the first final release on.

## Command line

```bash
pirilib generate sample --centers 20 --days 3
pirilib solve sample/problem.yaml -o plan.csv
pirilib check sample/problem.yaml plan.csv
pirilib forecast history.csv --network network.yaml --problem week.yaml --quantile 0.8
```

`generate` writes a synthetic problem to try, `solve` plans it with one row
per load, and `check` audits any plan against the rules. `solve` minimizes the objective that the problem file defines in its
`objective:` section; `generate` writes one, and there is no default.

## Your objective

The objective weighs the total cost of a plan against every rule it breaks.
You write it in the problem file: a weight for the cost, optional weights per
km, per dispatch and per unit of empty capacity, and a penalty per unit of
violation for every rule, by name or through `other`. The penalties decide
what gives way when not every rule can hold, for example whether a small
demand stays behind or travels in an under-filled vehicle. The page
[Building the objective](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/objective.md)
explains every field and the unit of every rule, works through the total cost
of a small plan, gives advice on choosing weights and writes the same
objective in Python. The scenarios in
[`examples/objectives`](https://github.com/mr-ozdemir/piriLIB/tree/master/examples/objectives)
show how the weights change a plan.

## Python

A problem is described in YAML, and large tables can come from CSV files, as in
[`examples/linehaul_mini`](https://github.com/mr-ozdemir/piriLIB/tree/master/examples/linehaul_mini):

```python
import pirilib

spec = pirilib.load_spec("examples/linehaul_mini/problem.yaml")
instance = spec.compile()
print(instance)
```

`load_spec` returns the validated, immutable document, and `compile` turns it
into read-only numpy arrays for the engines.

Every problem in a document is reported at once, with file and line for CSV
rows:

```text
problem.yaml: invalid problem specification
- demand.csv:8: destination: unknown center 'Bilecık'
- demand.csv:18: date: 2026-05-13 is outside the calendar 2026-05-11..2026-05-12
```

The JSON Schema of problem documents is in [`schemas/`](https://github.com/mr-ozdemir/piriLIB/tree/master/schemas). Editors with
YAML language support use it for completion and checks.

A plan is a list of dispatches. One evaluator computes its cost and how much it
violates every rule; your objective and your own constraints read the result.
The dispatch below is one spot kamyon (offer 1) from İstanbul (center 0) to
Kocaeli (center 1) on day 0, carrying 12000 desi of the first demand, and
`check_plan` explains the same rules in words:

```python
from pirilib import Dispatch, Load, Plan, check_plan, evaluate
from pirilib import objective as obj
from pirilib.constraints import MaxRouteKm

dispatch = Dispatch(
    day=0, kind="spot", source=1, origin=0, stops=(1,), loads=(Load(0, (12000.0,)),)
)
plan = Plan.from_dispatches([dispatch], dimensions=1)

evaluation = evaluate(instance, plan, constraints=[MaxRouteKm(400)])
goal = spec.build_objective()
print(evaluation.total_cost, evaluation.violated())
print(goal.breakdown(evaluation))
print(check_plan(instance, plan))
```

The ant colony searches for the plan that minimizes your objective. There is
no default one: build it from the `objective:` section of the problem file, or
write it with `pirilib.objective`. `print(result)` shows the objective value,
the seconds and the evaluation of the best plan. Any function of an evaluation
is an objective too; it is searched on whole plans, which is slower:

```python
from pirilib import ACOConfig, ACOSolver, solve

result = solve(instance, spec.build_objective(), seed=1)
result = solve(instance, spec.build_objective() + 500 * obj.dispatches("spot"), seed=1)
print(result)

result = solve(
    instance,
    objective=lambda e: goal(e) + 50 * len(e.plan),
    solver=ACOSolver(ACOConfig(ants=8)),
)
```

## Development setup

Create a virtual environment and activate it, with `.venv\Scripts\activate` on
Windows or `source .venv/bin/activate` on macOS and Linux:

```bash
python -m venv .venv
```

Then install and run the checks:

```bash
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

Pushing a version tag such as `v0.1.0a1` runs `.github/workflows/release.yml`,
which tests, builds and publishes the package with trusted publishing.
Pre-releases go to TestPyPI only, final releases to TestPyPI and then, after an
approval, to PyPI. The steps and the one-time setup are in
[CONTRIBUTING.md](https://github.com/mr-ozdemir/piriLIB/blob/master/CONTRIBUTING.md#releasing).

## Roadmap

pirilib is heading towards a general fleet optimization library: one solver,
tuner and objective core with problem types on top of it. Which problem types
follow line-haul is not decided yet.

| Version | Scope |
|---|---|
| 0.1.0a1 | First alpha, the code as it is today: the line-haul problem type (schema `pirilib/linehaul@0.2` with times, service, deferral and center limits), evaluator, constraint catalog and plan checker, numpy ACO with local search and an optional exact oracle, ABC tuner, demand forecasting with the forecast-to-plan bridge, benchmarks with PyVRP and OR-Tools baselines, CLI |
| Later alphas and betas | Changes from alpha feedback; the API and the internal structure may still change |
| 1.0 | Stable API, as listed in [Public API and stability](https://github.com/mr-ozdemir/piriLIB/blob/master/docs/stability.md) |

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
biçiminde (`--delimiter ";" --decimal ","`) yazılıp okunabilir. Bu ilk alfa
sürümüdür (0.1.0a1): 1.0'a kadar iç yapı dahil her şey değişebilir; hat taşıma
(line-haul) ilk problem türüdür, varsayılan alan genel olarak filo
optimizasyonudur.
