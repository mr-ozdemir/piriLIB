# pirilib user guide

pirilib plans line-haul transport: which vehicles leave which center on which
day, where they stop and what they carry, so that your objective is as small as
possible and your rules hold. It also forecasts the demand to plan. This guide
covers the pre-alpha versions 0.1 (planning) and 0.2 (forecasting).

## Install

```bash
pip install pirilib
pip install "pirilib[exact]"
pip install "pirilib[excel]"
pip install "pirilib[forecast]"
```

- `pirilib` is the core, with numpy, pydantic and PyYAML.
- `exact` adds the exact solver for small subproblems (scipy).
- `excel` adds writing plans as .xlsx (openpyxl).
- `forecast` adds the LightGBM and scikit-learn forecasters.

On macOS, LightGBM also needs OpenMP: `brew install libomp`.

Python 3.12 or newer.

## 1. Describe the problem

A problem is a YAML file (schema `pirilib/linehaul@0.2`; 0.1 files still load). Large tables can be
CSV files next to it. A complete example is in `examples/linehaul_mini`.

```yaml
schema: pirilib/linehaul@0.2
name: My network
calendar: {start: 2026-05-11, end: 2026-05-12}

centers:
  - {id: Istanbul, lat: 41.0082, lon: 28.9784}
  - {id: Ankara, lat: 39.9334, lon: 32.8597, banned_vehicle_types: [tir]}

matrix: {source: haversine, factor: 1.25}

vehicle_types:
  - {name: tir, capacity: {desi: 22000}}
  - {name: kamyon, capacity: {desi: 12000}}

fleet:
  owned:
    - {type: tir, origin: Istanbul, destination: Ankara, count: 2,
       cost: {fixed: 7500, per_km: 12}, mandatory: true, stopovers: false}
  spot:
    - {type: tir, cost: {fixed: 12000, per_km: 24}}
    - {type: kamyon, cost: {fixed: 8000, per_km: 20}, available: 5}

demand:
  - {origin: Istanbul, destination: Ankara, date: 2026-05-11, quantity: {desi: 18450}}

rules:
  primary_dimension: desi
  min_fill: {spot: 0.10, owned: 0.0}
  stopover: {enabled: true, max_stretch: 1.3, forward_only: true, max_stops: null}
  split_demand: true

objective:
  cost: 1
  penalties:
    coverage: 10000000
    other: 1000000
```

What the fields mean:

- **Name and calendar.** `name` is optional and used in reports. The calendar
  includes both its `start` and its `end` day.
- **Centers.** `id` is the name used everywhere. Ids are trimmed and Unicode
  NFC-normalized, and they are case sensitive. A banned vehicle type may not
  leave from, stop at or arrive at the center. Centers, like demand, can also
  come from a CSV file (see [Tables from CSV](#tables-from-csv)).
- **Distances.** With `haversine` they are great-circle distances from the
  coordinates, times `factor`. With `explicit` you give arcs
  (`{origin, destination, km}`) for every ordered pair. `symmetric: true`
  fills in each missing reverse direction.
- **Vehicle types.** Every type needs the same capacity dimensions (desi, kg,
  ...). Every demand must give every dimension.
- **Owned vehicles** are bound to a line.
  - `count` vehicles run on one line every day.
  - With `mandatory: true`, the default, their fixed cost is paid every day,
    used or not; a dispatch then only costs `per_km`. With `mandatory: false`
    a dispatch pays `fixed` too, and idle vehicles cost nothing.
  - With `stopovers: true` they may drop loads at centers on the way, but they
    always end at the line's destination.
- **Spot offers** are vehicles hired per dispatch.
  - Each dispatch costs `fixed + per_km x route km`.
  - `available` limits vehicles per day; leave it out for unlimited.
  - There is one offer per vehicle type.
- **Demand** is shipped on its date. Ids are generated (`D000001`, ...) when left out.
- **Routes** are one-way: from the origin through the stops in order. There is no return trip.
- **Minimum fill.** `min_fill` is the share of capacity a vehicle must carry,
  for spot and owned vehicles separately, in the `primary_dimension`. The
  primary dimension defaults to the first capacity dimension.
- **Split demand.** With `split_demand: true` a demand may travel in several
  vehicles.
- **Stopover rules.**
  - `max_stretch` bounds route length divided by the direct distance to the last stop.
  - `forward_only` requires each stop to be farther from the origin than the one before it.
  - `max_stops` bounds the number of stops.
- **Objective.** `objective:` says what to minimize: `cost` times the total
  cost, optional weights `distance_km`, `dispatches` and `unused_capacity`, and
  under `penalties` a weight per unit of violation of each rule (the names in
  the table of section 2). Every rule needs a penalty, or `other` gives one to
  every rule not named; a missing one is an error. The section is optional for
  validating and checking, but solving and tuning need it (see section 3 and
  [Building the objective](objective.md)).

### Times and service (schema 0.2)

All of these are optional; without them a problem behaves as in 0.1.

```yaml
vehicle_types:
  - {name: tir, capacity: {desi: 22000}, speed_kmh: 70}
fleet:
  spot:
    - {type: tir, cost: {fixed: 0, per_km: 20, per_hour: 400}}
centers:
  - {id: Istanbul, lat: 41.0, lon: 29.0, handling_capacity: 600000,
     vehicle_limits: {tir: 40}}
demand:
  - {origin: Istanbul, destination: Ankara, date: 2026-05-11, ready: "17:00",
     quantity: {desi: 5000}}
rules:
  service: {due_hours: 24, penalty: 0.4, max_deferral_days: 1}
  handling: {minutes_per_unit: 0.01}
```

A demand may also give its due time itself, as in `due: 2026-05-12T17:00`;
without it, the due time is `due_hours` after the demand is ready, and there is
none when `due_hours` is not set. A `minutes_per_unit` of
0.01 handles 100 desi a minute.

- **Timeline of a dispatch.**
  - Loading starts at the dispatch's `start` (minutes after 00:00 of its day;
    by default as soon as every load is ready).
  - Loading takes `minutes_per_unit × load` minutes.
  - Each leg takes `km / speed_kmh` hours, rounded up to whole minutes.
  - Unloading at a stop takes `minutes_per_unit × what is unloaded there`.
- **Hours** run from the start of loading to the end of the last unloading and
  are charged with `per_hour`.
- **Lateness.** A load is late when the unloading at its stop ends after the
  due time of its demand. Every started hour costs `penalty × quantity`. The
  total is `Evaluation.service_cost` and part of `total_cost`.
- **Deferral.** A demand may leave up to `max_deferral_days` days after its
  date. Its due time stays, so lateness grows.
- **Handling capacity.** Each center loads and unloads at most
  `handling_capacity` primary units per day, counted on the day handling
  starts.
- **Vehicle limits.** At most this many vehicles of a type start loading at or
  arrive at the center per day.

### Tables from CSV

`centers`, `demand` and `matrix.arcs` accept a file reference:

```yaml
demand:
  file: demand.csv
  delimiter: ";"
  encoding: cp1254
  decimal: ","
  date_format: "%d.%m.%Y"
  columns: {origin: Çıkış, destination: Varış, date: Tarih}
  quantity: {desi: Desi}
```

- `file` is relative to the YAML file.
- `encoding` defaults to `utf-8-sig`.
- With `decimal: ","`, a `.` is read as a thousands separator.
- `date_format` defaults to ISO dates.
- `columns` maps fields to column names. `quantity`, for demand only, maps
  each dimension to its column.

Lists inside a cell, such as banned vehicle types, are separated by `|`.

### Validation

`pirilib.load_spec("problem.yaml")` checks everything and reports every problem
at once, with file and line for CSV rows:

```text
problem.yaml: invalid problem specification
- demand.csv:8: destination: unknown center 'Bilecık'
- fleet.spot[1].type: unknown vehicle type 'huge'
```

Besides field values, validation checks:

- references between fields and duplicate ids;
- dimensions and the calendar;
- that distances are complete;
- that every demand has at least one vehicle that may serve it.

The JSON Schema in `schemas/` lets editors complete and check problem files.

## 2. Plans and their evaluation

```python
import pirilib

instance = pirilib.load_spec("problem.yaml").compile()
plan = pirilib.Plan.from_dispatches([...], dimensions=1)
evaluation = pirilib.evaluate(instance, plan)
evaluation.total_cost, evaluation.violated(), evaluation.feasible
print(pirilib.check_plan(instance, plan))
```

`compile()` turns the document into read-only numpy arrays. `check_plan` is an
independent checker that applies the same rules and explains every violation
in words. A dispatch is index-based: `Dispatch(day, kind, source, origin, stops, loads)`,
where `source` is the position of the owned group or spot offer in the problem
file. `evaluate` is the one place where costs and rule violations are computed.
Every rule has a violation amount that is zero when the rule holds:

| Constraint | Amount |
|---|---|
| `route` | dispatches without stops, stops at the origin, centers visited twice |
| `load_route` | loads whose demand does not match the dispatch's origin or stops |
| `ban` | origins and stops where the vehicle type is banned |
| `owned_line` | owned dispatches off their line |
| `owned_availability`, `spot_availability` | vehicles used beyond the daily count |
| `capacity` | load above capacity (quantity) |
| `min_fill` | load missing to the minimum fill (quantity) |
| `stopover`, `max_stops` | stopovers where not allowed, stops beyond the limit |
| `stretch` | route km above `max_stretch` x the direct km |
| `forward_only` | stops not farther than the one before |
| `split` | extra loads of demands that may not be split |
| `coverage`, `overdelivery` | quantity not carried, quantity carried beyond the demand |
| `ready` | loads that start loading before their demand is ready |
| `deferral` | loads that leave before their demand day or after `max_deferral_days` |
| `handling` | quantity handled at a center on a day beyond `handling_capacity` |
| `vehicle_limit` | vehicles of a type at a center on a day beyond `vehicle_limits` |

Constraints of your own are objects with a `name` and a
`violation(evaluation)` method. `MaxRouteKm`, `MaxDispatchesPerDay` and
`constraint(name, function)` are in `pirilib.constraints`.

## 3. Your objective

The objective is the number the solver minimizes. There is no default one:
`solve`, every solver, `tune` and `Benchmark` need it, and the command line
reads it from the `objective:` section of the problem file. Every rule a plan
can break needs a penalty weight that you chose, because the weights decide
what the solver gives up when not every rule can hold. The page
[Building the objective](objective.md) explains the section field by field,
the unit of every rule's penalty, how the total cost is computed, how to
choose weights, and common errors.

The usual way is to write the section in the document and build it:

```yaml
objective:
  cost: 1
  penalties:
    coverage: 10000000
    other: 1000000
```

```python
spec = pirilib.load_spec("problem.yaml")
goal = spec.build_objective()
goal.breakdown(evaluation)
```

`cost` weighs the total cost, `penalties` give a weight per unit of violation
of each rule of section 2 (`other` for the rules not named), and the optional
weights `distance_km`, `dispatches` and `unused_capacity` add terms.
`build_objective()` returns a `Sum` from `pirilib.objective`, and
`breakdown` gives the value of every term. A document without the section
makes `build_objective()` raise `SpecError`.

The same building blocks write objectives in Python. They add up and take
weights. The first line below is the section above in Python; the rules other
than `coverage` are named, because `violations()` without names includes
`coverage` and would weight it 1.1e7. The second adds a price per spot
dispatch to the document's objective:

```python
from pirilib import objective as obj
from pirilib.constraints import CATALOG

others = [name for name in CATALOG if name != "coverage"]
goal = obj.total_cost() + 1e7 * obj.violation("coverage") + 1e6 * obj.violations(others)
goal = spec.build_objective() + 500 * obj.dispatches("spot")
```

The available terms are:

- `total_cost`, `dispatch_cost(kind)`, `owned_fixed_cost`, `service_cost`;
- `distance_km(kind)`, `dispatches(kind)`, `unused_capacity(kind)`;
- `unserved(dimension)`;
- `violation(name)`, `violations(names)`.

An objective written in Python is used as it is; pirilib does not check that
it weights every rule, and a rule left out costs nothing to break.

Terms like these have a *linear form*, so the solver can optimize subproblem by
subproblem, which is fast. A plain function such as
`lambda e: goal(e) + 0.001 * e.total_km ** 2` is optimized on whole plans
instead, which is slower. Constraints of your own are handled the same way.

When not every rule can hold, the solver does what the objective prefers. For
example, it leaves a small demand uncarried rather than send a vehicle below
its minimum fill if the `coverage` penalty is the smaller one, and the other
way round if it is larger. `examples/objectives` shows both on the same data.

## 4. Solve

`solve` takes the instance and the objective, which has no default:

```python
result = pirilib.solve(instance, goal, seed=1)
result.plan, result.evaluation, result.objective, result.seconds
result.stats["mode"]
```

`result.objective` is the value of `goal` for the best plan, and
`result.evaluation` its costs and violations. `stats["mode"]` is `linear` when
the objective has a linear form and there are no constraints of your own (the
fast path), and `general` otherwise. Passing `None` as the objective raises
`ValueError`. Constraints of your own go in the third argument,
`pirilib.solve(instance, goal, [MaxRouteKm(400)])`, and need a term in the
objective, for example `goal + 1e6 * obj.violation("max_route_km")`.

The default solver is the ant colony (`pirilib.ACOSolver`):

- **Decomposition.** The problem is split into (day, origin) subproblems.
  Subproblems of similar size run side by side, vectorized with numpy.
- **Construction.** Each ant picks a vehicle and a first stop, then either
  extends the route or ends it. Pheromone is MAX-MIN in the hypercube
  framework.
- **Guarantees.** Every rule except minimum fill and coverage holds by
  construction.
- **Local search.** It then changes vehicles, merges, relocates and rebalances
  routes, serves leftover demand and drops routes that cost more than they
  carry.
- **Times and center limits.** With speeds, hourly costs, due times, deferral,
  handling capacity or vehicle limits, the subproblems of a day are planned one
  after another: first the most urgent (earliest due time), then the largest.
  A ledger keeps what every center can still handle per day; ants only use
  vehicle types with a slot left and load no more than the handling room, and
  every finished subproblem is timed and charged to the days its handling
  starts. Routes start as soon as their loads are ready; hourly costs and
  lateness are part of the scores. Demand that does not fit and may wait
  (`max_deferral_days`) joins the next day's subproblem with its due time
  unchanged; otherwise it stays uncarried. Subproblems that touch no limited
  center still run side by side.

`ACOConfig` settings:

| Setting | Default | Meaning |
|---|---|---|
| `ants` | 16 | solutions per iteration |
| `iterations`, `patience` | 60, 20 | iteration limit; stop after this many without improvement |
| `alpha`, `beta` | 1, 2 | weight of pheromone and of the heuristic |
| `evaporation` | 0.1 | pheromone evaporation |
| `exploitation` | 0 | chance of taking the best choice instead of sampling |
| `tau_min` | 0.01 | lower pheromone bound (upper bound is 1) |
| `update` | `iteration_best` | which solution lays pheromone |
| `local_search` | true | improve the best routes of every subproblem |
| `exact_nodes` | 0 | solve subproblems with at most this many destinations exactly (needs scipy; not with times or center limits) |

`pirilib.solvers.exact.ExactSolver` solves every subproblem to optimality by
mixed-integer programming. It is meant for small problems and for measuring
the colony.

## 5. Tune

```python
from pirilib.tuning import tune

result = tune([instance_a, instance_b], goal, budget=60, seeds=(0, 1))
better = pirilib.ACOSolver(result.config)
```

The artificial bee colony searches the space of ACO settings (`aco_space()`).
Each setting is scored by its mean relative difference to the base setting
over the training instances and seeds. A setting replaces the base only if it
is at least 0.1 % better. `RandomSearch` is the baseline tuner. With about 24
settings the bee colony and random search were close in our measurements;
give it 60 or more. Check tuned settings on instances you
did not tune on.

## 6. Forecast demand

`pirilib.forecast` forecasts daily quantities (desi) per line, a week ahead,
from a history CSV with origin, destination, date and quantity columns.

```python
from pirilib import forecast as fc

history = fc.load_history("history.csv", delimiter=";", decimal=",", date_format="%d.%m.%Y")
calendar = fc.Calendar.of(unusual=[...], holidays=[...])
print(fc.report(fc.compare(history, {"lightgbm": fc.lightgbm_tweedie()}, calendar=calendar)))
week = fc.lightgbm_tweedie().forecast(history, start, 7, calendar)
```

In the calendar, `unusual` days are not learned from or scored, and `holidays`
become features. Holidays are usually unusual as well, so list them in both.

- **Cleaning.** Unusual days and days without a record are filled with the
  median of the line on that weekday. Spikes above the median plus 4 MADs are
  cut. Only days before the forecast start are used, so backtests cannot see
  the future.
- **Features.** Lags of 7, 14, 21 and 28 days, rolling means and spread, the
  calendar, the line and distance to holidays. The horizon is at most 7 days.
- **Models.**
  - `SeasonalMean` (mean of the last 4 same weekdays) and `SeasonalNaive` are
    the baselines. `compare` always runs them.
  - `Regression(make, target)` turns any regressor with `fit`/`predict` into a
    forecaster. `target="log1p"` uses Duan smearing.
  - `lightgbm_tweedie()` is the prototype's tuned champion
    (`pip install "pirilib[forecast]"`).
- **Presence.** Lines are forecast only on weekdays on which they had a record
  in the last 35 days.
- **Backtest.** `backtest` and `compare` run walk-forward folds and pool WMAPE
  over them. `Backtester(folds, days, calendar)` keeps the fold settings for
  repeated runs: `.run(history, model)`, `.compare(history, models)`. `tune_forecaster` and `tune_lightgbm` tune with the bee colony on
  validation folds.
- **To planning.** `planning_document(network, forecast, factors=...)` turns
  a network (centers, fleet, rules) plus the forecast into a problem. With
  `factors=buffer_factors(backtest, 0.8, lines)`, each line is lifted to the
  80 % quantile of its past actual/forecast ratio. In tests the buffer covered
  the actual demand on about 80 % of unseen days.

**Time waves.** When demand becomes ready in waves (for example at 09:00 and
17:00), `WaveSplitter` learns the share of each wave per line and weekday from
a wave-level history (a CSV like the history with a `wave` column holding
`09:00`, `9:00` or an hour such as `9`). It uses the last 4 usual same weekdays
before the forecast start; a line-weekday with fewer than 2 such days falls
back to the line over all weekdays, and a line with fewer than 2 days to the
share of all lines. Passed to `planning_document`, every line and day becomes
one demand row per wave, with `ready` set to the wave time, so
`rules.service.due_hours` counts from the wave. Pieces below `minimum` (0.5)
are dropped; the rest add up to the daily forecast. `stability()` tells how
many shares fell back and how much they move from week to week.

```python
waves = fc.load_wave_history("waves.csv", wave="saat", delimiter=";", decimal=",")
shares = fc.WaveSplitter(weeks=4).fit(waves, start, calendar)
print(shares.stability().describe())
document = fc.planning_document(network, week, waves=shares)
```

The daily history may come from the same file: `load_history` adds up the
waves of a day.

## 7. Command line

```bash
pirilib generate sample --centers 20 --days 3
pirilib validate sample/problem.yaml
pirilib solve sample/problem.yaml -o plan.csv --summary summary.json
pirilib solve sample/problem.yaml -o plan.csv --delimiter ";" --decimal ","
pirilib check sample/problem.yaml plan.csv
pirilib tune sample/problem.yaml --budget 60 -o config.json
pirilib solve sample/problem.yaml --config config.json
pirilib schema linehaul.schema.json
pirilib backtest history.csv --holidays holidays.txt --models seasonal,naive,lightgbm
pirilib forecast history.csv --holidays holidays.txt -o forecast.csv \
    --network network.yaml --problem problem.yaml --quantile 0.8
pirilib forecast waves.csv --waves waves.csv --network network.yaml
pirilib solve problem.yaml -o plan.csv
```

- `generate` writes a synthetic problem.
- `--delimiter ";" --decimal ","` writes the plan for Turkish Excel.
- `check` exits with status 1 if a rule is broken.
- The last `solve` plans the week that `forecast --problem` wrote.

Date files (`--holidays`, `--unusual`) have one ISO date per line.

`solve`, `tune` and `bench` minimize the objective of the problem files (for
`bench`, the generated ones). Without an `objective:` section they stop with
exit status 2; `tune` also needs every training file to define the same one.
`generate` writes `cost: 1` with penalties `coverage: 10000000` and
`other: 1000000`, and `forecast --network` copies the network's section.

Plan files have one row per load. The columns are `dispatch`, `date`, `kind`,
`start` (loading start, `HH:MM` after 00:00 of the date, hours may pass 24),
`vehicle_type`, `source` (for example `fleet.spot[1]`), `origin`, `stops`
(separated by `|`), `route_km`, `dispatch_cost`, `demand`, `destination` and one
column per quantity dimension. `pirilib check` reads them back, so plans made
elsewhere can be audited too.

## 8. Benchmarks

`pirilib.bench` compares solvers on reproducible synthetic instances:

```python
from pirilib.bench import FAMILIES, SOLVERS, Benchmark
from pirilib.spec import parse_spec

instances = FAMILIES.get("small").instances(10)
goal = parse_spec(instances[0].document).build_objective()
benchmark = Benchmark(SOLVERS.make(["direct", "aco", "exact"]), objective=goal, seeds=(0, 1))
table = benchmark.run(instances)
table.to_csv("results.csv")
print(table.summary())
```

- **Families.** A `Family` holds ranges of settings for
  `pirilib.synthetic.generate`: centers, days, demand per day, owned lines,
  minimum fill, stretch, split or not, and daily spot limits. Instance `i`
  draws its settings from a stream seeded by the family's name, its seed and
  `i`. The built-in families are `tiny`, `small`, `routing`, `medium`,
  `no-split`, `spot-limits` and `mixed` (`pirilib bench` lists them).
- **Runs.** `Benchmark` takes any solvers with a `solve` method, by name.
  It evaluates every plan again with one objective. A solver that raises
  `ValueError` (it does not support the problem) or `ImportError` (its package
  is missing) is recorded as skipped.
- **Summary.** Per solver it gives the mean and median gap to the best
  objective any run reached on the instance, the share of runs that reached
  it, the share of solved runs without violations, and the mean seconds.
- **Baselines.**
  - `pirilib.solvers.direct.DirectShipping` sends every line straight: full
    vehicles first, then the cheapest vehicle that fits. It keeps bans, owned
    lines and spot limits.
  - `ExactSolver` is the oracle on small subproblems.
  - `pirilib.solvers.adapters.PyVRPSolver` (`pip install "pirilib[pyvrp]"`)
    and `ORToolsSolver` (`pip install "pirilib[ortools]"`) run external
    routing solvers on a reduced problem. Large demands go out as full direct
    trips first, and the rest of each destination is one stop. They refuse
    demand that may not be split, minimum fill, times and non-linear
    objectives.

```bash
pirilib bench small --solvers direct,aco,aco-exact,exact -o results.csv --summary summary.csv
```

## Library structure

Classes own state and settings: solvers, tuners, forecasters, backtesters,
plans, evaluations and the specification models. A public function is either a
stateless transformation (`load_spec`, `read_plan`, `synthetic.generate`) or a
thin entry point that builds a class and calls it (`solve`, `tune`, `evaluate`,
`backtest`). Extension points are protocols: `Solver`, `Tuner`, `Forecaster`.

## Known limits

- One-way line haul only: no return trips, working hours or driver rules.
- Costs are fixed, per km and per hour; there are no costs per unit carried.
- With times, deferral or center limits the ant colony plans greedily, day by
  day and subproblem by subproblem: an earlier subproblem never gives up room
  for a later one, and deferral does not look at the next day's capacity.
  While building routes it counts all handling on the departure day; the exact
  days are used only afterwards, so room after midnight is partly left unused
  and, rarely, a limit is exceeded (the penalty then shows it). Departures are
  always as early as the loads allow; the solver never sets `start` to wait.
  Planning one by one is slower: 2 to 2.5x the untimed time on a 10-center
  problem and 4.5x on 81 centers. The exact solver does not support these
  problems yet.
- Local search runs in plain Python; with cheap pre-checks and remembered
  rejections it takes about a tenth of solve time on large problems. Most of
  the rest is the colony's construction (a 30-day problem with 40,500 demands
  takes about 100 s).
- The exact solver needs splittable demand and up to about 9 destinations per
  subproblem.
- When spot offers have daily limits, origins of a day take vehicles one after
  another, greedily. `result.stats["spot_limits_reached"]` counts the (day,
  offer) pairs whose limit was used up; above zero, the greedy sharing may have
  bound the plan.
- Forecasts cover one quantity dimension, up to 7 days ahead, per day; time
  waves come from shares learned per line and weekday, not from a model. On the prototype data, walk-forward forecasts ran about 6 % low
  because of week-to-week swings around holidays. None of the corrections
  tried kept the accuracy (see `docs/olcumler.md`), so plan with a quantile
  buffer, for example 0.8. `Regression(scale=True)` (learning shapes relative
  to the line's recent level) is available as an experiment.
