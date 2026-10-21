# pirilib user guide

pirilib plans line-haul transport: which vehicles leave which center on which
day, where they stop and what they carry, so that your objective is as small as
possible and your rules hold. It also forecasts the demand to plan. This guide
covers the pre-alpha versions 0.1 (planning) and 0.2 (forecasting).

## Install

```bash
pip install pirilib                # core: numpy, pydantic, PyYAML
pip install "pirilib[exact]"       # + exact solver for small subproblems (scipy)
pip install "pirilib[excel]"       # + writing plans as .xlsx (openpyxl)
pip install "pirilib[forecast]"    # + LightGBM and scikit-learn forecasters
```

Python 3.12 or newer.

## 1. Describe the problem

A problem is a YAML file (schema `pirilib/linehaul@0.2`; 0.1 files still load). Large tables can be
CSV files next to it. A complete example is in `examples/linehaul_mini`.

```yaml
schema: pirilib/linehaul@0.2
name: My network                     # optional, used in reports
calendar: {start: 2026-05-11, end: 2026-05-12}   # both days included

centers:                             # or a CSV reference, see below
  - {id: Istanbul, lat: 41.0082, lon: 28.9784}
  - {id: Ankara, lat: 39.9334, lon: 32.8597, banned_vehicle_types: [tir]}

matrix: {source: haversine, factor: 1.25}        # or explicit arcs

vehicle_types:
  - {name: tir, capacity: {desi: 22000}}
  - {name: kamyon, capacity: {desi: 12000}}

fleet:
  owned:                             # vehicles bound to a line
    - {type: tir, origin: Istanbul, destination: Ankara, count: 2,
       cost: {fixed: 7500, per_km: 12}, mandatory: true, stopovers: false}
  spot:                              # vehicles hired per dispatch
    - {type: tir, cost: {fixed: 12000, per_km: 24}}
    - {type: kamyon, cost: {fixed: 8000, per_km: 20}, available: 5}

demand:                              # or a CSV reference
  - {origin: Istanbul, destination: Ankara, date: 2026-05-11, quantity: {desi: 18450}}

rules:
  primary_dimension: desi            # used by minimum fill; default: first capacity
  min_fill: {spot: 0.10, owned: 0.0} # share of capacity a vehicle must carry
  stopover: {enabled: true, max_stretch: 1.3, forward_only: true, max_stops: null}
  split_demand: true                 # a demand may travel in several vehicles
```

What the fields mean:

- **Centers.** `id` is the name used everywhere. Ids are trimmed and Unicode
  NFC-normalized, and they are case sensitive. A banned vehicle type may not
  leave from, stop at or arrive at the center.
- **Distances.** With `haversine` they are great-circle distances from the
  coordinates, times `factor`. With `explicit` you give arcs
  (`{origin, destination, km}`) for every ordered pair. `symmetric: true`
  fills in each missing reverse direction.
- **Vehicle types.** Every type needs the same capacity dimensions (desi, kg,
  ...). Every demand must give every dimension.
- **Owned vehicles.**
  - `count` vehicles run on one line every day.
  - With `mandatory: true` their fixed cost is paid every day, used or not; a
    dispatch then only costs `per_km`.
  - With `stopovers: true` they may drop loads at centers on the way, but they
    always end at the line's destination.
- **Spot offers.**
  - Each dispatch costs `fixed + per_km x route km`.
  - `available` limits vehicles per day; leave it out for unlimited.
  - There is one offer per vehicle type.
- **Demand** is shipped on its date. Ids are generated (`D000001`, ...) when left out.
- **Routes** are one-way: from the origin through the stops in order. There is no return trip.
- **Stopover rules.**
  - `max_stretch` bounds route length divided by the direct distance to the last stop.
  - `forward_only` requires each stop to be farther from the origin than the one before it.
  - `max_stops` bounds the number of stops.

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
     quantity: {desi: 5000}}             # due: 2026-05-12T17:00 also possible
rules:
  service: {due_hours: 24, penalty: 0.4, max_deferral_days: 1}
  handling: {minutes_per_unit: 0.01}     # 100 desi a minute
```

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
  file: demand.csv          # relative to the YAML file
  delimiter: ";"
  encoding: cp1254          # default utf-8-sig
  decimal: ","              # then "." is a thousands separator
  date_format: "%d.%m.%Y"   # default ISO dates
  columns: {origin: Çıkış, destination: Varış, date: Tarih}
  quantity: {desi: Desi}    # demand only: dimension -> column
```

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

instance = pirilib.load_spec("problem.yaml").compile()  # read-only numpy arrays
plan = pirilib.Plan.from_dispatches([...], dimensions=1)
evaluation = pirilib.evaluate(instance, plan)
evaluation.total_cost, evaluation.violated(), evaluation.feasible
print(pirilib.check_plan(instance, plan))  # the same rules, explained in words
```

A dispatch is index-based: `Dispatch(day, kind, source, origin, stops, loads)`,
where `source` is the position of the owned group or spot offer in the problem
file. `evaluate` is the one place where costs and rule violations are computed.
Every rule has a violation amount that is zero when the rule holds:

| Constraint | Amount |
|---|---|
| `route` | dispatches without stops, stops at the origin, centers visited twice |
| `load_route` | loads whose demand does not match the dispatch's origin, day or stops |
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

Constraints of your own are objects with a `name` and a
`violation(evaluation)` method. `MaxRouteKm`, `MaxDispatchesPerDay` and
`constraint(name, function)` are in `pirilib.constraints`.

## 3. Your objective

An objective is any function of an evaluation that returns a number to
minimize. The building blocks in `pirilib.objective` add up and take weights:

```python
from pirilib import objective as obj

goal = obj.total_cost() + 1e6 * obj.violations() + 500 * obj.dispatches("spot")
goal.breakdown(evaluation)  # the value of every term
```

The available terms are:

- `total_cost`, `dispatch_cost(kind)`, `owned_fixed_cost`;
- `distance_km(kind)`, `dispatches(kind)`, `unused_capacity(kind)`;
- `unserved(dimension)`;
- `violation(name)`, `violations(names)`.

Terms like these have a *linear form*, so the solver can optimize subproblem by
subproblem, which is fast. A plain function such as
`lambda e: e.total_cost ** 1.1` is optimized on whole plans instead, which is
slower. Constraints of your own are handled the same way.

The default objective is `total_cost() + 1e6 * violations()`, so every rule
behaves like a hard constraint. When a rule cannot be met, the solver does what
the objective prefers. For example, it may leave a small demand uncarried
rather than send a vehicle below its minimum fill, if that costs less in
penalties.

## 4. Solve

```python
result = pirilib.solve(instance, goal, seed=1)
result.plan, result.evaluation, result.objective, result.seconds
```

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
| `exact_nodes` | 0 | solve subproblems with at most this many destinations exactly (needs scipy) |

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
is at least 0.1 % better. `RandomSearch` is the baseline tuner. Budgets below
about 60 settings favour random search. Check tuned settings on instances you
did not tune on.

## 6. Forecast demand

`pirilib.forecast` forecasts daily quantities (desi) per line, a week ahead,
from a history CSV with origin, destination, date and quantity columns.

```python
from pirilib import forecast as fc

history = fc.load_history("history.csv", delimiter=";", decimal=",", date_format="%d.%m.%Y")
calendar = fc.Calendar.of(unusual=[...], holidays=[...])  # days not to learn from; holidays
print(fc.report(fc.compare(history, {"lightgbm": fc.lightgbm_tweedie()}, calendar=calendar)))
week = fc.lightgbm_tweedie().forecast(history, start, 7, calendar)
```

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

## 7. Command line

```bash
pirilib generate sample --centers 20 --days 3      # a synthetic problem
pirilib validate sample/problem.yaml
pirilib solve sample/problem.yaml -o plan.csv --summary summary.json
pirilib solve sample/problem.yaml -o plan.csv --delimiter ";" --decimal ","   # Turkish Excel
pirilib check sample/problem.yaml plan.csv         # exit status 1 if a rule is broken
pirilib tune sample/problem.yaml --budget 60 -o config.json
pirilib solve sample/problem.yaml --config config.json
pirilib schema linehaul.schema.json
pirilib backtest history.csv --holidays holidays.txt --models seasonal,naive,lightgbm
pirilib forecast history.csv --holidays holidays.txt -o forecast.csv \
    --network network.yaml --problem problem.yaml --quantile 0.8
pirilib solve problem.yaml -o plan.csv               # plan the forecast week
```

Date files (`--holidays`, `--unusual`) have one ISO date per line.

Plan files have one row per load. The columns are `dispatch`, `date`, `kind`,
`start` (loading start, `HH:MM` after 00:00 of the date, hours may pass 24),
`vehicle_type`, `source` (for example `fleet.spot[1]`), `origin`, `stops`
(separated by `|`), `route_km`, `dispatch_cost`, `demand`, `destination` and one
column per quantity dimension. `pirilib check` reads them back, so plans made
elsewhere can be audited too.

## Library structure

Classes own state and settings: solvers, tuners, forecasters, backtesters,
plans, evaluations and the specification models. A public function is either a
stateless transformation (`load_spec`, `read_plan`, `synthetic.generate`) or a
thin entry point that builds a class and calls it (`solve`, `tune`, `evaluate`,
`backtest`). Extension points are protocols: `Solver`, `Tuner`, `Forecaster`.

## Known limits

- One-way line haul only: no return trips, working hours or driver rules.
- Costs are fixed, per km and per hour; there are no costs per unit carried.
- The solvers do not yet plan with times, deferral, handling capacity or
  vehicle limits: the ant colony then searches whole plans and only the
  penalties steer it, so handling capacity can be exceeded; the exact solver
  refuses such problems. The evaluator and the checker measure them all.
- Local search runs in plain Python, which roughly doubles solve time on large
  problems.
- The exact solver needs splittable demand and up to about 9 destinations per
  subproblem.
- When spot offers have daily limits, origins of a day take vehicles one after
  another, greedily.
- Forecasts cover one quantity dimension, up to 7 days ahead, per day (no time
  waves). On the prototype data, walk-forward forecasts ran about 6 % low
  because of week-to-week swings around holidays. None of the corrections
  tried kept the accuracy (see `docs/olcumler.md`), so plan with a quantile
  buffer, for example 0.8. `Regression(scale=True)` (learning shapes relative
  to the line's recent level) is available as an experiment.
