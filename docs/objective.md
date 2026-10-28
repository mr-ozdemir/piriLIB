# Building the objective

The objective is the number a solver makes as small as possible. pirilib
computes what a plan costs and how much it breaks each rule; the objective says
how much each of those is worth to you. This page explains the `objective:`
section of problem documents, how the total cost is computed, how to choose
penalty weights, and how to write the same objective, or a different one, in
Python.

## Why there is no default

Earlier versions minimized the total cost plus 1,000,000 per unit of violation
of every rule. That looks harmless, but it makes decisions for you. Take a
40-desi demand that only a 400-desi vehicle can carry, with a minimum fill of
30 %. Carrying it leaves the vehicle 80 desi short of its minimum fill; leaving
it behind leaves 40 desi uncarried. With equal weights the second is cheaper,
so the solver quietly dropped the demand. Whether a missed delivery is worse
than an under-filled truck is a business decision, not a solver setting.

So there is no default objective. `pirilib.solve`, every solver, `tune` and
`Benchmark` need one, and the command line reads it from the problem document.
Every rule a plan can break needs a weight that someone wrote down.

## What is minimized

```text
objective = cost × total cost
          + distance_km × route km
          + dispatches × number of dispatches
          + unused_capacity × capacity left empty (primary dimension)
          + Σ over rules: penalty of the rule × violation amount of the rule
```

Each rule's violation amount is zero when the rule holds. The total cost and
the amounts come from `pirilib.evaluate`, the one place where plans are
measured, so solvers, reports and objectives always read the same numbers.
`pirilib.check_plan` is an independent checker, written separately, and the
test suite compares it with `evaluate`.

## The `objective:` section

```yaml
objective:
  cost: 1
  distance_km: 0
  dispatches: 0
  unused_capacity: 0
  penalties:
    coverage: 10000000
    min_fill: 1000
    other: 1000000
```

| Field | Required | Unit of the weight | Meaning |
|---|---|---|---|
| `cost` | yes, above 0 | per unit of money | Weight of the total cost: dispatch costs (fixed, per km, per hour), the fixed cost of mandatory owned vehicles and lateness. Use 1 unless you want the costs in other units. |
| `distance_km` | no, default 0 | per km | Extra weight per km of route, on top of the `per_km` costs, for example a price on emissions. |
| `dispatches` | no, default 0 | per dispatch | Extra weight per dispatch, owned or spot, for example the effort of arranging one. |
| `unused_capacity` | no, default 0 | per primary unit | Weight per unit of primary capacity left empty when a vehicle departs. |
| `penalties` | yes | per unit of the rule's amount | A weight per built-in rule, by name. `other` applies to every rule that is not named. Weights are 0 or more. |

Every rule must get a weight: either name all of them under `penalties`, or
add `other`. A rule without a weight is an error, and so is a name that is not
a rule. Rules of your own (Python constraints) are not part of the section; see
[Constraints of your own](#constraints-of-your-own).

The money unit is whatever the costs of your document are written in. The
section only adds weights, so penalties are in the same money unit: a coverage
penalty of 1000 means "one desi not carried is as bad as paying 1000".

## Rules and their units

A penalty is a weight per unit of the rule's violation amount, and the unit
differs from rule to rule: a count (of dispatches, loads, stops, vehicles), a
quantity (in the units of your capacity dimensions, such as desi) or km. A
penalty of 1000 on `capacity` costs 1000 per desi above capacity, while on
`ban` it costs 1000 per banned stop. The names are those of
`pirilib.constraints.CATALOG`.

With several capacity dimensions, such as desi and kg, `capacity`, `coverage`
and `overdelivery` add the amounts of all dimensions into one number, so one
penalty prices a desi and a kg alike. 1300 desi and 600 kg left behind are a
`coverage` amount of 1900. To weight dimensions separately, see
[One weight per dimension](#one-weight-per-dimension).

| Rule | Unit | Violation amount |
|---|---|---|
| `route` | count | Malformed routes: dispatches without stops, stops at the origin, and centers visited twice by one dispatch. |
| `load_route` | count | Loads whose demand does not leave from the dispatch origin, or whose destination is not a stop of the dispatch. |
| `ready` | count | Loads taken before their demand is ready. |
| `deferral` | count | Loads that leave before their demand's date, or more than `rules.service.max_deferral_days` after it. |
| `ban` | count | Origins and stops where the vehicle type is banned. |
| `owned_line` | count | Owned dispatches that do not leave from their line's origin or do not end at its destination. |
| `owned_availability` | count | Owned dispatches beyond the vehicle count of a group on a day. |
| `spot_availability` | count | Spot dispatches beyond the daily limit of an offer. |
| `handling` | quantity | Primary quantity loaded and unloaded at a center beyond its daily handling capacity. |
| `vehicle_limit` | count | Vehicles of a type at a center beyond its daily limit. |
| `capacity` | quantity | Load above vehicle capacity, summed over dimensions. |
| `min_fill` | quantity | Load missing to reach the minimum fill of the primary dimension. |
| `stopover` | count | Dispatches with more than one stop although stopovers are disabled, or not allowed for the owned group. |
| `max_stops` | count | Stops beyond `rules.stopover.max_stops`. |
| `stretch` | km | Route length above `max_stretch` times the direct distance to the last stop. |
| `forward_only` | count | Stops that are not farther from the origin than the stop before them. |
| `split` | count | Extra loads of demands that may not be split. |
| `coverage` | quantity | Demand quantity not carried, summed over dimensions. |
| `overdelivery` | quantity | Quantity carried beyond the demand, summed over dimensions. |

The ant colony keeps every rule except `min_fill` and `coverage` by
construction (on problems with handling capacity, a limit is rarely exceeded,
see the known limits in the [user guide](guide.md#known-limits)). The weights of
those rules still matter: they enter the objective value of every plan,
whichever solver made it, and `Benchmark` compares solvers by that value.
`pirilib.check_plan` does not use weights; it reports every violation in words.
`pirilib check` prints the same report, showing the first 50 violations and the
number of the others.

## How the total cost is computed

The total cost is `Evaluation.total_cost`, the sum of three parts. Owned
groups are mandatory by default: a group is `mandatory: true` unless you write
`mandatory: false`.

- **Dispatches.** Each dispatch costs `fixed + per_km × route km + per_hour ×
  hours`, with the costs of its owned group or spot offer. Hours run from the
  start of loading at the origin to the end of unloading at the last stop. A
  dispatch of a mandatory owned group does not pay `fixed`; that is paid in the
  next part. A dispatch of a group with `mandatory: false` pays `fixed` like a
  spot dispatch.
- **Mandatory owned vehicles.** Every mandatory owned group pays
  `fixed × count` on every day of the calendar, used or not
  (`Evaluation.owned_fixed_cost`). Write `mandatory: false` for vehicles whose
  fixed cost is paid only when they run.
- **Lateness.** A load is late when the unloading at its stop ends after the
  due time of its demand. It costs `rules.service.penalty × primary quantity
  × started hours late` (`Evaluation.service_cost`).

### A worked example

Three centers, one owned truck on the line A to B and spot trucks for
everything else. Demand AB (800 desi) is ready at 08:00; demand AC (500 desi)
is ready at 06:00 and due at 08:00, so it cannot arrive in time.

```yaml title="examples/objectives/worked_cost.yaml"
schema: pirilib/linehaul@0.2
name: Worked cost example

calendar: {start: 2026-06-01, end: 2026-06-01}

centers:
  - {id: A}
  - {id: B}
  - {id: C}

matrix:
  source: explicit
  symmetric: true
  arcs:
    - {origin: A, destination: B, km: 100}
    - {origin: A, destination: C, km: 140}
    - {origin: B, destination: C, km: 50}

vehicle_types:
  - {name: truck, capacity: {desi: 1000}, speed_kmh: 50}

fleet:
  owned:
    - {type: truck, origin: A, destination: B, count: 1, cost: {fixed: 400, per_km: 2}}
  spot:
    - {type: truck, cost: {fixed: 300, per_km: 3, per_hour: 20}}

demand:
  - {id: AB, origin: A, destination: B, date: 2026-06-01, ready: "08:00", quantity: {desi: 800}}
  - {id: AC, origin: A, destination: C, date: 2026-06-01, ready: "06:00",
     due: "2026-06-01T08:00", quantity: {desi: 500}}

rules:
  service: {penalty: 0.5}

objective:
  cost: 1
  penalties:
    coverage: 1000
    other: 100000
```

The owned group does not set `mandatory`, so it is mandatory by default: its
fixed cost of 400 is paid for the day, and its dispatch pays only per km. The
plan sends the owned truck to B and a spot truck to C:

| Part | Computation | Amount |
|---|---|---|
| Owned dispatch A to B | 100 km × 2 (its `fixed` is paid as owned fixed cost) | 200 |
| Spot dispatch A to C | 300 + 140 km × 3 + 2.8 h × 20 (140 km at 50 km/h, 06:00 to 08:48) | 776 |
| Mandatory owned vehicles | 400 × 1 vehicle × 1 day | 400 |
| Lateness of AC | 0.5 × 500 desi × 1 started hour (unloaded 48 minutes late) | 250 |
| Total cost | | 1626 |

No rule is broken, so the objective is `1 × 1626 = 1626`. The same numbers
from Python, where `source` is the position of the owned group or spot offer
in the document (here the first of each; see
[Plans and their evaluation](guide.md#2-plans-and-their-evaluation)):

```python
import pirilib
from pirilib import Dispatch, Load, Plan

spec = pirilib.load_spec("examples/objectives/worked_cost.yaml")
instance = spec.compile()
goal = spec.build_objective()

owned = Dispatch(day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (800.0,)),))
spot = Dispatch(day=0, kind="spot", source=0, origin=0, stops=(2,), loads=(Load(1, (500.0,)),))
evaluation = pirilib.evaluate(instance, Plan.from_dispatches([owned, spot], dimensions=1))

assert [round(value, 2) for value in evaluation.dispatch_cost] == [200.0, 776.0]
assert evaluation.owned_fixed_cost == 400.0
assert evaluation.service_cost == 250.0
assert round(evaluation.total_cost, 2) == 1626.0
assert evaluation.feasible
assert round(goal(evaluation), 2) == 1626.0
```

Without the spot truck the plan costs only 600, but 500 desi stay uncarried.
The `coverage` penalty of 1000 per desi adds 500,000:

```python
evaluation = pirilib.evaluate(instance, Plan.from_dispatches([owned], dimensions=1))

assert evaluation.total_cost == 600.0
assert evaluation.violated() == {"coverage": 500.0}
assert goal(evaluation) == 600.0 + 1000 * 500.0
```

The solver finds the first plan:

```python
result = pirilib.solve(instance, goal, seed=1)

assert round(result.objective, 2) == 1626.0
assert result.evaluation.feasible
```

## Choosing penalty weights

**Compare with your costs.** A penalty is money per unit of violation. Look at
what a single dispatch costs in your document, and at the unit of the rule.

- For a rule that must hold, make one unit of violation more expensive than
  anything breaking it could save: 10 to 1000 times the most expensive
  dispatch for count rules. For quantity rules, a weight per unit above the
  most expensive dispatch makes even one desi count. The generated documents
  (`pirilib generate`) use 10,000,000 for `coverage` and 1,000,000 for
  everything else, with dispatches that cost thousands to tens of thousands.
- Much larger weights do not make a rule any harder. When some violation
  cannot be avoided, a huge penalty dwarfs the costs, and plans that differ
  only in cost look alike to the solver's rounding. Stay well below 1e12.
- For a trade-off, use what the violation really costs you. A `coverage`
  penalty of 1000 per desi says "carry a load unless that costs more than 1000
  per desi"; above that, the solver leaves it behind. A `min_fill` penalty of
  1000 per missing desi says an under-filled truck is acceptable when the
  alternative is worse.

**`other`.** `other` gives one weight to every built-in rule you did not name.
It is the usual way to treat the rules that should never break alike, while
naming the few you want to trade off. Naming a rule always wins over `other`.
Without `other`, every rule of the table above must be named.

**A weight of 0.** The rule may be broken for free. The violation is still
measured and reported (`evaluation.violated()`, `pirilib check`), and
`evaluation.feasible` is false, but the solver does not try to avoid it. Use it
deliberately: `min_fill: 0` ignores minimum fill, and `coverage: 0` makes the
empty plan optimal, because leaving every demand behind costs nothing.

### Coverage against minimum fill

The two rules the solver trades against each other are `coverage` and
`min_fill`. A demand smaller than the minimum fill of every vehicle that may
carry it either goes in an under-filled vehicle or stays behind. With weights
`c` for coverage and `m` for minimum fill, carrying `q` units in a vehicle that
needs `f` units costs `m × (f − q)` plus the dispatch, and leaving them costs
`c × q`. The examples in `examples/objectives` show both choices on the same
data, with a 40-desi demand for a vehicle that needs 120:

- `every_load.yaml` (`coverage: 10000000`, `min_fill: 1000`): every load goes,
  and the plan breaks minimum fill by 80 desi.
- `hard_min_fill.yaml` (`coverage: 1000`, `min_fill: 10000000`): every vehicle
  is filled enough, and 40 desi stay behind.

Neither plan is feasible, because the data do not allow one. The objective says
which rule gives way.

## The same objective in Python

`spec.build_objective()` returns a `pirilib.objective.Sum`: the total cost
times `cost`, the extra terms with a weight above 0, and one penalty term per
built-in rule (also when its weight is 0). The same objective can be written
by hand with the term factories of `pirilib.objective`. Terms take weights by
multiplication and add up to a `Sum`; `breakdown` shows the value of every
term. Here `evaluation` is still the plan without the spot truck from the
worked example, and `goal` the objective of `worked_cost.yaml`:

```python
from pirilib import objective as obj
from pirilib.constraints import CATALOG

others = [name for name in CATALOG if name != "coverage"]
by_hand = obj.total_cost() + 1000 * obj.violation("coverage") + 100000 * obj.violations(others)

assert by_hand(evaluation) == goal(evaluation)
assert obj.linear_form(by_hand) == obj.linear_form(goal)
assert by_hand.breakdown(evaluation) == {
    "total_cost": 600.0,
    "violation[coverage]": 500000.0,
    "violations": 0.0,
}
```

The terms are:

| Term | Value | Document field |
|---|---|---|
| `total_cost()` | dispatch costs, mandatory owned fixed cost and lateness | `cost` |
| `dispatch_cost(kind)`, `owned_fixed_cost()`, `service_cost()` | the three parts of the total cost | |
| `distance_km(kind)` | km of all routes | `distance_km` |
| `dispatches(kind)` | number of dispatches | `dispatches` |
| `unused_capacity(kind)` | primary capacity left empty on departure | `unused_capacity` |
| `unserved(dimension)` | demand not carried, in one dimension or summed | |
| `violation(name)` | the amount of one rule, built-in or your own | `penalties` |
| `violations(names)` | the summed amounts of the named rules, or of all of them | `penalties` |

`kind` is `"owned"`, `"spot"` or left out for both, so the Python form can say
things the document cannot, such as a price per spot dispatch only.
`violations()` without names includes `coverage`: adding
`1e7 * obj.violation("coverage")` to `1e6 * obj.violations()` weights coverage
1.1e7. `unserved()` and `violation("coverage")` measure the same quantity.

A second scenario, in Python, with a price per spot dispatch on top of the
document's objective:

```python
from pirilib import ACOConfig, ACOSolver

spec = pirilib.load_spec("examples/objectives/every_load.yaml")
instance = spec.compile()
goal = spec.build_objective() + 200 * obj.dispatches("spot")
solver = ACOSolver(ACOConfig(ants=8, iterations=20))

result = solver.solve(instance, goal, seed=0)
print(goal.breakdown(result.evaluation))
assert result.evaluation.violations["coverage"] == 0.0
```

An objective written in Python is used as it is: pirilib does not check that
it weights every rule. A rule left out costs nothing to break.

### Linear forms and speed

Terms built with `pirilib.objective`, and sums of them, have a *linear form*:
their value written as a weighted sum of per-dispatch and per-demand features
(`obj.FEATURES`). With a linear form the ant colony splits the problem into
(day, origin) subproblems, optimizes each on its own, and improves the best
routes with local search. This is the fast path. An objective without a
linear form, such as a plain function, is optimized on whole plans: each ant
builds a complete plan, which is evaluated and scored, and there is no local
search. This is the general path, and it is much slower on large problems.

`obj.linear_form(goal)` returns the form, or `None`. `result.stats["mode"]`
says which path a solve took:

```python
assert obj.linear_form(goal) is not None
assert result.stats["mode"] == "linear"
```

### A plain function

Any function of an `Evaluation` that returns a number is an objective. Here the
objective above, the document's plus the price per spot dispatch, is kept and a
quadratic price on route length is added, which no linear form can express:

```python
def priced(evaluation):
    return goal(evaluation) + 0.5 * float((evaluation.route_km**2).sum())


slow = solver.solve(instance, priced, seed=0)
assert slow.stats["mode"] == "general"
```

The ant colony's `iterations` count whole plans on the general path, so keep
the settings small while you try a function out.

### Constraints of your own

Constraints written in Python (see the [user guide](guide.md#2-plans-and-their-evaluation))
are passed with `constraints=` and reported under their own name. The
`objective:` section cannot weight them, and `other` does not cover them: add
`weight * obj.violation(name)` to the objective in Python. Such a term has no
linear form, and constraints of your own always take the general path.

```python
from pirilib.constraints import MaxRouteKm

limit = MaxRouteKm(250)
bounded = goal + 10000 * obj.violation(limit.name)
result = solver.solve(instance, bounded, [limit], seed=0)

assert result.stats["mode"] == "general"
assert result.evaluation.violations["max_route_km"] == 0.0
```

### One weight per dimension

`unserved(dimension)` measures the demand not carried in one dimension, and it
has a linear form. To price a missing desi and a missing kg differently, give
`coverage` a weight of 0 in the document and add one term per dimension in
Python. The examples here have only desi, so this line is not run; for a
document with desi and kg and `coverage: 0` under `penalties`:

```{.python .no-run}
goal = spec.build_objective() + 1000 * obj.unserved("desi") + 50 * obj.unserved("kg")
```

`capacity` and `overdelivery` also add up all dimensions, but the ant colony
keeps both by construction, so in practice `coverage` is the one to split.

## Tuning and benchmarks

`tune` scores every ACO setting by its objective on the training instances, so
it takes the objective too. One run minimizes one objective. This example
tunes on `distance_aware.yaml`, whose penalties can all be avoided:

```python
from pirilib.tuning import tune

training = pirilib.load_spec("examples/objectives/distance_aware.yaml")
base = ACOConfig(ants=8, iterations=20)
tuned = tune([training.compile()], training.build_objective(), budget=6, base=base, seeds=(0, 1))
better = ACOSolver(tuned.config)
```

A real run needs a budget of 40 or more and several instances (see the
[user guide](guide.md#5-tune)). Settings are compared by their relative
difference to the base setting. A large penalty that no setting can avoid,
such as the 80 desi of minimum fill above, inflates every objective and makes
cost differences look small in relative terms; tune on instances where the
penalties are avoidable, or on their costs alone.

`Benchmark` scores every solver's plans again with its objective, so solvers
are compared on the same number whatever they report:

```python
from pirilib.bench import FAMILIES, SOLVERS, Benchmark
from pirilib.spec import parse_spec

instances = FAMILIES.get("tiny").instances(2)
family_goal = parse_spec(instances[0].document).build_objective()
table = Benchmark(SOLVERS.make(["direct", "aco"]), objective=family_goal).run(instances)
print(table.summary())
```

Any objective works here, including a plain function. Solvers that need a
linear form (`ExactSolver` and the PyVRP and OR-Tools adapters) are recorded
as skipped on other objectives.

## Command line

`pirilib solve`, `tune` and `bench` minimize the objective of the problem
documents; there is no option to change it.

- `solve` reads the document's `objective:` section.
- `tune` needs every training document to have the same `objective:` section.
- `bench` reads the section of the generated family documents.
- `pirilib generate` writes `cost: 1` with `coverage: 10000000` and
  `other: 1000000` (`pirilib.synthetic.OBJECTIVE`), and
  `pirilib forecast --network` copies the network's section into the problem
  it writes.
- `validate` and `check` do not need the section.

```bash
pirilib solve examples/objectives/every_load.yaml -o plan.csv
```

## Common errors

A document without an `objective:` section loads and validates, but solving
it stops. From Python, `spec.build_objective()` raises `pirilib.SpecError`;
`pirilib solve` prints the same message and exits with status 2:

```text
error: problem.yaml: the problem document has no objective: section; solving and tuning need one, for example

objective:
  cost: 1
  penalties:
    coverage: 10000000
    other: 1000000

Every rule a plan can break needs a penalty weight, by name or through 'other'.
```

A rule without a penalty, when `other` is missing:

```text
problem.yaml: invalid problem specification
- objective.penalties: no penalty for route, load_route, ready, deferral, ban, owned_line, owned_availability, spot_availability, handling, vehicle_limit, capacity, min_fill, stopover, max_stops, stretch, forward_only, split, overdelivery; every rule needs a weight: add them, or 'other' for every rule not named
```

A misspelt rule name:

```text
problem.yaml: invalid problem specification
- objective.penalties.coverge: unknown rule 'coverge'; known: route, load_route, ready, deferral, ban, owned_line, owned_availability, spot_availability, handling, vehicle_limit, capacity, min_fill, stopover, max_stops, stretch, forward_only, split, coverage, overdelivery, other
```

Invalid weights and fields, all reported at once:

```yaml
objective:
  cost: 0
  distance: 3
  penalties:
    min_fill: -1
    other: 1000000
```

```text
problem.yaml: invalid problem specification
- objective.cost: Input should be greater than 0
- objective.penalties.min_fill: Input should be greater than or equal to 0
- objective.distance: Extra inputs are not permitted
```

Training documents with different sections (`pirilib tune`, exit status 2):

```text
error: b.yaml and a.yaml define different objectives; one run minimizes one objective, so give them the same objective: section
```

From Python, leaving the objective out is a `TypeError`, and passing `None`
is a `ValueError`. These lines are not meant to run:

```{.python .no-run}
pirilib.solve(instance)
pirilib.solve(instance, None)
```

```text
TypeError: solve() missing 1 required positional argument: 'objective'
ValueError: solve needs an objective: build one with pirilib.objective, or from the objective: section of the problem document (LineHaulSpec.build_objective)
```

`tune` and `Benchmark` raise the same `ValueError`, with their own name. A
misspelt name in `obj.violation` fails when the first plan is evaluated, and a
term of an unknown kind fails before solving:

```text
KeyError: "no constraint named 'capacty'; known: route, load_route, ready, deferral, ban, owned_line, owned_availability, spot_availability, handling, vehicle_limit, capacity, min_fill, stopover, max_stops, stretch, forward_only, split, coverage, overdelivery"
ValueError: the objective uses unknown features ['dispatches[rented]']
```

## Example scenarios

[`examples/objectives`](https://github.com/mr-ozdemir/piriLIB/tree/master/examples/objectives)
holds small synthetic documents that differ mainly in their objective:

| Document | Objective | Result |
|---|---|---|
| `every_load.yaml` | `coverage` 10,000,000, `min_fill` 1000 | every load goes; minimum fill broken by 80 desi |
| `hard_min_fill.yaml` | `coverage` 1000, `min_fill` 10,000,000 | minimum fill holds; 40 desi stay behind |
| `distance_aware.yaml` | `distance_km` 20 | two direct trips (220 km) instead of one detour (300 km) |
| `worked_cost.yaml` | `cost` 1 | the worked example above |

In `distance_aware.yaml`, one van can serve B and C on a 300 km route for 1300,
or two vans can go direct, 220 km in all, for 2220. Without the distance weight
the detour is cheaper; at 20 per km the 80 km saved are worth 1600, so the
solver sends two vans.
