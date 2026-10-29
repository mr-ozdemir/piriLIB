# Public API and stability

This page records which names are meant to become the stable 1.0 API, which
stay experimental, and what has to happen before the API is frozen. It is
reviewed together with the decision log during the alpha.

## Alpha notice

pirilib 0.1.0a1 is the first alpha release. Until 1.0, everything may change
in any release, alphas and betas included: public names, the problem documents
and their schema ids, plan files, command line options and the internal
structure. The lists below are the intent for 1.0, not a promise for the
releases before it.

The default domain of pirilib is fleet optimization in general, built on one
core of solver, tuner, objective and adapters. Line-haul planning (schema
`pirilib/linehaul@0.2`, example `examples/linehaul_mini`) is the first
supported problem type, not the permanent shape of the library: its models
and arrays may be reorganized as other problem types are added, and the
library stays open to problems outside fleet optimization.

## Rules after 1.0

- Semantic versioning. Breaking changes only in a new major version.
- A name is public when it is exported by a public module's `__all__` and has
  no leading underscore. Modules whose names start with `_` are internal.
- A deprecated name keeps working for at least one minor release and warns with
  `DeprecationWarning` before it is removed in the next major version.
- Problem documents carry a schema id (`pirilib/linehaul@0.2`). Every 1.x
  release reads every schema id that a 1.0 release read.
- The plan file columns (`pirilib solve -o plan.csv`) and the command line
  options are part of the API.
- Every error caused by user input is a `PirilibError` (`SpecError`,
  `PlanError`).
- Supported Python versions: the three newest minor versions at release time,
  at least 3.12.

## Stable at 1.0

| Area | Names |
|---|---|
| Problem | `load_spec`, `parse_spec`, `compile_spec`, `LineHaulSpec` and its models (`Calendar`, `Center`, `VehicleType`, `Cost`, `Fleet`, `OwnedVehicles`, `SpotOffer`, `Demand`, `Rules`, `MinFill`, `Stopover`, `Service`, `Handling`, `ObjectiveSpec`, `HaversineMatrix`, `ExplicitMatrix`, `Arc`, `TableSource`), `SCHEMA_ID`, `UNLIMITED` |
| Plans | `Plan`, `Dispatch`, `Load`, `planio.read_plan`, `planio.write_plan` |
| Evaluation | `evaluate`, `Evaluation`, `check_plan`, `constraints.CATALOG`, `Constraint`, `constraint`, `MaxRouteKm`, `MaxDispatchesPerDay` |
| Objectives | every term factory in `pirilib.objective` (`total_cost`, `dispatch_cost`, `service_cost`, `owned_fixed_cost`, `distance_km`, `dispatches`, `unused_capacity`, `unserved`, `violation`, `violations`), `Term`, `Sum` |
| Solving | `solve`, `Solver` (protocol), `ACOSolver`, `ACOConfig`, `SolveResult`, `LineHaulSpec.build_objective` (there is no default objective) |
| Tuning | `tune`, `Tuner` (protocol), `BeeColony`, `RandomSearch`, `SearchSpace`, `Int`, `Float`, `Choice`, `aco_space`, `default_sources`, `TuningResult`, `Candidate`, `SearchResult`, `Trace` |
| Forecasting | `Forecaster` (protocol), `History`, `load_history`, `clean`, `Calendar`, `Forecast`, `SeasonalMean`, `SeasonalNaive`, `Regression`, `Backtester`, `backtest`, `compare`, `report`, `Backtest`, `Fold`, `Scores` |
| Command line | `validate`, `solve`, `check`, `tune`, `generate`, `backtest`, `forecast`, `schema` and their options |

`LineHaulInstance` is stable as a type. Its array attributes are what custom
constraints and solvers read; before 1.0 the documented subset becomes
stable and the rest is renamed with a leading underscore.

## Experimental at 1.0

These keep working but may change in a minor release; they are marked as such
in their docstrings.

- `pirilib.solvers.exact.ExactSolver` and `ACOConfig.exact_nodes`.
- Time, service and capacity rules of schema 0.2 in the solvers, until they
  have been measured on more than synthetic problems.
- `Regression(scale=True)`, `lightgbm_tweedie`, `PROTOTYPE_TWEEDIE`,
  `lightgbm_space`, `tune_forecaster`, `tune_lightgbm`, `ForecastTuning`.
- The forecast-to-plan bridge: `buffer_factors`, `planning_document`,
  `demand_rows`.
- `pirilib.synthetic`: generated problems may change between releases; only the
  function signatures are kept.
- `pirilib.bench`, `pirilib bench`, `pirilib.solvers.direct.DirectShipping` and the
  adapters in `pirilib.solvers.adapters`.
- `objective.linear_form`, `objective.FEATURES`: the contract between objectives
  and the fast solver path.

## Internal

`pirilib.solvers._colony`, `pirilib.solvers._routing`, `pirilib.solvers._timing`,
`pirilib.solvers.ledger`, `pirilib.solvers.decompose`,
`pirilib.solvers.local_search`, `pirilib.spec.tables`, `pirilib.spec._format`,
`pirilib.geo` and `pirilib.cli` beyond `main`.

## Before the freeze

1. Done: every public module has an `__all__`, so imports such as `np` or
   `dataclass` are no longer part of the API.
2. Decide the documented subset of `LineHaulInstance` arrays.
3. Rename the internal modules listed above with a leading underscore, or say
   in their docstrings that they are internal.
4. Mark the experimental names in their docstrings, and give the public
   methods that have none a docstring (`ACOSolver.solve`, `Plan.dispatches`,
   `cli.main` and others shown without text on the reference pages).
5. Settle the open decisions of the alpha review (`docs/kararlar.md`).
6. Freeze schema `pirilib/linehaul@1.0` and keep reading 0.1 and 0.2.

## Roadmap

pirilib is heading towards a general fleet optimization library: one solver,
tuner and objective core with problem types on top of it. Which problem types
follow line-haul is not decided yet.

| Version | Scope |
|---|---|
| 0.1.0a1 | The line-haul problem type (schema `pirilib/linehaul@0.2` with times, service, deferral and center limits), evaluator, constraint catalog and plan checker, numpy ACO with local search and an exact solver, ABC tuner, demand forecasting with the forecast-to-plan bridge, benchmarks with PyVRP and OR-Tools baselines, command line |
| Later alphas and betas | Changes from alpha feedback; the API and the internal structure may still change |
| 1.0 | Stable API, as listed on this page |
