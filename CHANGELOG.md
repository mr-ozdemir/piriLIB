# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/) written in PEP 440 form.

[Unreleased]: https://github.com/mr-ozdemir/piriLIB/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/mr-ozdemir/piriLIB/releases/tag/v0.1.0a1

## [Unreleased]

### Added

- `ExactSolver` plans problems with times, deferral, handling capacity and
  vehicle limits when handling takes no time (`rules.handling.minutes_per_unit`
  is 0).
- `read_plan` and `pirilib check` read .xlsx plans, as written by
  `pirilib solve -o plan.xlsx`.
- Test levels (`unit`, `integration`, `system`) on every test, fuzz tests of
  the plan and problem readers, a model-based test of the ledger, opt-in
  performance tests (`pytest --performance`), branch coverage in CI and a
  weekly quality workflow with mutation testing.
- `tune(validation=...)` and `pirilib tune --validation` score the best few
  settings on validation instances and keep the base unless one beats it there.
- `pirilib.tuning.default_sources` is public.
- `docs/testing.md`: how the tests are organised and written, and how to run
  the fuzz, model-based, performance and mutation tests.

### Changed

- `pirilib forecast --network` warns when no `--quantile` buffer is given.

## [0.1.0a1] - unreleased

### Added

- Project skeleton: src layout, hatchling build, Apache-2.0 license, typing
  marker, test and lint tooling, continuous integration.
- Line-haul problem specification `pirilib/linehaul@0.1` (`pirilib.spec`):
  calendar, centers, distance matrix (haversine or explicit arcs), vehicle types
  with capacity per quantity dimension, owned vehicles and spot offers, demand,
  and explicit business rules (minimum fill, stopovers, demand splitting).
- `pirilib.load_spec` reads YAML documents whose tables may come from CSV files
  with configurable delimiter, encoding, decimal separator, date format and
  column names. Every problem is reported at once, with file and line for CSV
  rows.
- Consistency checks for references, duplicate ids, dimensions, the calendar,
  distance coverage and whether every demand has an allowed vehicle.
- `LineHaulSpec.compile()` returns the problem as read-only numpy arrays.
- JSON Schema of problem documents in `schemas/linehaul-0.1.schema.json`.
- Synthetic example problem in `examples/linehaul_mini` (CC0-1.0).
- Plans (`pirilib.Plan`, `Dispatch`, `Load`): one-way multi-stop dispatches of
  owned or spot vehicles with loads per demand, in array form.
- `pirilib.evaluate`: the single, numpy-vectorized computation of plan cost,
  loads, fill and the violation amount of every built-in constraint
  (`pirilib.constraints.CATALOG`), plus constraints of your own.
- `pirilib.objective`: weighted terms (cost, distance, dispatches, unused
  capacity, unserved demand, violations) that add up to an objective; any
  function of an evaluation works too.
- `pirilib.check_plan`: an independent plain-Python checker that explains every
  violation in words; property tests compare it with the evaluator.
- `pirilib.solve` and `ACOSolver`: ant colony optimization over (day, origin)
  subproblems, vectorized with numpy across ants and subproblems, with a
  MAX-MIN hypercube pheromone model. Objectives with a linear form are
  optimized subproblem by subproblem; any other objective, or constraints of
  your own, are optimized on whole plans. All built-in rules except minimum
  fill and coverage hold by construction.
- Local search on every subproblem's best routes: change vehicle, merge,
  relocate, rebalance loads for minimum fill, serve leftover demand, and drop
  routes that cost more than the demand they carry.
- `pirilib.solvers.exact.ExactSolver`: optimal solutions of small subproblems
  by mixed-integer programming (extra `pirilib[exact]`, scipy/HiGHS), used as a
  benchmark oracle and, with `ACOConfig(exact_nodes=...)`, inside the colony.
- `pirilib.tuning`: an artificial bee colony tuner (rank-based onlookers,
  strict acceptance, modification rate, evaluation budget), random search as
  the baseline, search spaces over the unit cube, and `tune()` for ACO settings
  on training instances, with fixed seeds and a minimum-improvement guard.
- `pirilib` command line: `validate`, `solve`, `check`, `tune`, `generate`
  and `schema`.
- Plan files (`pirilib.planio`): one row per load, with names instead of
  indices, as CSV (Turkish spreadsheet format supported) or Excel (extra
  `pirilib[excel]`), and read back for checking.
- `pirilib.synthetic`: synthetic problems with a gravity demand model.
- User guide in `docs/guide.md`; release workflow with trusted publishing.
- `pirilib.forecast`: daily demand per line from a history CSV; cleaning
  without look-ahead, the prototype's features, seasonal baselines,
  `Regression` adapter for scikit-learn-style models, `lightgbm_tweedie` (extra
  `pirilib[forecast]`), walk-forward `backtest`/`compare`, tuning with the bee
  colony, and `planning_document` with quantile buffers to plan a forecast.
- `pirilib backtest` and `pirilib forecast` commands; synthetic histories in
  `pirilib.synthetic.history`.
- Schema `pirilib/linehaul@0.2` (0.1 files still load): vehicle speeds,
  hourly costs, demand ready and due times, a lateness penalty, deferral of
  demand by days, handling minutes, daily handling capacity and vehicle
  limits per center. Plans carry start times; the evaluator and the checker
  compute the timeline, hourly costs, lateness (`service_cost`, part of
  `total_cost`) and four new constraints (`ready`, `deferral`, `handling`,
  `vehicle_limit`). Plan files have a `start` column.
- `BeeColony` starts from a Latin hypercube and, unless `sources` is set,
  sizes the colony by the budget, which helps small tuning budgets.
- Local search is about ten times faster (plans and scores unchanged): cheap
  pre-checks and remembered rejections of move candidates.
- `pirilib.forecast.Backtester` keeps fold settings for repeated backtests and
  comparisons.
- Time waves in `pirilib.forecast`: `WaveSplitter` learns wave shares per line
  and weekday from a wave-level history (`load_wave_history`) without
  look-ahead, `planning_document(..., waves=...)` makes demand rows ready at
  the wave times, and `pirilib forecast --waves` splits from the command line.
- Documentation site (MkDocs, Material theme) with the user guide and an API
  reference generated from the docstrings; `docs` dependency group, and CI
  builds it with `mkdocs build --strict`.
- `pirilib.bench` and `pirilib bench`: reproducible synthetic instance
  families, a benchmark runner with CSV results and per-solver summaries, the
  `DirectShipping` baseline, and optional `PyVRPSolver` and `ORToolsSolver`
  adapters (extras `pirilib[pyvrp]`, `pirilib[ortools]`).
- The ant colony plans with times and center limits: on scheduled problems the
  subproblems run in urgency order against a ledger of handling capacity and
  vehicle slots per center and day, routes are timed (hourly costs and
  lateness in the scores, also in local search), and demand that does not fit
  waits for the next day within `max_deferral_days`. Objectives with a linear
  form keep the fast path; whole-plan search keeps a ledger per ant.
- Documentation page "Building the objective" (`docs/objective.md`): the
  `objective:` section field by field, the unit of every rule's penalty, a
  worked total-cost example, advice on choosing weights (including coverage
  against minimum fill), the same objective in Python, linear forms and speed,
  tuning and benchmarks, and the real error messages. Example scenarios in
  `examples/objectives` (every load goes, minimum fill as a hard rule,
  distance-aware, the worked example), and `tests/test_docs_examples.py`,
  which solves them, runs the page's Python blocks and checks its rule table
  against `CATALOG` and its error messages against the code.

### Changed

- The ant colony builds its solutions about 1.5 to 1.7 times faster, with the
  same plans and scores to the last bit: per-option and per-node values are
  laid out once per build, the weights of opening a route are kept and updated
  only where a row loads, and sampling and each step use fewer numpy calls.
  Large problems solve about a third faster in total.
- There is no default objective any more: `default_objective` and
  `DEFAULT_PENALTY` are gone, and `objective` is a required argument of
  `pirilib.solve`, `Solver.solve` (`ACOSolver`, `ExactSolver`, `DirectShipping`,
  the routing adapters), `tuning.tune` and `bench.Benchmark`. Problem documents
  define it in a new optional `objective:` section (`ObjectiveSpec`: `cost`,
  `distance_km`, `dispatches`, `unused_capacity` and a penalty per rule, or
  `other` for the rules not named; a rule without a penalty is an error), and
  `LineHaulSpec.build_objective()` turns it into a `Sum`. The command line
  `solve`, `tune` and `bench` read it from the documents, and their `--penalty`
  option was removed. Generated documents and the example carry an explicit
  `objective:` section.
- `BeeColony` without `sources` uses three food sources from 24 to 119
  evaluations and four from 120 (before: four from 32), which did better on
  test functions and on ACO tuning at the default budget of 40 settings. The
  default budget stays 40.
