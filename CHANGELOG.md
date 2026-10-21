# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
uses [Semantic Versioning](https://semver.org/) written in PEP 440 form.

## [Unreleased]

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
- Documentation site (MkDocs, Material theme) with the user guide and an API
  reference generated from the docstrings; `docs` dependency group, and CI
  builds it with `mkdocs build --strict`.
