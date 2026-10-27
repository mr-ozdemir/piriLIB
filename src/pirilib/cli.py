"""The ``pirilib`` command line.

Commands::

    pirilib validate problem.yaml
    pirilib solve problem.yaml -o plan.csv
    pirilib check problem.yaml plan.csv
    pirilib tune train1.yaml train2.yaml --budget 60 -o config.json
    pirilib generate folder --centers 20 --days 5
    pirilib backtest history.csv --models seasonal,lightgbm
    pirilib forecast history.csv -o forecast.csv --network network.yaml --problem problem.yaml
    pirilib forecast history.csv --waves waves.csv --network network.yaml
    pirilib schema linehaul.schema.json
    pirilib bench small --solvers direct,aco,exact -o results.csv --summary summary.csv

``solve``, ``tune`` and ``bench`` minimize the objective that the problem
documents define in their ``objective:`` section; there is no default one.

Exit status: 0 on success, 1 when an input is invalid or a plan breaks a rule
(``check``), 2 for wrong usage, including solving or tuning a problem whose
document has no ``objective:`` section.
"""

import argparse
import csv
import datetime as dt
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from pirilib import objective as obj
from pirilib._version import __version__
from pirilib.errors import PirilibError, SpecError
from pirilib.evaluation import evaluate
from pirilib.planio import read_plan, summary, write_plan
from pirilib.solvers import ACOConfig, ACOSolver
from pirilib.spec import LineHaulSpec, load_spec, parse_spec


class _UsageError(Exception):
    """Wrong usage found after the arguments are read; the command line exits with 2."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line with ``argv`` (default ``sys.argv``) and return the exit code."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result: int = arguments.run(arguments)
    except _UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (PirilibError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pirilib", description="Line-haul planning with ACO and ABC."
    )
    parser.add_argument("--version", action="version", version=f"pirilib {__version__}")
    commands = parser.add_subparsers(required=True, metavar="command")

    command = commands.add_parser("validate", help="check a problem file and summarize it")
    command.add_argument("problem")
    command.set_defaults(run=_validate)

    command = commands.add_parser("solve", help="plan a problem with the ant colony")
    command.add_argument("problem")
    command.add_argument("-o", "--output", help="plan file (.csv, or .xlsx with openpyxl)")
    command.add_argument("--summary", help="write the headline numbers as JSON")
    command.add_argument("--config", help="ACO settings as JSON, for example from 'tune'")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--ants", type=int)
    command.add_argument("--iterations", type=int)
    command.add_argument(
        "--exact-nodes", type=int, help="solve subproblems this small exactly (scipy)"
    )
    command.add_argument("--no-local-search", action="store_true")
    _table_options(command)
    command.set_defaults(run=_solve)

    command = commands.add_parser("check", help="check a plan file against a problem")
    command.add_argument("problem")
    command.add_argument("plan")
    _table_options(command)
    command.set_defaults(run=_check)

    command = commands.add_parser("tune", help="tune ACO settings on training problems")
    command.add_argument("problems", nargs="+")
    command.add_argument("-o", "--output", help="write the settings as JSON")
    command.add_argument("--budget", type=int, default=40, help="settings to try")
    command.add_argument("--seeds", type=int, default=1, help="runs per problem and setting")
    command.add_argument("--tuner", choices=("abc", "random"), default="abc")
    command.add_argument("--config", help="base ACO settings as JSON")
    command.set_defaults(run=_tune)

    command = commands.add_parser("generate", help="write a synthetic problem")
    command.add_argument("folder")
    command.add_argument("--centers", type=int, default=12)
    command.add_argument("--days", type=int, default=3)
    command.add_argument("--demand-per-day", type=int, default=60)
    command.add_argument("--seed", type=int, default=0)
    command.set_defaults(run=_generate)

    command = commands.add_parser("backtest", help="walk-forward backtest of forecasters")
    _history_options(command)
    command.add_argument("--models", default="seasonal,naive,lightgbm", help="comma-separated")
    command.add_argument("--folds", type=int, default=8)
    command.add_argument("--days", type=int, default=7)
    command.set_defaults(run=_backtest)

    command = commands.add_parser("forecast", help="forecast daily demand per line")
    _history_options(command)
    command.add_argument("--model", choices=sorted(_FORECASTERS), default="lightgbm")
    command.add_argument("--start", help="first forecast day (default: after the history)")
    command.add_argument("--days", type=int, default=7)
    command.add_argument(
        "-o", "--output", help="forecast CSV (origin, destination, date, quantity)"
    )
    command.add_argument("--network", help="problem YAML without demand, to plan the forecast")
    command.add_argument("--problem", help="where to write the problem YAML for the forecast")
    command.add_argument(
        "--quantile", type=float, help="buffer each line to this quantile of past errors"
    )
    command.add_argument(
        "--buffer-folds", type=int, default=4, help="backtest folds for the buffer"
    )
    command.add_argument(
        "--waves",
        help="CSV with origin, destination, date, wave and quantity, to split days into waves",
    )
    command.add_argument(
        "--wave-column", default="wave", help="wave time column of --waves (default wave)"
    )
    command.add_argument(
        "--wave-weeks", type=int, default=4, help="recent weeks to learn wave shares from"
    )
    command.set_defaults(run=_forecast)

    command = commands.add_parser("schema", help="print or write the JSON Schema of problem files")
    command.add_argument("path", nargs="?")
    command.set_defaults(run=_schema)

    command = commands.add_parser("bench", help="benchmark solvers on a synthetic instance family")
    command.add_argument("family", nargs="?", help="leave out to list the families and solvers")
    command.add_argument("--solvers", default="direct,aco", help="comma-separated solver names")
    command.add_argument("--count", type=int, help="instances (default: the family's size)")
    command.add_argument("--start", type=int, default=0, help="index of the first instance")
    command.add_argument("--seeds", type=int, default=1, help="runs per instance and solver")
    command.add_argument("-o", "--output", help="write one row per run as CSV")
    command.add_argument("--summary", help="write the summary per solver as CSV")
    command.set_defaults(run=_bench)
    return parser


def _table_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--delimiter", default=",", help="CSV delimiter of plan files")
    command.add_argument("--decimal", choices=(".", ","), default=".", help="decimal separator")


_FORECASTERS = ("seasonal", "naive", "lightgbm")


def _history_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("history", help="CSV with origin, destination, date and quantity")
    command.add_argument("--column", default="desi", help="quantity column (default desi)")
    command.add_argument("--delimiter", default=",")
    command.add_argument("--decimal", choices=(".", ","), default=".")
    command.add_argument("--date-format", help="strptime format, for example %%d.%%m.%%Y")
    command.add_argument("--unusual", help="file of dates (one per line) not to learn from")
    command.add_argument("--holidays", help="file of holiday dates (one per line)")


def _dates(path: str | None) -> list[dt.date]:
    if not path:
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        dt.date.fromisoformat(line.strip())
        for line in lines
        if line.strip() and not line.startswith("#")
    ]


def _csv_options(arguments: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {"delimiter": arguments.delimiter, "decimal": arguments.decimal}
    if arguments.date_format:
        options["date_format"] = arguments.date_format
    return options


def _load_history(arguments: argparse.Namespace) -> tuple[Any, Any]:
    from pirilib.forecast import Calendar, load_history

    options = _csv_options(arguments)
    history = load_history(arguments.history, dimension=arguments.column, **options)
    holidays = _dates(arguments.holidays)
    calendar = Calendar.of(unusual=[*_dates(arguments.unusual), *holidays], holidays=holidays)
    return history, calendar


def _forecaster(name: str) -> Any:
    from pirilib.forecast import SeasonalMean, SeasonalNaive, lightgbm_tweedie

    if name == "seasonal":
        return SeasonalMean()
    if name == "naive":
        return SeasonalNaive()
    if name == "lightgbm":
        return lightgbm_tweedie()
    raise ValueError(f"unknown model '{name}'; known: {', '.join(_FORECASTERS)}")


def _backtest(arguments: argparse.Namespace) -> int:
    from pirilib.forecast import compare, report

    history, calendar = _load_history(arguments)
    names = [name.strip() for name in arguments.models.split(",") if name.strip()]
    models = {name: _forecaster(name) for name in names}
    results = compare(
        history,
        models,
        baselines=False,
        folds=arguments.folds,
        days=arguments.days,
        calendar=calendar,
    )
    print(f"{history!r}")
    print(report(results, baseline="seasonal"))
    return 0


def _forecast(arguments: argparse.Namespace) -> int:
    from pirilib.forecast import (
        Forecast,
        WaveSplitter,
        backtest,
        buffer_factors,
        load_wave_history,
        planning_document,
    )
    from pirilib.spec.tables import materialize_tables

    history, calendar = _load_history(arguments)
    start = (
        dt.date.fromisoformat(arguments.start)
        if arguments.start
        else history.end + dt.timedelta(days=1)
    )
    model = _forecaster(arguments.model)
    forecast = model.forecast(history, start, arguments.days, calendar)
    print(f"{forecast!r}")
    factors = None
    if arguments.quantile is not None:
        past = backtest(
            history, model, folds=arguments.buffer_folds, days=arguments.days, calendar=calendar
        )
        factors = buffer_factors(past, arguments.quantile, history.lines)
        lifted = float((forecast.values * factors[:, None]).sum())
        print(
            f"buffered to quantile {arguments.quantile}: total {lifted:,.1f} {forecast.dimension}"
        )
    shares = None
    if arguments.waves:
        waves = load_wave_history(
            arguments.waves,
            dimension=arguments.column,
            wave=arguments.wave_column,
            **_csv_options(arguments),
        )
        shares = WaveSplitter(weeks=arguments.wave_weeks).fit(waves, start, calendar)
        print(f"{shares!r}")
        print(shares.stability().describe())
    if arguments.output:
        values = forecast.values if factors is None else forecast.values * factors[:, None]
        planned = Forecast(
            forecast.origins, forecast.destinations, forecast.start, values, forecast.dimension
        )
        with Path(arguments.output).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if shares is None:
                writer.writerow(["origin", "destination", "date", forecast.dimension])
                for origin, destination, date, quantity in planned.rows(0.5):
                    writer.writerow([origin, destination, date.isoformat(), round(quantity, 3)])
            else:
                writer.writerow(["origin", "destination", "date", "wave", forecast.dimension])
                for origin, destination, date, wave, quantity in shares.split(planned).rows(0.5):
                    writer.writerow(
                        [
                            origin,
                            destination,
                            date.isoformat(),
                            wave.isoformat("minutes"),
                            round(quantity, 3),
                        ]
                    )
        print(f"forecast written to {arguments.output}")
    if arguments.network:
        network_path = Path(arguments.network)
        network = yaml.safe_load(network_path.read_text(encoding="utf-8"))
        network, _ = materialize_tables(network, network_path.parent)
        document = planning_document(network, forecast, factors=factors, waves=shares)
        parse_spec(document, source=str(network_path))
        target = Path(arguments.problem or network_path.with_name("problem.yaml"))
        target.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), "utf-8")
        print(f"problem written to {target}")
    return 0


def _objective(problems: Sequence[tuple[str, LineHaulSpec]]) -> obj.Sum:
    """The objective the documents define; they must define the same one.

    A document without an ``objective:`` section, or documents with different
    sections, are usage errors.
    """
    if not problems:
        raise _UsageError("there is no problem to take the objective from")
    goals = []
    for label, spec in problems:
        try:
            goals.append(spec.build_objective())
        except SpecError as exc:
            raise _UsageError(f"{label}: {exc}") from exc
    first_label, first = problems[0]
    for label, spec in problems[1:]:
        if spec.objective != first.objective:
            raise _UsageError(
                f"{label} and {first_label} define different objectives; one run "
                "minimizes one objective, so give them the same objective: section"
            )
    return goals[0]


def _config(path: str | None, **overrides: Any) -> ACOConfig:
    values: dict[str, Any] = {}
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(ACOConfig)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"{path}: unknown ACO settings {unknown}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    return ACOConfig(**values)


def _print_summary(numbers: dict[str, Any]) -> None:
    state = "feasible" if numbers["feasible"] else f"violations {numbers['violations']}"
    print(
        f"total cost {numbers['total_cost']:,.2f}  dispatches {numbers['dispatches']} "
        f"({numbers['spot_dispatches']} spot)  km {numbers['total_km']:,.1f}  {state}"
    )


def _validate(arguments: argparse.Namespace) -> int:
    spec = load_spec(arguments.problem)
    instance = spec.compile()
    print(f"{arguments.problem}: valid")
    print(f"  {instance!r}")
    quantity = instance.demand_quantity.sum(axis=0)
    totals = ", ".join(
        f"{q:,.1f} {name}" for name, q in zip(instance.dimensions, quantity, strict=True)
    )
    print(f"  demand {totals}, {instance.days[0]} to {instance.days[-1]}")
    return 0


def _solve(arguments: argparse.Namespace) -> int:
    spec = load_spec(arguments.problem)
    goal = _objective([(arguments.problem, spec)])
    instance = spec.compile()
    config = _config(
        arguments.config,
        ants=arguments.ants,
        iterations=arguments.iterations,
        exact_nodes=arguments.exact_nodes,
        local_search=False if arguments.no_local_search else None,
    )
    result = ACOSolver(config).solve(instance, goal, seed=arguments.seed)
    numbers = summary(result.evaluation) | {
        "objective": result.objective,
        "seconds": result.seconds,
    }
    _print_summary(numbers)
    print(f"objective {result.objective:,.2f} in {result.seconds:.1f} s")
    if arguments.output:
        write_plan(
            result.evaluation,
            arguments.output,
            delimiter=arguments.delimiter,
            decimal=arguments.decimal,
        )
        print(f"plan written to {arguments.output}")
    if arguments.summary:
        Path(arguments.summary).write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    return 0


def _check(arguments: argparse.Namespace) -> int:
    from pirilib.validate import check_plan

    instance = load_spec(arguments.problem).compile()
    plan = read_plan(
        instance, arguments.plan, delimiter=arguments.delimiter, decimal=arguments.decimal
    )
    report = check_plan(instance, plan)
    _print_summary(summary(evaluate(instance, plan)))
    for item in report.violations[:50]:
        print(f"- {item.constraint}: {item.message}")
    if len(report.violations) > 50:
        print(f"- ... and {len(report.violations) - 50} more")
    return 0 if report.feasible else 1


def _tune(arguments: argparse.Namespace) -> int:
    from pirilib.tuning import BeeColony, RandomSearch, tune

    specs = [(path, load_spec(path)) for path in arguments.problems]
    goal = _objective(specs)
    result = tune(
        [spec.compile() for _, spec in specs],
        goal,
        budget=arguments.budget,
        tuner=BeeColony() if arguments.tuner == "abc" else RandomSearch(),
        base=_config(arguments.config),
        seeds=tuple(range(arguments.seeds)),
    )
    settings = asdict(result.config)
    print(f"score {result.score * 100:+.2f}% against the base settings, {result.seconds:.0f} s")
    print(json.dumps(settings, indent=2))
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return 0


def _generate(arguments: argparse.Namespace) -> int:
    from pirilib.synthetic import generate, write

    document = generate(
        centers=arguments.centers,
        days=arguments.days,
        demand_per_day=arguments.demand_per_day,
        seed=arguments.seed,
    )
    print(f"written {write(document, arguments.folder)}")
    return 0


def _bench(arguments: argparse.Namespace) -> int:
    from pirilib.bench import FAMILIES, SOLVERS, Benchmark, Record

    if not arguments.family:
        print("families:")
        for family in FAMILIES:
            print(f"  {family.name:<12} {family.size:>3} instances  {family.description}")
        print("solvers:")
        for name in SOLVERS.names():
            print(f"  {name:<12} {SOLVERS.descriptions.get(name, '')}")
        return 0
    if arguments.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    family = FAMILIES.get(arguments.family)
    names = [name.strip() for name in arguments.solvers.split(",") if name.strip()]
    instances = family.instances(arguments.count, start=arguments.start)
    goal = _objective([(item.name, parse_spec(item.document)) for item in instances])
    benchmark = Benchmark(SOLVERS.make(names), objective=goal, seeds=tuple(range(arguments.seeds)))

    def report(record: Record) -> None:
        head = f"{record.instance}  {record.solver:<10} seed {record.seed}"
        if not record.solved:
            print(f"{head}  skipped: {record.message}")
            return
        state = "feasible" if record.feasible else "violated " + ",".join(record.violated)
        print(
            f"{head}  objective {record.objective:,.2f}  {state}  {record.seconds:.2f} s",
            flush=True,
        )

    table = benchmark.run(instances, report)
    summary = table.summary()
    print()
    print(summary)
    if arguments.output:
        print(f"results written to {table.to_csv(arguments.output)}")
    if arguments.summary:
        print(f"summary written to {summary.to_csv(arguments.summary)}")
    return 0


def _schema(arguments: argparse.Namespace) -> int:
    from pirilib.spec.schema import main as schema_main

    return schema_main([arguments.path] if arguments.path else [])


if __name__ == "__main__":
    raise SystemExit(main())
