"""Running solvers on benchmark instances, and the solvers known by name."""

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from pirilib.bench.families import BenchInstance
from pirilib.bench.results import Record, ResultTable
from pirilib.constraints import Constraint
from pirilib.evaluation import evaluate
from pirilib.objective import Objective
from pirilib.solvers import ACOConfig, ACOSolver, Solver, default_objective
from pirilib.solvers.direct import DirectShipping


class Benchmark:
    """Solve instances with named solvers and record every run.

    Every plan is evaluated again here, with the benchmark's objective and
    constraints, so that solvers are scored the same way whatever they report.
    Times are wall-clock seconds of the ``solve`` call.

    A solver that raises one of ``skip`` (by default :class:`ValueError`, how
    solvers refuse problems they do not support, and :class:`ImportError`, for
    optional packages that are missing) is recorded as skipped on that
    instance; any other exception stops the benchmark.

    Attributes:
        solvers: By name, in the order they run on each instance
        objective: Scores every plan; defaults to :func:`~pirilib.solvers.default_objective`
        seeds: One run per seed, solver and instance
    """

    def __init__(
        self,
        solvers: Mapping[str, Solver],
        *,
        objective: Objective | None = None,
        constraints: Sequence[Constraint] = (),
        seeds: Sequence[int] = (0,),
        skip: tuple[type[Exception], ...] = (ValueError, ImportError),
    ) -> None:
        if not solvers:
            raise ValueError("a benchmark needs at least one solver")
        if not seeds:
            raise ValueError("a benchmark needs at least one seed")
        self.solvers = dict(solvers)
        self.objective = objective
        self.constraints = tuple(constraints)
        self.seeds = tuple(seeds)
        self.skip = skip

    def run(
        self,
        instances: Iterable[BenchInstance],
        progress: Callable[[Record], None] | None = None,
    ) -> ResultTable:
        """Run every solver on every instance; ``progress`` sees each record as it is made."""
        records = []
        for item in instances:
            for name, solver in self.solvers.items():
                for seed in self.seeds:
                    record = self.run_one(item, name, solver, seed)
                    records.append(record)
                    if progress is not None:
                        progress(record)
        return ResultTable(tuple(records))

    def run_one(self, item: BenchInstance, name: str, solver: Solver, seed: int) -> Record:
        """One run of ``solver`` on ``item``."""
        goal = default_objective() if self.objective is None else self.objective
        started = time.perf_counter()
        try:
            result = solver.solve(item.instance, self.objective, self.constraints, seed=seed)
        except self.skip as error:
            return Record(
                family=item.family,
                instance=item.name,
                index=item.index,
                instance_seed=item.seed,
                solver=name,
                seed=seed,
                status="skipped",
                objective=math.nan,
                total_cost=math.nan,
                feasible=False,
                violated=(),
                seconds=time.perf_counter() - started,
                dispatches=0,
                message=f"{type(error).__name__}: {error}",
            )
        seconds = time.perf_counter() - started
        evaluation = evaluate(item.instance, result.plan, self.constraints)
        return Record(
            family=item.family,
            instance=item.name,
            index=item.index,
            instance_seed=item.seed,
            solver=name,
            seed=seed,
            status="ok",
            objective=float(goal(evaluation)),
            total_cost=evaluation.total_cost,
            feasible=evaluation.feasible,
            violated=tuple(evaluation.violated()),
            seconds=seconds,
            dispatches=len(result.plan),
        )


def _exact() -> Solver:
    from pirilib.solvers.exact import ExactSolver

    return ExactSolver()


def _pyvrp() -> Solver:
    from pirilib.solvers.adapters import PyVRPSolver

    return PyVRPSolver()


def _ortools() -> Solver:
    from pirilib.solvers.adapters import ORToolsSolver

    return ORToolsSolver()


@dataclass(frozen=True, slots=True)
class SolverCatalog:
    """Solvers by name, made on demand, for the command line and quick benchmarks.

    Attributes:
        makers: Name to a function that makes the solver
        descriptions: Name to a short description
    """

    makers: Mapping[str, Callable[[], Solver]] = field(default_factory=dict)
    descriptions: Mapping[str, str] = field(default_factory=dict)

    def names(self) -> list[str]:
        """The names of the solvers, in registration order."""
        return list(self.makers)

    def make(self, names: Iterable[str]) -> dict[str, Solver]:
        """The solvers called ``names``; :class:`ValueError` lists the known names otherwise."""
        chosen = list(names)
        unknown = [name for name in chosen if name not in self.makers]
        if unknown:
            raise ValueError(f"unknown solvers {unknown}; known: {', '.join(self.names())}")
        return {name: self.makers[name]() for name in chosen}


SOLVERS: Final = SolverCatalog(
    makers=MappingProxyType(
        {
            "direct": DirectShipping,
            "aco": ACOSolver,
            "aco-exact": lambda: ACOSolver(ACOConfig(exact_nodes=9)),
            "exact": _exact,
            "pyvrp": _pyvrp,
            "ortools": _ortools,
        }
    ),
    descriptions=MappingProxyType(
        {
            "direct": "direct shipping, full vehicles then the cheapest that fits",
            "aco": "ant colony with default settings",
            "aco-exact": "ant colony, subproblems up to 9 destinations solved exactly (scipy)",
            "exact": "mixed-integer programming per subproblem (scipy; small subproblems)",
            "pyvrp": "PyVRP on the reduced routing problem (pirilib[pyvrp])",
            "ortools": "OR-Tools routing on the reduced routing problem (pirilib[ortools])",
        }
    ),
)
"""The built-in solver names."""
