"""A benchmark run against a baseline: what got better, what got worse.

Two :class:`~pirilib.bench.Report` objects are compared pair by pair: one
solver on one instance, by the median objective over its seeds. A comparison
also says what differs between the conditions of the two runs, because a
change only means something when the instances, the objective and the seeds
are the same, and times only when the machine is.

```python
from pirilib.bench import Report, compare

comparison = compare(Report.from_json("baseline.json"), report)
print(comparison)
if comparison.worse(tolerance=0.01):
    raise SystemExit(1)
```
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from pirilib.bench.report import Report
from pirilib.bench.results import Spread

__all__ = ["Change", "Comparison", "compare"]

_ABSENT: Final = "absent"
"""What a setting that one of two runs does not have is shown as."""


@dataclass(frozen=True, slots=True)
class Change:
    """One solver on one instance, in the baseline and in the new run.

    Attributes:
        before: Median objective over the solved runs of the baseline; NaN when
            none was solved
        after: The same for the new run
        change: ``(after - before) / max(|before|, 1)``, below 0 when the new run
            is better; NaN when either median is
        seconds_before: Median seconds of the baseline
        seconds_after: Median seconds of the new run
    """

    instance: str
    solver: str
    before: float
    after: float
    change: float
    seconds_before: float
    seconds_after: float


@dataclass(frozen=True, slots=True)
class Comparison:
    """A new benchmark run next to a baseline.

    Attributes:
        changes: One per (instance, solver) pair that both reports ran, in the
            order of the new run
        differences: What differs between the conditions of the two runs, in words
        missing: The pairs that only one of the reports ran, in words
    """

    changes: tuple[Change, ...]
    differences: tuple[str, ...]
    missing: tuple[str, ...]

    def worse(self, tolerance: float = 0.0) -> tuple[Change, ...]:
        """The pairs whose median objective rose by more than ``tolerance`` (relative).

        A pair the baseline solved and the new run did not is worse as well.
        """
        return tuple(
            item
            for item in self.changes
            if item.change > tolerance or (math.isnan(item.after) and not math.isnan(item.before))
        )

    def better(self, tolerance: float = 0.0) -> tuple[Change, ...]:
        """The pairs whose median objective fell by more than ``tolerance`` (relative)."""
        return tuple(item for item in self.changes if item.change < -tolerance)

    def describe(self, tolerance: float = 0.0) -> str:
        """The comparison in words: the counts, the differences and every changed pair."""
        worse, better = self.worse(tolerance), self.better(tolerance)
        same = len(self.changes) - len(worse) - len(better)
        lines = [
            f"{len(self.changes)} pairs of instance and solver: {len(worse)} worse, "
            f"{len(better)} better, {same} within {tolerance:.2%}"
        ]
        for title, notes in (
            ("conditions that differ", self.differences),
            ("not compared", self.missing),
        ):
            if notes:
                lines.append(f"{title}:")
                lines += [f"- {note}" for note in notes]
        for title, items in (("worse", worse), ("better", better)):
            if items:
                lines.append(f"{title}:")
                lines += [
                    f"  {item.instance}  {item.solver:<10} {item.before:,.2f} -> "
                    f"{item.after:,.2f}  {item.change:+.2%}"
                    for item in items
                ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def compare(baseline: Report, new: Report) -> Comparison:
    """Compare ``new`` with ``baseline``, pair by pair, and say what differs in their conditions."""
    before = {(item.instance, item.solver): item for item in baseline.spread()}
    after = {(item.instance, item.solver): item for item in new.spread()}
    changes = tuple(_change(before[pair], item) for pair, item in after.items() if pair in before)
    missing = [
        f"{instance} with {solver} is only in the {where}"
        for where, mine, other in (("baseline", before, after), ("new run", after, before))
        for instance, solver in mine
        if (instance, solver) not in other
    ]
    return Comparison(changes, tuple(_differences(baseline, new)), tuple(missing))


def _change(before: Spread, after: Spread) -> Change:
    return Change(
        instance=after.instance,
        solver=after.solver,
        before=before.median,
        after=after.median,
        change=(after.median - before.median) / max(abs(before.median), 1.0),
        seconds_before=before.median_seconds,
        seconds_after=after.median_seconds,
    )


def _differences(baseline: Report, new: Report) -> list[str]:
    """What differs between the conditions of two runs; empty when they ran alike."""
    notes: list[str] = []
    old, now = baseline.environment, new.environment
    for label, first, second in (
        ("pirilib", old.pirilib, now.pirilib),
        ("python", old.python, now.python),
        ("platform", old.platform, now.platform),
        ("processor", old.processor, now.processor),
        ("logical processors", old.cpus, now.cpus),
    ):
        if first != second:
            notes.append(f"{label}: {first} -> {second}")
    notes += _changed("package", old.packages, now.packages)
    if baseline.objective != new.objective:
        notes.append("the objectives have different terms or weights")
    if baseline.constraints != new.constraints:
        notes.append(f"constraints: {list(baseline.constraints)} -> {list(new.constraints)}")
    if baseline.seeds != new.seeds:
        notes.append(f"seeds: {list(baseline.seeds)} -> {list(new.seeds)}")
    for name, info in new.solvers.items():
        known = baseline.solvers.get(name)
        if known is None:
            continue
        if known.type != info.type:
            notes.append(f"solver {name}: {known.type} -> {info.type}")
        notes += _changed(f"solver {name}", known.settings, info.settings)
    generated = {item.name: item for item in baseline.instances}
    for item in new.instances:
        if item.name in generated and generated[item.name] != item:
            notes.append(f"instance {item.name} is not the same problem in the two runs")
    return notes


def _changed(label: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """One note per value that differs, looking into nested settings.

    A key that only one side has shows as ``absent`` on the other.
    """
    notes: list[str] = []
    for key in sorted({*before, *after}):
        first, second = before.get(key, _ABSENT), after.get(key, _ABSENT)
        if first == second:
            continue
        if isinstance(first, Mapping) and isinstance(second, Mapping):
            notes += _changed(f"{label} {key}", first, second)
        else:
            notes.append(f"{label} {key}: {first} -> {second}")
    return notes
