"""Reproducible families of synthetic benchmark instances.

A :class:`Family` describes ranges of problem settings; instance ``i`` of a
family draws its settings from a random stream seeded by the family's name,
its seed and ``i``, and is built with :func:`pirilib.synthetic.generate`. The
same name, seed and index always give the same problem, on every machine.
"""

import math
import zlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from pirilib.spec import parse_spec
from pirilib.spec.instance import LineHaulInstance
from pirilib.synthetic import generate


@dataclass(frozen=True, eq=False, slots=True)
class BenchInstance:
    """One instance of a family: its name, settings, problem document and compiled form.

    Attributes:
        index: Position in the family
        seed: Seed passed to :func:`pirilib.synthetic.generate`
        settings: The drawn settings, for reports
        document: The problem document
    """

    family: str
    index: int
    seed: int
    settings: Mapping[str, Any]
    document: Mapping[str, Any]
    instance: LineHaulInstance

    @property
    def name(self) -> str:
        """The label of the instance, ``family/index``."""
        return f"{self.family}/{self.index:03d}"

    def __repr__(self) -> str:
        return f"BenchInstance({self.name}, seed={self.seed}, {self.instance!r})"


@dataclass(frozen=True, slots=True)
class Family:
    """Ranges of synthetic problem settings that instances are drawn from.

    Integer ranges are ``(low, high)``, both included; the other settings are
    tuples of choices.

    Attributes:
        size: Instances in the family unless another count is asked for
        centers: Range of the number of centers
        days: Range of the number of days
        demand_per_day: Range of demand rows per day
        owned_lines: Range of owned vehicle lines
        min_fill: Choices of the spot vehicles' minimum fill
        max_stretch: Choices of the stopover stretch limit (None: no limit)
        split: Choices of whether demand may be split
        spot_limit: Choices of the daily limit of every spot offer, as a share of
            the demand rows per day (None: unlimited)
        seed: Base seed; families with the same ranges but other seeds differ
    """

    name: str
    description: str
    size: int = 10
    centers: tuple[int, int] = (6, 10)
    days: tuple[int, int] = (2, 2)
    demand_per_day: tuple[int, int] = (15, 30)
    owned_lines: tuple[int, int] = (0, 2)
    min_fill: tuple[float, ...] = (0.1,)
    max_stretch: tuple[float | None, ...] = (1.3,)
    split: tuple[bool, ...] = (True,)
    spot_limit: tuple[float | None, ...] = (None,)
    mean_desi: float = 6000.0
    seed: int = 0

    def __post_init__(self) -> None:
        problems = [
            f"{label} must be (low, high) with {minimum} <= low <= high"
            for label, (low, high), minimum in (
                ("centers", self.centers, 2),
                ("days", self.days, 1),
                ("demand_per_day", self.demand_per_day, 1),
                ("owned_lines", self.owned_lines, 0),
            )
            if not minimum <= low <= high
        ]
        problems += [
            f"{label} needs at least one choice"
            for label, choices in (
                ("min_fill", self.min_fill),
                ("max_stretch", self.max_stretch),
                ("split", self.split),
                ("spot_limit", self.spot_limit),
            )
            if not choices
        ]
        if self.size < 1:
            problems.append("size must be at least 1")
        if problems:
            raise ValueError(f"invalid family '{self.name}': " + "; ".join(problems))

    def settings(self, index: int) -> dict[str, Any]:
        """The settings of instance ``index``, drawn from the family's ranges."""
        stream = np.random.default_rng([zlib.crc32(self.name.encode("utf-8")), self.seed, index])

        def between(bounds: tuple[int, int]) -> int:
            return int(stream.integers(bounds[0], bounds[1] + 1))

        def pick[T](choices: tuple[T, ...]) -> T:
            return choices[int(stream.integers(len(choices)))]

        return {
            "centers": between(self.centers),
            "days": between(self.days),
            "demand_per_day": between(self.demand_per_day),
            "owned_lines": between(self.owned_lines),
            "min_fill": pick(self.min_fill),
            "max_stretch": pick(self.max_stretch),
            "split": pick(self.split),
            "spot_limit": pick(self.spot_limit),
            "seed": int(stream.integers(2**31)),
        }

    def instance(self, index: int) -> BenchInstance:
        """Instance ``index`` of the family (any index from 0 on, not only below :attr:`size`)."""
        if index < 0:
            raise ValueError("index must not be negative")
        settings = self.settings(index)
        document = generate(
            centers=settings["centers"],
            days=settings["days"],
            demand_per_day=settings["demand_per_day"],
            owned_lines=settings["owned_lines"],
            mean_desi=self.mean_desi,
            min_fill=settings["min_fill"],
            max_stretch=settings["max_stretch"],
            seed=settings["seed"],
        )
        document["name"] = f"{self.name}/{index:03d}"
        document["rules"]["split_demand"] = settings["split"]
        if not settings["split"]:
            document["demand"] = self._shipments(document)
        if settings["spot_limit"] is not None:
            limit = max(1, math.ceil(settings["spot_limit"] * settings["demand_per_day"]))
            for offer in document["fleet"]["spot"]:
                offer["available"] = limit
        return BenchInstance(
            family=self.name,
            index=index,
            seed=settings["seed"],
            settings=MappingProxyType(settings),
            document=MappingProxyType(document),
            instance=parse_spec(document).compile(),
        )

    @staticmethod
    def _shipments(document: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Demand rows cut into equal pieces that fit the largest vehicle allowed on their line.

        Without split demand, a row larger than every vehicle that may serve it
        would make the problem invalid; the pieces are separate shipments.
        """
        banned = {
            row["id"]: set(row.get("banned_vehicle_types", ())) for row in document["centers"]
        }
        capacity = {vehicle["name"]: vehicle["capacity"] for vehicle in document["vehicle_types"]}
        spot_types = {offer["type"] for offer in document["fleet"]["spot"]}
        rows = []
        for row in document["demand"]:
            closed = banned[row["origin"]] | banned[row["destination"]]
            usable = [capacity[name] for name in spot_types if name not in closed]
            pieces = max(
                1,
                *(
                    math.ceil(row["quantity"][name] / max(limit[name] for limit in usable))
                    for name in row["quantity"]
                ),
            )
            for _ in range(pieces):
                quantity = {
                    dimension: math.floor(value / pieces * 10) / 10
                    for dimension, value in row["quantity"].items()
                }
                rows.append({**row, "quantity": quantity})
        return rows

    def instances(self, count: int | None = None, start: int = 0) -> list[BenchInstance]:
        """``count`` instances (default :attr:`size`) from index ``start`` on."""
        total = self.size if count is None else count
        if total < 0:
            raise ValueError("count must not be negative")
        return [self.instance(index) for index in range(start, start + total)]

    def __iter__(self) -> Iterator[BenchInstance]:
        return iter(self.instances())


@dataclass(frozen=True, slots=True)
class FamilyCatalog:
    """Families by name.

    Attributes:
        families: In the order they are listed
    """

    families: tuple[Family, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        names = [family.name for family in self.families]
        doubles = sorted({name for name in names if names.count(name) > 1})
        if doubles:
            raise ValueError(f"family names used twice: {doubles}")

    def names(self) -> list[str]:
        """The names of the families, in catalog order."""
        return [family.name for family in self.families]

    def get(self, name: str) -> Family:
        """The family called ``name``; :class:`ValueError` lists the known names otherwise."""
        for family in self.families:
            if family.name == name:
                return family
        raise ValueError(f"unknown family '{name}'; known: {', '.join(self.names())}")

    def with_family(self, family: Family) -> "FamilyCatalog":
        """A catalog with ``family`` added, or replacing the one with its name."""
        kept = tuple(item for item in self.families if item.name != family.name)
        return FamilyCatalog((*kept, family))

    def __iter__(self) -> Iterator[Family]:
        return iter(self.families)


FAMILIES: Final = FamilyCatalog(
    (
        Family(
            "tiny",
            "3-5 centers, 1 day, a few demands; for tests and smoke runs",
            size=4,
            centers=(3, 5),
            days=(1, 1),
            demand_per_day=(5, 12),
            owned_lines=(0, 1),
            min_fill=(0.0,),
        ),
        Family(
            "small",
            "6-10 centers, 2 days; small enough for the exact solver",
            size=10,
            centers=(6, 10),
            days=(2, 2),
            demand_per_day=(15, 30),
            owned_lines=(0, 2),
            min_fill=(0.0, 0.1),
            max_stretch=(1.3, None),
        ),
        Family(
            "routing",
            "6-12 centers, 1-2 days, no minimum fill; the cases the routing adapters accept",
            size=10,
            centers=(6, 12),
            days=(1, 2),
            demand_per_day=(15, 40),
            owned_lines=(0, 2),
            min_fill=(0.0,),
            max_stretch=(1.2, 1.3, None),
        ),
        Family(
            "medium",
            "12-25 centers, 3 days, 40-120 demands per day",
            size=10,
            centers=(12, 25),
            days=(3, 3),
            demand_per_day=(40, 120),
            owned_lines=(1, 4),
            min_fill=(0.0, 0.1),
            max_stretch=(1.2, 1.3, None),
        ),
        Family(
            "no-split",
            "6-12 centers, 1-2 days; demand may not be split",
            size=10,
            centers=(6, 12),
            days=(1, 2),
            demand_per_day=(15, 40),
            min_fill=(0.0, 0.1),
            split=(False,),
        ),
        Family(
            "spot-limits",
            "6-12 centers, 2 days; every spot offer has a daily limit",
            size=10,
            centers=(6, 12),
            days=(2, 2),
            demand_per_day=(15, 40),
            min_fill=(0.0, 0.1),
            spot_limit=(0.2, 0.3, 0.5),
        ),
        Family(
            "mixed",
            "every setting varied: 4-20 centers, 1-3 days, split or not, limits or not",
            size=12,
            centers=(4, 20),
            days=(1, 3),
            demand_per_day=(10, 80),
            owned_lines=(0, 3),
            min_fill=(0.0, 0.1, 0.2),
            max_stretch=(1.1, 1.3, None),
            split=(True, False),
            spot_limit=(None, 0.3, 0.5),
        ),
    )
)
"""The built-in families."""
