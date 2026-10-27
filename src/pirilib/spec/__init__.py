"""Problem specification: validated models, YAML loading and the array form.

The JSON Schema of problem documents is in :mod:`pirilib.spec.schema`.
"""

from pirilib.spec.instance import UNLIMITED, LineHaulInstance, compile_spec
from pirilib.spec.loader import load_spec, parse_spec
from pirilib.spec.models import (
    SCHEMA_ID,
    Arc,
    Calendar,
    Center,
    Cost,
    Demand,
    ExplicitMatrix,
    Fleet,
    Handling,
    HaversineMatrix,
    LineHaulSpec,
    MinFill,
    ObjectiveSpec,
    OwnedVehicles,
    Rules,
    Service,
    SpotOffer,
    Stopover,
    VehicleType,
)
from pirilib.spec.tables import TableSource

__all__ = [
    "SCHEMA_ID",
    "UNLIMITED",
    "Arc",
    "Calendar",
    "Center",
    "Cost",
    "Demand",
    "ExplicitMatrix",
    "Fleet",
    "Handling",
    "HaversineMatrix",
    "LineHaulInstance",
    "LineHaulSpec",
    "MinFill",
    "ObjectiveSpec",
    "OwnedVehicles",
    "Rules",
    "Service",
    "SpotOffer",
    "Stopover",
    "TableSource",
    "VehicleType",
    "compile_spec",
    "load_spec",
    "parse_spec",
]
