import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pirilib import load_spec
from pirilib.spec import UNLIMITED, LineHaulInstance, parse_spec

ONE_DEGREE_KM = 6371.0 * math.pi / 180.0


def compile_document(document: dict[str, Any]) -> LineHaulInstance:
    return parse_spec(document).compile()


def test_arrays(document: dict[str, Any]) -> None:
    instance = compile_document(document)
    assert instance.centers == ("A", "B", "C")
    assert instance.vehicle_types == ("big", "small")
    assert instance.dimensions == ("desi",)
    assert instance.primary_dimension == 0
    assert instance.capacity.tolist() == [[1000.0], [400.0]]
    assert instance.banned.tolist() == [[False, False], [False, False], [True, False]]
    assert instance.spot_type.tolist() == [0, 1]
    assert instance.spot_fixed.tolist() == [300.0, 150.0]
    assert instance.spot_per_km.tolist() == [2.0, 1.5]
    assert instance.spot_available.tolist() == [UNLIMITED, 3]
    assert instance.owned_type.tolist() == [0]
    assert (instance.owned_origin.tolist(), instance.owned_destination.tolist()) == ([0], [1])
    assert instance.owned_count.tolist() == [1]
    assert instance.owned_mandatory.tolist() == [True]
    assert instance.owned_stopovers.tolist() == [False]
    assert instance.demand_ids == ("D000001", "D000002", "D000003")
    assert instance.demand_origin.tolist() == [0, 0, 1]
    assert instance.demand_destination.tolist() == [1, 2, 0]
    assert instance.demand_day.tolist() == [0, 0, 1]
    assert instance.demand_quantity.tolist() == [[900.0], [250.0], [120.0]]


def test_dtypes_and_read_only(document: dict[str, Any]) -> None:
    instance = compile_document(document)
    values = {field.name: getattr(instance, field.name) for field in dataclasses.fields(instance)}
    arrays = {name: value for name, value in values.items() if isinstance(value, np.ndarray)}
    assert len(arrays) == 26
    for name, array in arrays.items():
        assert array.dtype in (np.float64, np.int64, np.bool_), name
        assert not array.flags.writeable, name
    with pytest.raises(ValueError, match="read-only"):
        instance.distance_km[0, 1] = 0.0


def test_haversine_distances(document: dict[str, Any]) -> None:
    km = compile_document(document).distance_km
    assert km.shape == (3, 3)
    assert km[0, 1] == pytest.approx(ONE_DEGREE_KM, rel=1e-12)
    assert km[0, 2] == pytest.approx(2 * ONE_DEGREE_KM, rel=1e-12)
    assert np.array_equal(km, km.T)
    assert km.diagonal().tolist() == [0.0, 0.0, 0.0]


def test_distance_factor(document: dict[str, Any]) -> None:
    document["matrix"]["factor"] = 1.3
    km = compile_document(document).distance_km
    assert km[0, 1] == pytest.approx(1.3 * ONE_DEGREE_KM, rel=1e-12)


def test_explicit_matrix(document: dict[str, Any]) -> None:
    document["matrix"] = {
        "source": "explicit",
        "symmetric": True,
        "factor": 2.0,
        "arcs": [
            {"origin": "A", "destination": "B", "km": 10.0},
            {"origin": "A", "destination": "C", "km": 20.0},
            {"origin": "C", "destination": "A", "km": 25.0},
            {"origin": "B", "destination": "C", "km": 15.0},
        ],
    }
    km = compile_document(document).distance_km
    assert km.tolist() == [[0.0, 20.0, 40.0], [20.0, 0.0, 30.0], [50.0, 30.0, 0.0]]


def test_empty_demand(document: dict[str, Any]) -> None:
    document["demand"] = []
    instance = compile_document(document)
    assert instance.demand_quantity.shape == (0, 1)
    assert instance.demand_origin.shape == (0,)


def test_example_compiles(example_path: Path) -> None:
    instance = load_spec(example_path).compile()
    assert repr(instance) == (
        "LineHaulInstance(centers=6, days=2, vehicle_types=3, spot_offers=3, owned_groups=1, "
        "demands=18, dimensions=('desi',))"
    )
    assert instance.banned[instance.centers.index("Tekirdağ"), 0]
    assert instance.demand_quantity.sum() == pytest.approx(156_696.0)
