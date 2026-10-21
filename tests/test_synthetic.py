from pathlib import Path

import pytest

from pirilib import load_spec
from pirilib.spec import parse_spec
from pirilib.synthetic import generate, write


@pytest.mark.parametrize("seed", range(10))
def test_generated_problems_are_valid(seed: int) -> None:
    document = generate(centers=8, days=2, demand_per_day=20, seed=seed)
    spec = parse_spec(document)
    assert len(spec.centers) == 8
    assert len(spec.demand) == 40
    assert spec.days[-1] > spec.days[0]


def test_generation_is_deterministic() -> None:
    assert generate(seed=3) == generate(seed=3)
    assert generate(seed=3) != generate(seed=4)


def test_written_problem_loads_as_the_same_spec(tmp_path: Path) -> None:
    document = generate(centers=6, days=1, demand_per_day=15, seed=1)
    path = write(document, tmp_path / "problem")
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "centers.csv",
        "demand.csv",
        "problem.yaml",
    ]
    assert load_spec(path) == parse_spec(document)


def test_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="at least 2 centers"):
        generate(centers=1)
