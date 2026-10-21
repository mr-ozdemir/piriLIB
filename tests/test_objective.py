import math

import pytest

from pirilib import Dispatch, Evaluation, Load, Plan, evaluate
from pirilib import objective as obj
from pirilib.spec import LineHaulInstance

ONE = 6371.0 * math.pi / 180.0

DISPATCHES = [
    Dispatch(day=0, kind="owned", source=0, origin=0, stops=(1,), loads=(Load(0, (900.0,)),)),
    Dispatch(day=0, kind="spot", source=1, origin=0, stops=(2,), loads=(Load(1, (250.0,)),)),
]


@pytest.fixture
def evaluation(instance: LineHaulInstance) -> Evaluation:
    return evaluate(instance, Plan.from_dispatches(DISPATCHES, 1))


def test_cost_terms(evaluation: Evaluation) -> None:
    assert obj.total_cost()(evaluation) == evaluation.total_cost
    assert obj.owned_fixed_cost()(evaluation) == 200.0
    assert obj.dispatch_cost()(evaluation) == pytest.approx(ONE + 150 + 3 * ONE)
    assert obj.dispatch_cost("owned")(evaluation) == pytest.approx(ONE)
    assert obj.dispatch_cost("spot")(evaluation) == pytest.approx(150 + 3 * ONE)


def test_counting_terms(evaluation: Evaluation) -> None:
    assert obj.distance_km()(evaluation) == pytest.approx(3 * ONE)
    assert obj.distance_km("spot")(evaluation) == pytest.approx(2 * ONE)
    assert obj.dispatches()(evaluation) == 2.0
    assert obj.dispatches("owned")(evaluation) == 1.0
    assert obj.unused_capacity()(evaluation) == pytest.approx(100.0 + 150.0)
    assert obj.unserved()(evaluation) == 120.0
    assert obj.unserved("desi")(evaluation) == 120.0


def test_violation_terms(evaluation: Evaluation) -> None:
    assert obj.violation("coverage")(evaluation) == 120.0
    assert obj.violations()(evaluation) == 120.0
    assert obj.violations(["capacity", "coverage"])(evaluation) == 120.0
    with pytest.raises(KeyError, match="no constraint named 'capacty'"):
        obj.violation("capacty")(evaluation)
    with pytest.raises(KeyError, match="no quantity dimension 'kg'"):
        obj.unserved("kg")(evaluation)


def test_weights_and_sums(evaluation: Evaluation) -> None:
    goal = obj.total_cost() + 1000 * obj.violation("coverage") + obj.dispatches("spot") * 50
    assert goal(evaluation) == pytest.approx(evaluation.total_cost + 120_000 + 50)
    assert goal.breakdown(evaluation) == {
        "total_cost": pytest.approx(evaluation.total_cost),
        "violation[coverage]": 120_000.0,
        "dispatches[spot]": 50.0,
    }
    doubled = 2 * goal
    assert doubled(evaluation) == pytest.approx(2 * goal(evaluation))
    combined = goal + obj.distance_km()
    assert len(combined.terms) == 4


def test_terms_with_the_same_name_are_added_in_breakdowns(evaluation: Evaluation) -> None:
    goal = obj.dispatches() + 2 * obj.dispatches()
    assert goal.breakdown(evaluation) == {"dispatches": 6.0}


def test_plain_functions_are_objectives(evaluation: Evaluation) -> None:
    def goal(e: Evaluation) -> float:
        return e.total_cost + 2.0 * float(e.unserved.sum())

    objectives: list[obj.Objective] = [goal, obj.total_cost()]
    assert objectives[0](evaluation) == pytest.approx(evaluation.total_cost + 240.0)


def test_unknown_kind(evaluation: Evaluation) -> None:
    with pytest.raises(ValueError, match="kind must be 'owned', 'spot' or None"):
        obj.dispatches("rented")(evaluation)
