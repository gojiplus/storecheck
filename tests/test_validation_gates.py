import pytest
from test_comparison import compare

from storecheck import Case, EvalSet, Objective, Row, Store, weighted_error_cost


def test_cost_cap_blocks_net_accuracy_gain():
    baseline = Store([Row("apple", "a", id="a"), Row("banana", "b", id="b")])
    candidate = baseline.edit("a", label="b").edit("b", label="a")
    cases = [
        Case("high", "apple", "a", cost=100),
        Case("low1", "banana", "a", cost=1),
        Case("low2", "banana", "a", cost=1),
    ]
    result = compare(
        baseline, candidate, cases, objective=Objective(max_right_to_wrong_cost=5)
    )
    assert result.exit_code == 1
    assert result.to_dict()["overall"]["fixed"] == 2
    assert result.to_dict()["overall"]["broken"] == 1
    assert "cost" in result.summary()


def test_cost_ceiling_and_slice_sample_floor():
    store = Store([Row("apple", "a")])
    cases = [Case("c", "apple", "b", slices=frozenset({"rare"}), cost=10)]
    objective = Objective(
        primary=weighted_error_cost,
        primary_name="error cost",
        higher_is_better=False,
        max_slice_metric={"rare": 5},
    )
    assert compare(store, store, cases, objective=objective).exit_code == 1
    objective.min_slice_cases = {"rare": 2}
    assert compare(store, store, cases, objective=objective).exit_code == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_right_to_wrong_cost": -1},
        {"max_right_to_wrong_cost": float("nan")},
        {"min_slice_cases": {"s": 0}},
        {"min_slice_cases": {"s": True}},
        {"max_slice_metric": {"s": float("inf")}},
        {"min_slice_metric": {"s": 2}, "max_slice_metric": {"s": 1}},
    ],
)
def test_invalid_gate_settings(kwargs):
    with pytest.raises(ValueError, match=r"cost|slice|floor"):
        Objective(**kwargs)


def test_group_split_prevents_duplicate_and_group_leakage():
    cases = EvalSet(
        [
            Case("1", "same text", "a", metadata={"customer": "one"}),
            Case("2", " SAME  TEXT ", "a", metadata={"customer": "two"}),
            Case("3", "different", "a", metadata={"customer": "two"}),
            Case("4", "unique", "b", metadata={"customer": "three"}),
            Case("5", "final", "a", metadata={"customer": "four"}),
        ]
    )
    dev, holdout = cases.split(seed=10, group_key="customer")
    assert len(dev)
    assert len(holdout)
    dev.assert_disjoint(holdout, group_key="customer")
    assert {"1", "2", "3"}.issubset({c.id for c in dev}) or {"1", "2", "3"}.issubset(
        {c.id for c in holdout}
    )
    again, _ = cases.split(seed=10, group_key="customer")
    assert [c.id for c in again] == [c.id for c in dev]
    assert {c.id for c in dev} | {c.id for c in holdout} == {c.id for c in cases}


@pytest.mark.parametrize(
    ("other", "group_key"),
    [
        (Case("1", "different", "a"), None),
        (Case("2", " SAME  TEXT ", "b"), None),
        (Case("2", "different", "a", metadata={"customer": "one"}), "customer"),
    ],
)
def test_disjoint_rejects_leakage(other, group_key):
    dev = EvalSet([Case("1", "same text", "a", metadata={"customer": "one"})])
    with pytest.raises(ValueError, match="overlap"):
        dev.assert_disjoint(EvalSet([other]), group_key=group_key)


def test_group_key_requires_complete_metadata():
    with pytest.raises(ValueError, match="customer"):
        EvalSet([Case("1", "a", "a")]).split(group_key="customer")
