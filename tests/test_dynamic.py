from storecheck import Objective, Row, Severity
from storecheck.checks import (
    DeadRowCheck,
    DecisionLayerCheck,
    MutationCheck,
    RegressionCheck,
    RowInfluenceCheck,
    row_influence,
)
from storecheck.checks.base import CheckContext


def test_baseline_is_accurate(evaluator, store):
    run = evaluator.run(store)
    assert run.metric() >= 0.75, [(r.case.id, r.predicted) for r in run.results]


def test_context_diff_shortcut_limits_decider_calls(evaluator, store):
    base = evaluator.run(store)
    calls0 = evaluator.decider.calls
    # add a billing row; authn/authz cases should keep their evidence
    cand = store.add(Row("invoice shows a charge I do not recognize", "billing"))
    cmp = evaluator.compare(store, cand, baseline_run=base)
    assert cmp.context_changed < len(base.results)
    assert cmp.decider_calls == evaluator.decider.calls - calls0
    assert cmp.decider_calls <= cmp.context_changed


def test_mislabeled_row_has_positive_influence(evaluator, store):
    # poison: an authz example mislabeled as authn, phrased to be retrieved for
    # authz cases
    poison = Row(
        "logged in but access denied, permission denied on the "
        "admin page and reports folder",
        "authentication_failure",
        id="poison",
    )
    poisoned = store.add(poison)
    infl = row_influence(evaluator, poisoned, row_ids=["poison"])
    assert infl[0].net > 0, infl[0].comparison.summary()
    f = RowInfluenceCheck(evaluator, row_ids=["poison"]).run(CheckContext(poisoned))
    assert any(x.row_ids == ("poison",) and x.severity >= Severity.WARN for x in f)


def test_regression_check_blocks_bad_candidate(evaluator, store):
    poison = Row(
        "logged in but access denied, permission denied on the "
        "admin page and reports folder",
        "authentication_failure",
        id="poison",
    )
    objective = Objective(max_right_to_wrong=0, min_slice_metric={"authz": 0.9})
    f = RegressionCheck(evaluator, baseline=store, objective=objective).run(
        CheckContext(store.add(poison))
    )
    assert any(x.severity == Severity.ERROR for x in f), [x.message for x in f]


def test_regression_check_passes_neutral_candidate(evaluator, store):
    f = RegressionCheck(
        evaluator, baseline=store, objective=Objective(max_right_to_wrong=0)
    ).run(CheckContext(store.add(Row("refund my duplicate charge", "billing"))))
    assert not any(x.severity == Severity.ERROR for x in f), [x.message for x in f]


def test_dead_rows_and_decision_layer(evaluator, store):
    f = DeadRowCheck(evaluator).run(CheckContext(store))
    assert f
    f = DecisionLayerCheck(evaluator).run(CheckContext(store))
    assert f
    assert f[0].check == "decision_layer"


def test_mutation_check_detects_damage(evaluator, store):
    f = MutationCheck(evaluator).run(CheckContext(store))
    errors = [x for x in f if x.severity == Severity.ERROR]
    assert not errors, [x.message for x in errors]
