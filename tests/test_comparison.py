import json

import pytest

from storecheck import (
    Case,
    EmbeddingRetriever,
    EvalSet,
    HashingEmbedder,
    LabelDef,
    MajorityVoteDecider,
    Objective,
    Pipeline,
    Prediction,
    Row,
    Store,
    Taxonomy,
    compare_stores,
)


def compare(baseline, candidate, cases, **kwargs):
    retriever = EmbeddingRetriever(HashingEmbedder(), k=1)
    decider = MajorityVoteDecider()

    def run(query, snapshot):
        evidence = retriever.retrieve(query.text, snapshot, scope=query.scope)
        return Prediction(decider.decide(query.text, evidence), tuple(evidence))

    return compare_stores(
        baseline,
        candidate,
        EvalSet(cases),
        pipeline=Pipeline(
            run, {"retriever": "hashing-v2:1024:k1", "decider": "majority-v1"}
        ),
        **kwargs,
    )


def test_full_case_report_and_cost_gate(tmp_path):
    baseline = Store([Row("password", "login", id="login")])
    candidate = baseline.edit("login", label="billing")
    cases = [Case("c", "password", "login", slices=frozenset({"critical"}), cost=50)]
    result = compare(baseline, candidate, cases)
    data = result.to_dict()
    assert result.exit_code == 1
    assert data["status"] == "fail"
    assert data["overall"]["n"] == 1
    assert data["overall"]["baseline"]["error_cost"] == 0
    assert data["overall"]["candidate"]["error_cost"] == 50
    assert data["slices"]["critical"]["n"] == 1
    assert data["cases"][0]["transition"] == "right_to_wrong"
    assert data["cases"][0]["baseline"]["evidence"][0]["label"] == "login"
    assert data["cases"][0]["candidate"]["prediction"] == "billing"
    assert data["execution"]["pipeline_calls"] == 2
    path = tmp_path / "result.json"
    result.to_json(path)
    assert json.loads(path.read_text()) == data
    data["cases"].clear()
    assert len(result.to_dict()["cases"]) == 1
    assert "FAIL" in result.summary()


def test_identity_covers_order_eval_and_objective():
    store = Store([Row("same", "a", id="a"), Row("same", "b", id="b")])
    cases = [Case("c", "same", "a")]
    first = compare(store, store, cases).to_dict()
    again = compare(store, store, cases).to_dict()
    assert first["run_id"] == again["run_id"]
    assert first["execution"]["pipeline_calls"] == 2
    assert first["cases"][0]["transition"] == "unchanged"
    reversed_store = Store(list(reversed(list(store))))
    reordered = compare(reversed_store, store, cases).to_dict()
    assert first["run_id"] != reordered["run_id"]
    recosted = compare(store, store, [Case("c", "same", "a", cost=2)]).to_dict()
    assert first["run_id"] != recosted["run_id"]
    capped = compare(store, store, cases, objective=Objective(max_right_to_wrong=0))
    assert first["run_id"] != capped.to_dict()["run_id"]


def test_empty_and_missing_slice_are_unevaluable():
    store = Store([Row("same", "a")])
    empty = compare(store, store, [])
    assert empty.exit_code == 2
    assert empty.to_dict()["overall"]["candidate"]["accuracy"] is None
    missing = compare(
        store,
        store,
        [Case("c", "same", "a")],
        objective=Objective(min_slice_metric={"absent": 0.9}),
    )
    assert missing.exit_code == 2
    assert "absent" in missing.summary()


def test_identity_must_be_nonempty_json():
    with pytest.raises(ValueError, match="identity"):
        Pipeline(lambda query, snapshot: Prediction("a"), {})


def test_taxonomy_boundary_changes_identity_and_unknown_prediction_fails():
    store = Store([Row("query", "a")])
    cases = [Case("c", "query", "a")]
    tax = Taxonomy([LabelDef("a", "first", ("boundary one",))])
    changed = Taxonomy([LabelDef("a", "first", ("boundary two",))])
    first = compare(store, store, cases, taxonomy=tax).to_dict()
    second = compare(store, store, cases, taxonomy=changed).to_dict()
    assert first["run_id"] != second["run_id"]
    result = compare(store, Store([]), cases, taxonomy=tax)
    assert result.exit_code == 2
    assert "outside active taxonomy" in result.summary()
    with pytest.raises(ValueError, match="labels outside"):
        compare(store, store, [Case("c", "query", "b")], taxonomy=tax)


def test_discovery_validation_overlap_is_checked_before_deciding():
    store = Store([Row("query", "a")])
    cases = [Case("c", "query", "a", metadata={"customer": "one"})]
    discovery = EvalSet([Case("dev", "different", "a", metadata={"customer": "one"})])
    with pytest.raises(ValueError, match="overlap on group"):
        compare(store, store, cases, discovery=discovery, group_key="customer")
    result = compare(store, store, cases, discovery=discovery)
    assert result.to_dict()["manifest"]["discovery"] is not None
    with pytest.raises(ValueError, match="requires discovery"):
        compare(store, store, cases, group_key="customer")
    with pytest.raises(ValueError, match="must contain"):
        compare(store, store, cases, discovery=EvalSet([]))


def test_nonfinite_custom_metric_is_unevaluable():
    def unavailable(outcomes):
        return float("nan")

    store = Store([Row("query", "a")])
    cases = EvalSet([Case("c", "query", "a")])

    def run(query, snapshot):
        return Prediction("a")

    objective = Objective(primary=unavailable)
    with pytest.raises(ValueError, match=r"identity\.metric"):
        compare_stores(
            store,
            store,
            cases,
            pipeline=Pipeline(run, {"pipeline": "test"}),
            objective=objective,
        )
    result = compare_stores(
        store,
        store,
        cases,
        pipeline=Pipeline(run, {"pipeline": "test", "metric": "unavailable-v1"}),
        objective=objective,
    )
    assert result.exit_code == 2
    assert result.to_dict()["overall"]["candidate"]["primary"] is None
