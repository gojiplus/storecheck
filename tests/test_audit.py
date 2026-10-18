import numpy as np
import pytest

from storecheck import (
    CallableDecider,
    Case,
    EmbeddingRetriever,
    EvalSet,
    HashingEmbedder,
    MajorityVoteDecider,
    MemoizedDecider,
    Objective,
    Row,
    Severity,
    Store,
    run_static,
)
from storecheck.checks import CoverageCheck, Evaluator, RegressionCheck
from storecheck.checks.base import CheckContext
from storecheck.checks.static import PIICheck


def test_text_edit_discards_obsolete_embedding():
    row = Row("old", "a", embedding=np.array([1.0, 0.0]))
    assert row.with_(text="new").embedding is None


def test_embedding_change_affects_identity_and_diff():
    store = Store([Row("example", "a", id="r", embedding=np.array([1.0, 0.0]))])
    changed = store.edit("r", embedding=np.array([0.0, 1.0]))
    assert changed.version != store.version
    assert len(store.diff(changed).changed) == 1


def test_metadata_change_is_visible_to_decider():
    row = Row("example", "a", metadata={"answer": "a"})
    decider = MemoizedDecider(
        CallableDecider(lambda query, rows: rows[0].metadata["answer"])
    )
    assert decider.decide("q", [row]) == "a"
    assert decider.decide("q", [row.with_(metadata={"answer": "b"})]) == "b"


def test_retrieval_cache_respects_tie_order():
    rows = [Row("identical", "a", id="a"), Row("identical", "b", id="b")]
    retriever = EmbeddingRetriever(HashingEmbedder(), k=1)
    retriever.retrieve("identical", Store(rows))
    second = retriever.retrieve("identical", Store(rows[::-1]))
    fresh = EmbeddingRetriever(HashingEmbedder(), k=1).retrieve(
        "identical", Store(rows[::-1])
    )
    assert [r.id for r in second] == [r.id for r in fresh]


def test_retrieval_cache_does_not_expose_mutable_list():
    retriever = EmbeddingRetriever(HashingEmbedder(), k=1)
    store = Store([Row("example", "a")])
    retriever.retrieve("example", store).clear()
    assert len(retriever.retrieve("example", store)) == 1


def test_explicit_empty_cases_stay_empty(evaluator, store):
    assert evaluator.run(store, cases=[]).results == []


def test_explicit_empty_checks_stay_empty(store):
    assert run_static(store, checks=[]).findings == []


def test_explicit_empty_pii_patterns_stay_empty():
    assert PIICheck(patterns={}).run(CheckContext(Store([Row("x@y.com", "a")]))) == []


def test_scope_coverage_cannot_borrow_other_scope():
    findings = CoverageCheck(["example"], HashingEmbedder(), scopes=["private"]).run(
        CheckContext(Store([Row("example", "a", scope="public")]))
    )
    assert findings[0].severity == Severity.ERROR
    assert findings[0].data["uncovered_frac"] == 1


def test_empty_store_has_zero_coverage():
    findings = CoverageCheck(["example"], HashingEmbedder()).run(CheckContext(Store()))
    assert findings[0].severity == Severity.ERROR
    assert findings[0].data["uncovered_frac"] == 1


def test_required_missing_slice_blocks_gate(evaluator, store):
    findings = RegressionCheck(
        evaluator, store, Objective(min_slice_metric={"absent": 0.8})
    ).run(CheckContext(store))
    assert any(f.severity == Severity.ERROR and "absent" in f.message for f in findings)


def test_supplied_baseline_must_match_store(evaluator, store):
    run = evaluator.run(store)
    changed = store.add(Row("extra", "billing"))
    with pytest.raises(ValueError, match="baseline"):
        evaluator.compare(changed, changed, baseline_run=run)


@pytest.mark.parametrize("value", [-1, 0, 1.5, True])
def test_invalid_neighbor_count(value):
    with pytest.raises(ValueError, match="k"):
        EmbeddingRetriever(HashingEmbedder(), k=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_split_fraction(value):
    with pytest.raises(ValueError, match="dev_frac"):
        EvalSet([]).split(value)


def test_retriever_type_error_is_not_swallowed():
    class BrokenRetriever:
        k = 1

        def retrieve(self, query, store, k=None, scope=None):
            if scope is not None:
                raise TypeError("backend bug")
            return []

    evaluator = Evaluator(
        BrokenRetriever(),
        MajorityVoteDecider(),
        EvalSet([Case("c", "q", "a", scope="x")]),
    )
    with pytest.raises(TypeError, match="backend bug"):
        evaluator.run(Store())


def test_row_detaches_and_freezes_mutable_inputs():
    metadata = {"nested": [1]}
    embedding = np.array([1.0, 0.0])
    row = Row("example", "a", metadata=metadata, embedding=embedding)
    version = Store([row]).version
    metadata["nested"].append(2)
    embedding[:] = 0
    assert Store([row]).version == version
    with pytest.raises(TypeError):
        row.metadata["new"] = 1
    with pytest.raises(ValueError, match="read-only"):
        row.embedding[0] = 0


def test_changed_embedder_recomputes_vectors():
    store = Store([Row("example", "a")]).embed(HashingEmbedder(dim=2))
    updated = store.embed(HashingEmbedder(dim=4))
    assert updated.embeddings().shape == (1, 4)


def test_short_embedding_batch_is_rejected():
    class BrokenEmbedder:
        name = "broken"

        def embed(self, texts):
            return np.ones((1, 2))

    with pytest.raises(ValueError, match="one nonempty vector per text"):
        Store([Row("one", "a"), Row("two", "b")]).embed(BrokenEmbedder())


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf")])
def test_case_cost_requires_finite_nonnegative(cost):
    with pytest.raises(ValueError, match="cost"):
        Case("case", "query", "label", cost=cost)


def test_influence_retrieves_all_rows_for_global_retrievers():
    from storecheck.checks import row_influence

    class GlobalRetriever:
        k = 1

        def retrieve(self, query, store, k=None, scope=None):
            preferred = "a" if "switch" in store else "b"
            return [store.get(preferred) or store.rows[0]]

    store = Store(
        [
            Row("one", "a", id="a"),
            Row("two", "b", id="b"),
            Row("switch", "b", id="switch"),
        ]
    )
    evaluator = Evaluator(
        GlobalRetriever(), MajorityVoteDecider(), EvalSet([Case("c", "q", "b")])
    )
    influences = row_influence(evaluator, store)
    assert next(item.net for item in influences if item.row.id == "switch") == 1
    # A never-retrieved row can still control the evidence of a global retriever.
    # Restricting to retrieved rows requires an explicit deletion-stability guarantee.


def test_decision_layer_rejects_run_from_another_store(evaluator, store):
    from storecheck.checks import DecisionLayerCheck

    run = evaluator.run(store)
    changed = store.add(Row("extra example", "billing"))
    with pytest.raises(ValueError, match="baseline"):
        DecisionLayerCheck(evaluator, baseline_run=run).run(CheckContext(changed))


def test_influence_validates_baseline_even_without_rows(evaluator, store):
    from storecheck.checks import row_influence

    run = evaluator.run(store)
    changed = store.add(Row("extra example", "billing"))
    with pytest.raises(ValueError, match="baseline"):
        row_influence(evaluator, changed, row_ids=[], baseline_run=run)
