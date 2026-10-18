import json

import pytest

from storecheck import (
    Case,
    EvalSet,
    Pipeline,
    Prediction,
    Query,
    Row,
    Snapshot,
    Store,
    compare_stores,
)


def test_whole_application_runs_both_sides_even_when_evidence_is_identical():
    row = Row("same evidence", "a", id="r")
    calls = []

    def application(query, snapshot):
        calls.append((query.text, snapshot.revision))
        return Prediction(
            "a" if snapshot.revision == "v1" else "b",
            (row,),
            trace={"fusion_score": 0.8, "index_revision": snapshot.revision},
        )

    report = compare_stores(
        Snapshot("tickets", "v1"),
        Snapshot("tickets", "v2"),
        EvalSet([Case("c", "same query", "a")]),
        pipeline=Pipeline(application, {"application": "actual-v1"}),
    )
    data = report.to_dict()
    assert calls == [("same query", "v1"), ("same query", "v2")]
    assert report.exit_code == 1
    assert data["cases"][0]["context_changed"] is False
    assert data["cases"][0]["transition"] == "right_to_wrong"
    assert data["cases"][0]["candidate"]["trace"]["index_revision"] == "v2"
    assert data["execution"]["pipeline_calls"] == 2
    assert "decider_calls" not in data["execution"]
    assert data["manifest"]["baseline"]["revision"] == "v1"
    assert data["manifest"]["candidate"]["revision"] == "v2"


def test_application_receives_runtime_inputs_without_reference_answers():
    received = []

    def application(query, snapshot):
        received.append(query)
        assert isinstance(query, Query)
        assert not hasattr(query, "label")
        assert not hasattr(query, "cost")
        assert not hasattr(query, "slices")
        assert query.scope == "tenant-a"
        assert query.metadata == {"locale": "fr"}
        return Prediction("a")

    store = Store([])
    cases = EvalSet(
        [Case("c", "bonjour", "a", scope="tenant-a", metadata={"locale": "fr"})]
    )
    report = compare_stores(
        store, store, cases, pipeline=Pipeline(application, {"v": 1})
    )
    assert len(received) == 2
    assert report.exit_code == 0
    assert report.to_dict()["cases"][0]["candidate"]["evidence"] == []


def test_duplicate_queries_are_not_memoized():
    calls = []

    def application(query, snapshot):
        calls.append(query.text)
        return Prediction("a")

    store = Store([])
    report = compare_stores(
        store,
        store,
        EvalSet([Case("1", "same", "a"), Case("2", "same", "a")]),
        pipeline=Pipeline(application, {"v": 1}),
    )
    assert len(calls) == 4
    assert report.to_dict()["execution"]["pipeline_calls"] == 4


def test_snapshot_identity_covers_external_revision_and_metadata():
    pipeline = Pipeline(lambda query, snapshot: Prediction("a"), {"v": 1})
    cases = EvalSet([Case("c", "q", "a")])
    base = Snapshot("tickets", "v1", {"namespace": "one"})
    ids = {
        compare_stores(base, candidate, cases, pipeline=pipeline).to_dict()["run_id"]
        for candidate in (
            base,
            Snapshot("tickets", "v2", {"namespace": "one"}),
            Snapshot("tickets", "v1", {"namespace": "two"}),
        )
    }
    assert len(ids) == 3


def test_pipeline_identity_and_prediction_trace_are_detached():
    original = {"scores": [1, 2]}
    prediction = Prediction("a", trace=original)
    pipeline = Pipeline(lambda query, snapshot: prediction, original)
    original["scores"].clear()
    assert prediction.trace["scores"] == (1, 2)
    assert pipeline.identity["scores"] == (1, 2)


@pytest.mark.parametrize("label", [None, "", "  ", 3, ["a"]])
def test_bad_prediction_label(label):
    with pytest.raises(ValueError, match="label"):
        Prediction(label)


def test_malformed_application_result_cannot_pass():
    pipeline = Pipeline(lambda query, snapshot: "a", {"v": 1})
    with pytest.raises(ValueError, match=r"case 'c'.*Prediction"):
        compare_stores(
            Store([]), Store([]), EvalSet([Case("c", "q", "a")]), pipeline=pipeline
        )


def test_backend_error_is_not_retried_or_converted_to_a_pass():
    def broken(query, snapshot):
        raise TypeError("broken reranker")

    with pytest.raises(TypeError, match="broken reranker"):
        compare_stores(
            Store([]),
            Store([]),
            EvalSet([Case("c", "q", "a")]),
            pipeline=Pipeline(broken, {"v": 1}),
        )


def test_snapshot_json_roundtrip(tmp_path):
    path = tmp_path / "snapshot.json"
    descriptor = {
        "name": "tickets",
        "revision": "v1",
        "metadata": {"collection": "tickets-v1"},
    }
    path.write_text(json.dumps(descriptor))
    assert Snapshot.from_json(path).to_dict() == descriptor
    with pytest.raises(ValueError, match="revision"):
        Snapshot("tickets", "")


@pytest.mark.parametrize(
    "kwargs", [{"evidence": ["r"]}, {"trace": []}, {"trace": {"score": float("nan")}}]
)
def test_invalid_prediction_fields(kwargs):
    with pytest.raises(ValueError, match=r"evidence|trace|JSON"):
        Prediction("a", **kwargs)


def test_async_applications_are_rejected_without_running():
    async def run(query, snapshot):
        return Prediction("a")

    class AsyncApplication:
        async def __call__(self, query, snapshot):
            return Prediction("a")

    for application in (run, AsyncApplication(), None):
        with pytest.raises(ValueError, match="synchronous callable"):
            Pipeline(application, {"v": 1})


def test_invalid_inputs_are_rejected_before_application_calls():
    calls = []

    def application(query, snapshot):
        calls.append(query)
        return Prediction("a")

    pipeline = Pipeline(application, {"v": 1})
    cases = EvalSet([Case("c", "q", "a")])
    with pytest.raises(ValueError, match="overlap"):
        compare_stores(Store([]), Store([]), cases, pipeline=pipeline, discovery=cases)
    with pytest.raises(ValueError, match="Store or Snapshot"):
        compare_stores("live", Store([]), cases, pipeline=pipeline)
    with pytest.raises(ValueError, match="query text"):
        compare_stores(
            Store([]), Store([]), EvalSet([Case("c", "   ", "a")]), pipeline=pipeline
        )
    assert calls == []
