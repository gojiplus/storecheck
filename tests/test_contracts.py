import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from storecheck import (
    Case,
    EvalSet,
    Finding,
    HashingEmbedder,
    LabelDef,
    Objective,
    Report,
    Row,
    Severity,
    Store,
    Taxonomy,
    accuracy,
    run_checks,
    weighted_error_cost,
)
from storecheck.__main__ import main
from storecheck.checks import (
    CoverageCheck,
    MutationCheck,
    NearDuplicateCheck,
    NeighborhoodPurityCheck,
    SchemaCheck,
    ScopeBalanceCheck,
    StalenessCheck,
)
from storecheck.checks.base import CheckContext
from storecheck.checks.meta import Mutation
from storecheck.retriever import cosine_matrix
from storecheck.testing import assert_clean


def test_persistence_and_diff(tmp_path):
    row = Row(
        "café",
        "a",
        id="r",
        scope="s",
        metadata={"source": ["ticket"]},
        embedding=np.array([1.0, 2.0]),
        embedding_model="model",
    )
    store = Store([row])
    store.to_jsonl(tmp_path / "store.jsonl")
    restored = Store.from_jsonl(tmp_path / "store.jsonl")
    assert restored.version == store.version
    assert store.diff(restored).is_empty()
    assert row.id in store
    assert store.get("missing") is None
    assert store.scope("s").rows == store.rows
    assert len(store.scope("absent")) == 0
    assert "rows=1" in repr(store)
    changed = store.edit("r", scope="new")
    assert store.diff(changed).touched_ids == frozenset({"r"})
    assert store.diff(changed).touched_scopes == frozenset({"s", "new"})
    assert len(store.diff(store.remove("r")).removed) == 1
    with pytest.raises(KeyError, match="not in store"):
        store.remove("missing")
    with pytest.raises(ValueError, match="keeps the row id"):
        store.edit("r", id="new")
    with pytest.raises(ValueError, match="duplicate row ids"):
        Store([row, row])


def test_taxonomy_roundtrip(tmp_path):
    taxonomy = Taxonomy(
        [LabelDef("a", "first"), LabelDef("b", "second", deprecated=True)]
    )
    taxonomy.to_json(tmp_path / "taxonomy.json")
    restored = Taxonomy.from_json(tmp_path / "taxonomy.json")
    assert restored.version == taxonomy.version
    assert restored.ids == ["a", "b"]
    assert restored.active_ids == ["a"]
    assert restored.label_version("missing") is None
    with pytest.raises(ValueError, match="duplicate label ids"):
        Taxonomy([LabelDef("a", "one"), LabelDef("a", "two")])


def test_evalset_split_and_persistence(tmp_path):
    cases = [
        Case(str(i), f"query {i}", "a", frozenset({"slice"}), challenge=i == 0)
        for i in range(10)
    ]
    evalset = EvalSet(cases)
    assert evalset.slices == ["slice"]
    assert len(evalset.slice("slice")) == 10
    assert len(evalset.challenge) == 1
    dev, holdout = evalset.split(0.6, seed=42)
    assert len(dev) == 6
    assert len(holdout) == 4
    assert not {c.id for c in dev} & {c.id for c in holdout}
    assert evalset.split(0.6, seed=42)[0].cases == dev.cases
    evalset.to_jsonl(tmp_path / "eval.jsonl")
    assert EvalSet.from_jsonl(tmp_path / "eval.jsonl").cases == evalset.cases
    with pytest.raises(ValueError, match="duplicate case ids"):
        EvalSet([cases[0], cases[0]])


def test_metrics_and_report(tmp_path):
    outcomes = [
        (Case("a", "first", "yes", cost=4), "no"),
        (Case("b", "second", "yes"), "yes"),
    ]
    assert accuracy(outcomes) == 0.5
    assert weighted_error_cost(outcomes) == 4
    assert np.isnan(accuracy([]))
    assert Objective(primary=weighted_error_cost, higher_is_better=False).improved(4, 2)
    findings = [
        Finding("check", Severity.ERROR, "bad", ("row",), ("case",)) for _ in range(22)
    ]
    report = Report(findings, {"store": "example"})
    assert len(report.errors) == 22
    assert not report.warnings
    assert not report.passed()
    assert "more" in report.summary()
    report.to_json(tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text())["findings"][0][
        "case_ids"
    ] == ["case"]
    with pytest.raises(AssertionError, match="22 finding"):
        assert_clean(report)


def test_cli_diff_and_static_json(tmp_path, capsys):
    before = Store([Row("example", "a", id="r")])
    after = before.edit("r", label="b").add(Row("new row", "b", id="new"))
    before.to_jsonl(tmp_path / "before.jsonl")
    after.to_jsonl(tmp_path / "after.jsonl")
    assert (
        main(["diff", str(tmp_path / "before.jsonl"), str(tmp_path / "after.jsonl")])
        == 0
    )
    assert "changed=1" in capsys.readouterr().out
    assert (
        main(["diff", str(tmp_path / "after.jsonl"), str(tmp_path / "before.jsonl")])
        == 0
    )
    assert "removed=1" in capsys.readouterr().out
    assert (
        main(
            [
                "static",
                str(tmp_path / "before.jsonl"),
                "--json",
                str(tmp_path / "report.json"),
                "--fail-on",
                "WARN",
            ]
        )
        == 1
    )
    assert json.loads((tmp_path / "report.json").read_text())["context"]["rows"] == 1


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (SchemaCheck, {"min_chars": -1}),
        (SchemaCheck, {"min_chars": 5, "max_chars": 3}),
        (NearDuplicateCheck, {"threshold": float("nan")}),
        (NearDuplicateCheck, {"conflict_threshold": 1.1}),
        (NearDuplicateCheck, {"max_pairs": 0}),
        (NeighborhoodPurityCheck, {"k": 0}),
        (NeighborhoodPurityCheck, {"min_purity": 1.1}),
        (ScopeBalanceCheck, {"min_per_label": -1}),
        (StalenessCheck, {"ttl": timedelta(days=-1)}),
        (
            StalenessCheck,
            {"now": datetime(2020, 1, 1, tzinfo=UTC).replace(tzinfo=None)},
        ),
        (Objective, {"max_right_to_wrong": -1}),
        (Objective, {"max_right_to_wrong_frac": 1.1}),
        (Objective, {"min_primary_delta": float("inf")}),
        (Objective, {"min_slice_metric": {"slice": float("nan")}}),
        (HashingEmbedder, {"dim": 0}),
    ],
)
def test_invalid_settings_fail_at_construction(constructor, kwargs):
    with pytest.raises(ValueError, match=r"must|finite|exceed|timezone|nonnegative"):
        constructor(**kwargs)


def test_coverage_validation_and_scope_summary():
    embedder = HashingEmbedder()
    with pytest.raises(ValueError, match="scopes"):
        CoverageCheck(["one"], embedder, scopes=[])
    with pytest.raises(ValueError, match="min_similarity"):
        CoverageCheck(["one"], embedder, min_similarity=2)
    check = CoverageCheck(["one", "two"], embedder, scopes=["a", "b"])
    findings = check.run(CheckContext(Store([Row("one", "a", scope="a")])))
    assert findings[0].data["uncovered_frac"] == 0.5
    assert len(findings[0].data["by_scope"]) == 2
    assert (
        CoverageCheck([], embedder).run(CheckContext(Store()))[0].severity
        == Severity.ERROR
    )


def test_cosine_known_values_and_invalid_arrays():
    assert np.allclose(cosine_matrix(np.eye(2), np.eye(2)), np.eye(2))
    assert np.allclose(cosine_matrix(np.zeros((1, 2)), np.eye(2)), [[0, 0]])
    for a, b in [
        (np.ones(2), np.eye(2)),
        (np.eye(2), np.ones((1, 3))),
        (np.array([[np.nan, 0]]), np.eye(2)),
    ]:
        with pytest.raises(ValueError, match="cosine inputs"):
            cosine_matrix(a, b)


def test_mutation_contract_and_no_change(evaluator, store):
    with pytest.raises(ValueError, match="unknown mutation"):
        MutationCheck(
            evaluator, [Mutation("invalid", lambda store, evaluator: store, "invalid")]
        )
    report = run_checks(
        store,
        [
            MutationCheck(
                evaluator,
                [Mutation("identity", lambda store, evaluator: store, "no_change")],
            )
        ],
    )
    assert report.warnings


def test_cli_input_errors_are_actionable(tmp_path, capsys):
    assert main(["static", str(tmp_path / "missing.jsonl")]) == 2
    assert "missing.jsonl" in capsys.readouterr().err


def test_empty_eval_cannot_pass_even_with_constant_metric(store):
    from storecheck import EmbeddingRetriever, MajorityVoteDecider
    from storecheck.checks import Evaluator, RegressionCheck

    evaluator = Evaluator(
        EmbeddingRetriever(HashingEmbedder()), MajorityVoteDecider(), EvalSet([])
    )
    report = run_checks(
        store,
        [RegressionCheck(evaluator, store, Objective(primary=lambda outcomes: 1.0))],
    )
    assert not report.passed()


def test_nonfinite_metric_report_is_valid_json(evaluator, store):
    from storecheck.checks import RegressionCheck

    report = run_checks(
        store,
        [
            RegressionCheck(
                evaluator, store, Objective(primary=lambda outcomes: float("nan"))
            )
        ],
    )
    assert not report.passed()
    payload = report.to_json()
    assert "NaN" not in payload
    assert json.loads(payload)["findings"][0]["data"]["baseline"] is None
