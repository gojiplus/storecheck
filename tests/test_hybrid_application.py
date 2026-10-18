import json
import runpy
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from storecheck import Query, Row, Snapshot, Store

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hybrid_review.py"


def test_customer_hybrid_application_detects_regression_and_records_real_rankings(
    tmp_path,
):
    example = runpy.run_path(str(EXAMPLE))
    assert example["main"](["--output-dir", str(tmp_path)]) == 1
    report = json.loads(next(tmp_path.glob("*/comparison.json")).read_text())
    assert report["overall"]["baseline"]["accuracy"] == 1
    assert report["overall"]["candidate"]["accuracy"] == 2 / 3
    assert report["execution"]["pipeline_calls"] == 6
    changed = [
        case for case in report["cases"] if case["transition"] == "right_to_wrong"
    ]
    assert len(changed) == 1
    assert changed[0]["case"]["id"] == "reports"
    for side in ("baseline", "candidate"):
        prediction = changed[0][side]
        trace = prediction["trace"]
        scores = {}
        for ranking in (trace["lexical_ranking"], trace["vector_ranking"]):
            for rank, identifier in enumerate(ranking, start=1):
                scores[identifier] = scores.get(identifier, 0) + 1 / (60 + rank)
        assert trace["fusion_scores"] == scores
        assert prediction["evidence"][0]["id"] == max(scores, key=scores.get)


def test_application_honors_scope_and_refuses_mutated_snapshot(tmp_path):
    example = runpy.run_path(str(EXAMPLE))
    source = Store(
        [
            Row("same query", "a", id="a", scope="one"),
            Row("same query", "b", id="b", scope="two"),
        ]
    )
    path = tmp_path / "index.db"
    snapshot = example["build_snapshot"](source, path)
    app = example["HybridRouter"]()
    result = app(Query("same query", scope="two"), snapshot)
    assert result.label == "b"
    assert result.trace["lexical_ranking"] == ("b",)
    assert result.trace["vector_ranking"] == ("b",)
    assert app(Query("same query", scope="missing"), snapshot).label == "unknown"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM examples WHERE id = ?", ("b",))
        connection.commit()
    with pytest.raises(ValueError, match="pinned snapshot revision"):
        app(Query("same query"), snapshot)
    with pytest.raises(ValueError, match="external Snapshot"):
        app(Query("same query"), source)


def test_empty_query_and_snapshot_are_observed_without_fabricated_evidence(tmp_path):
    example = runpy.run_path(str(EXAMPLE))
    snapshot = example["build_snapshot"](Store([]), tmp_path / "empty.db")
    result = example["HybridRouter"]()(Query("?"), snapshot)
    assert isinstance(snapshot, Snapshot)
    assert result.label == "unknown"
    assert result.evidence == ()
    assert result.trace["fusion_scores"] == {}
