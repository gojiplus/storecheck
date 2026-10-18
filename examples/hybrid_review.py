"""Review an actual local hybrid-search application through one callable.

SQLite FTS5 BM25 and word-hash vectors feed reciprocal-rank fusion; the top
example supplies the routing label. This synthetic application demonstrates
integration, not semantic embedding quality or production-scale indexing.

References:
https://www.sqlite.org/fts5.html#the_bm25_function
https://qdrant.tech/documentation/search/hybrid-queries/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from storecheck import (
    EvalSet,
    HashingEmbedder,
    Objective,
    Pipeline,
    Prediction,
    Query,
    Row,
    Snapshot,
    Store,
    compare_stores,
)
from storecheck.retriever import cosine_matrix
from storecheck.validation import positive_int


def build_snapshot(store: Store, path: Path) -> Snapshot:
    """Create a separate SQLite database and identify it by its file digest.

    Args:
        store: Source examples for this revision.
        path: New database file in the demonstration directory.

    Returns:
        External snapshot descriptor pointing to the completed database.
    """
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE examples (id TEXT PRIMARY KEY, scope TEXT, record TEXT)"
        )
        connection.execute("CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED, text)")
        connection.executemany(
            "INSERT INTO examples VALUES (?, ?, ?)",
            [(row.id, row.scope, json.dumps(row.to_dict())) for row in store],
        )
        connection.executemany(
            "INSERT INTO search VALUES (?, ?)", [(row.id, row.text) for row in store]
        )
        connection.commit()
    return Snapshot(
        path.stem, hashlib.sha256(path.read_bytes()).hexdigest(), {"path": str(path)}
    )


class HybridRouter:
    """Customer-owned retrieval and classification in one application callable."""

    def __init__(self, pool_size: int = 5, rank_constant: int = 60):
        """Configure candidate depth and reciprocal-rank smoothing.

        Args:
            pool_size: Maximum results from each retrieval branch.
            rank_constant: Positive constant in 1 / (constant + one-based rank).
        """
        positive_int("pool_size", pool_size)
        positive_int("rank_constant", rank_constant)
        self.pool_size = pool_size
        self.rank_constant = rank_constant
        self.embedder = HashingEmbedder()

    def __call__(self, query: Query, snapshot: Store | Snapshot) -> Prediction:
        """Run BM25, vector search, fusion, and classification against the snapshot.

        Args:
            query: Runtime query and optional scope filter.
            snapshot: Immutable SQLite database descriptor.

        Returns:
            Label, selected evidence, and actual branch rankings and fusion scores.

        Raises:
            ValueError: If the snapshot is local data or its database was modified.
        """
        if not isinstance(snapshot, Snapshot):
            raise ValueError("HybridRouter requires an external Snapshot")
        path = Path(snapshot.metadata["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot.revision:
            raise ValueError(
                "database contents do not match the pinned snapshot revision"
            )
        with closing(
            sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            records = connection.execute(
                "SELECT record FROM examples "
                "WHERE (? IS NULL OR scope = ?) ORDER BY id",
                (query.scope, query.scope),
            ).fetchall()
            rows = {
                row.id: row
                for row in (Row.from_dict(json.loads(r[0])) for r in records)
            }
            terms = " OR ".join(f'"{term}"' for term in re.findall(r"\w+", query.text))
            lexical = (
                connection.execute(
                    "SELECT e.id FROM search JOIN examples AS e ON e.id = search.id "
                    "WHERE search MATCH ? AND (? IS NULL OR e.scope = ?) "
                    "ORDER BY bm25(search), e.id LIMIT ?",
                    (terms, query.scope, query.scope, self.pool_size),
                ).fetchall()
                if terms
                else []
            )
        vector = []
        if rows:
            similarities = cosine_matrix(
                self.embedder.embed([query.text]),
                self.embedder.embed([row.text for row in rows.values()]),
            )[0]
            vector = sorted(
                zip(rows, similarities, strict=True),
                key=lambda pair: (-pair[1], pair[0]),
            )[: self.pool_size]
        lexical_ids = [row[0] for row in lexical]
        vector_ids = [identifier for identifier, _ in vector]
        fused: dict[str, float] = {}
        for ranking in (lexical_ids, vector_ids):
            for rank, identifier in enumerate(ranking, start=1):
                fused[identifier] = fused.get(identifier, 0) + 1 / (
                    self.rank_constant + rank
                )
        ranked = sorted(fused, key=lambda identifier: (-fused[identifier], identifier))
        evidence = tuple(rows[identifier] for identifier in ranked[:1])
        return Prediction(
            evidence[0].label if evidence else "unknown",
            evidence,
            trace={
                "lexical_ranking": lexical_ids,
                "vector_ranking": vector_ids,
                "fusion_scores": fused,
                "snapshot_revision": snapshot.revision,
            },
        )

    def pipeline(self) -> Pipeline:
        """Expose the application's callable and declared configuration.

        Returns:
            Pipeline ready for storecheck's comparison engine.
        """
        return Pipeline(
            self,
            {
                "application": "sqlite-hybrid-router-v1",
                "sqlite": sqlite3.sqlite_version,
                "vector_model": self.embedder.name,
                "pool_size": self.pool_size,
                "rank_constant": self.rank_constant,
                "decision": "top-fused-example-label",
                "scope_filter": "exact",
            },
        )


def main(argv: list[str] | None = None) -> int:
    """Build isolated database revisions and gate a deliberately harmful edit.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Gate status; one is expected because the added example breaks a case.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(prefix="storecheck-hybrid-", dir=args.output_dir)
    )
    fixtures = Path(__file__).parent / "compare"
    source = Store.from_jsonl(str(fixtures / "baseline.jsonl"))
    proposed = source.add(
        Row(
            "logged in permission denied on reports",
            "authentication",
            id="ambiguous-example",
            created_at=source.rows[0].created_at,
        )
    )
    baseline = build_snapshot(source, destination / "baseline.db")
    candidate = build_snapshot(proposed, destination / "candidate.db")
    for name, snapshot in (("baseline", baseline), ("candidate", candidate)):
        (destination / f"{name}.json").write_text(
            json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8"
        )
    report = compare_stores(
        baseline,
        candidate,
        EvalSet.from_jsonl(str(fixtures / "validation.jsonl")),
        pipeline=HybridRouter().pipeline(),
        objective=Objective(max_right_to_wrong_cost=0),
    )
    report.to_json(destination / "comparison.json")
    print("Application: SQLite BM25 + hash-vector search + reciprocal-rank fusion.")
    print(report.summary())
    for case in report.to_dict()["cases"]:
        if case["transition"] == "unchanged":
            continue
        print(f"Query: {case['case']['text']}")
        for side in ("baseline", "candidate"):
            prediction = case[side]
            print(
                f"  {side}: {prediction['prediction']}; "
                f"evidence={[r['id'] for r in prediction['evidence']]}"
            )
            print(f"  rankings: {prediction['trace']}")
    print(f"Database snapshots, descriptors, and report: {destination}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
