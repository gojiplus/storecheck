from datetime import UTC, datetime, timedelta

from storecheck import LabelDef, Row, Severity, Store, Taxonomy, run_static
from storecheck.checks import (
    ExactDuplicateCheck,
    NearDuplicateCheck,
    NeighborhoodPurityCheck,
    PIICheck,
    SchemaCheck,
    StalenessCheck,
    TaxonomyValidityCheck,
)
from storecheck.checks.base import CheckContext
from storecheck.testing import assert_clean


def test_clean_store_passes(store, taxonomy, embedder):
    report = run_static(store, taxonomy, embedder)
    assert_clean(report)


def test_store_is_immutable_and_versioned(store):
    v0 = store.version
    s2 = store.add(Row("new example about refunds", "billing"))
    assert store.version == v0
    assert s2.version != v0
    assert len(s2) == len(store) + 1
    d = store.diff(s2)
    assert len(d.added) == 1
    assert not d.removed
    assert not d.changed
    s3 = s2.edit(d.added[0].id, label="authorization_failure")
    assert len(s2.diff(s3).changed) == 1


def test_schema_flags_empty_text(taxonomy):
    s = Store([Row("   ", "billing", id="x")])
    f = SchemaCheck().run(CheckContext(s, taxonomy))
    assert any(x.severity == Severity.ERROR and "empty text" in x.message for x in f)


def test_taxonomy_unknown_and_orphaned_labels(store, taxonomy):
    bad = store.add(Row("something about shipping", "shipping"))
    f = TaxonomyValidityCheck().run(CheckContext(bad, taxonomy))
    assert any("not in taxonomy" in x.message for x in f)

    # redefine a label: rows written under the old definition become orphans
    new_tax = Taxonomy(
        [
            LabelDef(item.id, item.definition + " (revised)")
            if item.id == "billing"
            else item
            for item in taxonomy
        ]
    )
    f = TaxonomyValidityCheck().run(CheckContext(store, new_tax))
    orphan = [x for x in f if "older definition" in x.message]
    assert len(orphan) == 1
    assert len(orphan[0].row_ids) == 3


def test_exact_duplicates_conflict_vs_redundant(store):
    r = store.rows[0]
    conflict = store.add(Row(r.text.upper(), "billing"))
    f = ExactDuplicateCheck().run(CheckContext(conflict))
    assert any(x.severity == Severity.ERROR for x in f)
    redundant = store.add(Row(r.text + "  ", r.label, id="dup"))
    f = ExactDuplicateCheck().run(CheckContext(redundant))
    assert any(x.severity == Severity.WARN for x in f)


def test_near_duplicates_with_conflicting_labels(store, embedder):
    r = store.rows[0]
    s = store.add(Row(r.text + " please", "authorization_failure"))
    f = NearDuplicateCheck(threshold=0.9).run(CheckContext(s, embedder=embedder))
    assert any(
        x.severity == Severity.ERROR and "different labels" in x.message for x in f
    )


def test_staleness():
    old = Row(
        "ancient example", "billing", created_at=datetime.now(UTC) - timedelta(days=800)
    )
    f = StalenessCheck(ttl=timedelta(days=365)).run(
        CheckContext(Store([old, Row("fresh", "billing")]))
    )
    assert len(f) == 1
    assert f[0].row_ids == (old.id,)


def test_pii():
    s = Store(
        [Row("my email is someone@example.com and card 4111 1111 1111 1111", "billing")]
    )
    f = PIICheck().run(CheckContext(s))
    assert f
    assert set(f[0].data["kinds"]) >= {"email", "card"}


def test_neighborhood_purity_flags_mislabel(store, embedder):
    r = store.rows[0]  # an authn example; relabel it as billing
    s = store.edit(r.id, label="billing")
    f = NeighborhoodPurityCheck(k=3, min_purity=0.5).run(
        CheckContext(s, embedder=embedder)
    )
    flagged = {x.row_ids[0] for x in f if x.row_ids}
    assert r.id in flagged


def test_cli_roundtrip(tmp_path, store, taxonomy):
    import subprocess
    import sys

    sp, tp = tmp_path / "s.jsonl", tmp_path / "t.json"
    store.to_jsonl(sp)
    taxonomy.to_json(tp)
    assert len(Store.from_jsonl(str(sp))) == len(store)
    out = subprocess.run(  # noqa: S603 - invokes this installed package
        [sys.executable, "-m", "storecheck", "static", str(sp), "--taxonomy", str(tp)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "storecheck report" in out.stdout
