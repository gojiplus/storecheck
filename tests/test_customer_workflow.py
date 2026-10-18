import json
import runpy
from pathlib import Path

from storecheck import Store

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "release_review.py"


def test_release_review_rejects_hidden_regression_then_validates_revision(tmp_path):
    example = runpy.run_path(str(EXAMPLE))
    assert example["main"](["--output-dir", str(tmp_path)]) == 0

    def artifact(name):
        return json.loads((tmp_path / f"{name}.json").read_text())

    proposal = artifact("discovery-proposed")
    assert proposal["status"] == "fail"
    assert proposal["overall"]["baseline"]["accuracy"] == 2 / 3
    assert proposal["overall"]["candidate"]["accuracy"] == 2 / 3
    assert proposal["overall"]["baseline"]["error_cost"] == 1
    assert proposal["overall"]["candidate"]["error_cost"] == 50
    assert proposal["overall"]["fixed"] == 1
    assert proposal["overall"]["broken"] == 1
    broken = next(c for c in proposal["cases"] if c["transition"] == "right_to_wrong")
    assert broken["baseline"]["evidence"][0]["id"] == "reports"
    assert broken["candidate"]["evidence"][0]["id"] == "sso-example"

    revision = artifact("discovery-revised")
    validation = artifact("validation-revised")
    assert revision["status"] == "pass"
    assert validation["status"] == "pass"
    assert validation["overall"]["n"] == 8
    assert validation["overall"]["baseline"]["accuracy"] == 7 / 8
    assert validation["overall"]["candidate"]["accuracy"] == 1
    assert validation["overall"]["broken"] == 0
    assert validation["manifest"]["discovery"] == revision["manifest"]["evalset"]
    assert validation["manifest"]["evalset"] != revision["manifest"]["evalset"]
    assert validation["manifest"]["candidate"] == revision["manifest"]["candidate"]
    baseline = Store.from_jsonl(str(tmp_path / "baseline.jsonl"))
    revised = Store.from_jsonl(str(tmp_path / "revised.jsonl"))
    assert len(baseline) == 6
    assert len(revised) == 7
    assert baseline.diff(revised).added[0].id == "sso-example"


def test_discovery_failure_keeps_validation_closed(monkeypatch):
    example = runpy.run_path(str(EXAMPLE))
    original = example["evaluate"]
    calls = []

    def fail_review(baseline, candidate, cases, taxonomy, discovery=None):
        calls.append(cases.name)
        return original(
            baseline,
            candidate.edit("sso-example", text="SSO login access denied"),
            cases,
            taxonomy,
            discovery,
        )

    monkeypatch.setitem(example["main"].__globals__, "evaluate", fail_review)
    assert example["main"]([]) == 1
    assert calls == ["discovery", "discovery"]
