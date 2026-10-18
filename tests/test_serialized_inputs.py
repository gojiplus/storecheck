import json
from datetime import UTC, datetime, timedelta

import pytest

from storecheck import Case, Row, Store, Taxonomy
from storecheck.checks.base import CheckContext
from storecheck.checks.static import StalenessCheck


def test_missing_creation_time_is_stable_and_unknown():
    record = {"id": "r", "text": "query", "label": "a"}
    first = Row.from_dict(record)
    second = Row.from_dict(record)
    assert first.created_at is None
    assert first.to_dict() == second.to_dict()
    assert Store([first]).version == Store([second]).version
    assert Row.from_dict(first.to_dict()).to_dict() == first.to_dict()
    findings = StalenessCheck().run(CheckContext(Store([first])))
    assert any("unknown" in f.message for f in findings)
    assert findings[0].row_ids == ("r",)


def test_unknown_time_does_not_hide_old_rows():
    unknown = Row.from_dict({"id": "u", "text": "query", "label": "a"})
    old = Row("old", "a", id="old", created_at=datetime.now(UTC) - timedelta(days=800))
    findings = StalenessCheck().run(CheckContext(Store([unknown, old])))
    assert len(findings) == 2
    assert any(f.row_ids == ("old",) for f in findings)


def test_serialized_slice_string_is_not_a_sequence_of_slices():
    with pytest.raises(ValueError, match="slices"):
        Case.from_dict({"id": "c", "text": "query", "label": "a", "slices": "critical"})


def test_serialized_boundary_string_is_not_a_sequence_of_examples(tmp_path):
    path = tmp_path / "taxonomy.json"
    path.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "id": "a",
                        "definition": "first",
                        "examples_of_boundary": "boundary",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="examples_of_boundary"):
        Taxonomy.from_json(str(path))
