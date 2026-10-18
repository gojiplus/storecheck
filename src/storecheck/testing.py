"""Helpers for using storecheck inside pytest.

from storecheck.testing import assert_clean
def test_store_is_clean():
report = run_static(store, taxonomy, embedder)
assert_clean(report)
"""

from __future__ import annotations

from .report import Report, Severity


def assert_clean(report: Report, fail_on: Severity = Severity.ERROR) -> None:
    """Raise AssertionError when findings reach the failure threshold.

    Args:
        report: Report whose findings must meet the failure threshold.
        fail_on: Minimum finding severity that fails the gate.

    Raises:
        AssertionError: If findings reach the failure threshold.
    """
    if report.passed(fail_on):
        return
    bad = [f for f in report.findings if f.severity >= fail_on]
    lines = [f"{len(bad)} finding(s) at or above {fail_on.name}:"]
    for f in bad[:20]:
        ids = f" rows={list(f.row_ids[:5])}" if f.row_ids else ""
        cases = f" cases={list(f.case_ids[:5])}" if f.case_ids else ""
        lines.append(f"  [{f.check}] {f.severity.name}: {f.message}{ids}{cases}")
    if len(bad) > 20:
        lines.append(f"  ... {len(bad) - 20} more")
    raise AssertionError("\n".join(lines))
