"""Findings and reports: the output of every check."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def _json_values(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    return value


class Severity(IntEnum):
    """Finding levels in increasing order of impact."""

    INFO = 0
    WARN = 1
    ERROR = 2


@dataclass
class Finding:
    """One diagnostic with affected rows, cases, and structured details."""

    check: str
    severity: Severity
    message: str
    row_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record.

        Returns:
            a JSON-compatible record.
        """
        return {
            "check": self.check,
            "severity": self.severity.name,
            "message": self.message,
            "row_ids": list(self.row_ids),
            "case_ids": list(self.case_ids),
            "data": self.data,
        }


@dataclass
class Report:
    """Collected findings and the state against which checks ran."""

    findings: list[Finding] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def extend(self, findings: Iterable[Finding]) -> None:
        """Append findings to this report.

        Args:
            findings: Findings to append.
        """
        self.findings.extend(findings)

    def by_severity(self, s: Severity) -> list[Finding]:
        """Return findings at the specified severity.

        Args:
            s: Severity to select.

        Returns:
            findings at the specified severity.
        """
        return [f for f in self.findings if f.severity == s]

    @property
    def errors(self) -> list[Finding]:
        """Return error findings.

        Returns:
            error findings.
        """
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> list[Finding]:
        """Return warning findings.

        Returns:
            warning findings.
        """
        return self.by_severity(Severity.WARN)

    def passed(self, fail_on: Severity = Severity.ERROR) -> bool:
        """Return whether no finding meets or exceeds the failure threshold.

        Args:
            fail_on: Minimum finding severity that fails the gate.

        Returns:
            whether no finding meets or exceeds the failure threshold.
        """
        return not any(f.severity >= fail_on for f in self.findings)

    def by_check(self) -> dict[str, list[Finding]]:
        """Group findings by check name.

        Returns:
            Group findings by check name.
        """
        out: dict[str, list[Finding]] = {}
        for f in self.findings:
            out.setdefault(f.check, []).append(f)
        return out

    def summary(self) -> str:
        """Return a readable summary of the recorded results.

        Returns:
            a readable summary of the recorded results.
        """
        lines = [
            f"storecheck report  errors={len(self.errors)}  "
            f"warnings={len(self.warnings)}  "
            f"info={len(self.by_severity(Severity.INFO))}"
        ]
        for k, v in self.context.items():
            lines.append(f"  {k}: {v}")
        for check, fs in self.by_check().items():
            lines.append(f"- {check}: {len(fs)} finding(s)")
            for f in fs[:5]:
                ids = (
                    f"  "
                    f"[{', '.join(f.row_ids[:3])}{'...' if len(f.row_ids) > 3 else ''}]"
                    if f.row_ids
                    else ""
                )
                lines.append(f"    {f.severity.name:5} {f.message}{ids}")
            if len(fs) > 5:
                lines.append(f"    ... {len(fs) - 5} more")
        return "\n".join(lines)

    def to_json(self, path: str | None = None) -> str:
        """Serialize the report or taxonomy as JSON.

        Args:
            path: Input or output file path.

        Returns:
            Serialize the report or taxonomy as JSON.
        """
        s = json.dumps(
            _json_values(
                {
                    "context": self.context,
                    "findings": [f.to_dict() for f in self.findings],
                }
            ),
            indent=2,
            allow_nan=False,
        )
        if path:
            with Path(path).open("w", encoding="utf-8") as fh:
                fh.write(s)
        return s
