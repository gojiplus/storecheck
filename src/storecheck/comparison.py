"""Reproducible comparison artifacts shared by the Python API and CLI."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, fields
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checks.dynamic import CaseResult, Comparison, EvalRun, Transition, judge
from .objective import Objective, accuracy, weighted_error_cost
from .pipeline import Pipeline, Prediction, Query, Snapshot
from .report import Finding, Severity, _json_values
from .store import Store, _thaw

if TYPE_CHECKING:
    from .evalset import EvalSet
    from .taxonomy import Taxonomy


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _metrics(results: list[CaseResult], objective: Objective) -> dict[str, Any]:
    outcomes = [(r.case, r.predicted) for r in results]
    return {
        "accuracy": accuracy(outcomes),
        "error_cost": weighted_error_cost(outcomes),
        "primary": objective.primary(outcomes),
    }


def _paired_metrics(
    before: list[CaseResult], after: list[CaseResult], objective: Objective
) -> dict[str, Any]:
    return {
        "n": len(before),
        "baseline": _metrics(before, objective),
        "candidate": _metrics(after, objective),
        "fixed": sum(
            not b.correct and a.correct for b, a in zip(before, after, strict=True)
        ),
        "broken": sum(
            b.correct and not a.correct for b, a in zip(before, after, strict=True)
        ),
        "new_error_cost": sum(
            b.case.cost
            for b, a in zip(before, after, strict=True)
            if b.correct and not a.correct
        ),
    }


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4g}"


def _prediction(result: CaseResult) -> dict[str, Any]:
    return {
        "prediction": result.predicted,
        "correct": result.correct,
        "error_cost": 0 if result.correct else result.case.cost,
        "evidence": [row.to_dict() for row in result.evidence],
        "trace": _thaw(result.trace),
    }


@dataclass(frozen=True)
class ComparisonReport:
    """An immutable JSON snapshot of a completed comparison."""

    _json: str

    def to_dict(self) -> dict[str, Any]:
        """Return an independent copy of the artifact.

        Returns:
            Manifest, findings, metrics, and all paired case results.
        """
        return json.loads(self._json)

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize the artifact, optionally writing it as UTF-8.

        Args:
            path: Optional output file.

        Returns:
            Strict JSON with unavailable metrics represented as null.
        """
        if path is not None:
            Path(path).write_text(self._json + "\n", encoding="utf-8")
        return self._json

    @property
    def exit_code(self) -> int:
        """Return the process status for this result.

        Returns:
            Zero for pass, one for fail, or two for unevaluable.
        """
        return {"pass": 0, "fail": 1, "unevaluable": 2}[self.to_dict()["status"]]

    def summary(self) -> str:
        """Describe the verdict, sample size, costs, and failed requirements.

        Returns:
            Human-readable CLI output.
        """
        data = self.to_dict()
        overall = data["overall"]
        before, after = overall["baseline"], overall["candidate"]
        lines = [
            f"{data['status'].upper()} — {overall['n']} cases; "
            f"fixed={overall['fixed']} broken={overall['broken']}",
            f"  {data['manifest']['objective']['primary_name']}: "
            f"{_number(before['primary'])} -> {_number(after['primary'])}",
            f"  error cost: {before['error_cost']} -> {after['error_cost']}; "
            f"new error cost={overall['new_error_cost']}",
            f"  pipeline calls={data['execution']['pipeline_calls']}; "
            f"run={data['run_id'][:16]}",
        ]
        lines.extend(
            f"  {f['message']}" for f in data["findings"] if f["severity"] == "ERROR"
        )
        for name, metrics in data["slices"].items():
            lines.append(
                f"  slice {name!r}: n={metrics['n']}; "
                f"{_number(metrics['baseline']['primary'])} -> "
                f"{_number(metrics['candidate']['primary'])}"
            )
        broken = [
            c["case"]["id"]
            for c in data["cases"]
            if c["transition"] == "right_to_wrong"
        ]
        if broken:
            lines.append(f"  broken cases: {', '.join(broken[:10])}")
        return "\n".join(lines)


def compare_stores(
    baseline: Store | Snapshot,
    candidate: Store | Snapshot,
    evalset: EvalSet,
    *,
    pipeline: Pipeline,
    objective: Objective | None = None,
    taxonomy: Taxonomy | None = None,
    discovery: EvalSet | None = None,
    group_key: str | None = None,
) -> ComparisonReport:
    """Run the customer's application against both revisions for every case.

    No prediction is reused, even for unchanged evidence or duplicate queries.
    External snapshots are caller-declared immutable references; the application
    must honor their revisions. This is one observed paired run, not a statistical
    guarantee about future or stochastic behavior. Runtime metadata must not
    contain reference answers. Input validation completes before application calls.

    Args:
        baseline: Current local data or external snapshot reference.
        candidate: Proposed local data or external snapshot reference.
        evalset: Fixed labeled evaluation cases.
        pipeline: Application callable and its complete nonsecret configuration.
        objective: Gate requirements; defaults to nondecreasing accuracy.
        taxonomy: Optional label definitions for validation and run identity.
        discovery: Optional nonempty, disjoint cases used to select the edit.
        group_key: Optional metadata field for discovery/validation overlap checks.

    Returns:
        Immutable comparison artifact with a pass, fail, or unevaluable verdict.

    Raises:
        ValueError: If inputs, reference labels, or application outputs are invalid.
    """
    if not isinstance(pipeline, Pipeline):
        raise ValueError("pipeline must be a Pipeline instance")
    identity = _thaw(pipeline.identity)
    baseline_identity = _snapshot_identity(baseline)
    candidate_identity = _snapshot_identity(candidate)
    queries = [Query(case.text, case.scope, case.metadata) for case in evalset]
    if group_key is not None and discovery is None:
        raise ValueError("group_key requires discovery cases")
    if discovery is not None:
        if not len(discovery):
            raise ValueError("discovery must contain cases when supplied")
        discovery.assert_disjoint(evalset, group_key=group_key)
    objective = objective if objective is not None else Objective()
    if objective.primary not in (accuracy, weighted_error_cost) and not identity.get(
        "metric"
    ):
        raise ValueError("identity.metric is required for a custom primary metric")
    labels = None if taxonomy is None else set(taxonomy.active_ids)
    if labels is not None:
        records = list(evalset)
        invalid = {record.label for record in records} - labels
        for snapshot in (baseline, candidate):
            if isinstance(snapshot, Store):
                invalid.update(row.label for row in snapshot if row.label not in labels)
        if invalid:
            raise ValueError(f"labels outside active taxonomy: {sorted(invalid)}")
    objective_config = {
        f.name: getattr(objective, f.name)
        for f in fields(objective)
        if f.name != "primary"
    }
    metric_name = getattr(
        objective.primary, "__qualname__", type(objective.primary).__qualname__
    )
    objective_config["primary_callable"] = (
        f"{objective.primary.__module__}.{metric_name}"
    )
    manifest = {
        "baseline": baseline_identity,
        "candidate": candidate_identity,
        "evaluation": {"mode": "paired_single_run", "prediction_reuse": False},
        "evalset": _digest([c.to_dict() for c in evalset]),
        "discovery": None
        if discovery is None
        else _digest([c.to_dict() for c in discovery]),
        "group_key": group_key,
        "taxonomy": None
        if taxonomy is None
        else [
            {
                "id": label.id,
                "definition": label.definition,
                "examples_of_boundary": list(label.examples_of_boundary),
                "deprecated": label.deprecated,
            }
            for label in taxonomy
        ],
        "pipeline": identity,
        "objective": objective_config,
        "versions": {
            "storecheck": version("storecheck"),
            "numpy": version("numpy"),
            "python": platform.python_version(),
        },
    }
    run_id = _digest(manifest)
    started = time.perf_counter()
    before: list[CaseResult] = []
    after: list[CaseResult] = []
    transitions: list[Transition] = []
    changed = 0
    for case, query in zip(evalset, queries, strict=True):
        for side, snapshot, results in (
            ("baseline", baseline, before),
            ("candidate", candidate, after),
        ):
            result = pipeline.run(query, snapshot)
            if not isinstance(result, Prediction):
                raise ValueError(
                    f"case {case.id!r} {side}: pipeline must return Prediction"
                )
            results.append(
                CaseResult(case, result.evidence, result.label, result.trace)
            )
        b, a = before[-1], after[-1]
        changed += [row.to_dict() for row in b.evidence] != [
            row.to_dict() for row in a.evidence
        ]
        transition = Transition(
            case, b.predicted, a.predicted, b.evidence_ids, a.evidence_ids
        )
        if transition.kind != "unchanged":
            transitions.append(transition)
    elapsed = time.perf_counter() - started
    comparison = Comparison(
        EvalRun(_digest(baseline_identity), before),
        EvalRun(_digest(candidate_identity), after),
        changed,
        None,
        transitions,
    )
    findings = judge(comparison, objective)
    if labels is not None:
        invalid_cases = tuple(
            sorted({r.case.id for r in [*before, *after] if r.predicted not in labels})
        )
        if invalid_cases:
            findings.append(
                Finding(
                    "regression",
                    Severity.ERROR,
                    "predictions outside active taxonomy",
                    case_ids=invalid_cases,
                    data={"unevaluable": True},
                )
            )
    status = "pass"
    if any(f.severity == Severity.ERROR for f in findings):
        status = "fail"
    if any(f.data.get("unevaluable") for f in findings):
        status = "unevaluable"
    cases = []
    for b, a in zip(before, after, strict=True):
        transition = Transition(
            b.case, b.predicted, a.predicted, b.evidence_ids, a.evidence_ids
        )
        cases.append(
            {
                "case": b.case.to_dict(),
                "baseline": _prediction(b),
                "candidate": _prediction(a),
                "transition": transition.kind,
                "context_changed": [r.to_dict() for r in b.evidence]
                != [r.to_dict() for r in a.evidence],
            }
        )
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "manifest": manifest,
        "status": status,
        "overall": _paired_metrics(before, after, objective),
        "slices": {
            s: _paired_metrics(
                [r for r in before if s in r.case.slices],
                [r for r in after if s in r.case.slices],
                objective,
            )
            for s in evalset.slices
        },
        "cases": cases,
        "findings": [f.to_dict() for f in findings],
        "execution": {
            "elapsed_seconds": elapsed,
            "pipeline_calls": 2 * len(evalset),
            "context_changed": comparison.context_changed,
        },
    }
    return ComparisonReport(
        json.dumps(_json_values(payload), indent=2, allow_nan=False)
    )


def _snapshot_identity(snapshot: Store | Snapshot) -> dict[str, Any]:
    if isinstance(snapshot, Store):
        return {
            "kind": "local",
            "fingerprint": _digest([r.to_dict() for r in snapshot]),
            "rows": len(snapshot),
        }
    if isinstance(snapshot, Snapshot):
        return {"kind": "external", **snapshot.to_dict()}
    raise ValueError("snapshots must be Store or Snapshot instances")
