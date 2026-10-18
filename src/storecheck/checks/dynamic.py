"""Dynamic tier: checks that run retrieval and (sparingly) the decider.

Core operation: hold the eval set fixed, change the store, and compare.
Retrieval runs for both store versions; the decider runs only on cases
whose retrieved context changed. The MemoizedDecider makes that automatic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..decider import Decider, MemoizedDecider
from ..objective import Objective, Outcome, accuracy
from ..report import Finding, Severity
from ..retriever import Embedder, Retriever, cosine_matrix
from ..validation import fraction, positive_int

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from ..evalset import Case, EvalSet
    from ..store import Row, Store
    from .base import CheckContext


@dataclass
class CaseResult:
    """Prediction and ordered retrieved evidence for one labeled query."""

    case: Case
    evidence: tuple[Row, ...]
    predicted: str
    trace: Mapping[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        """Return whether the predicted label equals the reference label.

        Returns:
            whether the predicted label equals the reference label.
        """
        return self.predicted == self.case.label

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return retrieved identifiers in evidence order.

        Returns:
            retrieved identifiers in evidence order.
        """
        return tuple(r.id for r in self.evidence)


@dataclass
class EvalRun:
    """Predictions and evidence recorded against one store version."""

    store_version: str
    results: list[CaseResult]

    @property
    def outcomes(self) -> list[Outcome]:
        """Return paired reference cases and predictions.

        Returns:
            paired reference cases and predictions.
        """
        return [(r.case, r.predicted) for r in self.results]

    def metric(self, fn: Callable[[Sequence[Outcome]], float] = accuracy) -> float:
        """Evaluate the requested metric on the recorded outcomes.

        Args:
            fn: Callable used to compute the result.

        Returns:
            Evaluate the requested metric on the recorded outcomes.
        """
        return fn(self.outcomes)

    def by_slice(
        self, fn: Callable[[Sequence[Outcome]], float] = accuracy
    ) -> dict[str, float]:
        """Summarize results separately for each represented slice.

        Args:
            fn: Callable used to compute the result.

        Returns:
            Summarize results separately for each represented slice.
        """
        groups: dict[str, list[Outcome]] = defaultdict(list)
        for r in self.results:
            for s in r.case.slices:
                groups[s].append((r.case, r.predicted))
        return {s: fn(o) for s, o in sorted(groups.items())}

    def retrieval_counts(self) -> dict[str, int]:
        """Count how many times each row appears in retrieved evidence.

        Returns:
            Count how many times each row appears in retrieved evidence.
        """
        c: dict[str, int] = defaultdict(int)
        for r in self.results:
            for row in r.evidence:
                c[row.id] += 1
        return dict(c)


@dataclass
class Transition:
    """A case prediction before and after a store change."""

    case: Case
    before: str
    after: str
    evidence_before: tuple[str, ...]
    evidence_after: tuple[str, ...]

    @property
    def kind(self) -> str:
        """Classify the before/after prediction transition.

        Returns:
            Classify the before/after prediction transition.
        """
        b, a = self.before == self.case.label, self.after == self.case.label
        if not b and a:
            return "wrong_to_right"
        if b and not a:
            return "right_to_wrong"
        if not b and not a and self.before != self.after:
            return "wrong_to_wrong"
        return "unchanged"


@dataclass
class Comparison:
    """Paired runs and prediction transitions caused by a store change."""

    baseline: EvalRun
    candidate: EvalRun
    context_changed: int
    decider_calls: int | None
    transitions: list[Transition] = field(default_factory=list)

    def count(self, kind: str) -> int:
        """Count transitions of the specified kind.

        Args:
            kind: Transition kind to count.

        Returns:
            Count transitions of the specified kind.
        """
        return sum(t.kind == kind for t in self.transitions)

    @property
    def wrong_to_right(self) -> list[Transition]:
        """Return cases corrected by the candidate.

        Returns:
            cases corrected by the candidate.
        """
        return [t for t in self.transitions if t.kind == "wrong_to_right"]

    @property
    def right_to_wrong(self) -> list[Transition]:
        """Return cases broken by the candidate.

        Returns:
            cases broken by the candidate.
        """
        return [t for t in self.transitions if t.kind == "right_to_wrong"]

    def by_slice(self) -> dict[str, dict[str, int]]:
        """Summarize results separately for each represented slice.

        Returns:
            Summarize results separately for each represented slice.
        """
        out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for t in self.transitions:
            for s in t.case.slices:
                out[s][t.kind] += 1
        return {s: dict(v) for s, v in out.items()}

    def summary(self) -> str:
        """Return a readable summary of the recorded results.

        Returns:
            a readable summary of the recorded results.
        """
        n = len(self.baseline.results)
        return (
            f"{self.baseline.store_version} -> {self.candidate.store_version}: "
            f"accuracy {self.baseline.metric():.3f} -> {self.candidate.metric():.3f}  "
            f"wrong->right={self.count('wrong_to_right')} "
            f"right->wrong={self.count('right_to_wrong')} "
            f"wrong->wrong={self.count('wrong_to_wrong')}  context "
            f"changed {self.context_changed}/{n}, decider calls "
            f"{self.decider_calls if self.decider_calls is not None else 'unobserved'}"
        )


class Evaluator:
    """Runs an EvalSet against a Store through a Retriever and Decider."""

    def __init__(self, retriever: Retriever, decider: Decider, evalset: EvalSet):
        """Configure Evaluator with the supplied inputs.

        Args:
            retriever: Retrieval implementation used for each query.
            decider: Deterministic decision implementation.
            evalset: Fixed labeled evaluation set.
        """
        self.retriever = retriever
        self.decider = (
            decider
            if isinstance(decider, MemoizedDecider)
            else MemoizedDecider(decider)
        )
        self.evalset = evalset

    def _retrieve(self, case: Case, store: Store) -> list[Row]:
        return self.retriever.retrieve(case.text, store, scope=case.scope)

    def run(self, store: Store, cases: Iterable[Case] | None = None) -> EvalRun:
        """Retrieve evidence and predict a label for each selected case.

        Args:
            store: Example store to inspect.
            cases: Labeled evaluation cases.

        Returns:
            Predictions and retrieved evidence in case order.
        """
        results = []
        for c in self.evalset if cases is None else cases:
            ev = tuple(self._retrieve(c, store))
            results.append(CaseResult(c, ev, self.decider.decide(c.text, ev)))
        return EvalRun(store.version, results)

    def compare(
        self, baseline: Store, candidate: Store, baseline_run: EvalRun | None = None
    ) -> Comparison:
        """Retrieval runs on both stores; the decider only where context differs.

        Args:
            baseline: Reference store or metric value.
            candidate: Proposed store or metric value.
            baseline_run: Optional previously computed reference run.

        Returns:
            Paired results, transitions, context changes, and additional decider calls.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        base = baseline_run if baseline_run is not None else self.run(baseline)
        if base.store_version != baseline.version:
            raise ValueError("baseline_run does not match the baseline store")
        calls_before = self.decider.calls
        results: list[CaseResult] = []
        changed = 0
        transitions: list[Transition] = []
        for br in base.results:
            ev = tuple(self._retrieve(br.case, candidate))
            if [r.to_dict() for r in ev] == [r.to_dict() for r in br.evidence]:
                results.append(CaseResult(br.case, ev, br.predicted))
                continue
            changed += 1
            pred = self.decider.decide(br.case.text, ev)
            results.append(CaseResult(br.case, ev, pred))
            t = Transition(
                br.case, br.predicted, pred, br.evidence_ids, tuple(r.id for r in ev)
            )
            if t.kind != "unchanged":
                transitions.append(t)
        cand = EvalRun(candidate.version, results)
        return Comparison(
            base, cand, changed, self.decider.calls - calls_before, transitions
        )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def judge(
    cmp: Comparison, objective: Objective, check_name: str = "regression"
) -> list[Finding]:
    """Turn a Comparison into findings against an Objective.

    Args:
        cmp: Paired evaluation results.
        objective: Metric and transition requirements.
        check_name: Name attached to emitted findings.

    Returns:
        Informational comparison details and any violated objective requirements.
    """
    out: list[Finding] = []
    b, c = (
        cmp.baseline.metric(objective.primary),
        cmp.candidate.metric(objective.primary),
    )
    n = len(cmp.baseline.results)
    data = {
        "baseline": b,
        "candidate": c,
        "transitions": {
            k: cmp.count(k)
            for k in ("wrong_to_right", "right_to_wrong", "wrong_to_wrong")
        },
        "by_slice": cmp.by_slice(),
    }
    out.append(Finding(check_name, Severity.INFO, cmp.summary(), data=data))

    if (
        not n
        or not np.isfinite(b)
        or not np.isfinite(c)
        or not objective.improved(b, c)
    ):
        out.append(
            Finding(
                check_name,
                Severity.ERROR,
                f"{objective.primary_name} {b:.3f} -> {c:.3f} fails "
                f"min delta {objective.min_primary_delta}",
                case_ids=tuple(t.case.id for t in cmp.right_to_wrong),
                data={
                    "unevaluable": not n or not bool(np.isfinite(b) and np.isfinite(c))
                },
            )
        )
    r2w = cmp.count("right_to_wrong")
    if objective.max_right_to_wrong is not None and r2w > objective.max_right_to_wrong:
        out.append(
            Finding(
                check_name,
                Severity.ERROR,
                f"{r2w} right->wrong transitions exceed cap "
                f"{objective.max_right_to_wrong}",
                case_ids=tuple(t.case.id for t in cmp.right_to_wrong),
            )
        )
    if (
        objective.max_right_to_wrong_frac is not None
        and n
        and r2w / n > objective.max_right_to_wrong_frac
    ):
        out.append(
            Finding(
                check_name,
                Severity.ERROR,
                f"right->wrong fraction {r2w / n:.3f} exceeds cap "
                f"{objective.max_right_to_wrong_frac}",
                case_ids=tuple(t.case.id for t in cmp.right_to_wrong),
            )
        )
    cand_slices = cmp.candidate.by_slice(objective.primary)
    for s, floor in objective.min_slice_metric.items():
        v = cand_slices.get(s)
        if v is None:
            out.append(
                Finding(
                    check_name,
                    Severity.ERROR,
                    f"slice {s!r} has no cases; floor {floor} unchecked",
                    data={"unevaluable": True},
                )
            )
        elif not np.isfinite(v) or v < floor:
            out.append(
                Finding(
                    check_name,
                    Severity.ERROR,
                    f"slice {s!r} {objective.primary_name} {v:.3f} below floor {floor}",
                    data={
                        "slice": s,
                        "value": v,
                        "unevaluable": not bool(np.isfinite(v)),
                    },
                )
            )
    for name, ceiling in objective.max_slice_metric.items():
        value = cand_slices.get(name)
        unavailable = value is None or not np.isfinite(value)
        if value is None or not np.isfinite(value) or value > ceiling:
            out.append(
                Finding(
                    check_name,
                    Severity.ERROR,
                    f"slice {name!r} {objective.primary_name} {value} exceeds ceiling "
                    f"{ceiling} or cannot be evaluated",
                    data={
                        "slice": name,
                        "value": value,
                        "unevaluable": bool(unavailable),
                    },
                )
            )
    for name, minimum in objective.min_slice_cases.items():
        count = sum(name in r.case.slices for r in cmp.candidate.results)
        if count < minimum:
            out.append(
                Finding(
                    check_name,
                    Severity.ERROR,
                    f"slice {name!r} has {count} cases; requires at least {minimum}",
                    data={"slice": name, "n": count, "unevaluable": True},
                )
            )
    cost = sum(t.case.cost for t in cmp.right_to_wrong)
    if (
        objective.max_right_to_wrong_cost is not None
        and cost > objective.max_right_to_wrong_cost
    ):
        out.append(
            Finding(
                check_name,
                Severity.ERROR,
                f"new error cost {cost} exceeds cap "
                f"{objective.max_right_to_wrong_cost}",
                case_ids=tuple(t.case.id for t in cmp.right_to_wrong),
                data={"new_error_cost": cost},
            )
        )
    return out


@dataclass
class RowInfluence:
    """Observed effect of removing one row from a fixed store."""

    row: Row
    comparison: Comparison  # baseline = with row, candidate = without row

    @property
    def net(self) -> int:
        """Return cases fixed minus cases broken by removing the row.

        Positive values indicate observed net harm on these cases.

        Returns:
            Cases fixed minus cases broken when the row is removed.
        """
        return self.comparison.count("wrong_to_right") - self.comparison.count(
            "right_to_wrong"
        )

    @property
    def helps(self) -> int:
        """Return cases broken when this row is removed.

        Returns:
            cases broken when this row is removed.
        """
        return self.comparison.count("right_to_wrong")

    @property
    def hurts(self) -> int:
        """Return cases corrected when this row is removed.

        Returns:
            cases corrected when this row is removed.
        """
        return self.comparison.count("wrong_to_right")

    @property
    def retrieved_for(self) -> int:
        """Return the number of cases whose context changed after removal.

        Returns:
            the number of cases whose context changed after removal.
        """
        return self.comparison.context_changed


def row_influence(
    evaluator: Evaluator,
    store: Store,
    row_ids: Sequence[str] | None = None,
    baseline_run: EvalRun | None = None,
) -> list[RowInfluence]:
    """Leave-one-out influence of each row on the eval set.

    Cost: one retrieval pass per row, plus decider calls only on cases whose
    evidence changed. Rows never retrieved are skipped only when the
    retriever declares deletion_stable=True: removing an unretrieved row
    cannot change its ranked results. Other retrievers inspect every row.

    Args:
        evaluator: Configured retrieval and decision evaluator.
        store: Example store to inspect.
        row_ids: Row identifiers to inspect or remove.
        baseline_run: Optional previously computed reference run.

    Returns:
        Row-removal comparisons sorted by decreasing net cases fixed.

    Raises:
        ValueError: If the supplied baseline run belongs to another store.
    """
    base = baseline_run or evaluator.run(store)
    if base.store_version != store.version:
        raise ValueError("baseline_run does not match the baseline store")
    counts = base.retrieval_counts()
    deletion_stable = getattr(evaluator.retriever, "deletion_stable", False)
    ids = (
        list(row_ids)
        if row_ids is not None
        else [r.id for r in store if not deletion_stable or counts.get(r.id, 0) > 0]
    )
    out: list[RowInfluence] = []
    for rid in ids:
        cmp = evaluator.compare(store, store.remove(rid), baseline_run=base)
        out.append(RowInfluence(store[rid], cmp))
    out.sort(key=lambda x: -x.net)
    return out


class RowInfluenceCheck:
    """Flags rows whose removal fixes more cases than it breaks."""

    name = "row_influence"
    tier = "dynamic"

    def __init__(
        self,
        evaluator: Evaluator,
        min_net_harm: int = 1,
        row_ids: Sequence[str] | None = None,
    ):
        """Configure RowInfluenceCheck with the supplied inputs.

        Args:
            evaluator: Configured retrieval and decision evaluator.
            min_net_harm: Minimum net cases fixed by removing a row.
            row_ids: Row identifiers to inspect or remove.
        """
        positive_int("min_net_harm", min_net_harm)
        self.evaluator, self.min_net_harm, self.row_ids = (
            evaluator,
            min_net_harm,
            row_ids,
        )

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Flag rows whose removal fixes more cases than it breaks.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        infl = row_influence(self.evaluator, ctx.store, self.row_ids)
        out: list[Finding] = []
        for i in infl:
            if i.net >= self.min_net_harm:
                sev = Severity.ERROR if i.helps == 0 else Severity.WARN
                hint = (
                    "test removal on held-out cases"
                    if i.helps == 0
                    else "edit or add a competing example (row also helps some cases)"
                )
                out.append(
                    Finding(
                        self.name,
                        sev,
                        f"removing row fixes {i.hurts}, breaks "
                        f"{i.helps} (net {i.net}); retrieved for "
                        f"{i.retrieved_for} cases; suggest: {hint}",
                        (i.row.id,),
                        case_ids=tuple(t.case.id for t in i.comparison.transitions),
                        data={
                            "net": i.net,
                            "fixes": i.hurts,
                            "breaks": i.helps,
                            "retrieved_for": i.retrieved_for,
                        },
                    )
                )
        out.append(
            Finding(
                self.name,
                Severity.INFO,
                f"influence computed for {len(infl)} rows; decider "
                f"calls so far {self.evaluator.decider.calls}",
            )
        )
        return out


class RegressionCheck:
    """Judge a candidate against a baseline using fixed labeled cases."""

    name = "regression"
    tier = "dynamic"

    def __init__(
        self, evaluator: Evaluator, baseline: Store, objective: Objective | None = None
    ):
        """Configure RegressionCheck with the supplied inputs.

        Args:
            evaluator: Configured retrieval and decision evaluator.
            baseline: Reference store or metric value.
            objective: Metric and transition requirements.
        """
        self.evaluator, self.baseline, self.objective = (
            evaluator,
            baseline,
            objective or Objective(),
        )

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Compare the candidate with the baseline and enforce the objective.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        cmp = self.evaluator.compare(self.baseline, ctx.store)
        return judge(cmp, self.objective, self.name)


class CoverageCheck:
    """Empty neighborhoods: traffic queries whose nearest store example is far.

    Coverage is measured against traffic, the way code coverage is measured
    against execution, so it needs a sample of production inputs (unlabeled
    is fine). Reports the fraction uncovered overall and per scope.
    """

    name = "coverage"
    tier = "dynamic"

    def __init__(
        self,
        traffic: Sequence[str],
        embedder: Embedder,
        min_similarity: float = 0.3,
        max_uncovered_frac: float = 0.1,
        scopes: Sequence[str | None] | None = None,
    ):
        """Configure CoverageCheck with the supplied inputs.

        Args:
            traffic: Queries representing production traffic.
            embedder: Text embedding implementation.
            min_similarity: Minimum cosine similarity counted as coverage.
            max_uncovered_frac: Largest permitted fraction of uncovered queries.
            scopes: Optional scope for each traffic query.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        if not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be finite and in [-1, 1]")
        fraction("max_uncovered_frac", max_uncovered_frac)
        if scopes is not None and len(scopes) != len(traffic):
            raise ValueError("scopes must contain one scope per traffic query")
        (
            self.traffic,
            self.embedder,
            self.min_similarity,
            self.max_uncovered_frac,
            self.scopes,
        ) = list(traffic), embedder, min_similarity, max_uncovered_frac, scopes

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Measure traffic queries lacking a close example in their scope.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        if not self.traffic:
            return [
                Finding(self.name, Severity.ERROR, "no traffic; coverage unchecked")
            ]
        store = ctx.store.embed(self.embedder)
        queries = self.embedder.embed(self.traffic)
        uncovered = []
        by_scope: dict[str | None, list[bool]] = defaultdict(list)
        for index, query in enumerate(self.traffic):
            scope = self.scopes[index] if self.scopes is not None else None
            candidates = store.scope(scope) if self.scopes is not None else store
            best = (
                float(
                    cosine_matrix(
                        queries[index : index + 1], candidates.embeddings()
                    ).max()
                )
                if len(candidates)
                else -float("inf")
            )
            missing = best < self.min_similarity
            by_scope[scope].append(missing)
            if missing:
                uncovered.append(query)
        frac = len(uncovered) / len(self.traffic)
        severity = Severity.ERROR if frac > self.max_uncovered_frac else Severity.INFO
        return [
            Finding(
                self.name,
                severity,
                f"{frac:.1%} of {len(self.traffic)} traffic queries "
                "lack a near neighbor",
                data={
                    "uncovered_frac": frac,
                    "uncovered_examples": uncovered[:10],
                    "by_scope": [
                        {
                            "scope": scope,
                            "queries": len(values),
                            "uncovered_frac": sum(values) / len(values),
                        }
                        for scope, values in by_scope.items()
                    ],
                },
            )
        ]


class DeadRowCheck:
    """Rows never retrieved for any eval case or traffic query."""

    name = "dead_rows"
    tier = "dynamic"

    def __init__(self, evaluator: Evaluator, traffic: Sequence[str] = ()):
        """Configure DeadRowCheck with the supplied inputs.

        Args:
            evaluator: Configured retrieval and decision evaluator.
            traffic: Queries representing production traffic.
        """
        self.evaluator, self.traffic = evaluator, list(traffic)

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Find rows absent from evaluation and traffic retrieval results.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        seen = {
            row.id
            for case in self.evaluator.evalset
            for row in self.evaluator.retriever.retrieve(
                case.text, ctx.store, scope=case.scope
            )
        }
        for q in self.traffic:
            for r in self.evaluator.retriever.retrieve(q, ctx.store):
                seen.add(r.id)
        dead = [r.id for r in ctx.store if r.id not in seen]
        if not dead:
            return [
                Finding(
                    self.name, Severity.INFO, "every row was retrieved at least once"
                )
            ]
        return [
            Finding(
                self.name,
                Severity.WARN,
                f"{len(dead)} of {len(ctx.store)} rows never "
                f"retrieved for eval cases or traffic",
                tuple(dead),
            )
        ]


class DecisionLayerCheck:
    """Cases where the evidence was right and the decision was wrong.

    This is a triage heuristic: a majority of gold-labeled examples does
    not establish evidence quality or isolate the cause of an error.
    """

    name = "decision_layer"
    tier = "dynamic"

    def __init__(self, evaluator: Evaluator, baseline_run: EvalRun | None = None):
        """Configure DecisionLayerCheck with the supplied inputs.

        Args:
            evaluator: Configured retrieval and decision evaluator.
            baseline_run: Optional previously computed reference run.
        """
        self.evaluator, self.baseline_run = evaluator, baseline_run

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Flag prediction errors despite majority-gold retrieved labels.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.

        Raises:
            ValueError: If the supplied baseline run belongs to another store.
        """
        run = self.baseline_run or self.evaluator.run(ctx.store)
        if run.store_version != ctx.store.version:
            raise ValueError("baseline_run does not match the baseline store")
        bad = []
        for r in run.results:
            if r.correct or not r.evidence:
                continue
            share = sum(e.label == r.case.label for e in r.evidence) / len(r.evidence)
            if share > 0.5:
                bad.append((r, share))
        wrong = [r for r in run.results if not r.correct]
        if not wrong:
            return [Finding(self.name, Severity.INFO, "no errors on eval set")]
        frac = len(bad) / len(wrong)
        return [
            Finding(
                self.name,
                Severity.WARN if bad else Severity.INFO,
                f"{len(bad)} of {len(wrong)} errors had "
                "majority-correct evidence; inspect the decision rule "
                "and evidence quality",
                case_ids=tuple(r.case.id for r, _ in bad),
                data={"majority_correct_evidence_frac_of_errors": frac},
            )
        ]
