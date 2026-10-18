"""Static tier: diagnostics that do not call the decider.

These are the lint layer. They catch rows that are malformed, out of
type (label missing or redefined), redundant, stale, leaky, or sitting
in a neighborhood that disagrees with their label. None of them can say
whether a row helps decisions; that is the dynamic tier's job.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..report import Finding, Severity
from ..retriever import cosine_matrix
from ..validation import fraction, positive_int

if TYPE_CHECKING:
    from .base import CheckContext


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class SchemaCheck:
    """Required fields present, text within length bounds, label well formed."""

    name = "schema"
    tier = "static"

    def __init__(
        self,
        min_chars: int = 3,
        max_chars: int = 4000,
        require_scope: bool = False,
        require_author: bool = False,
    ):
        """Configure SchemaCheck with the supplied inputs.

        Args:
            min_chars: Minimum example length in characters.
            max_chars: Maximum example length in characters.
            require_scope: Whether every row must declare a scope.
            require_author: Whether missing authors should be flagged.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        positive_int("min_chars", min_chars, allow_zero=True)
        positive_int("max_chars", max_chars)
        if min_chars > max_chars:
            raise ValueError("min_chars must not exceed max_chars")
        self.min_chars, self.max_chars = min_chars, max_chars
        self.require_scope, self.require_author = require_scope, require_author

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Check text length, label presence, and required provenance fields.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        out: list[Finding] = []
        for r in ctx.store:
            n = len(r.text.strip())
            if n == 0:
                out.append(Finding(self.name, Severity.ERROR, "empty text", (r.id,)))
            elif n < self.min_chars:
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        f"text shorter than {self.min_chars} chars ({n})",
                        (r.id,),
                    )
                )
            elif n > self.max_chars:
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        f"text longer than {self.max_chars} chars "
                        f"({n}); will crowd a fixed context",
                        (r.id,),
                    )
                )
            if not r.label or not r.label.strip():
                out.append(Finding(self.name, Severity.ERROR, "empty label", (r.id,)))
            if self.require_scope and r.scope is None:
                out.append(
                    Finding(
                        self.name, Severity.ERROR, "scope required but missing", (r.id,)
                    )
                )
            if self.require_author and not r.author:
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        "no author (blame unavailable)",
                        (r.id,),
                    )
                )
        return out


class TaxonomyValidityCheck:
    """Check label existence, deprecation, and definition versions."""

    name = "taxonomy"
    tier = "static"

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Check unknown, deprecated, and outdated label definitions.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        if ctx.taxonomy is None:
            return [
                Finding(
                    self.name,
                    Severity.WARN,
                    "no taxonomy supplied; label validity unchecked",
                )
            ]
        tax = ctx.taxonomy
        out: list[Finding] = []
        orphans_by_label: dict[str, list[str]] = defaultdict(list)
        for r in ctx.store:
            if r.label not in tax:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"label {r.label!r} not in taxonomy",
                        (r.id,),
                    )
                )
                continue
            if tax[r.label].deprecated:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"label {r.label!r} is deprecated",
                        (r.id,),
                    )
                )
            cur = tax.label_version(r.label)
            if r.taxonomy_version is None:
                out.append(
                    Finding(
                        self.name,
                        Severity.INFO,
                        "row records no taxonomy version; cannot detect orphaning",
                        (r.id,),
                    )
                )
            elif r.taxonomy_version != cur:
                orphans_by_label[r.label].append(r.id)
        for label, ids in orphans_by_label.items():
            out.append(
                Finding(
                    self.name,
                    Severity.WARN,
                    f"{len(ids)} row(s) labeled {label!r} were "
                    f"written under an older definition of the "
                    f"label; re-review",
                    tuple(ids),
                    data={"label": label, "current_version": tax.label_version(label)},
                )
            )
        return out


class ExactDuplicateCheck:
    """Same normalized text appearing more than once.

    Same label: redundancy (WARN); redundant rows crowd better evidence out
    of a fixed context. Different labels: contradiction (ERROR); the store
    is telling the decider two things about one input.
    """

    name = "exact_duplicates"
    tier = "static"

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Group normalized identical texts and flag label conflicts.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        groups: dict[str, list] = defaultdict(list)
        for r in ctx.store:
            groups[_norm(r.text)].append(r)
        out: list[Finding] = []
        for rows in groups.values():
            if len(rows) < 2:
                continue
            labels = {r.label for r in rows}
            ids = tuple(r.id for r in rows)
            if len(labels) > 1:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"identical text with conflicting labels {sorted(labels)}",
                        ids,
                        data={"labels": sorted(labels)},
                    )
                )
            else:
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        f"{len(rows)} identical rows with label {rows[0].label!r}",
                        ids,
                    )
                )
        return out


class NearDuplicateCheck:
    """Pairs above a cosine threshold.

    Conflicting labels across a near-duplicate pair are the cheapest signal of a
    mislabel or a taxonomy distinction the
    representation cannot see.
    """

    name = "near_duplicates"
    tier = "static"

    def __init__(
        self,
        threshold: float = 0.95,
        conflict_threshold: float | None = None,
        max_pairs: int = 200,
    ):
        """Configure NearDuplicateCheck with the supplied inputs.

        Args:
            threshold: Cosine threshold for same-label redundancy.
            conflict_threshold: Cosine threshold for conflicting-label pairs.
            max_pairs: Maximum number of pairs inspected.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        for name, value in (
            ("threshold", threshold),
            ("conflict_threshold", conflict_threshold),
        ):
            if value is not None and (not np.isfinite(value) or not -1 <= value <= 1):
                raise ValueError(f"{name} must be finite and in [-1, 1]")
        positive_int("max_pairs", max_pairs)
        self.threshold = threshold
        self.conflict_threshold = (
            conflict_threshold
            if conflict_threshold is not None
            else max(-1.0, threshold - 0.1)
        )
        self.max_pairs = max_pairs

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Inspect the highest-similarity pairs for redundancy or conflict.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        store = ctx.store
        if len(store) < 2:
            return []
        if ctx.embedder is not None:
            store = store.embed(ctx.embedder)
        try:
            embeddings = store.embeddings()
        except ValueError as e:
            return [Finding(self.name, Severity.WARN, f"skipped: {e}")]
        similarities = cosine_matrix(embeddings, embeddings)
        np.fill_diagonal(similarities, -1.0)
        rows = store.rows
        iu = np.triu_indices(len(rows), k=1)
        sims = similarities[iu]
        order = np.argsort(-sims)
        out: list[Finding] = []
        n_same = 0
        for idx in order[: self.max_pairs]:
            s = float(sims[idx])
            if s < min(self.threshold, self.conflict_threshold):
                break
            i, j = int(iu[0][idx]), int(iu[1][idx])
            a, b = rows[i], rows[j]
            if a.label != b.label and s >= self.conflict_threshold:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"near-duplicates (cos={s:.3f}) carry "
                        f"different labels {a.label!r} vs "
                        f"{b.label!r}",
                        (a.id, b.id),
                        data={"cosine": s},
                    )
                )
            elif a.label == b.label and s >= self.threshold:
                n_same += 1
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        f"near-duplicates (cos={s:.3f}) with same "
                        f"label; likely redundant",
                        (a.id, b.id),
                        data={"cosine": s},
                    )
                )
        return out


class StalenessCheck:
    """Rows older than a TTL.

    Age is a hypothesis generator: old rows may describe products, policies, or
        phrasings that traffic no longer uses.
    """

    name = "staleness"
    tier = "static"

    def __init__(
        self, ttl: timedelta = timedelta(days=365), now: datetime | None = None
    ):
        """Configure StalenessCheck with the supplied inputs.

        Args:
            ttl: Maximum permitted example age.
            now: Reference time; None uses the current UTC time.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        if ttl.total_seconds() < 0:
            raise ValueError("ttl must be nonnegative")
        if now is not None and now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        self.ttl = ttl
        self.now = now

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Find examples older than the configured age limit.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        now = self.now or datetime.now(UTC)
        known = [(r, r.created_at) for r in ctx.store if r.created_at is not None]
        stale = [(row, created) for row, created in known if now - created > self.ttl]
        unknown = tuple(r.id for r in ctx.store if r.created_at is None)
        out = []
        if unknown:
            out.append(
                Finding(
                    self.name,
                    Severity.WARN,
                    f"{len(unknown)} row(s) have unknown creation time; age unchecked",
                    row_ids=unknown,
                )
            )
        if stale:
            out.append(
                Finding(
                    self.name,
                    Severity.WARN,
                    f"{len(stale)} row(s) older than {self.ttl.days} days",
                    tuple(row.id for row, _ in stale),
                    data={"oldest": min(created for _, created in stale).isoformat()},
                )
            )
        return out


class PIICheck:
    """Patterns that should never sit in an example store.

    Extend `patterns` for domain identifiers (order numbers, member ids).
    """

    name = "pii"
    tier = "static"

    default_patterns: ClassVar[dict[str, re.Pattern[str]]] = {
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "phone": re.compile(
            r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
        ),
        "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    }

    def __init__(self, patterns: dict[str, re.Pattern] | None = None):
        """Configure PIICheck with the supplied inputs.

        Args:
            patterns: Named regular expressions for identifier detection.
        """
        self.patterns = self.default_patterns if patterns is None else patterns

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Find rows matching configured identifier patterns.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        out: list[Finding] = []
        for r in ctx.store:
            hits = [k for k, p in self.patterns.items() if p.search(r.text)]
            if hits:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"possible PII: {', '.join(hits)}",
                        (r.id,),
                        data={"kinds": hits},
                    )
                )
        return out


class NeighborhoodPurityCheck:
    """Measure the fraction of neighbors sharing each row label.

    Low purity means one of three things: the row is mislabeled, it
    sits on a real boundary (keep it; it is doing work), or the embedding
    collapses a distinction the taxonomy draws. The check cannot tell these
    apart; the dynamic tier's row-influence check can.
    """

    name = "neighborhood_purity"
    tier = "static"

    def __init__(self, k: int = 5, min_purity: float = 0.4):
        """Configure NeighborhoodPurityCheck with the supplied inputs.

        Args:
            k: Maximum number of neighbors.
            min_purity: Minimum fraction of neighbors sharing a row label.
        """
        positive_int("k", k)
        fraction("min_purity", min_purity)
        self.k, self.min_purity = k, min_purity

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Measure each row's same-label fraction among nearest neighbors.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        store = ctx.store
        if len(store) <= self.k:
            return []
        if ctx.embedder is not None:
            store = store.embed(ctx.embedder)
        try:
            embeddings = store.embeddings()
        except ValueError as e:
            return [Finding(self.name, Severity.WARN, f"skipped: {e}")]
        similarities = cosine_matrix(embeddings, embeddings)
        np.fill_diagonal(similarities, -np.inf)
        rows = store.rows
        labels = np.array([r.label for r in rows])
        nn = np.argsort(-similarities, axis=1)[:, : self.k]
        out: list[Finding] = []
        purities = []
        for i, r in enumerate(rows):
            p = float(np.mean(labels[nn[i]] == r.label))
            purities.append(p)
            if p < self.min_purity:
                neigh = [rows[j].id for j in nn[i]]
                neigh_labels = sorted(set(labels[nn[i]].tolist()))
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        f"purity {p:.2f} < {self.min_purity}: "
                        f"neighbors labeled {neigh_labels}",
                        (r.id,),
                        data={
                            "purity": p,
                            "neighbors": neigh,
                            "neighbor_labels": neigh_labels,
                        },
                    )
                )
        out.append(
            Finding(
                self.name,
                Severity.INFO,
                f"mean neighborhood purity {np.mean(purities):.3f} "
                f"over {len(rows)} rows",
                data={"mean_purity": float(np.mean(purities))},
            )
        )
        return out


class ScopeBalanceCheck:
    """Report rows per scope and per label.

    Flag labels with fewer than `min_per_label` examples. Class balance is a hypothesis
    generator, so this is INFO/WARN.
    """

    name = "balance"
    tier = "static"

    def __init__(self, min_per_label: int = 3):
        """Configure ScopeBalanceCheck with the supplied inputs.

        Args:
            min_per_label: Minimum expected number of examples per label.
        """
        positive_int("min_per_label", min_per_label, allow_zero=True)
        self.min_per_label = min_per_label

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Report scope and label counts and flag underrepresented labels.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        by_label: dict[str, int] = defaultdict(int)
        by_scope: dict[str | None, int] = defaultdict(int)
        for r in ctx.store:
            by_label[r.label] += 1
            by_scope[r.scope] += 1
        out = [
            Finding(
                self.name,
                Severity.INFO,
                "rows per label / per scope",
                data={
                    "by_label": dict(by_label),
                    "by_scope": {str(k): v for k, v in by_scope.items()},
                },
            )
        ]
        thin = {item: n for item, n in by_label.items() if n < self.min_per_label}
        if ctx.taxonomy is not None:
            for item in ctx.taxonomy.active_ids:
                if item not in by_label:
                    thin[item] = 0
        for item, n in sorted(thin.items()):
            out.append(
                Finding(
                    self.name,
                    Severity.WARN,
                    f"label {item!r} has {n} example(s) (< {self.min_per_label})",
                    data={"label": item, "count": n},
                )
            )
        return out


DEFAULT_STATIC_CHECKS = [
    SchemaCheck(),
    TaxonomyValidityCheck(),
    ExactDuplicateCheck(),
    NearDuplicateCheck(),
    StalenessCheck(),
    PIICheck(),
    NeighborhoodPurityCheck(),
    ScopeBalanceCheck(),
]
