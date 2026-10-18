"""Meta tier: mutation testing for the evaluator.

Perturb a copy of a store and compare the observed effect with a declared
expectation. Default expectations are diagnostic hypotheses, not guarantees
that a particular perturbation damages every classifier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..report import Finding, Severity
from ..store import Row, Store
from ..validation import positive_int

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .base import CheckContext
    from .dynamic import Comparison, Evaluator


@dataclass
class Mutation:
    """A store perturbation with a caller-specified expected signal."""

    name: str
    apply: Callable[[Store, Evaluator], Store]
    expect: str  # "degrade" | "change" | "no_change"


def _most_retrieved(store: Store, evaluator: Evaluator) -> Row | None:
    counts = evaluator.run(store).retrieval_counts()
    if not counts:
        return None
    return store[max(counts, key=counts.__getitem__)]


def flip_influential_label(store: Store, evaluator: Evaluator) -> Store:
    """Relabel the most-retrieved row to a different label present in the store.

    Args:
        store: Example store to inspect.
        evaluator: Configured retrieval and decision evaluator.

    Returns:
        A new store with one influential label changed, or the original if inapplicable.
    """
    r = _most_retrieved(store, evaluator)
    if r is None:
        return store
    others = sorted({x.label for x in store} - {r.label})
    if not others:
        return store
    return store.edit(r.id, label=others[0])


def inject_redundancy(store: Store, evaluator: Evaluator, copies: int = 5) -> Store:
    """Duplicate the most-retrieved row several times (crowds context).

    Args:
        store: Example store to inspect.
        evaluator: Configured retrieval and decision evaluator.
        copies: Number of redundant copies to inject.

    Returns:
        A store with copies of the most-retrieved row, or the original if empty.
    """
    positive_int("copies", copies, allow_zero=True)
    r = _most_retrieved(store, evaluator)
    if r is None:
        return store
    dupes = [
        r.with_(id=f"{r.id}-dup{i}", text=r.text + " " * (i + 1)) for i in range(copies)
    ]
    return store.add(*dupes)


def remove_label_region(store: Store, evaluator: Evaluator) -> Store:  # noqa: ARG001
    """Drop every example of the most common label (an uncovered region).

    Args:
        store: Example store to inspect.
        evaluator: Configured retrieval and decision evaluator.

    Returns:
        A store without examples of its most frequent label.
    """
    counts: dict[str, int] = {}
    for r in store:
        counts[r.label] = counts.get(r.label, 0) + 1
    if not counts:
        return store
    target = max(counts, key=counts.__getitem__)
    return store.filter(lambda r: r.label != target)


def shuffle_labels(store: Store, evaluator: Evaluator, seed: int = 0) -> Store:  # noqa: ARG001
    """Permute store labels using a reproducible random seed.

    Args:
        store: Example store to inspect.
        evaluator: Configured retrieval and decision evaluator.
        seed: Seed for reproducible pseudorandom sampling.

    Returns:
        Permute store labels using a reproducible random seed.
    """
    rng = random.Random(seed)  # noqa: S311 - reproducible sampling, not security
    labels = [r.label for r in store]
    rng.shuffle(labels)
    return Store(
        [r.with_(label=item) for r, item in zip(store, labels, strict=False)],
        name=store.name,
    )


DEFAULT_MUTATIONS = [
    Mutation("flip_influential_label", flip_influential_label, "degrade"),
    Mutation("inject_redundancy", inject_redundancy, "change"),
    Mutation("remove_label_region", remove_label_region, "degrade"),
    Mutation("shuffle_labels", shuffle_labels, "degrade"),
]


class MutationCheck:
    """Each mutation must produce the expected signal in the Comparison."""

    name = "mutation"
    tier = "meta"

    def __init__(
        self,
        evaluator: Evaluator,
        mutations: Sequence[Mutation] = DEFAULT_MUTATIONS,
        min_transitions: int = 1,
    ):
        """Configure MutationCheck with the supplied inputs.

        Args:
            evaluator: Configured retrieval and decision evaluator.
            mutations: Perturbations and their expected signals.
            min_transitions: Minimum observed changes required by a mutation.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        positive_int("min_transitions", min_transitions)
        if any(m.expect not in {"degrade", "change", "no_change"} for m in mutations):
            raise ValueError("unknown mutation expectation")
        self.evaluator, self.mutations, self.min_transitions = (
            evaluator,
            list(mutations),
            min_transitions,
        )

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Compare each store mutation with its declared expected signal.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings with affected identifiers and diagnostic values.
        """
        out: list[Finding] = []
        base_run = self.evaluator.run(ctx.store)
        for m in self.mutations:
            mutated = m.apply(ctx.store, self.evaluator)
            if mutated.version == ctx.store.version:
                out.append(
                    Finding(
                        self.name,
                        Severity.WARN,
                        f"{m.name}: mutation produced no change in "
                        f"store (nothing to test)",
                    )
                )
                continue
            cmp: Comparison = self.evaluator.compare(
                ctx.store, mutated, baseline_run=base_run
            )
            n_trans = len(cmp.transitions)
            b, c = cmp.baseline.metric(), cmp.candidate.metric()
            data = {
                "baseline": b,
                "mutated": c,
                "transitions": n_trans,
                "context_changed": cmp.context_changed,
            }
            # "degrade" is judged on right->wrong transitions, never on the aggregate:
            # a flipped label can break one case and accidentally fix another, leaving
            # accuracy unchanged. The transition report is what must catch it.
            if (
                m.expect == "degrade"
                and cmp.count("right_to_wrong") < self.min_transitions
            ):
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"{m.name}: expected degradation was not observed "
                        f" ({b:.3f} -> {c:.3f}, "
                        f"right->wrong={cmp.count('right_to_wrong')})",
                        data=data,
                    )
                )
            elif m.expect == "change" and cmp.context_changed < self.min_transitions:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"{m.name}: evaluator saw no context change",
                        data=data,
                    )
                )
            elif m.expect == "no_change" and n_trans > 0:
                out.append(
                    Finding(
                        self.name,
                        Severity.ERROR,
                        f"{m.name}: expected no transitions, saw {n_trans}",
                        data=data,
                    )
                )
            else:
                out.append(
                    Finding(
                        self.name,
                        Severity.INFO,
                        f"{m.name}: detected ({b:.3f} -> {c:.3f}, "
                        f"transitions={n_trans})",
                        data=data,
                    )
                )
        return out
