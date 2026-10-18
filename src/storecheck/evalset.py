"""EvalSet: the cases that define the decision problem.

Three kinds of case, per the operating post: representative labeled
traffic (overall performance), tagged slices (where aggregates hide
regressions), and challenge cases (boundaries and high-cost errors).
Cases used to discover a change should be split from those used to
validate it; `split()` supports that.
"""

from __future__ import annotations

import json
import math
import random
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .store import _freeze, _thaw
from .validation import fraction

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Case:
    """A labeled evaluation query with slices and error cost."""

    id: str
    text: str
    label: str
    slices: frozenset[str] = frozenset()
    challenge: bool = False
    cost: float = 1.0  # weight of getting this case wrong
    scope: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Validate case fields and detach mutable metadata.

        Raises:
            ValueError: If fields or error cost are invalid.
        """
        if any(
            not isinstance(v, str) or not v for v in (self.id, self.text, self.label)
        ):
            raise ValueError("case id, text, and label must be nonempty strings")
        if isinstance(self.cost, bool) or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("cost must be finite and nonnegative")
        if not isinstance(self.challenge, bool):
            raise ValueError("challenge must be a boolean")
        if self.scope is not None and not isinstance(self.scope, str):
            raise ValueError("scope must be a string or None")
        if isinstance(self.slices, str) or any(
            not isinstance(v, str) for v in self.slices
        ):
            raise ValueError("slices must be a collection of strings")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "slices", frozenset(self.slices))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record.

        Returns:
            a JSON-compatible record.
        """
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "slices": sorted(self.slices),
            "challenge": self.challenge,
            "cost": self.cost,
            "scope": self.scope,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        """Load a record from its serialized fields.

        Args:
            d: Serialized record fields.

        Returns:
            Load a record from its serialized fields.
        """
        return cls(
            id=d["id"],
            text=d["text"],
            label=d["label"],
            slices=d.get("slices", ()),
            challenge=d.get("challenge", False),
            cost=d.get("cost", 1.0),
            scope=d.get("scope"),
            metadata=d.get("metadata", {}),
        )


class EvalSet:
    """An ordered collection of uniquely identified evaluation cases."""

    def __init__(self, cases: list[Case], name: str = "evalset"):
        """Configure EvalSet with the supplied inputs.

        Args:
            cases: Labeled evaluation cases.
            name: Human-readable collection or slice name.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        ids = [c.id for c in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate case ids")
        self.cases = tuple(cases)
        self.name = name

    def __len__(self) -> int:
        """Return the number of records.

        Returns:
            the number of records.
        """
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        """Iterate over records in insertion order.

        Returns:
            Iterate over records in insertion order.
        """
        return iter(self.cases)

    @property
    def slices(self) -> list[str]:
        """Return the sorted slice names represented by evaluation cases.

        Returns:
            the sorted slice names represented by evaluation cases.
        """
        return sorted({s for c in self.cases for s in c.slices})

    def slice(self, name: str) -> EvalSet:
        """Return cases belonging to the named slice.

        Args:
            name: Human-readable collection or slice name.

        Returns:
            cases belonging to the named slice.
        """
        return EvalSet(
            [c for c in self.cases if name in c.slices], name=f"{self.name}[{name}]"
        )

    @property
    def challenge(self) -> EvalSet:
        """Return the challenge cases.

        Returns:
            the challenge cases.
        """
        return EvalSet(
            [c for c in self.cases if c.challenge], name=f"{self.name}[challenge]"
        )

    def split(
        self, dev_frac: float = 0.5, seed: int = 0, *, group_key: str | None = None
    ) -> tuple[EvalSet, EvalSet]:
        """Split independent groups into discovery and validation cases.

        Equal text after Unicode, case, and whitespace normalization stays
        together. An optional metadata key also joins related cases, including
        transitive links through duplicate text. The fraction applies to groups,
        so case counts can differ substantially from the requested proportion.

        Args:
            dev_frac: Fraction of independent groups assigned to discovery.
            seed: Seed for reproducible pseudorandom sampling.
            group_key: Optional metadata field identifying independent groups.

        Returns:
            Discovery and holdout sets retaining original case order. With at
            least two groups and an interior fraction, both sets are nonempty.

        Raises:
            ValueError: If group_key is not a nonempty string when supplied.
        """
        if group_key is not None and (not isinstance(group_key, str) or not group_key):
            raise ValueError("group_key must be a nonempty string")
        fraction("dev_frac", dev_frac)
        parents = list(range(len(self.cases)))

        def root(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        seen: dict[tuple[str, str], int] = {}
        for i, case in enumerate(self.cases):
            for key in _leakage_keys(case, group_key):
                if key in seen:
                    parents[root(i)] = root(seen[key])
                else:
                    seen[key] = i
        groups = list(dict.fromkeys(root(i) for i in range(len(self.cases))))
        rng = random.Random(seed)  # noqa: S311 - reproducible sampling
        rng.shuffle(groups)
        n = int(len(groups) * dev_frac)
        if len(groups) > 1 and 0 < dev_frac < 1:
            n = max(1, min(n, len(groups) - 1))
        selected = set(groups[:n])
        return (
            EvalSet(
                [c for i, c in enumerate(self.cases) if root(i) in selected],
                name=f"{self.name}[dev]",
            ),
            EvalSet(
                [c for i, c in enumerate(self.cases) if root(i) not in selected],
                name=f"{self.name}[holdout]",
            ),
        )

    def assert_disjoint(self, other: EvalSet, *, group_key: str | None = None) -> None:
        """Reject shared IDs, normalized text, or declared groups between sets.

        Args:
            other: Cases used for the other data role.
            group_key: Optional metadata field identifying independent groups.

        Raises:
            ValueError: If either set lacks a requested group or the sets overlap.
        """
        if group_key is not None and (not isinstance(group_key, str) or not group_key):
            raise ValueError("group_key must be a nonempty string")
        own = {key for case in self for key in _leakage_keys(case, group_key)}
        theirs = {key for case in other for key in _leakage_keys(case, group_key)}
        overlap = own & theirs
        if overlap:
            kinds = sorted({kind for kind, _ in overlap})
            raise ValueError(f"discovery and validation overlap on {', '.join(kinds)}")

    def to_jsonl(self, path: str) -> None:
        """Write records as UTF-8 JSON Lines.

        Args:
            path: Input or output file path.
        """
        with Path(path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(c.to_dict()) + "\n" for c in self.cases)

    @classmethod
    def from_jsonl(cls, path: str, name: str | None = None) -> EvalSet:
        """Read records from a UTF-8 JSON Lines file.

        Args:
            path: Input or output file path.
            name: Human-readable collection or slice name.

        Returns:
            Read records from a UTF-8 JSON Lines file.
        """
        with Path(path).open(encoding="utf-8") as f:
            cases = [Case.from_dict(json.loads(item)) for item in f if item.strip()]
        return cls(cases, name=name or path)


def _leakage_keys(case: Case, group_key: str | None) -> list[tuple[str, str]]:
    text = " ".join(unicodedata.normalize("NFKC", case.text).casefold().split())
    keys = [("id", case.id), ("text", text)]
    if group_key is not None:
        group = case.metadata.get(group_key)
        if (
            not isinstance(group_key, str)
            or not group_key
            or not isinstance(group, str)
            or not group
        ):
            raise ValueError(
                f"case {case.id!r} requires nonempty metadata {group_key!r}"
            )
        keys.append(("group", group))
    return keys
