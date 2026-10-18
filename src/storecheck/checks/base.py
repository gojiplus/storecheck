"""Check protocol shared by all tiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..report import Finding
    from ..retriever import Embedder
    from ..store import Store
    from ..taxonomy import Taxonomy


@dataclass
class CheckContext:
    """Context shared by checks.

    Static checks use the store, taxonomy, and embedder. Dynamic checks
    receive an evaluator when constructed.
    """

    store: Store
    taxonomy: Taxonomy | None = None
    embedder: Embedder | None = None  # Embedder; optional for checks that need vectors


@runtime_checkable
class Check(Protocol):
    """Interface implemented by static, dynamic, and mutation checks."""

    name: str
    tier: str  # "static" | "dynamic" | "meta"

    def run(self, ctx: CheckContext) -> list[Finding]:
        """Return findings for the supplied check context.

        Args:
            ctx: Store and optional taxonomy and embedder for this check.

        Returns:
            Findings produced by the check.
        """
        ...
