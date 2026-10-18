"""Retriever and Embedder protocols, plus reference implementations.

The retriever is the function from (query, store) to the evidence the
decider sees. Everything the dynamic tier does is expressed through this
interface, so a production retriever (dense + lexical + rerank) plugs in
by implementing `retrieve`.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .validation import positive_int

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .store import Row, Store


@runtime_checkable
class Embedder(Protocol):
    """Interface for mapping text batches into equal-width vectors."""

    @property
    def name(self) -> str:
        """Return a stable identity including all embedding configuration."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Map the text batch to a matrix with one vector per text.

        Args:
            texts: Text batch to embed.

        Returns:
            The result defined by the protocol.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """Interface for retrieving ordered evidence for a query."""

    k: int

    def retrieve(
        self,
        query: str,
        store: Store,
        k: int | None = None,
        *,
        scope: str | None = None,
    ) -> list[Row]:
        """Return up to k examples ranked by similarity to the query.

        Args:
            query: Text to classify.
            store: Example store to inspect.
            k: Maximum number of neighbors.
            scope: Optional partition used by scope-aware retrievers.

        Returns:
            The result defined by the protocol.
        """
        ...


class HashingEmbedder:
    """Dependency-free bag-of-words hashing embedder for tests and demos.

    Swap for a real sentence embedder in production. The `name` field is
    the pinned dependency: changing it invalidates every stored embedding.
    """

    @property
    def name(self) -> str:
        """Return an identity covering hashing dimensions and stopwords."""
        words = "\n".join(sorted(self.stopwords)).encode()
        fingerprint = hashlib.sha256(words).hexdigest()[:16]
        return f"hashing-bow-v2:{self.dim}:{fingerprint}"

    STOPWORDS = frozenset(
        [
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "from",
            "had",
            "has",
            "have",
            "i",
            "if",
            "in",
            "into",
            "is",
            "it",
            "its",
            "me",
            "my",
            "of",
            "on",
            "or",
            "so",
            "that",
            "the",
            "this",
            "to",
            "was",
            "we",
            "when",
            "which",
            "with",
            "you",
            "your",
            "am",
            "do",
            "does",
            "did",
            "not",
            "no",
            "can",
            "cannot",
            "could",
            "would",
            "should",
            "will",
            "still",
            "even",
            "though",
            "fine",
            "please",
        ]
    )

    def __init__(self, dim: int = 1024, stopwords: frozenset | None = None):
        """Configure HashingEmbedder with the supplied inputs.

        Args:
            dim: Number of hashing dimensions.
            stopwords: Tokens omitted before hashing.
        """
        positive_int("dim", dim)
        self.dim = dim
        self.stopwords = self.STOPWORDS if stopwords is None else stopwords
        self._tok = re.compile(r"[a-z0-9]+")

    def _stem(self, tok: str) -> str:
        for suf in ("ing", "ed", "es", "s"):
            if len(tok) > 4 and tok.endswith(suf):
                return tok[: -len(suf)]
        return tok

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Map the text batch to a matrix with one vector per text.

        Args:
            texts: Text batch to embed.

        Returns:
            Map the text batch to a matrix with one vector per text.
        """
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            toks = [
                self._stem(x)
                for x in self._tok.findall(t.lower())
                if x not in self.stopwords
            ]
            for tok in toks:
                h = int(
                    hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16
                )
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities, treating zero vectors as zero.

    Args:
        a: First matrix, with one vector per row.
        b: Second matrix, with matching vector width.

    Returns:
        pairwise cosine similarities, treating zero vectors as zero.

    Raises:
        ValueError: If inputs violate the documented constraints.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or not a.shape[1]:
        raise ValueError("cosine inputs must be matrices with equal nonzero width")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine inputs must be finite")
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return an @ bn.T


class EmbeddingRetriever:
    """Dense kNN over row embeddings with optional scope restriction."""

    deletion_stable = True

    def __init__(self, embedder: Embedder, k: int = 5, scope_aware: bool = False):
        """Configure EmbeddingRetriever with the supplied inputs.

        Args:
            embedder: Text embedding implementation.
            k: Maximum number of neighbors.
            scope_aware: Whether retrieval restricts evidence to the supplied scope.
        """
        positive_int("k", k)
        self.embedder = embedder
        self.k = k
        self.scope_aware = scope_aware
        self._cache: dict[tuple, tuple[Row, ...]] = {}

    def retrieve(
        self, query: str, store: Store, k: int | None = None, scope: str | None = None
    ) -> list[Row]:
        """Return up to k examples ranked by similarity to the query.

        Args:
            query: Text to classify.
            store: Example store to inspect.
            k: Maximum number of neighbors.
            scope: Scope to select; None denotes unscoped rows.

        Returns:
            up to k examples ranked by similarity to the query.
        """
        k = self.k if k is None else k
        positive_int("k", k)
        if len(store) == 0:
            return []
        candidates = (
            store.scope(scope) if (self.scope_aware and scope is not None) else store
        )
        if len(candidates) == 0:
            return []
        key = (
            query,
            candidates.version,
            k,
            tuple(row.id for row in candidates),
            self.embedder.name,
        )
        if key in self._cache:
            return list(self._cache[key])
        candidates = candidates.embed(self.embedder)
        q = self.embedder.embed([query])
        sims = cosine_matrix(q, candidates.embeddings())[0]
        order = np.argsort(-sims, kind="stable")[:k]
        rows = [candidates.rows[i] for i in order]
        self._cache[key] = tuple(rows)
        return rows
