"""Store: the example store treated as versioned model state.

A Store is immutable. Every edit returns a new Store with a new content
hash, so a store version can be diffed, bisected, and pinned the way a
code commit can. Rows carry provenance (who, when, under which taxonomy
version) and an optional scope that bounds the blast radius of an edit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from .retriever import Embedder


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        json.dumps(value, allow_nan=False)
        return value
    raise ValueError("metadata must contain only finite JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Row:
    """One labeled example with immutable JSON metadata and an optional vector.

    Attributes:
        text: Example text supplied to retrieval and classification.
        label: Taxonomy label identifier.
        id: Stable identifier, derived from text and label when omitted.
        scope: Optional partition used by scope-aware retrieval.
        created_at: Creation time, or None if unknown; naive times mean UTC.
        author: Author or source responsible for the example.
        taxonomy_version: Label definition version reviewed by the author.
        embedding: Optional finite vector, copied into read-only storage.
        metadata: Immutable JSON-compatible provenance or application fields.
        embedding_model: Embedder identity; None means externally supplied vectors.
    """

    text: str
    label: str
    id: str = ""
    scope: str | None = None
    created_at: datetime | None = field(default_factory=_utcnow)
    author: str | None = None
    taxonomy_version: str | None = None
    embedding: np.ndarray | None = field(default=None, compare=False, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    embedding_model: str | None = None

    def __post_init__(self) -> None:
        """Normalize derived fields after construction.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        for name in ("text", "label", "id"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        for name in ("scope", "author", "taxonomy_version", "embedding_model"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.embedding is not None:
            vector = np.asarray(self.embedding, dtype=float)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError("embedding must be a finite nonempty vector")
            object.__setattr__(
                self, "embedding", np.frombuffer(vector.tobytes(), dtype=float)
            )
        if not self.id:
            object.__setattr__(self, "id", content_id(self.text, self.label))
        if self.created_at is not None and not isinstance(self.created_at, datetime):
            raise ValueError("created_at must be a datetime or None")
        if self.created_at is not None and self.created_at.tzinfo is None:
            object.__setattr__(self, "created_at", self.created_at.replace(tzinfo=UTC))

    def with_(self, **changes: Any) -> Row:
        """Return a replacement row with the requested fields changed.

        Args:
            **changes: Replacement field values.

        Returns:
            a replacement row with the requested fields changed.
        """
        if "text" in changes and changes["text"] != self.text:
            changes.setdefault("embedding", None)
            changes.setdefault("embedding_model", None)
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record.

        Returns:
            a JSON-compatible record.
        """
        d: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "scope": self.scope,
            "created_at": None
            if self.created_at is None
            else self.created_at.isoformat(),
            "author": self.author,
            "taxonomy_version": self.taxonomy_version,
            "metadata": _thaw(self.metadata),
            "embedding_model": self.embedding_model,
        }
        if self.embedding is not None:
            d["embedding"] = [float(x) for x in self.embedding]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Row:
        """Load a record from its serialized fields.

        Args:
            d: Serialized record fields.

        Returns:
            Load a record from its serialized fields.
        """
        emb = d.get("embedding")
        return cls(
            text=d["text"],
            label=d["label"],
            id=d.get("id", ""),
            scope=d.get("scope"),
            created_at=datetime.fromisoformat(d["created_at"])
            if d.get("created_at") is not None
            else None,
            author=d.get("author"),
            taxonomy_version=d.get("taxonomy_version"),
            embedding=np.asarray(emb, dtype=float) if emb is not None else None,
            metadata=d.get("metadata", {}),
            embedding_model=d.get("embedding_model"),
        )


def content_id(text: str, label: str) -> str:
    """Derive a stable identifier from the text and label.

    Args:
        text: Example text.
        label: Reference label identifier.

    Returns:
        Derive a stable identifier from the text and label.
    """
    return hashlib.sha1(
        f"{label}\x00{text}".encode(), usedforsecurity=False
    ).hexdigest()[:12]


@dataclass(frozen=True)
class StoreDiff:
    """Added, removed, and edited rows between two stores."""

    added: tuple[Row, ...]
    removed: tuple[Row, ...]
    changed: tuple[tuple[Row, Row], ...]  # (before, after) pairs sharing an id

    @property
    def touched_ids(self) -> frozenset[str]:
        """Return identifiers added, removed, or edited.

        Returns:
            identifiers added, removed, or edited.
        """
        return frozenset(
            [r.id for r in self.added]
            + [r.id for r in self.removed]
            + [a.id for a, _ in self.changed]
        )

    @property
    def touched_scopes(self) -> frozenset[str | None]:
        """Return scopes affected before or after the change.

        Returns:
            scopes affected before or after the change.
        """
        rows = (
            list(self.added)
            + list(self.removed)
            + [b for _, b in self.changed]
            + [a for a, _ in self.changed]
        )
        return frozenset(r.scope for r in rows)

    def is_empty(self) -> bool:
        """Return whether the diff contains no changes.

        Returns:
            whether the diff contains no changes.
        """
        return not (self.added or self.removed or self.changed)


class Store:
    """An immutable, content-addressed collection of Rows."""

    def __init__(self, rows: Iterable[Row] = (), name: str = "store"):
        """Configure Store with the supplied inputs.

        Args:
            rows: Rows to include in the store.
            name: Human-readable collection or slice name.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        rows = list(rows)
        ids = [r.id for r in rows]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate row ids: {dupes[:5]}")
        self._rows: tuple[Row, ...] = tuple(rows)
        self._by_id: dict[str, Row] = {r.id: r for r in rows}
        self.name = name

    # -- container protocol ------------------------------------------------
    def __len__(self) -> int:
        """Return the number of records.

        Returns:
            the number of records.
        """
        return len(self._rows)

    def __iter__(self) -> Iterator[Row]:
        """Iterate over records in insertion order.

        Returns:
            Iterate over records in insertion order.
        """
        return iter(self._rows)

    def __contains__(self, row_id: str) -> bool:
        """Return whether the identifier exists.

        Args:
            row_id: Stable identifier of the row.

        Returns:
            whether the identifier exists.
        """
        return row_id in self._by_id

    def __getitem__(self, row_id: str) -> Row:
        """Look up a record by its identifier.

        Args:
            row_id: Stable identifier of the row.

        Returns:
            Look up a record by its identifier.
        """
        return self._by_id[row_id]

    @property
    def rows(self) -> tuple[Row, ...]:
        """Return the stored rows in insertion order.

        Returns:
            the stored rows in insertion order.
        """
        return self._rows

    def get(self, row_id: str) -> Row | None:
        """Look up a row, returning None when absent.

        Args:
            row_id: Stable identifier of the row.

        Returns:
            Look up a row, returning None when absent.
        """
        return self._by_id.get(row_id)

    # -- identity ----------------------------------------------------------
    @property
    def version(self) -> str:
        """Content hash over all serialized row fields, independent of row order.

        Returns:
            the content-derived version.
        """
        h = hashlib.sha256()
        for r in sorted(self._rows, key=lambda r: r.id):
            h.update(json.dumps(r.to_dict(), sort_keys=True, allow_nan=False).encode())
        return h.hexdigest()[:16]

    # -- edits (each returns a new Store) ----------------------------------
    def add(self, *rows: Row) -> Store:
        """Return a new store containing the additional rows.

        Args:
            *rows: Rows to include in the store.

        Returns:
            a new store containing the additional rows.
        """
        return Store(list(self._rows) + list(rows), name=self.name)

    def remove(self, *row_ids: str) -> Store:
        """Return a new store without the specified row identifiers.

        Args:
            *row_ids: Row identifiers to inspect or remove.

        Returns:
            a new store without the specified row identifiers.

        Raises:
            KeyError: If a requested row identifier is absent.
        """
        drop = set(row_ids)
        missing = drop - set(self._by_id)
        if missing:
            raise KeyError(f"rows not in store: {sorted(missing)}")
        return Store([r for r in self._rows if r.id not in drop], name=self.name)

    def edit(self, row_id: str, **changes: Any) -> Store:
        """Return a new store with one row replaced, keeping its identifier.

        Args:
            row_id: Stable identifier of the row.
            **changes: Replacement field values.

        Returns:
            a new store with one row replaced, keeping its identifier.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        if "id" in changes:
            raise ValueError("edit keeps the row id; use remove + add to replace")
        new = self._by_id[row_id].with_(**changes)
        return Store([new if r.id == row_id else r for r in self._rows], name=self.name)

    def filter(self, pred: Callable[[Row], bool]) -> Store:
        """Return a store containing only rows accepted by the predicate.

        Args:
            pred: Predicate deciding which rows to retain.

        Returns:
            a store containing only rows accepted by the predicate.
        """
        return Store([r for r in self._rows if pred(r)], name=self.name)

    def scope(self, scope: str | None) -> Store:
        """Return a store containing only the requested scope.

        Args:
            scope: Scope to select; None denotes unscoped rows.

        Returns:
            a store containing only the requested scope.
        """
        return self.filter(lambda r: r.scope == scope)

    # -- diff --------------------------------------------------------------
    def diff(self, other: Store) -> StoreDiff:
        """Rows that differ between self (before) and other (after).

        Args:
            other: Candidate store to compare with this baseline.

        Returns:
            Added and removed rows and ordered before/after pairs for edits.
        """
        added = tuple(r for r in other if r.id not in self._by_id)
        removed = tuple(r for r in self if r.id not in other._by_id)
        changed = tuple(
            (self._by_id[r.id], r)
            for r in other
            if r.id in self._by_id and self._by_id[r.id].to_dict() != r.to_dict()
        )
        return StoreDiff(added=added, removed=removed, changed=changed)

    # -- embeddings --------------------------------------------------------
    def embeddings(self) -> np.ndarray:
        """Stacked embeddings; raises if any row lacks one.

        Returns:
            A matrix with one vector per row in store order.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        missing = [r.id for r in self._rows if r.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} rows lack embeddings; call store.embed(embedder) first"
            )
        return np.stack([r.embedding for r in self._rows if r.embedding is not None])

    def embed(self, embedder: Embedder, overwrite: bool = False) -> Store:
        """Return a Store where every row carries an embedding.

        Args:
            embedder: Text embedding implementation.
            overwrite: Whether existing embeddings should be recomputed.

        Returns:
            Map the text batch to a matrix with one vector per text.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        todo = [
            r
            for r in self._rows
            if overwrite
            or r.embedding is None
            or (r.embedding_model is not None and r.embedding_model != embedder.name)
        ]
        if not todo:
            return self
        vecs = np.asarray(embedder.embed([r.text for r in todo]), dtype=float)
        if vecs.ndim != 2 or vecs.shape[0] != len(todo) or vecs.shape[1] == 0:
            raise ValueError("embedder must return one nonempty vector per text")
        if not np.isfinite(vecs).all():
            raise ValueError("embeddings must be finite")
        new = {
            r.id: r.with_(
                embedding=np.asarray(v, dtype=float), embedding_model=embedder.name
            )
            for r, v in zip(todo, vecs, strict=False)
        }
        return Store([new.get(r.id, r) for r in self._rows], name=self.name)

    # -- persistence -------------------------------------------------------
    def to_jsonl(self, path: str) -> None:
        """Write records as UTF-8 JSON Lines.

        Args:
            path: Input or output file path.
        """
        with Path(path).open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r.to_dict()) + "\n" for r in self._rows)

    @classmethod
    def from_jsonl(cls, path: str, name: str | None = None) -> Store:
        """Read records from a UTF-8 JSON Lines file.

        Args:
            path: Input or output file path.
            name: Human-readable collection or slice name.

        Returns:
            Read records from a UTF-8 JSON Lines file.
        """
        with Path(path).open(encoding="utf-8") as f:
            rows = [Row.from_dict(json.loads(line)) for line in f if line.strip()]
        return cls(rows, name=name or path)

    def __repr__(self) -> str:
        """Return a concise store description.

        Returns:
            a concise store description.
        """
        return f"Store({self.name!r}, rows={len(self)}, version={self.version})"
