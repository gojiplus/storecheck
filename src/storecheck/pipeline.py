"""Application inputs, snapshot references, and observed predictions."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .store import Row, Store, _freeze, _thaw


def _json_object(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return _freeze(value)


@dataclass(frozen=True)
class Query:
    """Application input without reference labels, evaluation slices, or costs.

    Metadata must contain only inputs available to the application at runtime.
    """

    text: str
    scope: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate text and scope and detach metadata.

        Raises:
            ValueError: If text, scope, or metadata have invalid types.
        """
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("query text must be a nonempty string")
        if self.scope is not None and not isinstance(self.scope, str):
            raise ValueError("query scope must be a string or None")
        object.__setattr__(self, "metadata", _json_object("metadata", self.metadata))


@dataclass(frozen=True)
class Snapshot:
    """Reference to an immutable external dataset/index revision.

    The application must route to this exact revision. Storecheck records the
    descriptor but cannot verify the contents of the external service. Metadata
    may identify collections, index revisions, or namespaces; omit credentials.
    """

    name: str
    revision: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the external identity and detach metadata.

        Raises:
            ValueError: If a required identifier is blank or not a string.
        """
        for name, value in (("name", self.name), ("revision", self.revision)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"snapshot {name} must be a nonempty string")
        object.__setattr__(self, "metadata", _json_object("metadata", self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return the external snapshot descriptor.

        Returns:
            Name, revision, and detached JSON metadata.
        """
        return {
            "name": self.name,
            "revision": self.revision,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> Snapshot:
        """Read a snapshot descriptor from a JSON file.

        Args:
            path: Descriptor file, not a dump of database rows.

        Returns:
            Validated external snapshot reference.

        Raises:
            ValueError: If the file does not contain a descriptor object.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("snapshot descriptor must be a JSON object")
        return cls(**data)


@dataclass(frozen=True)
class Prediction:
    """Observed label, final ordered evidence, and optional application trace.

    Evidence should be what the application actually used after filtering,
    fusion, reranking, and truncation. Trace can retain scores, formatted context,
    and nonsecret provider/configuration details. Empty evidence is permitted
    when an application cannot expose it; no evidence diagnosis is then possible.
    """

    label: str
    evidence: tuple[Row, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the output and detach the evidence container and trace.

        Raises:
            ValueError: If the label, evidence, or trace is malformed.
        """
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("prediction label must be a nonempty string")
        if not isinstance(self.evidence, (tuple, list)) or any(
            not isinstance(row, Row) for row in self.evidence
        ):
            raise ValueError("prediction evidence must be a sequence of Row objects")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "trace", _json_object("trace", self.trace))


@dataclass(frozen=True)
class Pipeline:
    """One synchronous application callable plus its nonsecret configuration.

    The callable accepts Query and Store or Snapshot, and returns Prediction.
    It owns all retrieval and decision behavior. Identity should include code,
    model, prompt, retrieval, and sampling revisions. Calls are never memoized
    by the comparison engine; one paired run does not estimate stochastic risk.
    """

    run: Callable[[Query, Store | Snapshot], Prediction]
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require a synchronous callable and nonempty JSON identity.

        Raises:
            ValueError: If the callable or identity is invalid.
        """
        if (
            not callable(self.run)
            or inspect.iscoroutinefunction(self.run)
            or inspect.iscoroutinefunction(self.run.__call__)
        ):
            raise ValueError("pipeline.run must be a synchronous callable")
        if not self.identity:
            raise ValueError("pipeline identity must be a nonempty JSON object")
        object.__setattr__(self, "identity", _json_object("identity", self.identity))
