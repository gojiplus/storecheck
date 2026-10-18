"""Taxonomy: the type system for rows.

Each label has a definition. The hash of that definition is the label's
version. A row records the label version its author saw; when the
definition changes, rows written under the old definition become
orphans and the TaxonomyValidity check flags them for re-review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class LabelDef:
    """A label definition and its review boundary examples."""

    id: str
    definition: str
    examples_of_boundary: tuple[
        str, ...
    ] = ()  # free text: what this label is often confused with
    deprecated: bool = False

    def __post_init__(self) -> None:
        """Validate label definition fields.

        Raises:
            ValueError: If label fields have invalid types or empty values.
        """
        if any(
            not isinstance(v, str) or not v.strip() for v in (self.id, self.definition)
        ):
            raise ValueError("label id and definition must be nonempty strings")
        if not isinstance(self.deprecated, bool):
            raise ValueError("deprecated must be a boolean")
        if isinstance(self.examples_of_boundary, str) or any(
            not isinstance(value, str) for value in self.examples_of_boundary
        ):
            raise ValueError("examples_of_boundary must contain strings")
        object.__setattr__(
            self, "examples_of_boundary", tuple(self.examples_of_boundary)
        )

    @property
    def version(self) -> str:
        """Return the content-derived version.

        Returns:
            the content-derived version.
        """
        return hashlib.sha1(
            f"{self.id}\x00{self.definition}".encode(), usedforsecurity=False
        ).hexdigest()[:8]


class Taxonomy:
    """Named label definitions with content-derived versions."""

    def __init__(self, labels: list[LabelDef], name: str = "taxonomy"):
        """Configure Taxonomy with the supplied inputs.

        Args:
            labels: Label definitions to register.
            name: Human-readable collection or slice name.

        Raises:
            ValueError: If inputs violate the documented constraints.
        """
        ids = [label_def.id for label_def in labels]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate label ids")
        self._labels = {label_def.id: label_def for label_def in labels}
        self.name = name

    def __contains__(self, label_id: str) -> bool:
        """Return whether the identifier exists.

        Args:
            label_id: Label identifier to look up.

        Returns:
            whether the identifier exists.
        """
        return label_id in self._labels

    def __getitem__(self, label_id: str) -> LabelDef:
        """Look up a record by its identifier.

        Args:
            label_id: Label identifier to look up.

        Returns:
            Look up a record by its identifier.
        """
        return self._labels[label_id]

    def __iter__(self) -> Iterator[LabelDef]:
        """Iterate over records in insertion order.

        Returns:
            Iterate over records in insertion order.
        """
        return iter(self._labels.values())

    def __len__(self) -> int:
        """Return the number of records.

        Returns:
            the number of records.
        """
        return len(self._labels)

    @property
    def ids(self) -> list[str]:
        """Return all label identifiers.

        Returns:
            all label identifiers.
        """
        return list(self._labels)

    @property
    def active_ids(self) -> list[str]:
        """Return identifiers of labels that are not deprecated.

        Returns:
            identifiers of labels that are not deprecated.
        """
        return [
            label_def.id
            for label_def in self._labels.values()
            if not label_def.deprecated
        ]

    @property
    def version(self) -> str:
        """Return the content-derived version.

        Returns:
            the content-derived version.
        """
        h = hashlib.sha256()
        for label_def in sorted(
            self._labels.values(), key=lambda label_def: label_def.id
        ):
            h.update(
                f"{label_def.id}:{label_def.version}:{int(label_def.deprecated)}\n".encode()
            )
        return h.hexdigest()[:12]

    def label_version(self, label_id: str) -> str | None:
        """Return a label definition version, or None if unknown.

        Args:
            label_id: Label identifier to look up.

        Returns:
            a label definition version, or None if unknown.
        """
        label_def = self._labels.get(label_id)
        return label_def.version if label_def else None

    # -- persistence -------------------------------------------------------
    def to_json(self, path: str) -> None:
        """Serialize the report or taxonomy as JSON.

        Args:
            path: Input or output file path.
        """
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "labels": [
                        {
                            "id": label_def.id,
                            "definition": label_def.definition,
                            "examples_of_boundary": list(
                                label_def.examples_of_boundary
                            ),
                            "deprecated": label_def.deprecated,
                        }
                        for label_def in self
                    ],
                },
                f,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: str) -> Taxonomy:
        """Load label definitions from a JSON file.

        Args:
            path: Input or output file path.

        Returns:
            Load label definitions from a JSON file.
        """
        with Path(path).open(encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            [
                LabelDef(
                    id=x["id"],
                    definition=x["definition"],
                    examples_of_boundary=x.get("examples_of_boundary", ()),
                    deprecated=x.get("deprecated", False),
                )
                for x in d["labels"]
            ],
            name=d.get("name", "taxonomy"),
        )
