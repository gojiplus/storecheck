"""Decider: the function from (query, evidence) to a label.

In production this is an LLM call. The dynamic tier treats it as opaque
and expensive, so it is memoized on the query and full ordered evidence. A case whose
retrieved context did not change between two store versions never
triggers a second call.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .store import Row


@runtime_checkable
class Decider(Protocol):
    """Interface for predicting a label from ordered evidence."""

    def decide(self, query: str, examples: Sequence[Row]) -> str:
        """Predict a label from the query and ordered examples.

        Args:
            query: Text to classify.
            examples: Retrieved evidence in presentation order.

        Returns:
            The result defined by the protocol.
        """
        ...


class MajorityVoteDecider:
    """kNN decision rule. Deterministic; useful for tests and as a floor."""

    def __init__(self, fallback: str = "unknown"):
        """Configure MajorityVoteDecider with the supplied inputs.

        Args:
            fallback: Label returned when evidence is empty.
        """
        self.fallback = fallback

    def decide(self, query: str, examples: Sequence[Row]) -> str:  # noqa: ARG002
        """Predict a label from the query and ordered examples.

        Args:
            query: Text to classify.
            examples: Retrieved evidence in presentation order.

        Returns:
            Predict a label from the query and ordered examples.
        """
        if not examples:
            return self.fallback
        counts = Counter(r.label for r in examples)
        top = counts.most_common()
        # ties broken by first retrieved example (rank order)
        best = top[0][1]
        for r in examples:
            if counts[r.label] == best:
                return r.label
        return top[0][0]


class CallableDecider:
    """Wrap any function(query, examples) -> label, e.g. an LLM prompt."""

    def __init__(self, fn: Callable[[str, Sequence[Row]], str]):
        """Configure CallableDecider with the supplied inputs.

        Args:
            fn: Callable used to compute the result.
        """
        self.fn = fn

    def decide(self, query: str, examples: Sequence[Row]) -> str:
        """Predict a label from the query and ordered examples.

        Args:
            query: Text to classify.
            examples: Retrieved evidence in presentation order.

        Returns:
            Predict a label from the query and ordered examples.
        """
        return self.fn(query, examples)


class MemoizedDecider:
    """Cache deterministic decisions on full ordered evidence and count real calls."""

    def __init__(self, inner: Decider):
        """Configure MemoizedDecider with the supplied inputs.

        Args:
            inner: Deterministic decider whose results will be cached.
        """
        self.inner = inner
        self._cache: dict[tuple, str] = {}
        self.calls = 0

    def decide(self, query: str, examples: Sequence[Row]) -> str:
        # key on evidence content, not ids: an edited row keeps its id but
        # changes the decision
        """Predict a label from the query and ordered examples.

        Args:
            query: Text to classify.
            examples: Retrieved evidence in presentation order.

        Returns:
            Predict a label from the query and ordered examples.
        """
        key = (
            query,
            tuple(
                json.dumps(r.to_dict(), sort_keys=True, allow_nan=False)
                for r in examples
            ),
        )
        if key not in self._cache:
            self.calls += 1
            self._cache[key] = self.inner.decide(query, examples)
        return self._cache[key]
