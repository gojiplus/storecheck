"""Objective: what a store change is judged against.

A primary metric to improve, subject to bounds: minimum accuracy on
named slices and a cap on right-to-wrong transitions. A custom primary
metric can measure total error cost. Similarity, novelty, and class balance are
hypothesis generators, so they live in checks, never here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .evalset import Case
from .validation import fraction, positive_int

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Outcome = tuple[Case, str]  # (case, predicted label)


def accuracy(outcomes: Sequence[Outcome]) -> float:
    """Return the fraction of correct predictions, or NaN for no cases.

    Args:
        outcomes: Reference cases paired with predicted labels.

    Returns:
        the fraction of correct predictions, or NaN for no cases.
    """
    if not outcomes:
        return float("nan")
    return sum(c.label == p for c, p in outcomes) / len(outcomes)


def weighted_error_cost(outcomes: Sequence[Outcome]) -> float:
    """Return the total error cost of incorrectly predicted cases.

    Args:
        outcomes: Reference cases paired with predicted labels.

    Returns:
        the total error cost of incorrectly predicted cases.
    """
    return sum(c.cost for c, p in outcomes if c.label != p)


@dataclass
class Objective:
    """Metric improvement and regression limits required of a candidate."""

    primary: Callable[[Sequence[Outcome]], float] = accuracy
    primary_name: str = "accuracy"
    higher_is_better: bool = True
    min_slice_metric: dict[str, float] = field(
        default_factory=dict
    )  # slice -> floor on primary
    max_right_to_wrong: int | None = None  # absolute cap on regressions
    max_right_to_wrong_frac: float | None = None  # cap as fraction of eval set
    max_right_to_wrong_cost: float | None = None
    max_slice_metric: dict[str, float] = field(default_factory=dict)
    min_slice_cases: dict[str, int] = field(default_factory=dict)
    min_primary_delta: float = 0.0  # candidate must improve by at least this much

    def __post_init__(self) -> None:
        """Reject invalid transition caps and non-finite metric thresholds.

        Raises:
            ValueError: If a configured metric threshold is not finite.
        """
        if self.max_right_to_wrong is not None:
            positive_int("max_right_to_wrong", self.max_right_to_wrong, allow_zero=True)
        if self.max_right_to_wrong_frac is not None:
            fraction("max_right_to_wrong_frac", self.max_right_to_wrong_frac)
        if not math.isfinite(self.min_primary_delta):
            raise ValueError("min_primary_delta must be finite")
        for bounds in (self.min_slice_metric, self.max_slice_metric):
            if not isinstance(bounds, dict) or any(
                not isinstance(k, str)
                or not k
                or isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for k, v in bounds.items()
            ):
                raise ValueError(
                    "slice thresholds must map nonempty names to finite numbers"
                )
        for name in self.min_slice_metric.keys() & self.max_slice_metric.keys():
            if self.min_slice_metric[name] > self.max_slice_metric[name]:
                raise ValueError(f"slice {name!r} floor exceeds ceiling")
        if not isinstance(self.min_slice_cases, dict):
            raise ValueError("min_slice_cases must be a mapping")
        for name, count in self.min_slice_cases.items():
            if not isinstance(name, str) or not name:
                raise ValueError("min_slice_cases requires nonempty slice names")
            positive_int("min_slice_cases", count)
        cost = self.max_right_to_wrong_cost
        if cost is not None and (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError("max_right_to_wrong_cost must be finite and nonnegative")

    def improved(self, baseline: float, candidate: float) -> bool:
        """Return whether the metric change meets the required improvement.

        Args:
            baseline: Reference store or metric value.
            candidate: Proposed store or metric value.

        Returns:
            whether the metric change meets the required improvement.
        """
        delta = (
            (candidate - baseline) if self.higher_is_better else (baseline - candidate)
        )
        return delta >= self.min_primary_delta
