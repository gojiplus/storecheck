"""Regression gates and diagnostics for classifier example stores."""

from importlib.metadata import version

from .comparison import ComparisonReport, compare_stores
from .decider import CallableDecider, Decider, MajorityVoteDecider, MemoizedDecider
from .evalset import Case, EvalSet
from .objective import Objective, accuracy, weighted_error_cost
from .pipeline import Pipeline, Prediction, Query, Snapshot
from .report import Finding, Report, Severity
from .retriever import Embedder, EmbeddingRetriever, HashingEmbedder, Retriever
from .runner import run_checks, run_static
from .store import Row, Store, StoreDiff
from .taxonomy import LabelDef, Taxonomy

__version__ = version("storecheck")

__all__ = [
    "CallableDecider",
    "Case",
    "ComparisonReport",
    "Decider",
    "Embedder",
    "EmbeddingRetriever",
    "EvalSet",
    "Finding",
    "HashingEmbedder",
    "LabelDef",
    "MajorityVoteDecider",
    "MemoizedDecider",
    "Objective",
    "Pipeline",
    "Prediction",
    "Query",
    "Report",
    "Retriever",
    "Row",
    "Severity",
    "Snapshot",
    "Store",
    "StoreDiff",
    "Taxonomy",
    "accuracy",
    "compare_stores",
    "run_checks",
    "run_static",
    "weighted_error_cost",
]
