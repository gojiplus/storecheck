"""Strict TOML configuration and adapter factories for comparison runs."""

from __future__ import annotations

import importlib
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

from .comparison import ComparisonReport, compare_stores
from .decider import MajorityVoteDecider
from .evalset import EvalSet
from .objective import Objective, accuracy, weighted_error_cost
from .pipeline import Pipeline, Prediction, Query, Snapshot
from .retriever import EmbeddingRetriever, HashingEmbedder
from .store import Store
from .taxonomy import Taxonomy


def _keys(data: dict[str, Any], allowed: set[str], section: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be a table")
    unknown = data.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {section} settings: {', '.join(sorted(unknown))}")


def _path(parent: Path, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("file paths must be nonempty strings")
    return str(parent / value)


def _pipeline(settings: dict[str, Any]) -> Pipeline:
    kind = settings.get("kind")
    if kind == "reference":
        _keys(
            settings, {"kind", "k", "dimension", "scope_aware", "fallback"}, "pipeline"
        )
        k = settings.get("k", 5)
        dimension = settings.get("dimension", 1024)
        scope_aware = settings.get("scope_aware", False)
        fallback = settings.get("fallback", "unknown")
        if not isinstance(scope_aware, bool):
            raise ValueError("pipeline.scope_aware must be a boolean")
        if not isinstance(fallback, str) or not fallback:
            raise ValueError("pipeline.fallback must be a nonempty string")
        embedder = HashingEmbedder(dimension)
        retriever = EmbeddingRetriever(embedder, k=k, scope_aware=scope_aware)
        decider = MajorityVoteDecider(fallback)

        def run(query: Query, snapshot: Store | Snapshot) -> Prediction:
            if not isinstance(snapshot, Store):
                raise ValueError("reference pipeline requires local Store inputs")
            evidence = retriever.retrieve(query.text, snapshot, scope=query.scope)
            return Prediction(decider.decide(query.text, evidence), tuple(evidence))

        return Pipeline(
            run,
            {
                "kind": kind,
                "embedder": embedder.name,
                "k": k,
                "scope_aware": scope_aware,
                "decider": "majority-v1",
                "fallback": fallback,
            },
        )
    if kind == "custom":
        _keys(settings, {"kind", "factory", "options"}, "pipeline")
        factory = settings.get("factory", "")
        if not isinstance(factory, str) or factory.count(":") != 1:
            raise ValueError("pipeline.factory must be 'installed_module:function'")
        options = settings.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("pipeline.options must be a table")
        module_name, function_name = factory.split(":")
        function = getattr(importlib.import_module(module_name), function_name, None)
        if not callable(function):
            raise ValueError(f"pipeline factory {factory!r} is not callable")
        pipeline = function(options)
        if not isinstance(pipeline, Pipeline) or not pipeline.identity:
            raise ValueError(
                "pipeline factory must return Pipeline with a nonempty identity"
            )
        return Pipeline(pipeline.run, {**pipeline.identity, "factory": factory})
    raise ValueError("pipeline.kind must be 'reference' or 'custom'")


def compare_from_config(
    baseline: Store | Snapshot, candidate: Store | Snapshot, path: str | Path
) -> ComparisonReport:
    """Load a TOML configuration and execute the shared comparison engine.

    Data paths are relative to the configuration file. Custom factories are
    imported Python code; run only configurations and adapters you trust.

    Args:
        baseline: Current data or immutable external snapshot reference.
        candidate: Proposed data or immutable external snapshot reference.
        path: Configuration file with schema_version, evalset, and pipeline.

    Returns:
        The same comparison artifact produced by the Python API.

    Raises:
        ValueError: If settings are unknown, missing, or inconsistent.
    """
    path = Path(path)
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    _keys(
        config,
        {
            "schema_version",
            "evalset",
            "taxonomy",
            "pipeline",
            "objective",
            "discovery",
            "group_key",
        },
        "root",
    )
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    evalset = EvalSet.from_jsonl(_path(path.parent, config.get("evalset")))
    discovery = (
        EvalSet.from_jsonl(_path(path.parent, config["discovery"]))
        if "discovery" in config
        else None
    )
    taxonomy = (
        Taxonomy.from_json(_path(path.parent, config["taxonomy"]))
        if "taxonomy" in config
        else None
    )
    settings = config.get("objective", {})
    allowed = {f.name for f in fields(Objective)} - {
        "primary",
        "primary_name",
        "higher_is_better",
    }
    _keys(settings, allowed | {"metric"}, "objective")
    metric_name = settings.get("metric", "accuracy")
    if metric_name not in ("accuracy", "weighted_error_cost"):
        raise ValueError("objective.metric must be 'accuracy' or 'weighted_error_cost'")
    objective = Objective(
        primary=accuracy if metric_name == "accuracy" else weighted_error_cost,
        primary_name=metric_name,
        higher_is_better=metric_name == "accuracy",
        **{key: value for key, value in settings.items() if key != "metric"},
    )
    pipeline_settings = config.get("pipeline", {})
    if not isinstance(pipeline_settings, dict):
        raise ValueError("pipeline must be a table")
    pipeline = _pipeline(pipeline_settings)
    return compare_stores(
        baseline,
        candidate,
        evalset,
        pipeline=pipeline,
        objective=objective,
        taxonomy=taxonomy,
        discovery=discovery,
        group_key=config.get("group_key"),
    )
