"""Run a list of checks against a store and collect a Report."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .checks.base import Check, CheckContext
from .checks.static import DEFAULT_STATIC_CHECKS
from .report import Report

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .retriever import Embedder
    from .store import Store
    from .taxonomy import Taxonomy


def run_checks(
    store: Store,
    checks: Iterable[Check],
    taxonomy: Taxonomy | None = None,
    embedder: Embedder | None = None,
) -> Report:
    """Run each check against the supplied store and collect findings.

    Args:
        store: Example store to inspect.
        checks: Checks to run; None selects the default static checks.
        taxonomy: Current label definitions, when available.
        embedder: Text embedding implementation.

    Returns:
        Run each check against the supplied store and collect findings.
    """
    ctx = CheckContext(store=store, taxonomy=taxonomy, embedder=embedder)
    report = Report(
        context={
            "store": store.name,
            "store_version": store.version,
            "rows": len(store),
        }
    )
    if taxonomy is not None:
        report.context["taxonomy_version"] = taxonomy.version
    if embedder is not None and getattr(embedder, "name", None):
        report.context["embedder"] = embedder.name
    for check in checks:
        report.extend(check.run(ctx))
    return report


def run_static(
    store: Store,
    taxonomy: Taxonomy | None = None,
    embedder: Embedder | None = None,
    checks: Iterable[Check] | None = None,
) -> Report:
    """Run the configured static diagnostics against a store.

    Args:
        store: Example store to inspect.
        taxonomy: Current label definitions, when available.
        embedder: Text embedding implementation.
        checks: Checks to run; None selects the default static checks.

    Returns:
        Run the configured static diagnostics against a store.
    """
    return run_checks(
        store,
        DEFAULT_STATIC_CHECKS if checks is None else checks,
        taxonomy=taxonomy,
        embedder=embedder,
    )
