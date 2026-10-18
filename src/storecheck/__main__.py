"""Inspect stores and gate edits from the command line."""

from __future__ import annotations

import argparse
import sys

from .config import compare_from_config
from .pipeline import Snapshot
from .report import Severity
from .retriever import HashingEmbedder
from .runner import run_static
from .store import Store
from .taxonomy import Taxonomy


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit status.

    Args:
        argv: CLI arguments; None reads the process arguments.

    Returns:
        Run the CLI and return its exit status.
    """
    p = argparse.ArgumentParser(prog="storecheck")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static", help="run the static tier")
    s.add_argument("store")
    s.add_argument("--taxonomy")
    s.add_argument("--json", dest="json_out")
    s.add_argument("--fail-on", choices=["WARN", "ERROR"], default="ERROR")
    d = sub.add_parser("diff", help="diff two store versions")
    d.add_argument("before")
    d.add_argument("after")
    c = sub.add_parser("compare", help="gate a store edit on fixed labeled cases")
    c.add_argument("before", help="baseline data or snapshot descriptor file")
    c.add_argument("after", help="candidate data or snapshot descriptor file")
    c.add_argument(
        "--input-format",
        choices=["jsonl", "snapshot"],
        default="jsonl",
        help="local example JSONL or external snapshot descriptor JSON",
    )
    c.add_argument("--config", required=True, help="evaluation TOML configuration")
    c.add_argument(
        "--json",
        dest="json_out",
        help="write the full comparison artifact; - writes JSON to stdout",
    )
    a = p.parse_args(argv)

    try:
        return _run(a)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"storecheck: {exc}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace) -> int:
    if a.cmd == "compare":
        read = Snapshot.from_json if a.input_format == "snapshot" else Store.from_jsonl
        result = compare_from_config(read(a.before), read(a.after), a.config)
        if a.json_out == "-":
            print(result.to_json())
        else:
            if a.json_out:
                result.to_json(a.json_out)
            print(result.summary())
        return result.exit_code
    if a.cmd == "static":
        store = Store.from_jsonl(a.store)
        tax = Taxonomy.from_json(a.taxonomy) if a.taxonomy else None
        report = run_static(store, tax, HashingEmbedder())
        print(report.summary())
        if a.json_out:
            report.to_json(a.json_out)
        return 0 if report.passed(Severity[a.fail_on]) else 1
    if a.cmd == "diff":
        diff = Store.from_jsonl(a.before).diff(Store.from_jsonl(a.after))
        print(
            f"added={len(diff.added)} removed={len(diff.removed)} "
            f"changed={len(diff.changed)} "
            f"scopes={sorted(map(str, diff.touched_scopes))}"
        )
        for r in diff.added:
            print(f"+ {r.id} [{r.label}] {r.text[:60]}")
        for r in diff.removed:
            print(f"- {r.id} [{r.label}] {r.text[:60]}")
        for b, af in diff.changed:
            print(f"~ {b.id} [{b.label} -> {af.label}] {af.text[:60]}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
