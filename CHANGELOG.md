# Changelog

## [Unreleased]

### Changed

- The release-review API now takes `Pipeline(run, identity)` and calls the whole
  application for every case on both revisions. The custom factory contract now
  returns that pipeline instead of separate retrieval/decision adapters.
- Comparison artifacts use schema version 2, record application traces and
  snapshot descriptors, and count pipeline calls instead of inferred model calls.

- Adopt py-canon v1.3.0: uv, src layout, Python >=3.12, Ruff, Pyright,
  pydoclint, coverage, shared documentation, and reusable workflow shims.
- Store versions and diffs include full serialized row content. Metadata is
  immutable JSON data; embeddings are detached and read-only. Version hashes change.
- Text edits invalidate embeddings; generated vectors record their embedder identity.
- Custom retrievers accept the scope keyword. Backend TypeError exceptions propagate.
- Required missing slices, empty evaluations, and non-finite metrics fail the gate.
- Validate numeric settings, embedding output shape, and evaluation error costs.
- Describe majority-correct evidence and mutation expectations as diagnostics.

### Fixed

- JSONL imports preserve unknown creation times, keeping repeated reads stable.
- Reject strings where serialized slice and boundary-example collections are required.

- Retrieval cache order dependence and mutation of cached lists.
- Stale decisions after evidence metadata changes.
- Ignored empty cases, checks, and PII pattern selections.
- Scope-blind coverage and empty-store coverage handling.
- Unchecked reuse of baseline evaluation runs.
- Influence screening for retrievers with global deletion effects.
- CLI tracebacks for ordinary input errors and nonstandard NaN report JSON.

### Added

- External `Snapshot` descriptors and CLI `--input-format snapshot` for applications
  that own database/index revisions. `Query` omits reference answers and evaluation
  settings; `Prediction` preserves final evidence and application trace.
- A runnable SQLite BM25/vector/fusion example and integration tests for snapshot
  routing, unchanged-evidence regressions, and validation separation.

- TOML-driven `storecheck compare` and a shared Python comparison API with
  immutable JSON artifacts, input/configuration identity, paired evidence,
  slice denominators, error costs, and explicit unevaluable results.
- Custom adapter factories, cost-sensitive regression caps, slice ceilings,
  and minimum validation slice sizes.
- Grouped discovery/validation splits and overlap checks for IDs, normalized
  duplicate text, and declared metadata groups.

- Regression tests, installed-wheel validation workflow, and Docker checks.
