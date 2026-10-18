# storecheck

[![PyPI](https://img.shields.io/pypi/v/storecheck.svg)](https://pypi.org/project/storecheck/)
[![Python](https://img.shields.io/pypi/pyversions/storecheck.svg)](https://pypi.org/project/storecheck/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)

Review changes to the examples used by a retrieval-augmented classifier before
releasing them. See which cases improve, which regress, and whether the proposed
revision meets your application's release requirements.

Our working primary customer is the application engineer responsible for routing
support tickets. Support operations maintains the labeled examples; the engineer
owns the release decision. This customer hypothesis has not been validated with
customer research.

The job: **When support operations changes the examples, help me decide whether
to release that revision, show me regressions, and provide evidence for review.**
Package design starts from that job, the user's circumstances, and the outcome
that matters. Checks and integrations must justify their role in that workflow.

This is a local prototype. Install from this checkout; a hosted repository,
documentation site, and PyPI release have not been verified. Production use needs
your actual retrieval and classification pipeline, labeled evaluation cases,
and immutable baseline/candidate snapshots. Storecheck produces review evidence;
your infrastructure builds indexes and deploys the approved snapshot.

## Work through a release review

Python 3.12 or later and [uv](https://docs.astral.sh/uv/) are required for development.

```bash
uv sync --all-groups --frozen
uv run python examples/release_review.py --output-dir /tmp/storecheck-review
```

Support operations adds `SSO login access denied` as an authentication example.
It fixes a login case but also captures a permissions case that was previously
routed correctly. Schema and label checks pass. Overall discovery accuracy stays
at 2/3, while illustrative error cost rises from 1 to 50. The release gate rejects
the proposal and shows the changed retrieved evidence.

The scripted reviewer clarifies the example to
`SSO identity provider rejects credentials before session starts`. Discovery
improves to 3/3 with no newly broken cases. The script then freezes that revision
and evaluates eight separate validation cases once: accuracy improves from 7/8
to 8/8, with no regressions. A failed discovery review leaves validation unopened.

The output directory contains the source snapshots, label definitions, discovery
and validation cases, and three JSON comparison artifacts. The final artifact
identifies the exact candidate tested. The script exits zero when the final gate
passes; the rejected first proposal is intentional. It does not deploy anything.

These are synthetic cases and illustrative business costs. The example uses
local hashing vectors and nearest-label classification so it runs without keys.
It demonstrates the review workflow, not production quality, hybrid-search
behavior, statistical confidence, or worst-case protection. In a real application,
run the same review against your production pipeline and keep the validation
cases out of edit selection.

## Try the comparison CLI

```bash
uv run storecheck compare examples/compare/baseline.jsonl examples/compare/candidate.jsonl \
  --config examples/compare/evaluation.toml --json /tmp/storecheck-comparison.json
```

This smaller example deliberately exits **1**: relabeling a permission example
breaks one of three validation cases. To see a passing gate, use `baseline.jsonl`
for both store arguments. For individual diagnostic checks and row influence,
run `uv run python examples/quickstart.py`.

## Configure a comparison

The shipped `examples/compare/evaluation.toml` uses this structure. Data paths
are relative to the TOML file; store arguments are relative to the working directory.

```toml
schema_version = 1
evalset = "validation.jsonl"
discovery = "discovery.jsonl"
group_key = "customer"
taxonomy = "taxonomy.json"

[pipeline]
kind = "reference"
k = 1

[objective]
metric = "accuracy"
max_right_to_wrong = 0
max_right_to_wrong_cost = 0

[objective.min_slice_metric]
authorization = 1.0

[objective.min_slice_cases]
authorization = 1
```

`compare` exits **0** when all requirements pass, **1** when an edit fails a
requirement, and **2** for invalid input or an unevaluable gate. Empty evaluations,
missing required slices, insufficient slice samples, and non-finite primary
metrics cannot pass. `--json -` writes only JSON to standard output.

`objective.metric` supports `accuracy` (higher is better) and
`weighted_error_cost` (lower is better). `min_primary_delta` is the required
improvement in the selected direction and defaults to zero. `min_slice_metric`
sets floors; `max_slice_metric` sets ceilings, useful for error cost.
`max_right_to_wrong_cost` caps the total cost of newly broken cases even when
overall accuracy improves. `max_right_to_wrong_frac` caps newly broken cases as
a fraction of all evaluation cases. Error costs are caller-supplied weights,
not API spending estimates.

The reference pipeline uses hashing and majority vote. It accepts `k`,
`dimension`, `scope_aware`, and `fallback`. To evaluate your production pipeline,
set `kind = "custom"` and `factory = "your_package.adapters:build"` under
`[pipeline]`. The installed Python factory receives the `[pipeline.options]`
table and returns `storecheck.Pipeline(run, identity)`. The synchronous `run`
callable receives a `Query` and either a `Store` or external `Snapshot`, and returns
`Prediction(label, evidence, trace)`. It owns the whole application: hybrid search,
filters, fusion, reranking, context formatting, and classification. Storecheck
calls it once per case per revision and never reuses a prediction.

`Query` exposes text, scope, and metadata, without the reference label, cost, or
evaluation slice tags. Metadata must contain only runtime inputs. `Prediction`
retains the actual final evidence as ordered `Row` objects; its optional JSON
trace can record scores, formatted context, or provider revisions. If evidence
is unavailable, an empty tuple is allowed and outcome comparison still works,
but the report cannot explain retrieval changes.

Include code, model, prompt, retrieval settings, and sampling settings in the
nonsecret pipeline identity. Read credentials from the application's environment.
A custom factory executes Python code from that module. Application errors abort
the run; they are not scored as successes or silently retried.

## Gate an edit in Python

```python
from storecheck import (
    Case,
    EmbeddingRetriever,
    EvalSet,
    HashingEmbedder,
    MajorityVoteDecider,
    Objective,
    Pipeline,
    Prediction,
    Row,
    Store,
    compare_stores,
)

baseline = Store([Row("password rejected", "authentication", id="login")])
candidate = baseline.edit("login", label="billing")
cases = EvalSet([Case("login-case", "password rejected", "authentication")])
retriever = EmbeddingRetriever(HashingEmbedder(), k=1)
decider = MajorityVoteDecider()


def application(query, snapshot):
    rows = retriever.retrieve(query.text, snapshot, scope=query.scope)
    return Prediction(decider.decide(query.text, rows), tuple(rows))


report = compare_stores(
    baseline,
    candidate,
    cases,
    pipeline=Pipeline(
        application,
        {"retriever": "hashing-bow-v2:1024:k1", "decider": "majority-v1"},
    ),
    objective=Objective(max_right_to_wrong=0),
)
print(report.summary())
assert report.exit_code == 1
```

`compare_stores()` returns the same `ComparisonReport` as the CLI. Use
`to_dict()` to inspect it and `to_json(path)` to save it. Custom primary metrics
are supported through `Objective`; include their configuration in
`pipeline.identity["metric"]`. The lower-level `RegressionCheck` and
`storecheck.testing.assert_clean()` remain available for collected check reports.

## Evaluate database snapshots and hybrid search

You can pass `Snapshot(name, revision, metadata)` instead of loading every row
into a `Store`. The application must route to that exact, immutable revision of
both data and indexes. A descriptor might be:

```json
{
  "name": "support-tickets",
  "revision": "release-2026-10-17-001",
  "metadata": {"collection": "support-2026-10-17-001"}
}
```

With an installed custom application factory configured in TOML:

```bash
uv run storecheck compare baseline.json candidate.json \
  --input-format snapshot --config evaluation.toml --json comparison.json
```

Storecheck records these external identities but cannot verify a service's
contents or enforce its routing. Pin immutable versions instead of aliases such
as `latest`. Schema/duplicate checks on the entire collection and local row
influence still require a `Store`; a remote descriptor does not load its rows.

A runnable database example is included:

```bash
uv run python examples/hybrid_review.py --output-dir /tmp/storecheck-hybrid
```

It creates two separate SQLite databases, runs BM25 and hash-vector retrieval,
fuses their rankings, and predicts the top example's label. The whole application
is supplied as one callable. The proposed addition breaks the permissions case;
the command deliberately exits **1**, showing 3/3 → 2/3 accuracy and error cost
0 → 50. Its report includes both branch rankings, fusion scores, and final evidence.
The generated directory retains the databases, descriptors, and report. This
example checks each database's file digest and refuses modified snapshots.

The lexical branch uses [SQLite FTS5 BM25](https://www.sqlite.org/fts5.html#the_bm25_function).
The fusion step uses [reciprocal-rank fusion](https://qdrant.tech/documentation/search/hybrid-queries/).
The vector branch uses toy hashing vectors; this demonstrates integration and
observed regressions, not production embedding quality or scalable indexing.

## Inspect files from the command line

Write a store with `Store.to_jsonl()` and label definitions with
`Taxonomy.to_json()`, then run:

```bash
uv run storecheck static store.jsonl --taxonomy taxonomy.json --json report.json
uv run storecheck static store.jsonl --taxonomy taxonomy.json --fail-on WARN
uv run storecheck diff baseline.jsonl candidate.jsonl
```

`static` exits 0 when findings stay below the selected threshold, 1 when findings
reach it, and 2 for invalid input. Warnings do not fail the default gate. Omitting
the taxonomy therefore does not make the command fail, but leaves label validity
unchecked. `diff` prints additions, removals, and edits; differences alone do not
produce a failing exit status. Influence checks use the Python API. There is no
web interface.

## What the checks establish

| Layer | Available checks | Interpretation |
| --- | --- | --- |
| Static | Schema, taxonomy validity, exact and near duplicates, staleness, identifier patterns, neighborhood purity, scope/label counts | Finds review candidates. Similarity and balance do not establish prediction quality. Embedding checks can invoke an embedder. |
| Dynamic | Regression, row influence, coverage, unused rows, majority-correct evidence | Measures outcomes or retrieval on the supplied queries. A finite sample does not establish behavior on all future traffic. |
| Mutation | Label flips, redundancy, region removal, label shuffling, custom mutations | Tests caller-declared expectations. A default mutation need not damage every classifier. |

The release-review API combines `Pipeline`, `Store` or `Snapshot`, `EvalSet`,
and `Objective`. `Query` carries runtime inputs and `Prediction` records the
application output. Stores retain immutable rows. Store versions cover
serialized content, including provenance and embeddings; row order is preserved
for retrieval and included in retrieval cache identity. JSONL imports preserve
missing creation times as unknown; the age check reports those rows separately.
Text edits discard stale
embeddings. Generated vectors record the embedder identity and are recomputed
when that identity changes. External vectors without an identity remain the
caller's responsibility.

Custom retrievers implement
`retrieve(query, store, k=None, *, scope=None) -> list[Row]`. Custom deciders
implement `decide(query, examples) -> str`; `CallableDecider` wraps a function.
These separate retriever/decider interfaces serve the lower-level `Evaluator`
and row diagnostics. That evaluator memoizes deterministic decisions and must
be recreated when configuration changes. The primary `compare_stores()` API
uses the whole application callable and performs every paired call.

## Limits and next steps

Use discovery cases to choose an edit, freeze that choice, then validate it on
separate cases. `EvalSet.split(seed=0, group_key="customer")` keeps normalized
duplicate text and shared metadata groups together, including transitive links.
The fraction applies to groups, so case counts can be uneven. Without a group key,
the split still keeps duplicate text together. A single connected group cannot
produce two nonempty independent sets.

Pass `discovery` and optionally `group_key` to `compare_stores()`, or set them in
TOML, to reject overlap in IDs, normalized text, and declared groups. Omit them
only when independence is checked elsewhere; the artifact records whether these
inputs were supplied. This does not detect all semantic near-duplicates, preserve
rare slices, or protect a holdout reused to select many edits. Minimum slice
counts make insufficient validation data visible. Row influence suggests edits
for discovery; it does not justify automatic deletion.

Run IDs identify ordered input content, objective, taxonomy, supplied pipeline
identity, and runtime versions. They exclude timing and predictions. External snapshots are recorded as
descriptors; local snapshots are fingerprinted from ordered serialized rows.
Neither a declared application identity nor an external descriptor proves the
service honored that configuration. Each case runs once per revision, so a
stochastic difference cannot be attributed solely to the edit. Repeated runs
and uncertainty estimates remain future work. JSON comparison artifacts use
schema version 2 and count application calls; internal model calls are not known. The reference similarity checks
form dense pairwise matrices, and influence can rerun retrieval once per row;
production-scale indexes are also future work.

## Development

```bash
make ci
make ci-docker PYTHON=3.12
make ci-docker PYTHON=3.14
```

The project uses py-canon's uv build backend, src layout, Ruff, Pyright,
pydoclint, pytest coverage, and shared Sphinx/workflow configuration. The local
coverage floor is 85%. `uv.lock` pins dependencies; docs include this README.

## Method references

The production-readiness rubric follows
[The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/).
Row deletion is a finite intervention on a fixed store; it differs from both
[influence-function approximations](https://proceedings.mlr.press/v70/koh17a.html)
and [Data Shapley](https://proceedings.mlr.press/v97/ghorbani19c.html).
Evidence order belongs in the evaluation contract because few-shot predictions
can depend on it. [Lu et al., 2022](https://aclanthology.org/2022.acl-long.556/).
