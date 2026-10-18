import json

import pytest

from storecheck import Case, EvalSet, Row, Store
from storecheck.__main__ import main


def files(tmp_path, *, candidate_label="a", extra=""):
    baseline = Store([Row("password", "a", id="row")])
    baseline.to_jsonl(str(tmp_path / "before.jsonl"))
    baseline.edit("row", label=candidate_label).to_jsonl(str(tmp_path / "after.jsonl"))
    EvalSet([Case("c", "password", "a")]).to_jsonl(str(tmp_path / "eval.jsonl"))
    config = tmp_path / "eval.toml"
    config.write_text(
        'schema_version = 1\nevalset = "eval.jsonl"\n'
        '[pipeline]\nkind = "reference"\nk = 1\n' + extra
    )
    return [
        "compare",
        str(tmp_path / "before.jsonl"),
        str(tmp_path / "after.jsonl"),
        "--config",
        str(config),
    ]


@pytest.mark.parametrize(
    ("label", "status", "word"), [("a", 0, "PASS"), ("b", 1, "FAIL")]
)
def test_compare_cli_pass_fail_and_json(tmp_path, capsys, label, status, word):
    args = files(tmp_path, candidate_label=label)
    output = tmp_path / "out.json"
    assert main([*args, "--json", str(output)]) == status
    assert word in capsys.readouterr().out
    data = json.loads(output.read_text())
    assert data["status"] == word.lower()
    assert data["manifest"]["pipeline"]["kind"] == "reference"
    assert data["overall"]["n"] == 1


@pytest.mark.parametrize("extra", ["bogus = 1\n", 'scope_aware = "yes"\n', "k = 0\n"])
def test_compare_cli_bad_config_is_actionable(tmp_path, capsys, extra):
    args = files(tmp_path, extra=extra)
    assert main(args) == 2
    assert "storecheck:" in capsys.readouterr().err


def test_compare_cli_missing_slice_is_unevaluable(tmp_path, capsys):
    args = files(tmp_path, extra="[objective.min_slice_metric]\nmissing = 0.8\n")
    assert main(args) == 2
    assert "UNEVALUABLE" in capsys.readouterr().out


def test_compare_cli_custom_factory(tmp_path, monkeypatch, capsys):
    args = files(tmp_path)
    module = tmp_path / "sample_adapter.py"
    module.write_text(
        "from storecheck import Pipeline, Prediction\n"
        "from storecheck import EmbeddingRetriever, HashingEmbedder\n"
        "from storecheck import MajorityVoteDecider\n"
        "def build(options):\n"
        '    retriever = EmbeddingRetriever(HashingEmbedder(), k=options["k"])\n'
        "    decider = MajorityVoteDecider()\n"
        "    def run(query, snapshot):\n"
        "        rows = retriever.retrieve(query.text, snapshot, scope=query.scope)\n"
        "        return Prediction(decider.decide(query.text, rows), tuple(rows))\n"
        '    return Pipeline(run, {"revision": "test-v1"})\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "eval.toml").write_text(
        'schema_version = 1\nevalset = "eval.jsonl"\n[pipeline]\n'
        'kind = "custom"\nfactory = "sample_adapter:build"\n'
        "[pipeline.options]\nk = 1\n"
    )
    assert main(args) == 0
    assert "PASS" in capsys.readouterr().out


def test_json_stdout_and_cost_metric(tmp_path, capsys):
    args = files(
        tmp_path,
        candidate_label="b",
        extra=(
            '[objective]\nmetric = "weighted_error_cost"\nmax_right_to_wrong_cost = 0\n'
        ),
    )
    assert main([*args, "--json", "-"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["manifest"]["objective"]["higher_is_better"] is False
    assert result["overall"]["candidate"]["primary"] == 1


def test_cli_rejects_discovery_overlap(tmp_path, capsys):
    args = files(tmp_path)
    config = tmp_path / "eval.toml"
    config.write_text(
        config.read_text().replace("[pipeline]", 'discovery = "eval.jsonl"\n[pipeline]')
    )
    assert main(args) == 2
    assert "overlap" in capsys.readouterr().err


@pytest.mark.parametrize(
    "config",
    [
        "schema_version = 2\n",
        "schema_version = true\n",
        "schema_version = 1\nevalset = 0\n",
        'schema_version = 1\nevalset = "eval.jsonl"\npipeline = 0\n',
        'schema_version = 1\nevalset = "eval.jsonl"\n[objective]\nmetric = "oops"\n',
        'schema_version = 1\nevalset = "eval.jsonl"\n'
        "[objective]\nmin_slice_metric = 2\n",
        'schema_version = 1\nevalset = "eval.jsonl"\n[pipeline]\nkind = "missing"\n',
        'schema_version = 1\nevalset = "eval.jsonl"\n'
        '[pipeline]\nkind = "custom"\nfactory = "no-colon"\n',
        'schema_version = 1\nevalset = "eval.jsonl"\n'
        '[pipeline]\nkind = "custom"\nfactory = "storecheck:no_such_factory"\n',
    ],
)
def test_invalid_configuration_does_not_traceback(tmp_path, capsys, config):
    args = files(tmp_path)
    (tmp_path / "eval.toml").write_text(config)
    assert main(args) == 2
    assert "storecheck:" in capsys.readouterr().err


def test_external_snapshots_use_custom_application_factory(
    tmp_path, monkeypatch, capsys
):
    args = files(tmp_path)
    for name, revision in (("before", "v1"), ("after", "v2")):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps(
                {
                    "name": "tickets",
                    "revision": revision,
                    "metadata": {"collection": f"tickets-{revision}"},
                }
            )
        )
    (tmp_path / "snapshot_application.py").write_text(
        "from storecheck import Pipeline, Prediction\n"
        "def build(options):\n"
        "    def run(query, snapshot):\n"
        '        label = "a" if snapshot.revision == "v1" else "b"\n'
        '        trace = {"collection": snapshot.metadata["collection"]}\n'
        "        return Prediction(label, trace=trace)\n"
        '    return Pipeline(run, {"app_revision": "1", "model": options["model"]})\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "eval.toml").write_text(
        'schema_version = 1\nevalset = "eval.jsonl"\n[pipeline]\n'
        'kind = "custom"\nfactory = "snapshot_application:build"\n'
        '[pipeline.options]\nmodel = "test-model"\n'
    )
    assert main([*args, "--input-format", "snapshot", "--json", "-"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 2
    assert report["execution"]["pipeline_calls"] == 2
    assert report["manifest"]["pipeline"]["factory"] == "snapshot_application:build"
    assert report["cases"][0]["candidate"]["trace"]["collection"] == "tickets-v2"


def test_reference_pipeline_rejects_external_snapshot(tmp_path, capsys):
    args = files(tmp_path)
    for name in ("before", "after"):
        (tmp_path / f"{name}.jsonl").write_text(
            json.dumps({"name": name, "revision": "v1"})
        )
    assert main([*args, "--input-format", "snapshot"]) == 2
    assert "requires local Store" in capsys.readouterr().err
