"""Smoke tests for the trufflepig CLI.

After the analysis migration (trufflepig#1), `trufflepig run` and
`trufflepig compare` invoke ``trufflepig.main.analyze`` /
``trufflepig.main.compare_analyze`` natively — no bridge to
``pirlygenes`` is left. These tests assert that the CLI passes the
user-supplied arguments through and records native-mode metadata in
the workspace.
"""

import json
import subprocess
import sys


def test_version():
    result = subprocess.run(
        [sys.executable, "-m", "trufflepig.cli", "--version"],
        capture_output=True, text=True, check=True,
    )
    assert "trufflepig" in result.stdout


def test_list_stages_includes_expected_names():
    result = subprocess.run(
        [sys.executable, "-m", "trufflepig.cli", "list-stages"],
        capture_output=True, text=True, check=True,
    )
    for expected in [
        "sample_context", "analyze", "decompose", "ranges",
        "confidence", "render_brief", "render_provenance", "bundle",
    ]:
        assert expected in result.stdout, f"missing stage in list-stages: {expected}"


def test_stage_subcommand_still_scaffolded(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "trufflepig.cli", "stage", "analyze",
            "--workspace", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "not wired" in result.stderr


def test_pipeline_dependencies_are_sound():
    from trufflepig.pipeline import STAGE_ORDER, required_upstream

    seen = set()
    for stage in STAGE_ORDER:
        up = required_upstream(stage)
        for dep in up:
            assert dep in seen, f"{stage} depends on {dep} but {dep} not seen yet"
        seen.add(stage)


def test_run_dispatches_to_native_analyze(tmp_path, monkeypatch):
    from trufflepig import cli

    captured = {}

    def fake_analyze(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("trufflepig.main.analyze", fake_analyze)

    sample = tmp_path / "sample.tsv"
    sample.write_text("gene_id\tTPM\nENSG00000141510\t12.3\n")
    workspace = tmp_path / "ws"

    rc = cli.main([
        "run",
        "--sample", str(sample),
        "--workspace", str(workspace),
        "--cancer-type", "BLCA",
    ])

    assert rc == 0
    assert captured["input_path"] == str(sample)
    assert captured["cancer_type"] == "BLCA"
    assert captured["output_dir"] == str(workspace / "analyze")
    assert (workspace / "analyze").is_dir()

    meta = json.loads((workspace / "meta.json").read_text())
    assert meta["pipeline_mode"] == cli.NATIVE_MODE
    assert meta["command"] == "run"
    assert meta["sample_path"] == str(sample)
    assert meta["analyze_output_dir"] == str(workspace / "analyze")
    assert meta["args"]["cancer_type"] == "BLCA"
    assert meta["trufflepig_version"]
    assert meta["pirlygenes_version"]


def test_run_forwards_optional_pirlygenes_flags(tmp_path, monkeypatch):
    from trufflepig import cli

    captured = {}
    monkeypatch.setattr("trufflepig.main.analyze", lambda **kw: captured.update(kw))

    sample = tmp_path / "sample.tsv"
    sample.write_text("gene_id\tTPM\n")
    workspace = tmp_path / "ws"

    rc = cli.main([
        "run",
        "--sample", str(sample),
        "--workspace", str(workspace),
        "--hla-types", "A*02:01,B*07:02",
        "--fusions", "/tmp/fusions.tsv",
        "--variants", "/tmp/variants.tsv",
        "--alignment-qc", "/tmp/aqc.tsv",
        "--sample-mode", "tumor_bulk",
        "--tumor-context", "primary",
        "--site-hint", "bladder",
        "--met-site", "liver",
        "--force",
    ])
    assert rc == 0
    assert captured["hla_types"] == "A*02:01,B*07:02"
    assert captured["fusions"] == "/tmp/fusions.tsv"
    assert captured["variants"] == "/tmp/variants.tsv"
    assert captured["alignment_qc"] == "/tmp/aqc.tsv"
    assert captured["sample_mode"] == "tumor_bulk"
    assert captured["tumor_context"] == "primary"
    assert captured["site_hint"] == "bladder"
    assert captured["met_site"] == "liver"
    assert captured["force"] is True


def test_compare_dispatches_to_native_compare(tmp_path, monkeypatch):
    from trufflepig import cli

    captured = {}
    monkeypatch.setattr(
        "trufflepig.main.compare_analyze",
        lambda **kw: captured.update(kw),
    )

    # Two prior trufflepig workspaces — `compare` should descend into
    # their analyze/ subdirs since that's where the analyze report
    # artifacts live.
    ws_a = tmp_path / "a"
    (ws_a / "analyze").mkdir(parents=True)
    ws_b = tmp_path / "b"
    (ws_b / "analyze").mkdir(parents=True)
    out_ws = tmp_path / "cmp"

    rc = cli.main([
        "compare",
        "--workspace", str(out_ws),
        "--inputs", f"{ws_a},{ws_b}",
        "--title", "Patient X longitudinal",
    ])

    assert rc == 0
    forwarded = captured["output_dirs"].split(",")
    assert forwarded == [str(ws_a / "analyze"), str(ws_b / "analyze")]
    assert captured["output_path"] == str(out_ws / "comparison.md")
    assert captured["title"] == "Patient X longitudinal"

    meta = json.loads((out_ws / "meta.json").read_text())
    assert meta["pipeline_mode"] == cli.COMPARE_MODE
    assert meta["command"] == "compare"
    assert meta["args"]["inputs"] == [str(ws_a), str(ws_b)]
    assert meta["resolved_inputs"] == [str(ws_a / "analyze"), str(ws_b / "analyze")]


def test_compare_accepts_raw_pirlygenes_output_dirs(tmp_path, monkeypatch):
    """A directory that *isn't* a trufflepig workspace should pass through
    unchanged so users can compare legacy `pirlygenes analyze` runs."""
    from trufflepig import cli

    captured = {}
    monkeypatch.setattr(
        "trufflepig.main.compare_analyze",
        lambda **kw: captured.update(kw),
    )

    raw_a = tmp_path / "raw_a"
    raw_a.mkdir()
    raw_b = tmp_path / "raw_b"
    raw_b.mkdir()
    out_ws = tmp_path / "cmp"

    rc = cli.main([
        "compare",
        "--workspace", str(out_ws),
        "--inputs", f"{raw_a},{raw_b}",
    ])
    assert rc == 0
    assert captured["output_dirs"] == f"{raw_a},{raw_b}"


def test_compare_requires_at_least_two_inputs(tmp_path, monkeypatch, capsys):
    from trufflepig import cli

    monkeypatch.setattr("trufflepig.main.compare_analyze", lambda **kw: None)
    rc = cli.main([
        "compare",
        "--workspace", str(tmp_path / "cmp"),
        "--inputs", str(tmp_path / "only_one"),
    ])
    assert rc == 2
    captured = capsys.readouterr()
    assert "at least two" in captured.err
