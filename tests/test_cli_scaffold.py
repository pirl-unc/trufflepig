"""Smoke tests for the CLI scaffold."""

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


def test_run_is_scaffolded_not_implemented(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "-m", "trufflepig.cli", "run",
            "--sample", "nonexistent.tsv",
            "--workspace", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "scaffolded" in result.stderr or "not wired" in result.stderr


def test_pipeline_dependencies_are_sound():
    from trufflepig.pipeline import required_upstream, STAGE_ORDER

    # Every stage's upstream set must be a subset of stages listed
    # earlier in STAGE_ORDER (no cycles, valid topological order).
    seen = set()
    for stage in STAGE_ORDER:
        up = required_upstream(stage)
        for dep in up:
            assert dep in seen, f"{stage} depends on {dep} but {dep} not seen yet"
        seen.add(stage)
