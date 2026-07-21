"""Tests for the FastAPI web UI.

These tests exercise the API surface end-to-end with a stub subprocess
runner — the real ``trufflepig run`` invocation is replaced with a
synchronous in-process function that writes a small synthetic
workspace. That way the test suite covers the run lifecycle (submit →
status → log → report rendering) without paying for a real analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from trufflepig.web import WebSettings, create_app
from trufflepig.web.runs import RunStore, _write_status


def _make_settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        runs_root=tmp_path / "runs",
        uploads_root=tmp_path / "uploads",
    )


def _post_sample(
    client: TestClient,
    sample_path: Path,
    *,
    filename: str | None = None,
    content_type: str = "text/plain",
    data: dict[str, str] | None = None,
):
    """Submit one multipart sample while deterministically closing its stream."""
    with sample_path.open("rb") as sample_file:
        return client.post(
            "/api/run",
            files={
                "sample": (
                    filename or sample_path.name,
                    sample_file,
                    content_type,
                )
            },
            data=data or {},
        )


def _stub_runner_factory(workspace_writer):
    """Return a runner suitable for ``RunStore.submit_*(runner=...)``.

    The runner ignores the actual ``cmd`` and synchronously writes a
    workspace via ``workspace_writer(workspace_path)`` then sets status
    to ok. The workspace path is inferred from the ``--workspace`` arg
    so we can drive both analyze and compare runs with one helper.
    """

    def _runner(cmd, log_path, status_path):
        # Find --workspace path in cmd
        try:
            ws_idx = cmd.index("--workspace")
            workspace = Path(cmd[ws_idx + 1])
        except (ValueError, IndexError):
            workspace = Path(log_path).parent / "workspace"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("$ " + " ".join(cmd) + "\nstage: load_expression\nstage: analyze\nstage: bundle\n")
        workspace_writer(workspace, cmd)
        _write_status(status_path, {"state": "ok", "returncode": 0, "finished_at": time.time()})

    return _runner


def _write_analyze_workspace(workspace: Path, cmd: list[str]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    analyze = workspace / "analyze"
    analyze.mkdir(parents=True, exist_ok=True)
    (workspace / "meta.json").write_text(
        json.dumps(
            {
                "tool": "trufflepig",
                "pipeline_mode": "trufflepig_analyze_native",
                "command": "run",
            }
        )
    )
    (analyze / "test-summary.md").write_text(
        "# Summary: test\n\n**Cancer call:** BLCA — moderate confidence\n\n"
        "## Top candidate therapies\n\n"
        "- **NECTIN4** — enfortumab vedotin (Approved, urothelial). 200.0 tumor-source bulk TPM.\n"
    )
    (analyze / "test-analysis.md").write_text("# Analysis: test\n\nDetailed analysis content.")
    (analyze / "test-brief.md").write_text("# Brief: test\n\nClinician-facing brief.")


def _write_compare_workspace(workspace: Path, cmd: list[str]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "meta.json").write_text(
        json.dumps({"tool": "trufflepig", "command": "compare"})
    )
    (workspace / "comparison.md").write_text("# Comparison\n\nDeltas across samples.")
    (workspace / "deltas.json").write_text("[]")


def test_root_serves_html(tmp_path):
    app = create_app(_make_settings(tmp_path))
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "trufflepig" in res.text
    assert "<form" in res.text  # upload form present


def test_static_files_served(tmp_path):
    app = create_app(_make_settings(tmp_path))
    client = TestClient(app)
    res = client.get("/static/style.css")
    assert res.status_code == 200
    assert "card" in res.text


def test_runs_empty_initially(tmp_path):
    app = create_app(_make_settings(tmp_path))
    client = TestClient(app)
    res = client.get("/api/runs")
    assert res.status_code == 200
    assert res.json() == []


def test_submit_analyze_run_lifecycle(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        "trufflepig.web.runs._spawn",
        _stub_runner_factory(_write_analyze_workspace),
    )
    app = create_app(settings)
    client = TestClient(app)

    sample = tmp_path / "sample.tsv"
    sample.write_text("gene_id\tTPM\nENSG0000\t12\n")

    res = _post_sample(
        client,
        sample,
        filename="sample.tsv",
        content_type="text/tab-separated-values",
        data={"cancer_type": "BLCA", "title": "BLCA sample"},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    run_id = payload["run_id"]
    assert payload["kind"] == "analyze"
    assert payload["state"] == "ok"
    assert payload["title"] == "BLCA sample"

    # The run appears in the list endpoint.
    listing = client.get("/api/runs").json()
    assert any(r["run_id"] == run_id for r in listing)

    # The run-specific endpoint resolves.
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["state"] == "ok"

    # The log endpoint returns command + stages.
    log = client.get(f"/api/runs/{run_id}/log").json()["log"]
    assert "stage: analyze" in log

    # The report endpoint renders summary.md to HTML.
    rep = client.get(f"/api/runs/{run_id}/report/summary")
    assert rep.status_code == 200
    assert "BLCA" in rep.text
    assert "<table" in rep.text or "<h1" in rep.text

    # Brief is also available.
    brief = client.get(f"/api/runs/{run_id}/report/brief")
    assert brief.status_code == 200

    # An artifact (the raw markdown) can be downloaded.
    art = client.get(f"/api/runs/{run_id}/artifact/meta.json")
    assert art.status_code == 200
    assert json.loads(art.content)["tool"] == "trufflepig"

    # Path traversal is blocked.
    bad = client.get(f"/api/runs/{run_id}/artifact/../../etc/passwd")
    assert bad.status_code in (400, 404)


def test_submit_compare_run(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)

    # First submit two analyze runs so we have something to compare.
    monkeypatch.setattr(
        "trufflepig.web.runs._spawn",
        _stub_runner_factory(_write_analyze_workspace),
    )
    app = create_app(settings)
    client = TestClient(app)

    sample = tmp_path / "sample.tsv"
    sample.write_text("x\n")

    ids = []
    for label in ("a", "b"):
        res = _post_sample(
            client,
            sample,
            filename=f"{label}.tsv",
            data={"title": label},
        )
        ids.append(res.json()["run_id"])

    # Swap the runner for the compare workspace shape and submit.
    monkeypatch.setattr(
        "trufflepig.web.runs._spawn",
        _stub_runner_factory(_write_compare_workspace),
    )
    res = client.post(
        "/api/compare",
        data={"run_ids": ",".join(ids), "title": "A vs B"},
    )
    assert res.status_code == 200, res.text
    cmp_id = res.json()["run_id"]
    assert res.json()["state"] == "ok"

    rep = client.get(f"/api/runs/{cmp_id}/report/comparison")
    assert rep.status_code == 200
    assert "Comparison" in rep.text

    deltas = client.get(f"/api/runs/{cmp_id}/artifact/deltas.json")
    assert deltas.status_code == 200
    assert deltas.json() == []


def test_compare_rejects_missing_run_ids(tmp_path):
    app = create_app(_make_settings(tmp_path))
    client = TestClient(app)
    res = client.post("/api/compare", data={"run_ids": "nonexistent"})
    assert res.status_code == 400 or res.status_code == 404


def test_unknown_run_id_returns_404(tmp_path):
    app = create_app(_make_settings(tmp_path))
    client = TestClient(app)
    assert client.get("/api/runs/notarealid").status_code == 404
    assert client.get("/api/runs/notarealid/log").status_code == 404


# Regression coverage for the security fixes.


def test_upload_filename_traversal_is_neutralized(tmp_path, monkeypatch):
    """`../../etc/passwd` as a filename must not escape uploads_root."""
    settings = _make_settings(tmp_path)
    captured = {}

    def _capture_spawn(cmd, log_path, status_path):
        # Record the --sample arg so we can assert where the upload landed.
        try:
            captured["sample"] = cmd[cmd.index("--sample") + 1]
        except (ValueError, IndexError):
            captured["sample"] = None
        _write_status(status_path, {"state": "ok", "returncode": 0})
        log_path.write_text("ok\n")

    monkeypatch.setattr("trufflepig.web.runs._spawn", _capture_spawn)
    app = create_app(settings)
    client = TestClient(app)

    sample = tmp_path / "src"
    sample.write_text("x\n")
    res = _post_sample(
        client,
        sample,
        filename="../../etc/passwd",
    )
    assert res.status_code == 200, res.text

    # The file landed under uploads_root, with no path components from
    # the malicious name preserved.
    sample_path = Path(captured["sample"])
    sample_path.relative_to(settings.uploads_root.resolve())  # raises if escaped
    assert "etc" not in sample_path.parts
    assert "passwd" in sample_path.name


def test_upload_size_limit_aborts_oversized_payload(tmp_path, monkeypatch):
    """Uploads above the configured cap return 413 and don't keep the partial file."""
    settings = _make_settings(tmp_path)
    settings.max_upload_bytes = 16  # tiny cap for the test
    monkeypatch.setattr("trufflepig.web.runs._spawn", lambda *a, **k: None)
    app = create_app(settings)
    client = TestClient(app)

    big = tmp_path / "big.tsv"
    big.write_bytes(b"A" * 1024)
    res = _post_sample(
        client,
        big,
        filename="big.tsv",
    )
    assert res.status_code == 413
    # No file left behind under uploads_root.
    leftovers = list(settings.uploads_root.glob("*"))
    assert leftovers == []


def test_flag_injection_in_cancer_type_is_rejected(tmp_path, monkeypatch):
    """A form value starting with `-` is rejected as a likely flag injection."""
    settings = _make_settings(tmp_path)
    monkeypatch.setattr("trufflepig.web.runs._spawn", lambda *a, **k: None)
    app = create_app(settings)
    client = TestClient(app)

    sample = tmp_path / "x.tsv"
    sample.write_text("x\n")
    res = _post_sample(
        client,
        sample,
        filename="x.tsv",
        data={"cancer_type": "--force"},
    )
    assert res.status_code == 400
    assert "flag" in res.text.lower()


def test_markdown_html_in_source_is_escaped(tmp_path, monkeypatch):
    """A `<script>` tag in the rendered markdown must not survive into the page."""
    settings = _make_settings(tmp_path)

    def _xss_workspace(workspace, cmd):
        analyze = workspace / "analyze"
        analyze.mkdir(parents=True, exist_ok=True)
        (analyze / "x-summary.md").write_text(
            "# Summary: x\n\n"
            "Normal text. <script>window.__pwned=1</script>"
            ' <img src=x onerror="alert(1)">\n'
        )

    monkeypatch.setattr(
        "trufflepig.web.runs._spawn",
        _stub_runner_factory(_xss_workspace),
    )
    app = create_app(settings)
    client = TestClient(app)
    sample = tmp_path / "x.tsv"
    sample.write_text("x\n")
    rid = _post_sample(client, sample, filename="x.tsv").json()["run_id"]

    page = client.get(f"/api/runs/{rid}/report/summary").text
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    # The literal substring "onerror=" survives inside the escaped img
    # tag (`&lt;img src=x onerror="alert(1)"&gt;`), but no live HTML
    # tag with an executable attribute is emitted.
    assert "<img " not in page  # raw img tag with attribute would be live
    assert "&lt;img " in page


def test_run_title_is_html_escaped_in_report_page(tmp_path, monkeypatch):
    """User-supplied `title` is interpolated safely into the wrapper HTML."""
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        "trufflepig.web.runs._spawn",
        _stub_runner_factory(_write_analyze_workspace),
    )
    app = create_app(settings)
    client = TestClient(app)

    sample = tmp_path / "x.tsv"
    sample.write_text("x\n")
    res = _post_sample(
        client,
        sample,
        filename="x.tsv",
        data={"title": "<script>alert('xss')</script>"},
    )
    rid = res.json()["run_id"]
    page = client.get(f"/api/runs/{rid}/report/summary").text
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_atomic_run_json_write_survives_partial_failure(tmp_path):
    """A crash mid-write must not leave run.json corrupt."""
    from trufflepig.web.runs import _atomic_write_json

    target = tmp_path / "run.json"
    _atomic_write_json(target, {"run_id": "abc", "kind": "analyze"})
    assert json.loads(target.read_text())["run_id"] == "abc"
    # The temp file is gone after rename.
    assert not (target.with_suffix(".json.tmp")).exists()


def test_reconcile_orphaned_runs_relabels_stuck_running(tmp_path):
    """Stuck `running` status from a prior server gets relabelled on startup."""
    from trufflepig.web.runs import _atomic_write_json

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    rid = "stuckrun123"
    rdir = runs_root / rid
    rdir.mkdir()
    _atomic_write_json(
        rdir / "run.json",
        {
            "run_id": rid,
            "kind": "analyze",
            "workspace": str(rdir / "workspace"),
            "log_path": str(rdir / "run.log"),
            "status_path": str(rdir / "status.json"),
            "title": "stuck",
            "created_at": 0.0,
        },
    )
    _atomic_write_json(rdir / "status.json", {"state": "running"})

    store = RunStore(runs_root)
    relabelled = store.reconcile_orphaned_runs()
    assert relabelled == 1
    assert store.get(rid).status()["state"] == "orphaned"


def test_stream_endpoint_emits_events(tmp_path, monkeypatch):
    settings = _make_settings(tmp_path)
    monkeypatch.setattr(
        "trufflepig.web.runs._spawn",
        _stub_runner_factory(_write_analyze_workspace),
    )
    app = create_app(settings)
    client = TestClient(app)

    sample = tmp_path / "sample.tsv"
    sample.write_text("x\n")
    res = _post_sample(
        client,
        sample,
        filename="sample.tsv",
        data={"title": "stream-test"},
    )
    run_id = res.json()["run_id"]

    # The run already finished (stub runner is synchronous), so the
    # stream should emit any buffered log lines then a status event.
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        body = "".join(chunk for chunk in response.iter_text())
    assert "data: " in body
    assert "event: status" in body
    assert "ok" in body
