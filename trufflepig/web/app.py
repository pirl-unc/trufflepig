"""FastAPI application.

Endpoint summary:

    GET  /                  — landing page (upload form + run list)
    POST /api/run           — start an analyze run (multipart upload)
    POST /api/compare       — start a compare run (form-encoded inputs)
    GET  /api/runs          — list all known runs
    GET  /api/runs/{id}     — run descriptor + current status
    GET  /api/runs/{id}/log — tail of the run log
    GET  /api/runs/{id}/stream
                            — server-sent events stream of progress
                              (each event is one stdout line)
    GET  /api/runs/{id}/report/{name}
                            — render a markdown artifact as HTML
                              (name ∈ summary, analysis, brief,
                              actionable, evidence, comparison, ...)
    GET  /api/runs/{id}/artifact/{path}
                            — raw artifact (PNG/PDF/TSV/MD/JSON)
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .runs import RUN_KIND_ANALYZE, RUN_KIND_COMPARE, RunStore, WebRun, iter_log_events


@dataclasses.dataclass
class WebSettings:
    runs_root: Path
    uploads_root: Path

    @classmethod
    def from_env(cls) -> "WebSettings":
        default_root = Path.home() / "trufflepig-web-runs"
        root = Path(os.environ.get("TRUFFLEPIG_WEB_ROOT", str(default_root)))
        return cls(
            runs_root=root / "runs",
            uploads_root=root / "uploads",
        )


def _markdown_to_html(text: str) -> str:
    import markdown as _md

    return _md.markdown(
        text,
        extensions=["tables", "fenced_code", "toc"],
    )


def create_app(settings: Optional[WebSettings] = None) -> FastAPI:
    settings = settings or WebSettings.from_env()
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    settings.uploads_root.mkdir(parents=True, exist_ok=True)
    store = RunStore(settings.runs_root)

    app = FastAPI(
        title="trufflepig",
        version=__import__("trufflepig").__version__,
        description="RNA tumor analysis web UI",
    )

    static_dir = Path(__file__).parent / "static"
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (static_dir / "index.html").read_text()

    @app.get("/api/runs")
    def api_runs():
        return JSONResponse([r.to_dict() for r in store.list_runs()])

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")
        return run.to_dict()

    @app.get("/api/runs/{run_id}/log")
    def api_run_log(run_id: str, lines: int = 200):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")
        return {"log": run.log_tail(lines)}

    @app.post("/api/run")
    def api_submit_run(
        sample: UploadFile = File(...),
        cancer_type: Optional[str] = Form(None),
        hla_types: Optional[str] = Form(None),
        sample_id_value: Optional[str] = Form(None),
        title: Optional[str] = Form(None),
    ):
        if not sample.filename:
            raise HTTPException(400, "Sample file is required")
        dest = settings.uploads_root / f"{os.urandom(6).hex()}_{sample.filename}"
        with dest.open("wb") as f:
            shutil.copyfileobj(sample.file, f)

        extra: list[str] = []
        if hla_types:
            extra += ["--hla-types", hla_types]
        if sample_id_value:
            extra += ["--sample-id-value", sample_id_value]

        run = store.submit_analyze(
            sample_path=dest,
            cancer_type=cancer_type,
            title=title or sample.filename,
            extra_flags=extra,
        )
        return run.to_dict()

    @app.post("/api/compare")
    def api_submit_compare(
        run_ids: str = Form(...),
        title: Optional[str] = Form(None),
    ):
        ids = [s.strip() for s in run_ids.split(",") if s.strip()]
        if len(ids) < 2:
            raise HTTPException(400, "Provide at least two comma-separated run_ids")
        workspaces: list[Path] = []
        for rid in ids:
            r = store.get(rid)
            if r is None or r.kind != RUN_KIND_ANALYZE:
                raise HTTPException(404, f"Analyze run not found: {rid}")
            workspaces.append(r.workspace)
        run = store.submit_compare(
            workspaces=workspaces,
            title=title or "Comparison",
        )
        return run.to_dict()

    @app.get("/api/runs/{run_id}/stream")
    def api_stream(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")

        def gen():
            for line in iter_log_events(run):
                # Server-sent events format: each event is a `data:` line.
                # Avoid blank lines inside the payload — they'd terminate
                # an SSE event prematurely.
                safe = line.replace("\n", " ").replace("\r", " ")
                yield f"data: {safe}\n\n"
            # Final status push so the client can stop listening.
            status = run.status()
            yield f"event: status\ndata: {status.get('state', 'unknown')}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/report/{name}", response_class=HTMLResponse)
    def api_report(run_id: str, name: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")
        # For analyze runs, reports live under workspace/analyze/<prefix>-<name>.md
        # For compare runs, comparison.md is at workspace/comparison.md.
        candidates: list[Path] = []
        if run.kind == RUN_KIND_COMPARE:
            candidates = [run.workspace / "comparison.md"]
        else:
            analyze_dir = run.workspace / "analyze"
            candidates = list(analyze_dir.glob(f"*-{name}.md"))
            if not candidates:
                candidates = list(analyze_dir.glob(f"{name}.md"))
        candidates = [c for c in candidates if c.is_file()]
        if not candidates:
            raise HTTPException(404, f"No {name}.md in this workspace")
        text = candidates[0].read_text(errors="replace")
        return HTMLResponse(_render_report_page(name, run, text))

    @app.get("/api/runs/{run_id}/artifact/{relpath:path}")
    def api_artifact(run_id: str, relpath: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")
        target = (run.workspace / relpath).resolve()
        try:
            target.relative_to(run.workspace.resolve())
        except ValueError:
            raise HTTPException(400, "Path escapes workspace") from None
        if not target.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(str(target))

    return app


def _render_report_page(name: str, run: WebRun, body_md: str) -> str:
    body = _markdown_to_html(body_md)
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>trufflepig {run.title or run.run_id} — {name}</title>
<link rel="stylesheet" href="/static/style.css">
</head><body>
<header class="report-header">
  <a href="/">&larr; trufflepig</a>
  <strong>{run.title or run.run_id}</strong>
  <code>{run.run_id}</code>
  <span class="badge">{name}</span>
</header>
<main class="markdown-body">
{body}
</main>
</body></html>"""
