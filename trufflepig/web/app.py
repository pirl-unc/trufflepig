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
import html
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .runs import (
    RUN_KIND_ANALYZE,
    RUN_KIND_COMPARE,
    RunStore,
    WebRun,
    aiter_log_events,
)


# Maximum upload size in bytes (default 200 MB — typical TPM file is
# 5–50 MB; an upper bound protects against accidental directory uploads
# or hostile clients). Override with TRUFFLEPIG_WEB_MAX_UPLOAD_BYTES.
DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# Filename safety: keep alphanumerics, dot, dash, underscore. Anything
# else is replaced with `_`. Path components are stripped via
# Path(name).name before this runs.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(raw: str) -> str:
    """Reduce a user-supplied filename to its safe leaf form.

    Strips path components (so ``../../etc/passwd`` becomes ``passwd``),
    replaces shell-unfriendly characters, and caps the length. Empty
    or all-dot inputs fall back to ``upload.bin``.
    """
    leaf = Path(raw).name or ""
    cleaned = _SAFE_FILENAME_RE.sub("_", leaf).strip(".")
    if not cleaned:
        cleaned = "upload.bin"
    return cleaned[:128]


def _reject_flag_value(name: str, value: Optional[str]) -> None:
    """Refuse form values that argparse would re-interpret as a flag.

    Anything starting with ``-`` (or with leading whitespace then ``-``)
    could pollute the analyze subprocess's argument parser. We reject
    such values upfront rather than relying on the child to ignore them.
    """
    if value is None:
        return
    stripped = value.lstrip()
    if stripped.startswith("-"):
        raise HTTPException(
            400, f"{name!r} must not start with '-' (looks like a CLI flag)"
        )


@dataclasses.dataclass
class WebSettings:
    runs_root: Path
    uploads_root: Path
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES

    @classmethod
    def from_env(cls) -> "WebSettings":
        default_root = Path.home() / "trufflepig-web-runs"
        root = Path(os.environ.get("TRUFFLEPIG_WEB_ROOT", str(default_root)))
        max_bytes_env = os.environ.get("TRUFFLEPIG_WEB_MAX_UPLOAD_BYTES")
        try:
            max_bytes = int(max_bytes_env) if max_bytes_env else DEFAULT_MAX_UPLOAD_BYTES
        except ValueError:
            max_bytes = DEFAULT_MAX_UPLOAD_BYTES
        return cls(
            runs_root=root / "runs",
            uploads_root=root / "uploads",
            max_upload_bytes=max_bytes,
        )


# Allowed URL schemes for markdown links / autolinks. Anything else
# (``javascript:``, ``data:``, ``vbscript:``, ``file:`` …) gets the
# href rewritten to ``#`` so the rendered page can't carry an active
# payload back to the browser.
_SAFE_URL_SCHEMES = ("http://", "https://", "mailto:", "#", "/", "./", "../")

# Report-name URL whitelist. ``name`` is glob-interpolated against the
# workspace's analyze dir; without this allowlist, ``?`` and ``[``
# would match unintended files. Keep in sync with the names
# trufflepig.main actually emits.
_ALLOWED_REPORT_NAMES = frozenset(
    {"summary", "analysis", "brief", "actionable", "evidence", "comparison"}
)

# File suffixes safe to serve inline. Anything else gets
# ``Content-Disposition: attachment`` so a workspace ``.html``,
# ``.svg``, or ``.xml`` can't render with active content.
_INLINE_SAFE_SUFFIXES = frozenset(
    {"png", "jpg", "jpeg", "gif", "webp", "pdf", "json", "tsv", "csv", "md", "txt"}
)


def _markdown_to_html(text: str) -> str:
    """Render Markdown to safe HTML.

    ``markdown.markdown`` passes raw HTML in the source through as-is
    by default — that's stored XSS waiting to happen when the source
    is derived from user-supplied form fields or filenames. We turn
    off raw-HTML passthrough with ``_HtmlStripPreprocessor`` so any
    literal ``<`` is rendered as ``&lt;``, and we post-render to
    neutralize ``[click](javascript:...)`` / ``<javascript:...>``
    autolinks by replacing any non-allowlisted ``href=`` value.
    """
    import markdown as _md

    md = _md.Markdown(extensions=["tables", "fenced_code", "toc"])
    md.preprocessors.register(_HtmlStripPreprocessor(md), "_trufflepig_strip_html", 30)
    return _strip_unsafe_hrefs(md.convert(text))


def _strip_unsafe_hrefs(html_text: str) -> str:
    """Rewrite ``href="javascript:..."`` (and other unsafe schemes) to ``#``.

    Catches the ``[text](javascript:alert(1))`` markdown form that the
    HTML-block strip preprocessor doesn't see — inline links go
    through the link tree-processor at a later stage. Conservative
    allowlist: only ``http``, ``https``, ``mailto:``, anchor (``#``),
    and same-origin relative paths survive.
    """
    import re

    def _sub(match):
        quote = match.group(1)
        value = match.group(2)
        value_stripped = value.strip()
        lowered = value_stripped.lower()
        if lowered.startswith(_SAFE_URL_SCHEMES):
            return match.group(0)
        # Non-allowlisted scheme — neutralize.
        return f'href={quote}#{quote}'

    return re.sub(r'href=([\'"])([^\'"]*)\1', _sub, html_text, flags=re.IGNORECASE)


class _HtmlStripPreprocessor:
    """Escape raw HTML in markdown source before any extension sees it."""

    def __init__(self, md):
        self.md = md

    def run(self, lines):
        return [html.escape(line, quote=False) for line in lines]


def create_app(settings: Optional[WebSettings] = None) -> FastAPI:
    settings = settings or WebSettings.from_env()
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    settings.uploads_root.mkdir(parents=True, exist_ok=True)
    store = RunStore(settings.runs_root)
    # If a previous server died mid-run, the subprocess thread that
    # would have flipped status.json is gone — relabel those once.
    store.reconcile_orphaned_runs()

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

        # Form values that would parse as CLI flags inside the analyze
        # subprocess get rejected before we shell them out.
        for name, value in (
            ("cancer_type", cancer_type),
            ("hla_types", hla_types),
            ("sample_id_value", sample_id_value),
        ):
            _reject_flag_value(name, value)

        safe_name = _sanitize_filename(sample.filename)
        dest = settings.uploads_root / f"{os.urandom(6).hex()}_{safe_name}"

        # Cap upload size: we stream into the file but abort once we
        # cross the limit so a 10 GB sample doesn't fill the disk.
        written = 0
        chunk_size = 1024 * 1024
        with dest.open("wb") as f:
            while True:
                chunk = sample.file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"Upload exceeds {settings.max_upload_bytes} bytes; "
                        f"raise TRUFFLEPIG_WEB_MAX_UPLOAD_BYTES if intentional.",
                    )
                f.write(chunk)

        extra: list[str] = []
        if hla_types:
            extra += ["--hla-types", hla_types]
        if sample_id_value:
            extra += ["--sample-id-value", sample_id_value]

        run = store.submit_analyze(
            sample_path=dest,
            cancer_type=cancer_type,
            title=title or safe_name,
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
    async def api_stream(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")

        # Cap per-event payload so a single misbehaving log line can't
        # OOM the server (or the client). 4 KB is plenty for stage
        # banners and progress messages.
        _MAX_EVENT_BYTES = 4096

        async def gen():
            async for line in aiter_log_events(run):
                safe = line.replace("\n", " ").replace("\r", " ")
                if len(safe) > _MAX_EVENT_BYTES:
                    safe = safe[:_MAX_EVENT_BYTES] + " …[truncated]"
                yield f"data: {safe}\n\n"
            # Final status push so the client can stop listening. The
            # status comes from disk so the previous "aiter_log_events
            # returned" can be due either to the run finishing or to
            # an idle-timeout — surface both explicitly.
            status = run.status()
            state = status.get("state", "unknown")
            if state == "running":
                yield "event: status\ndata: timeout\n\n"
            else:
                yield f"event: status\ndata: {state}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/report/{name}", response_class=HTMLResponse)
    def api_report(run_id: str, name: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run id")
        # Whitelist the report names trufflepig actually emits — the
        # URL value is glob-interpolated below, and ``?``/``[`` in
        # ``name`` would otherwise match unintended files in the
        # workspace.
        if name not in _ALLOWED_REPORT_NAMES:
            raise HTTPException(400, f"Unknown report name {name!r}")
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
        # Force ``Content-Disposition: attachment`` for anything that
        # isn't an inline-safe image/PDF/JSON/TSV/CSV/MD/TXT. Without
        # this, a workspace ``.html`` or ``.svg`` artifact would render
        # with active content when fetched in the browser.
        suffix = target.suffix.lower().lstrip(".")
        if suffix in _INLINE_SAFE_SUFFIXES:
            return FileResponse(str(target))
        return FileResponse(
            str(target),
            headers={
                "Content-Disposition": f'attachment; filename="{target.name}"'
            },
        )

    return app


def _render_report_page(name: str, run: WebRun, body_md: str) -> str:
    body = _markdown_to_html(body_md)
    # Every piece of user-controlled or arbitrary text that lands in the
    # HTML wrapper gets HTML-escaped — title, name, and run_id all flow
    # through this template and could otherwise carry markup.
    title = html.escape(run.title or run.run_id)
    run_id = html.escape(run.run_id)
    name_safe = html.escape(name)
    return (
        "<!doctype html>\n"
        "<html><head>\n"
        '<meta charset="utf-8">\n'
        f"<title>trufflepig {title} — {name_safe}</title>\n"
        '<link rel="stylesheet" href="/static/style.css">\n'
        "</head><body>\n"
        '<header class="report-header">\n'
        '  <a href="/">&larr; trufflepig</a>\n'
        f"  <strong>{title}</strong>\n"
        f"  <code>{run_id}</code>\n"
        f'  <span class="badge">{name_safe}</span>\n'
        "</header>\n"
        '<main class="markdown-body">\n'
        f"{body}\n"
        "</main>\n"
        "</body></html>\n"
    )
