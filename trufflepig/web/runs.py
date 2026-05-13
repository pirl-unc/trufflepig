"""Run-state management for the web UI.

A *run* is one workspace produced by ``trufflepig run`` or
``trufflepig compare``. The web app launches each run as a subprocess
so the FastAPI event loop stays responsive, captures stdout + stderr
to a per-run log file, and exposes the workspace + log + status as
plain files. Run state is reconstructed by reading those files —
nothing lives in memory, so a server restart doesn't lose anything.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


RUN_KIND_ANALYZE = "analyze"
RUN_KIND_COMPARE = "compare"


@dataclass(frozen=True)
class WebRun:
    """In-memory view of a single web run.

    All persistent state lives on disk inside :attr:`workspace`; this
    record is reconstructed each time the API hands it to a caller.
    """

    run_id: str
    kind: str           # RUN_KIND_ANALYZE or RUN_KIND_COMPARE
    workspace: Path
    log_path: Path
    status_path: Path
    created_at: float
    title: str = ""

    def status(self) -> dict:
        if not self.status_path.exists():
            return {"state": "pending"}
        try:
            return json.loads(self.status_path.read_text())
        except Exception:
            return {"state": "unknown"}

    def log_tail(self, lines: int = 200) -> str:
        if not self.log_path.exists():
            return ""
        text = self.log_path.read_text(errors="replace").splitlines()
        return "\n".join(text[-lines:])

    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "kind": self.kind,
            "workspace": str(self.workspace),
            "title": self.title,
            "created_at": self.created_at,
            **self.status(),
        }
        return d


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _spawn(
    cmd: list[str],
    log_path: Path,
    status_path: Path,
) -> None:
    _write_status(status_path, {"state": "running", "started_at": time.time()})
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _runner():
        try:
            with log_path.open("w") as f:
                f.write("$ " + " ".join(cmd) + "\n\n")
                f.flush()
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                rc = proc.wait()
            _write_status(
                status_path,
                {
                    "state": "ok" if rc == 0 else "failed",
                    "returncode": rc,
                    "finished_at": time.time(),
                },
            )
        except Exception as exc:  # pragma: no cover - subprocess plumbing
            _write_status(
                status_path,
                {"state": "failed", "error": repr(exc), "finished_at": time.time()},
            )

    threading.Thread(target=_runner, daemon=True).start()


@dataclass
class RunStore:
    """Filesystem-backed registry of web runs."""

    root: Path

    def __post_init__(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def list_runs(self) -> list[WebRun]:
        out: list[WebRun] = []
        for path in sorted(self.root.iterdir(), reverse=True):
            descriptor = path / "run.json"
            if not descriptor.exists():
                continue
            try:
                payload = json.loads(descriptor.read_text())
                out.append(
                    WebRun(
                        run_id=payload["run_id"],
                        kind=payload["kind"],
                        workspace=Path(payload["workspace"]),
                        log_path=Path(payload["log_path"]),
                        status_path=Path(payload["status_path"]),
                        created_at=payload.get("created_at", 0.0),
                        title=payload.get("title", ""),
                    )
                )
            except Exception:
                continue
        return out

    def get(self, run_id: str) -> WebRun | None:
        descriptor = self._run_dir(run_id) / "run.json"
        if not descriptor.exists():
            return None
        payload = json.loads(descriptor.read_text())
        return WebRun(
            run_id=payload["run_id"],
            kind=payload["kind"],
            workspace=Path(payload["workspace"]),
            log_path=Path(payload["log_path"]),
            status_path=Path(payload["status_path"]),
            created_at=payload.get("created_at", 0.0),
            title=payload.get("title", ""),
        )

    def submit_analyze(
        self,
        sample_path: Path,
        cancer_type: str | None = None,
        title: str = "",
        extra_flags: list[str] | None = None,
        runner=None,
    ) -> WebRun:
        run_id = self._new_id()
        run_root = self._run_dir(run_id)
        workspace = run_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = run_root / "run.log"
        status_path = run_root / "status.json"

        cmd = [
            sys.executable, "-m", "trufflepig.cli", "run",
            "--sample", str(sample_path),
            "--workspace", str(workspace),
        ]
        if cancer_type:
            cmd += ["--cancer-type", cancer_type]
        if extra_flags:
            cmd += list(extra_flags)

        descriptor = {
            "run_id": run_id,
            "kind": RUN_KIND_ANALYZE,
            "workspace": str(workspace),
            "log_path": str(log_path),
            "status_path": str(status_path),
            "title": title or sample_path.name,
            "created_at": time.time(),
            "command": cmd,
        }
        (run_root / "run.json").write_text(json.dumps(descriptor, indent=2))

        (runner or _spawn)(cmd, log_path, status_path)
        return self.get(run_id)

    def submit_compare(
        self,
        workspaces: list[Path],
        title: str = "",
        runner=None,
    ) -> WebRun:
        run_id = self._new_id()
        run_root = self._run_dir(run_id)
        workspace = run_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = run_root / "run.log"
        status_path = run_root / "status.json"

        cmd = [
            sys.executable, "-m", "trufflepig.cli", "compare",
            "--workspace", str(workspace),
            "--inputs", ",".join(str(p) for p in workspaces),
        ]
        if title:
            cmd += ["--title", title]

        descriptor = {
            "run_id": run_id,
            "kind": RUN_KIND_COMPARE,
            "workspace": str(workspace),
            "log_path": str(log_path),
            "status_path": str(status_path),
            "title": title or "Comparison",
            "created_at": time.time(),
            "command": cmd,
            "inputs": [str(p) for p in workspaces],
        }
        (run_root / "run.json").write_text(json.dumps(descriptor, indent=2))

        (runner or _spawn)(cmd, log_path, status_path)
        return self.get(run_id)


def iter_log_events(run: WebRun, max_seconds: float = 600.0) -> Iterator[str]:
    """Stream new stdout lines from a run's log as they arrive.

    Yields raw lines (without trailing newline). Terminates when the run
    status moves out of ``running`` OR after ``max_seconds`` of idle wait.
    """
    start = time.time()
    offset = 0
    while True:
        if run.log_path.exists():
            with run.log_path.open("rb") as f:
                f.seek(offset)
                chunk = f.read()
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    offset += len(chunk)
                    for line in text.splitlines():
                        yield line
        status = run.status()
        if status.get("state") in {"ok", "failed", "unknown"}:
            return
        if time.time() - start > max_seconds:
            return
        time.sleep(0.4)
