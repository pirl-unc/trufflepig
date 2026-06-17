#!/usr/bin/env python3
"""Replay a pirlygenes ``local_reports`` manifest through ``trufflepig run``.

This is the trufflepig-side equivalent of
``pirlygenes/local_reports/regenerate_from_manifest.py`` (kept private to
each researcher's machine — the report directories are gitignored and
contain patient-level data).

Default behaviour:

* Reads the most recent pirlygenes manifest unless ``--source`` is
  given.
* Translates each ``pirlygenes analyze ...`` invocation into
  ``trufflepig run --sample ... --workspace ...`` preserving the same
  arguments.
* Writes every workspace under ``--root`` (default:
  ``$HOME/trufflepig-local-reports/<timestamp>/``) so outputs **never
  land inside the repo**.
* Emits a per-run log, an aggregate manifest, and a Markdown summary
  with the same shape as the pirlygenes harness.

The script reuses the existing pirlygenes ``compare-analyze`` step
where present (each comparison runs through ``trufflepig compare``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MANIFEST = Path(
    "/Users/iskander/code/pirlygenes/local_reports/rnaseq-20260504-130527/manifest.json"
)
DEFAULT_ROOT = Path.home() / "trufflepig-local-reports"


# Flags the pirlygenes CLI exposed that map 1:1 to trufflepig run flags.
# All others (e.g. --output-dir, --output-image-prefix consumed by
# pirlygenes' analyze for its own layout) are absorbed here.
_PASSTHROUGH = {
    "--cancer-type", "--sample-mode", "--tumor-context", "--site-hint",
    "--met-site", "--hla-types", "--fusions", "--alterations",
    "--alignment-qc", "--decomposition-templates", "--output-image-prefix",
    "--sample-id-col", "--sample-id-value", "--gene-id-col",
    "--gene-name-col", "--label-genes", "--genes", "--transcripts",
    "--expression-qc-rescue", "--therapy-target-top-k",
    "--therapy-target-tpm-threshold", "--output-dpi",
    "--plot-height", "--plot-aspect",
}
_FLAGS_NO_VALUE = {
    "--force", "--aggregate-gene-expression",
    "--expression-qc-remove-noncoding", "--deprecated-figures",
    "--no-figures",
}


def _translate_command(name, run, workspace: Path, *, blind: bool = False) -> list[str]:
    """Translate a manifest ``command`` array into a trufflepig invocation.

    Two manifest formats are supported:

    * **legacy** pirlygenes ``analyze``::

        [<python>, '-m', 'pirlygenes.cli', 'analyze',
         '--output-dir', '...', '--cancer-type', 'XXX', input_path]

    * **current** trufflepig ``run`` (what the pirlygenes harness now emits)::

        [<trufflepig>, 'run', '--workspace', '...', '--sample-id-value', 'S',
         '--gene-id-col', 'gene_id', '--sample', input_path, '--no-figures']

    Earlier this function only understood the ``analyze`` form and stripped
    everything up to that token; against a ``run`` command (no ``analyze``
    token) it discarded **every flag**, which silently dropped
    ``--sample-id-value`` (so multi-sample inputs like pfo017 / the NUTM1
    long-format files loaded the wrong column or failed outright) and all the
    clinical-context flags (``--cancer-type`` / ``--fusions`` / ``--hla-types``
    / ``--alterations``). We now locate the sub-command (``analyze`` *or*
    ``run``) and preserve the flags that follow.

    The output goes into ``workspace``; ``--workspace`` / ``--output-dir`` are
    substituted, and the input path comes from ``--sample`` (run form) or the
    trailing positional (analyze form).
    """
    cmd = [str(tok) for tok in run["command"]]

    # Drop the launcher header up to and including the sub-command token.
    sub_idx = next(
        (i for i, tok in enumerate(cmd) if tok in ("analyze", "run")), None
    )
    if sub_idx is not None:
        cmd = cmd[sub_idx + 1:]

    input_path = run.get("input")
    args_after = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        has_value = i + 1 < len(cmd)
        if tok in ("--output-dir", "--workspace"):
            i += 2  # we substitute our own --workspace
            continue
        if tok == "--sample" and has_value:
            input_path = cmd[i + 1]
            i += 2
            continue
        if blind and tok == "--cancer-type":
            # blind mode: drop the clinically-known diagnosis so the report
            # scope is driven entirely by the RNA-inferred top candidate.
            i += 2
            continue
        if tok in _FLAGS_NO_VALUE:
            args_after.append(tok)
            i += 1
            continue
        if tok in _PASSTHROUGH and has_value:
            args_after.extend([tok, cmd[i + 1]])
            i += 2
            continue
        if tok.startswith("--"):
            # Unknown (or value-less trailing) flag → skip it without consuming a
            # following positional we might need (e.g. the input path).
            print(f"[{name}] dropping unsupported flag {tok!r}", file=sys.stderr)
            i += 2 if has_value else 1
            continue
        # positional — that's the input file (legacy analyze form)
        if input_path is None:
            input_path = tok
        i += 1

    out = [
        sys.executable, "-m", "trufflepig.cli", "run",
        "--workspace", str(workspace),
        "--sample", str(input_path),
        *args_after,
    ]
    # Bulk sweeps for the cancer-type / target / markdown-consistency review need
    # only markdown + TSV; figure rendering is ~62% of analyze runtime (matplotlib
    # text layout), so skip it unless the manifest explicitly requested figures.
    if "--no-figures" not in out and "--deprecated-figures" not in out:
        out.append("--no-figures")
    return out


def _translate_comparison(comp, workspace: Path, run_dirs: dict[str, Path]) -> list[str]:
    """Translate a `pirlygenes compare-analyze` command to `trufflepig compare`."""
    cmd = list(comp["command"])
    while cmd and cmd[0] != "compare-analyze":
        cmd.pop(0)
    if cmd and cmd[0] == "compare-analyze":
        cmd.pop(0)

    output_dirs_arg = cmd.pop(0) if cmd else ""
    raw_dirs = [d.strip() for d in output_dirs_arg.split(",") if d.strip()]
    # Map raw pirlygenes paths back to our regenerated workspace paths
    # when names match. Otherwise pass through.
    resolved = []
    for d in raw_dirs:
        run_name = Path(d).name
        resolved.append(str(run_dirs.get(run_name, d)))

    title = "Local comparison"
    i = 0
    while i < len(cmd):
        if cmd[i] == "--title":
            title = cmd[i + 1]
            i += 2
        else:
            i += 1

    return [
        sys.executable, "-m", "trufflepig.cli", "compare",
        "--workspace", str(workspace),
        "--inputs", ",".join(resolved),
        "--title", title,
    ]


def _run(name, cmd, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    elapsed = round(time.time() - start, 1)
    print(f"[{name}] rc={rc} elapsed={elapsed}s log={log_path}")
    return rc, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to a pirlygenes manifest.json (default: %(default)s).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Output root for regenerated workspaces. Default: "
            "$HOME/trufflepig-local-reports/<timestamp>/. Never inside the repo."
        ),
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of run names to regenerate.",
    )
    parser.add_argument(
        "--skip-comparisons",
        action="store_true",
        help="Run only individual analyses; skip compare-analyze.",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help=(
            "Strip --cancer-type from every run so the report scope is driven "
            "by the RNA-inferred top candidate instead of the clinically-known "
            "diagnosis. Use to evaluate blind classification end-to-end."
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        sys.exit(f"Manifest not found: {source}")
    repo_root = Path(__file__).resolve().parents[1]

    with source.open() as f:
        source_manifest = json.load(f)

    root = args.root
    if root is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        root = DEFAULT_ROOT / f"rnaseq-{stamp}"
    root = root.resolve()

    # Guard against accidental in-repo output.
    try:
        root.relative_to(repo_root)
        sys.exit(
            f"Refusing to write reports inside the trufflepig repo: {root}\n"
            "Pass --root pointing somewhere outside the repo."
        )
    except ValueError:
        pass

    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    only = set(args.only.split(",")) if args.only else None

    manifest = {
        "root": str(root),
        "source_manifest": str(source),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tool": "trufflepig",
        "runs": [],
        "comparisons": [],
    }

    run_dirs: dict[str, Path] = {}
    for run in source_manifest.get("runs", []):
        name = run["name"]
        if only is not None and name not in only:
            continue
        if not run.get("input") or not Path(run["input"]).exists():
            print(f"[{name}] skipping — input not present locally")
            manifest["runs"].append({"name": name, "status": "skipped",
                                      "reason": "input missing"})
            continue
        ws = root / name
        run_dirs[name] = ws
        cmd = _translate_command(name, run, ws, blind=args.blind)
        log_path = logs / f"{name}.log"
        rc, elapsed = _run(name, cmd, log_path)
        manifest["runs"].append({
            "name": name,
            "input": run.get("input"),
            "cancer_type": run.get("cancer_type"),
            "workspace": str(ws),
            "command": cmd,
            "log": str(log_path),
            "returncode": rc,
            "status": "ok" if rc == 0 else "failed",
            "elapsed_seconds": elapsed,
        })

    if not args.skip_comparisons:
        for comp in source_manifest.get("comparisons", []):
            name = comp["name"]
            ws = root / name
            cmd = _translate_comparison(comp, ws, run_dirs)
            log_path = logs / f"{name}.log"
            rc, elapsed = _run(name, cmd, log_path)
            manifest["comparisons"].append({
                "name": name,
                "workspace": str(ws),
                "command": cmd,
                "log": str(log_path),
                "returncode": rc,
                "status": "ok" if rc == 0 else "failed",
                "elapsed_seconds": elapsed,
            })

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = ["# trufflepig local-report regeneration", "",
               f"Source: {source}", f"Root: {root}", "",
               "| Run | Status | Seconds |", "|---|---:|---:|"]
    for r in manifest["runs"]:
        summary.append(f"| {r['name']} | {r.get('status', '?')} | "
                       f"{r.get('elapsed_seconds', '-')} |")
    for c in manifest["comparisons"]:
        summary.append(f"| {c['name']} (compare) | {c.get('status', '?')} | "
                       f"{c.get('elapsed_seconds', '-')} |")
    (root / "README.md").write_text("\n".join(summary) + "\n")

    failed = [r for r in manifest["runs"] + manifest["comparisons"]
              if r.get("status") == "failed"]
    print(f"[done] root={root}  ok={len(manifest['runs']) - len(failed)}  "
          f"failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
