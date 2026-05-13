"""Command-line entry point.

After the migration from pirlygenes (trufflepig#1), every subcommand
calls into :mod:`trufflepig.main` natively. ``pirlygenes`` ships
gene-set and reference data only.

Shape:

    trufflepig run --sample path/to/quant.sf --workspace out/sample_X --cancer-type BLCA
    trufflepig compare --workspace out/longitudinal --inputs out/A,out/B
    trufflepig data
    trufflepig cancers [--family ...] [--tissue ...] [--details]
    trufflepig plot-cancer-cohorts [--output-prefix ...]
    trufflepig list-stages
    trufflepig stage <name> --workspace ...    # scaffolded; per-stage extraction is trufflepig#2..#13

`run` and `compare` write a ``meta.json`` describing the run. The
`stage` / `list-stages` commands sit on top of the stage DAG in
:mod:`trufflepig.pipeline`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .version import __version__
from .workspace import Workspace

NATIVE_MODE = "trufflepig_analyze_native"
COMPARE_MODE = "trufflepig_compare_native"


def _pirlygenes_version() -> str:
    try:
        from pirlygenes.version import __version__ as v
        return v
    except Exception as exc:  # pragma: no cover - pirlygenes must be installed
        return f"unknown ({exc!r})"


def _add_run_parser(sub):
    p = sub.add_parser(
        "run",
        help="Run the full analysis pipeline against a sample.",
    )
    p.add_argument("--sample", required=True, help="Path to sample TPM TSV / quant file.")
    p.add_argument("--workspace", required=True, help="Output workspace directory.")
    p.add_argument("--cancer-type", default=None)
    p.add_argument("--sample-mode", default="auto")
    p.add_argument("--tumor-context", default="auto")
    p.add_argument("--site-hint", default=None)
    p.add_argument("--met-site", default=None)
    p.add_argument("--hla-types", default=None)
    p.add_argument("--fusions", default=None)
    p.add_argument("--alterations", default=None)
    p.add_argument("--alignment-qc", default=None)
    p.add_argument("--decomposition-templates", default=None)
    p.add_argument(
        "--output-image-prefix",
        default=None,
        help="Override the figure-filename prefix (otherwise derived from sample id).",
    )
    p.add_argument("--sample-id-col", default=None)
    p.add_argument("--sample-id-value", default=None)
    p.add_argument("--gene-id-col", default=None)
    p.add_argument("--gene-name-col", default=None)
    p.add_argument("--label-genes", default=None)
    p.add_argument("--genes", default=None, help="Explicit gene-level input path.")
    p.add_argument("--transcripts", default=None, help="Explicit transcript-level input path.")
    p.add_argument("--aggregate-gene-expression", action="store_true")
    p.add_argument("--expression-qc-rescue", default="auto")
    p.add_argument("--expression-qc-remove-noncoding", action="store_true")
    p.add_argument("--therapy-target-top-k", type=int, default=10)
    p.add_argument("--therapy-target-tpm-threshold", type=float, default=30.0)
    p.add_argument("--output-dpi", type=int, default=300)
    p.add_argument("--plot-height", type=float, default=14.0)
    p.add_argument("--plot-aspect", type=float, default=1.4)
    p.add_argument("--deprecated-figures", action="store_true")
    p.add_argument("--force", action="store_true", help="Ignore advisory output-dir lock.")
    return p


def _add_compare_parser(sub):
    p = sub.add_parser(
        "compare",
        help="Multi-sample longitudinal comparison across analyze workspaces.",
    )
    p.add_argument("--workspace", required=True)
    p.add_argument(
        "--inputs",
        required=True,
        help=(
            "Comma-separated analyze output directories. New trufflepig "
            "workspaces will be auto-resolved to their analyze/ subdir; "
            "raw pirlygenes-style output dirs pass through unchanged."
        ),
    )
    p.add_argument("--title", default="Analyze Sample Comparison")
    return p


def _resolve_analyze_dir(path_str: str) -> Path:
    """Map a workspace path to its analyze subdir if present."""
    p = Path(path_str).expanduser()
    nested = p / "analyze"
    if nested.is_dir():
        return nested
    return p


def cmd_run(args) -> int:
    from .main import analyze

    ws = Workspace.open(args.workspace)
    analyze_dir = ws.root / "analyze"
    analyze_dir.mkdir(parents=True, exist_ok=True)

    sample_path = str(Path(args.sample).expanduser())

    flags = {
        "cancer_type": args.cancer_type,
        "sample_mode": args.sample_mode,
        "tumor_context": args.tumor_context,
        "site_hint": args.site_hint,
        "met_site": args.met_site,
        "hla_types": args.hla_types,
        "fusions": args.fusions,
        "alterations": args.alterations,
        "alignment_qc": args.alignment_qc,
        "decomposition_templates": args.decomposition_templates,
        "output_image_prefix": args.output_image_prefix,
        "sample_id_col": args.sample_id_col,
        "sample_id_value": args.sample_id_value,
        "gene_id_col": args.gene_id_col,
        "gene_name_col": args.gene_name_col,
        "label_genes": args.label_genes,
        "genes": args.genes,
        "transcripts": args.transcripts,
        "aggregate_gene_expression": bool(args.aggregate_gene_expression),
        "expression_qc_rescue": args.expression_qc_rescue,
        "expression_qc_remove_noncoding": bool(args.expression_qc_remove_noncoding),
        "therapy_target_top_k": args.therapy_target_top_k,
        "therapy_target_tpm_threshold": args.therapy_target_tpm_threshold,
        "output_dpi": args.output_dpi,
        "plot_height": args.plot_height,
        "plot_aspect": args.plot_aspect,
        "deprecated_figures": bool(args.deprecated_figures),
        "force": bool(args.force),
    }

    meta = {
        "tool": "trufflepig",
        "trufflepig_version": __version__,
        "pirlygenes_version": _pirlygenes_version(),
        "pipeline_mode": NATIVE_MODE,
        "command": "run",
        "args": {"sample": sample_path, "workspace": str(ws.root), **flags},
        "sample_path": sample_path,
        "analyze_output_dir": str(analyze_dir),
    }
    ws.write_meta(meta)

    analyze(input_path=sample_path, output_dir=str(analyze_dir), **flags)
    return 0


def cmd_compare(args) -> int:
    from .analyze import (
        compute_longitudinal_delta_sets,
        load_analyze_summary_record,
        write_deltas_json,
    )
    from .main import compare_analyze

    ws = Workspace.open(args.workspace)

    raw_inputs = [s.strip() for s in args.inputs.split(",") if s.strip()]
    if len(raw_inputs) < 2:
        print(
            "[trufflepig] compare needs at least two --inputs directories",
            file=sys.stderr,
        )
        return 2

    resolved_inputs = [str(_resolve_analyze_dir(s)) for s in raw_inputs]
    output_path = ws.root / "comparison.md"
    deltas_path = ws.root / "deltas.json"

    meta = {
        "tool": "trufflepig",
        "trufflepig_version": __version__,
        "pirlygenes_version": _pirlygenes_version(),
        "pipeline_mode": COMPARE_MODE,
        "command": "compare",
        "args": {
            "workspace": str(ws.root),
            "inputs": raw_inputs,
            "title": args.title,
        },
        "resolved_inputs": resolved_inputs,
        "comparison_output": str(output_path),
        "deltas_output": str(deltas_path),
    }
    ws.write_meta(meta)

    compare_analyze(
        output_dirs=",".join(resolved_inputs),
        output_path=str(output_path),
        title=args.title,
    )

    # Companion machine-readable record: typed longitudinal observations.
    # Only emit when every input has a parseable summary — print a
    # one-line notice if we have to skip so the user knows deltas.json
    # is missing and why.
    try:
        records = [load_analyze_summary_record(p) for p in resolved_inputs]
    except FileNotFoundError as exc:
        print(
            f"[deltas] skipped: {exc} — comparison.md was still written.",
            file=sys.stderr,
        )
        return 0
    delta_sets = compute_longitudinal_delta_sets(records)
    write_deltas_json(deltas_path, delta_sets)
    total = sum(len(ds.deltas) for ds in delta_sets)
    print(f"[deltas] Wrote {deltas_path} ({total} typed observations)")
    return 0


def cmd_data(args) -> int:
    from .main import print_dataset_info

    print_dataset_info()
    return 0


def cmd_cancers(args) -> int:
    from .main import print_cancer_registry

    print_cancer_registry(
        family=args.family,
        tissue=args.tissue,
        show_all=args.show_all,
        details=args.details,
    )
    return 0


def cmd_plot_cancer_cohorts(args) -> int:
    from .main import plot_cancer_cohorts

    plot_cancer_cohorts(
        output_prefix=args.output_prefix,
        output_dpi=args.output_dpi,
    )
    return 0


def cmd_stage(args) -> int:
    print(
        "[trufflepig] per-stage extraction is not wired yet — `trufflepig run` "
        "currently runs the migrated full pipeline. Track stage extraction in "
        "pirl-unc/trufflepig#2..#13.",
        file=sys.stderr,
    )
    return 2


def cmd_list_stages(args) -> int:
    from .pipeline import STAGE_DEPS, STAGE_ORDER

    for stage in STAGE_ORDER:
        deps = STAGE_DEPS[stage]
        dep_str = ", ".join(deps) if deps else "(root)"
        print(f"{stage:20s}  <- {dep_str}")
    return 0


def cmd_serve(args) -> int:
    """Run the web UI locally (dev convenience)."""
    try:
        import uvicorn
    except ImportError:
        print(
            "[trufflepig] `trufflepig serve` needs the web extras — "
            "install with `pip install 'trufflepig[web]'`.",
            file=sys.stderr,
        )
        return 2
    from .web import create_app

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trufflepig")
    parser.add_argument("--version", action="version", version=f"trufflepig {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    _add_run_parser(sub)
    _add_compare_parser(sub)

    sub.add_parser("data", help="Print dataset summary.")

    cancers_p = sub.add_parser("cancers", help="Print cancer-type registry.")
    cancers_p.add_argument("--family", default=None)
    cancers_p.add_argument("--tissue", default=None)
    cancers_p.add_argument(
        "--no-show-all",
        dest="show_all",
        action="store_false",
        default=True,
        help="Hide rows without expression data.",
    )
    cancers_p.add_argument("--details", action="store_true")

    pc_p = sub.add_parser(
        "plot-cancer-cohorts",
        help="Plot cancer-cohort gene-set distributions across TCGA (no sample needed).",
    )
    pc_p.add_argument("--output-prefix", default=None)
    pc_p.add_argument("--output-dpi", type=int, default=300)

    stage_p = sub.add_parser(
        "stage",
        help="Run a single pipeline stage against an existing workspace.",
    )
    stage_p.add_argument("name")
    stage_p.add_argument("--workspace", required=True)

    sub.add_parser("list-stages", help="List pipeline stages and their dependencies.")

    serve_p = sub.add_parser("serve", help="Run the web UI locally.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)

    dispatch = {
        "run": cmd_run,
        "compare": cmd_compare,
        "data": cmd_data,
        "cancers": cmd_cancers,
        "plot-cancer-cohorts": cmd_plot_cancer_cohorts,
        "stage": cmd_stage,
        "list-stages": cmd_list_stages,
        "serve": cmd_serve,
    }
    fn = dispatch.get(args.cmd)
    if fn is None:
        parser.print_help()
        return 0
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
