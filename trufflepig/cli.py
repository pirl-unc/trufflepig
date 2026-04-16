"""Command-line entry point.

Each sub-command corresponds to a single pipeline stage. The umbrella
``run`` command invokes the whole DAG against a workspace directory.

Examples:

    trufflepig run --sample path/to/tpm.tsv --workspace /tmp/sample_X
    trufflepig stage sample_context --workspace /tmp/sample_X
    trufflepig stage render_brief --workspace /tmp/sample_X

The stages themselves are intentionally thin wrappers around pirlygenes
functions — no analytical logic should live here. The point of this
repo is to be the orchestration + serialization + CLI layer so a
website frontend can stream stage outputs back to a user incrementally.
"""

from __future__ import annotations

import argparse
import sys

from .version import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trufflepig")
    parser.add_argument("--version", action="version", version=f"trufflepig {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    # `trufflepig run` — full pipeline
    run_p = sub.add_parser(
        "run",
        help="Run the full analysis pipeline against a sample.",
    )
    run_p.add_argument("--sample", required=True, help="Path to sample TPM TSV.")
    run_p.add_argument("--workspace", required=True, help="Output workspace directory.")
    run_p.add_argument("--cancer-type", default=None, help="Optional cancer-type hint.")

    # `trufflepig stage <name>` — run a single stage
    stage_p = sub.add_parser(
        "stage",
        help="Run a single pipeline stage against an existing workspace.",
    )
    stage_p.add_argument("name", help="Stage name (see `trufflepig list-stages`).")
    stage_p.add_argument("--workspace", required=True, help="Workspace directory.")

    # `trufflepig list-stages` — print the DAG
    sub.add_parser("list-stages", help="List pipeline stages and their dependencies.")

    args = parser.parse_args(argv)

    if args.cmd == "list-stages":
        from .pipeline import STAGE_ORDER, STAGE_DEPS

        for stage in STAGE_ORDER:
            deps = STAGE_DEPS[stage]
            dep_str = ", ".join(deps) if deps else "(root)"
            print(f"{stage:20s}  <- {dep_str}")
        return 0

    # The actual stage implementations will be wired in subsequent
    # migration issues (pirl-unc/trufflepig#2..#13). For now this is a
    # scaffold — running a stage raises so the follow-up issues have a
    # clear hook.
    if args.cmd in ("run", "stage"):
        print(
            "[trufflepig] stage implementations are not wired yet — this "
            "repo is the scaffolded target for the pirlygenes.analyze "
            "migration. See pirl-unc/trufflepig issues.",
            file=sys.stderr,
        )
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
