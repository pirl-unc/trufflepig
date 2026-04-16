"""Pipeline stages.

Each stage is a thin wrapper around a pirlygenes entry point that reads
its input records from the workspace, calls the underlying library code,
writes its output records, and returns a short status dict for the CLI
to echo.

Stage order (DAG):

    load_expression      →  parses sample TSV/TPM into a canonical frame
    sample_context       →  library prep, preservation, degradation
    analyze              →  cancer-type call + purity (wraps analyze_sample)
    decompose            →  compartment fit (wraps decompose_sample)
    ranges               →  per-target expression ranges + #108 attribution
    confidence           →  compute_purity_confidence + target tiers
    render_targets       →  writes targets.md (pirlygenes renderer)
    render_summary       →  summary.md
    render_analysis      →  analysis.md
    render_provenance    →  provenance.md + provenance.png
    render_brief         →  brief.md + actionable.md
    bundle               →  collects figures into PDF, writes meta

The DAG is declared in :mod:`trufflepig.pipeline`, not hard-coded here,
so a user (or website) can invoke a single sub-command with the minimal
set of upstream records.
"""
