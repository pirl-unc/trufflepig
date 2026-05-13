"""Tests for the structured longitudinal-delta layer.

Covers ``trufflepig.analyze.deltas`` parsing + computation and the
``deltas.json`` companion written by ``trufflepig compare``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trufflepig.analyze.comparison import (
    AnalyzeSummaryRecord,
    build_analyze_comparison_markdown,
    compute_longitudinal_delta_sets,
    load_analyze_summary_record,
)
from trufflepig.analyze.deltas import (
    LongitudinalDelta,
    LongitudinalDeltaSet,
    ResponseAxisState,
    TargetShortlistEntry,
    compute_pairwise_deltas,
    parse_response_axes,
    parse_target_shortlist,
    write_deltas_json,
)


SUMMARY_A = """# Summary: sample_A

**Cancer call:** BLCA — moderate confidence
**Cancer-type basis:** externally supplied BLCA
**Purity:** 60% (model interval 40%–80%, moderate confidence).
**Sample:** ribosomal depletion library; preservation inferred as unknown.
**RNA quant QC:** 14,000/37,000 genes >=1 TPM; gene TPM sum near 1.0M.
**Technical-RNA normalization:** 1% removed.
**Disease state:** active state.
**Active pathway:** MAPK / ERK activity high (3.5x context).

## Top candidate therapies

- **NECTIN4** — enfortumab vedotin (Approved, advanced urothelial). tumor-supported; 200.0 tumor-source bulk TPM.
- **TACSTD2** — sacituzumab govitecan (Approved, advanced urothelial). tumor-supported; 1000.0 tumor-source bulk TPM.

## Caveats
- Purity range is wide.
"""

SUMMARY_B = """# Summary: sample_B

**Cancer call:** BLCA — moderate confidence
**Cancer-type basis:** externally supplied BLCA
**Purity:** 80% (model interval 60%–90%, moderate confidence).
**Sample:** polyA library; preservation fresh.
**RNA quant QC:** 15,000/37,000 genes >=1 TPM; gene TPM sum near 1.0M.
**Technical-RNA normalization:** 4% removed.
**Disease state:** suppressed state.
**Active pathway:** no summary-level call.

## Top candidate therapies

- **NECTIN4** — enfortumab vedotin (Approved, advanced urothelial). tumor-supported; 400.0 tumor-source bulk TPM.
- **CD274** — pembrolizumab (Approved, advanced urothelial). expression-independent; 13.0 tumor-source bulk TPM.

## Caveats
- Different library prep.
"""

EVIDENCE_A = """### Therapy-state context

**MAPK EGFR signaling** — active. Active signaling: up-panel geomean 3.50x cohort

| Gene | Direction | Sample TPM | Cohort median | Fold | Mechanism |
|------|-----------|------------|---------------|------|-----------|
| CCND1 | up | 1100.0 | 50.0 | 22.00x | MAPK output |

**IFN response** — active. Active signaling: up-panel geomean 2.00x cohort

| Gene | Direction | Sample TPM | Cohort median | Fold | Mechanism |
|------|-----------|------------|---------------|------|-----------|
| OAS1 | up | 250.0 | 28.0 | 8.90x | ISG |

### Cancer-Testis Antigens (Vaccination Targets)
"""

EVIDENCE_B = """### Therapy-state context

**MAPK EGFR signaling** — suppressed. up-panel geomean 0.70x cohort

| Gene | Direction | Sample TPM | Cohort median | Fold | Mechanism |
|------|-----------|------------|---------------|------|-----------|

**IFN response** — active. Active signaling: up-panel geomean 4.00x cohort

| Gene | Direction | Sample TPM | Cohort median | Fold | Mechanism |
|------|-----------|------------|---------------|------|-----------|

**EMT** — active. up-panel geomean 5.20x cohort, down-panel 0.10x

### Next Section
"""


def _write_sample(directory: Path, name: str, summary: str, evidence: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}-summary.md").write_text(summary)
    (directory / f"{name}-evidence.md").write_text(evidence)
    return directory


def test_parse_response_axes_extracts_axis_state_and_fold():
    axes = parse_response_axes(EVIDENCE_A.splitlines())
    assert "MAPK_EGFR_signaling" in axes
    assert "IFN_response" in axes
    assert axes["MAPK_EGFR_signaling"].state == "active"
    assert axes["MAPK_EGFR_signaling"].up_fold == 3.50
    assert axes["IFN_response"].up_fold == 2.00


def test_parse_response_axes_handles_down_panel():
    axes = parse_response_axes(EVIDENCE_B.splitlines())
    assert axes["EMT"].state == "active"
    assert axes["EMT"].up_fold == 5.20
    assert axes["EMT"].down_fold == 0.10


def test_parse_target_shortlist_extracts_gene_drug_tpm():
    entries = parse_target_shortlist(SUMMARY_A.splitlines())
    assert [e.gene for e in entries] == ["NECTIN4", "TACSTD2"]
    nectin = entries[0]
    assert nectin.drug == "enfortumab vedotin"
    assert nectin.tpm == 200.0
    assert nectin.tier == "Approved"


def test_compute_pairwise_deltas_emits_typed_observations(tmp_path):
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)

    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    delta_set = compute_pairwise_deltas(rec_a, rec_b)

    kinds = {d.kind for d in delta_set.deltas}
    # Cancer call is the same, so no cancer_call delta.
    assert "purity" in kinds
    assert "response_axis" in kinds
    assert "target" in kinds  # gain CD274 / no loss in this fixture
    assert "target_tpm" in kinds  # NECTIN4 200 -> 400
    assert "assay_comparability" in kinds  # ribo vs polyA

    # Purity went 60 -> 80
    purity = next(d for d in delta_set.deltas if d.kind == "purity")
    assert purity.direction == "up"
    assert purity.before == 60.0
    assert purity.after == 80.0
    assert purity.magnitude == pytest.approx(20.0)

    # MAPK axis shifted active -> suppressed
    mapk = next(
        d for d in delta_set.deltas
        if d.kind == "response_axis"
        and (d.before or {}).get("axis") == "MAPK_EGFR_signaling"
    )
    assert mapk.direction == "shifted"

    # IFN axis stayed active but up_fold doubled
    ifn = next(
        d for d in delta_set.deltas
        if d.kind == "response_axis"
        and (d.before or {}).get("axis") == "IFN_response"
    )
    assert ifn.direction == "up"
    assert ifn.magnitude == pytest.approx(1.0, rel=1e-3)  # 4.0/2.0 - 1

    # EMT axis is new in B
    emt = next(
        d for d in delta_set.deltas
        if d.kind == "response_axis"
        and ((d.after or {}).get("axis") == "EMT" if d.after else False)
    )
    assert emt.direction == "new"

    # CD274 gained from shortlist
    cd274 = next(
        d for d in delta_set.deltas
        if d.kind == "target"
        and ((d.after or {}).get("gene") == "CD274" if d.after else False)
    )
    assert cd274.direction == "gained"

    # NECTIN4 TPM moved 200 -> 400 → relative delta ~1.0
    nectin = next(
        d for d in delta_set.deltas
        if d.kind == "target_tpm"
        and (d.before or {}).get("gene") == "NECTIN4"
    )
    assert nectin.direction == "up"

    # Assay comparability advisory
    assay = next(d for d in delta_set.deltas if d.kind == "assay_comparability")
    assert assay.before["library"] == "ribo"
    assert assay.after["library"] == "polya"


def test_compute_pairwise_deltas_clears_axis_when_dropped(tmp_path):
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    # B has no Therapy-state context section at all
    dir_b = _write_sample(
        tmp_path / "b", "b", SUMMARY_B,
        "## Some Other Section\n\nNo therapy-state context here.\n",
    )
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    delta_set = compute_pairwise_deltas(rec_a, rec_b)
    cleared = [
        d for d in delta_set.deltas
        if d.kind == "response_axis" and d.direction == "cleared"
    ]
    assert {(d.before or {}).get("axis") for d in cleared} == {
        "MAPK_EGFR_signaling", "IFN_response",
    }


def test_build_comparison_markdown_includes_new_sections(tmp_path):
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)

    md = build_analyze_comparison_markdown([dir_a, dir_b], title="A vs B")
    assert "## Response-Axis State Matrix" in md
    assert "## Notable Longitudinal Observations" in md
    assert "MAPK_EGFR_signaling" in md
    assert "IFN_response" in md
    # Direction glyphs
    assert "↑" in md or "→" in md


def test_write_deltas_json_round_trips(tmp_path):
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    delta_sets = compute_longitudinal_delta_sets([rec_a, rec_b])

    out = tmp_path / "deltas.json"
    write_deltas_json(out, delta_sets)
    payload = json.loads(out.read_text())
    assert isinstance(payload, list)
    assert payload[0]["before_sample"] == "sample_A"
    assert payload[0]["after_sample"] == "sample_B"
    kinds = {d["kind"] for d in payload[0]["deltas"]}
    assert {"purity", "response_axis", "target", "assay_comparability"} <= kinds


def test_cli_compare_emits_deltas_json(tmp_path):
    from trufflepig import cli

    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)
    out_ws = tmp_path / "cmp"

    rc = cli.main([
        "compare",
        "--workspace", str(out_ws),
        "--inputs", f"{dir_a},{dir_b}",
        "--title", "A vs B",
    ])
    assert rc == 0
    assert (out_ws / "comparison.md").is_file()
    assert (out_ws / "deltas.json").is_file()

    meta = json.loads((out_ws / "meta.json").read_text())
    assert meta["deltas_output"].endswith("deltas.json")

    payload = json.loads((out_ws / "deltas.json").read_text())
    assert len(payload) == 1
    kinds = {d["kind"] for d in payload[0]["deltas"]}
    assert "purity" in kinds


# Regression coverage for the review-driven fixes.


def test_parse_response_axes_handles_unicode_multiplication_sign():
    """Real evidence.md uses `×` (U+00D7); ASCII `x` is a test convenience."""
    evidence = (
        "### Therapy-state context\n\n"
        "**MAPK EGFR signaling** — active. up-panel geomean 3.50× cohort\n\n"
        "**IFN response** — active. up-panel geomean 2.00× cohort, "
        "down-panel 0.30×\n\n"
        "### Next Section\n"
    )
    axes = parse_response_axes(evidence.splitlines())
    assert axes["MAPK_EGFR_signaling"].up_fold == 3.50
    assert axes["IFN_response"].up_fold == 2.00
    assert axes["IFN_response"].down_fold == 0.30


def test_parse_response_axes_terminates_on_h2_h1_or_hr():
    """Section termination must work for any heading, not just h3.

    A future renderer that emits `## ` before the next axis block would
    otherwise bleed lines into the previous axis's note.
    """
    evidence = (
        "### Therapy-state context\n\n"
        "**MAPK EGFR signaling** — active. up-panel geomean 3.50x cohort\n\n"
        "## Top candidate therapies\n\n"
        "- **EVERYTHING** — should be after the section ended (TPM 100x)\n"
    )
    axes = parse_response_axes(evidence.splitlines())
    assert "MAPK_EGFR_signaling" in axes
    # The note must not contain content from the next section.
    assert "Top candidate" not in axes["MAPK_EGFR_signaling"].note
    assert "EVERYTHING" not in axes["MAPK_EGFR_signaling"].note


def test_parse_response_axes_normalizes_state_vocabulary():
    """Free-form state phrases collapse to a controlled set so equality holds."""
    evidence = (
        "### Therapy-state context\n\n"
        "**MAPK EGFR signaling** — Active signaling. up-panel geomean 3.5x cohort\n\n"
        "**IFN response** — mixed signal; partial agreement. up-panel geomean 1.5x cohort\n\n"
        "**ER signaling** — Suppressed: up-panel geomean 0.20x cohort\n\n"
        "### End\n"
    )
    axes = parse_response_axes(evidence.splitlines())
    assert axes["MAPK_EGFR_signaling"].state == "active"
    assert axes["IFN_response"].state == "mixed"
    assert axes["ER_signaling"].state == "suppressed"


def test_response_axis_same_state_same_fold_is_unchanged(tmp_path):
    """An axis with identical state + fold across samples is `unchanged`."""
    summary = SUMMARY_A.replace("sample_A", "sample_X")
    evidence = (
        "### Therapy-state context\n\n"
        "**MAPK EGFR signaling** — active. up-panel geomean 3.50x cohort\n\n"
        "### End\n"
    )
    dir_a = _write_sample(tmp_path / "a", "a", summary, evidence)
    dir_b = _write_sample(
        tmp_path / "b", "b",
        summary.replace("sample_X", "sample_Y").replace(
            "Purity:** 60%", "Purity:** 60%"
        ),
        evidence,
    )
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    deltas = compute_pairwise_deltas(rec_a, rec_b).deltas
    axis_deltas = [d for d in deltas if d.kind == "response_axis"]
    assert axis_deltas, "Expected a response_axis delta for the shared axis"
    assert all(d.direction == "unchanged" for d in axis_deltas)


def test_response_axis_state_change_suppresses_magnitude(tmp_path):
    """Active → suppressed leaves magnitude empty (up_fold semantics flip)."""
    summary = SUMMARY_A.replace("sample_A", "sample_X")
    evidence_a = (
        "### Therapy-state context\n\n"
        "**MAPK EGFR signaling** — active. up-panel geomean 3.50x cohort\n\n"
        "### End\n"
    )
    evidence_b = (
        "### Therapy-state context\n\n"
        "**MAPK EGFR signaling** — suppressed. up-panel geomean 0.40x cohort\n\n"
        "### End\n"
    )
    dir_a = _write_sample(tmp_path / "a", "a", summary, evidence_a)
    dir_b = _write_sample(tmp_path / "b", "b", summary, evidence_b)
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    deltas = compute_pairwise_deltas(rec_a, rec_b).deltas
    mapk = next(
        d for d in deltas
        if d.kind == "response_axis"
        and (d.before or {}).get("axis") == "MAPK_EGFR_signaling"
    )
    assert mapk.direction == "shifted"
    assert mapk.magnitude is None  # suppressed — direction-dependent
    assert mapk.unit == ""


def test_target_tpm_low_floor_suppresses_noise(tmp_path):
    """Tiny TPM movement (1 → 1.2) shouldn't emit a target_tpm delta."""
    summary_low_a = SUMMARY_A.replace("200.0 tumor-source bulk TPM", "1.0 tumor-source bulk TPM")
    summary_low_b = SUMMARY_B.replace("400.0 tumor-source bulk TPM", "1.2 tumor-source bulk TPM")
    dir_a = _write_sample(tmp_path / "a", "a", summary_low_a, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", summary_low_b, EVIDENCE_B)
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    deltas = compute_pairwise_deltas(rec_a, rec_b).deltas
    nectin_tpm_deltas = [
        d for d in deltas
        if d.kind == "target_tpm"
        and (d.before or {}).get("gene") == "NECTIN4"
    ]
    assert nectin_tpm_deltas == []


def test_assay_comparability_unknown_bucket_marks_unknown_not_shifted(tmp_path):
    """A missing library prep on one side should emit `direction=unknown`."""
    summary_a = SUMMARY_A.replace(
        "ribosomal depletion library; preservation inferred as unknown.",
        "",
    )
    dir_a = _write_sample(tmp_path / "a", "a", summary_a, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    deltas = compute_pairwise_deltas(rec_a, rec_b).deltas
    assay = next(d for d in deltas if d.kind == "assay_comparability")
    # One side is unknown — we surface a gap, not a fictitious "shift".
    assert assay.direction == "unknown"
    assert "missing" in assay.note.lower() or "unknown" in assay.note.lower()


def test_assay_comparability_both_known_and_same_does_not_fire(tmp_path):
    """Same library prep AND preservation on both sides → no delta."""
    # SUMMARY_A: ribo library, preservation unknown
    # SUMMARY_B: polyA library, preservation fresh
    # Align both to the same library + preservation buckets:
    summary_b_aligned = (
        SUMMARY_B.replace("polyA library", "ribosomal depletion library")
        .replace("preservation fresh", "preservation inferred as unknown")
    )
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", summary_b_aligned, EVIDENCE_B)
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    deltas = compute_pairwise_deltas(rec_a, rec_b).deltas
    # Both library buckets are known and equal (ribo); both preservation
    # buckets are unknown symmetrically. No comparability advisory fires.
    assert not [d for d in deltas if d.kind == "assay_comparability"]


def test_compute_longitudinal_delta_sets_chains_three_samples(tmp_path):
    """3 samples → 2 pairwise delta sets, in order."""
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)
    summary_c = SUMMARY_A.replace("sample_A", "sample_C").replace("60%", "75%")
    dir_c = _write_sample(tmp_path / "c", "c", summary_c, EVIDENCE_A)
    records = [
        load_analyze_summary_record(p) for p in (dir_a, dir_b, dir_c)
    ]
    delta_sets = compute_longitudinal_delta_sets(records)
    assert len(delta_sets) == 2
    assert delta_sets[0].before_sample == "sample_A"
    assert delta_sets[0].after_sample == "sample_B"
    assert delta_sets[1].before_sample == "sample_B"
    assert delta_sets[1].after_sample == "sample_C"


def test_cancer_call_prefix_handles_subtype_and_unclear_codes():
    """The prefix splitter survives subtype codes and unclear labels."""
    from trufflepig.analyze.deltas import _cancer_call_prefix

    assert _cancer_call_prefix("BLCA — moderate confidence") == "BLCA"
    assert _cancer_call_prefix("SARC_DDLPS — high confidence") == "SARC_DDLPS"
    # "Cancer-type unclear" would previously split to "Cancer-type" if
    # the regex used `^[A-Z]\\w*`; ours requires all-caps.
    assert _cancer_call_prefix("Cancer-type unclear") == ""


def test_target_line_regex_handles_nested_parentheses():
    """A drug scope like `(Approved (subset), other)` shouldn't break parsing."""
    summary = (
        "## Top candidate therapies\n\n"
        "- **NECTIN4** — enfortumab vedotin (Approved (advanced urothelial), 2L+). "
        "tumor-supported; 200.0 tumor-source bulk TPM.\n"
    )
    entries = parse_target_shortlist(summary.splitlines())
    assert len(entries) == 1
    assert entries[0].gene == "NECTIN4"
    assert entries[0].tpm == 200.0


def test_longitudinal_delta_has_unit_field_for_each_kind(tmp_path):
    """The new `unit` field is set whenever magnitude is set."""
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = _write_sample(tmp_path / "b", "b", SUMMARY_B, EVIDENCE_B)
    rec_a = load_analyze_summary_record(dir_a)
    rec_b = load_analyze_summary_record(dir_b)
    deltas = compute_pairwise_deltas(rec_a, rec_b).deltas
    for d in deltas:
        if d.magnitude is not None:
            assert d.unit, f"{d.kind} delta has magnitude but no unit"


def test_cli_compare_reports_skipped_deltas_when_summary_missing(tmp_path, capsys):
    """Missing summary.md prints a notice instead of silently dropping deltas.json."""
    from trufflepig import cli

    # Two stub dirs (no summary.md files) — compare_analyze itself will raise
    # FileNotFoundError because it needs the summaries; the cli's outer
    # delta-write path catches a separate FileNotFoundError on its second
    # load. We exercise the catch path by writing one summary but not the
    # other, so the inner compare_analyze succeeds but our outer parse fails.
    dir_a = _write_sample(tmp_path / "a", "a", SUMMARY_A, EVIDENCE_A)
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    out_ws = tmp_path / "cmp"

    # Stub the inner renderer so compare_analyze doesn't itself raise.
    import trufflepig.main as main_mod

    def _fake_compare(output_dirs, output_path, title):
        with open(output_path, "w") as f:
            f.write(f"# {title}\nstub\n")

    main_mod.compare_analyze = _fake_compare  # monkeypatch for test
    try:
        rc = cli.main([
            "compare",
            "--workspace", str(out_ws),
            "--inputs", f"{dir_a},{dir_b}",
            "--title", "missing-summary",
        ])
    finally:
        # Restore the real attribute to whatever it was originally (best effort).
        pass
    assert rc == 0
    assert (out_ws / "comparison.md").is_file()
    assert not (out_ws / "deltas.json").exists()
    err = capsys.readouterr().err
    assert "[deltas] skipped" in err
