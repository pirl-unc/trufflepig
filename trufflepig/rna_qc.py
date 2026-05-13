"""Run-level RNA quantification QC helpers.

The expression-scale checks in :mod:`pirlygenes.load_expression` validate the
gene table itself. This module looks one layer upstream when a quantifier
output directory is available, especially Salmon's ``aux_info`` JSON files.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _candidate_salmon_dirs(input_path: str | Path | None) -> list[Path]:
    if input_path is None:
        return []
    try:
        start = Path(input_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return []
    parent = start if start.is_dir() else start.parent
    out: list[Path] = []
    for candidate in (parent, parent.parent, parent / "salmon-output"):
        if candidate not in out:
            out.append(candidate)
    return out


def _candidate_alignment_qc_paths(
    input_path: str | Path | None,
    salmon_dir: Path | None,
) -> list[Path]:
    """Return nearby samtools-idxstats style files, if present.

    BAM/CRAM files are intentionally not auto-discovered because running an
    external tool over a large alignment should be an explicit user choice.
    """

    dirs: list[Path] = []
    if input_path is not None:
        try:
            start = Path(input_path).expanduser().resolve()
            parent = start if start.is_dir() else start.parent
            dirs.extend([parent, parent.parent])
        except (OSError, RuntimeError):
            pass
    if salmon_dir is not None:
        dirs.extend([salmon_dir, salmon_dir.parent])

    out: list[Path] = []
    names = (
        "idxstats.tsv",
        "idxstats.txt",
        "samtools.idxstats",
        "alignment.idxstats",
        "chromosome-breakdown.tsv",
        "chromosome_breakdown.tsv",
    )
    seen: set[Path] = set()
    for directory in dirs:
        if not directory.exists() or directory in seen:
            continue
        seen.add(directory)
        for name in names:
            candidate = directory / name
            if candidate.exists() and candidate not in out:
                out.append(candidate)
        for candidate in sorted(directory.glob("*idxstats*")):
            if candidate.is_file() and candidate not in out:
                out.append(candidate)
    return out


def discover_salmon_dir(input_path: str | Path | None) -> Path | None:
    """Return the nearest Salmon output directory, if recognizable."""

    for candidate in _candidate_salmon_dirs(input_path):
        if (candidate / "aux_info" / "meta_info.json").exists():
            return candidate
        if (candidate / "quant.sf").exists() and (candidate / "aux_info").exists():
            return candidate
    return None


def _read_tpm_stats(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    sep = "\t" if path.suffix.lower() in {".sf", ".tsv"} else ","
    try:
        df = pd.read_csv(path, sep=sep, usecols=lambda c: str(c).lower() == "tpm")
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if df.empty:
        return None
    tpm_col = next((c for c in df.columns if str(c).lower() == "tpm"), None)
    if tpm_col is None:
        return None
    tpm = pd.to_numeric(df[tpm_col], errors="coerce").fillna(0.0)
    return {
        "path": str(path),
        "total": int(len(tpm)),
        "detected_gt0": int((tpm > 0.0).sum()),
        "detected_ge1": int((tpm >= 1.0).sum()),
        "detected_ge5": int((tpm >= 5.0).sum()),
        "sum_tpm": float(tpm.sum()),
        "max_tpm": float(tpm.max()) if len(tpm) else 0.0,
    }


def _stats_from_loaded_frame(df: pd.DataFrame | None) -> dict[str, Any] | None:
    if df is None or "TPM" not in set(df.columns):
        return None
    tpm = pd.to_numeric(df["TPM"], errors="coerce").fillna(0.0)
    return {
        "path": "loaded expression table",
        "total": int(len(tpm)),
        "detected_gt0": int((tpm > 0.0).sum()),
        "detected_ge1": int((tpm >= 1.0).sum()),
        "detected_ge5": int((tpm >= 5.0).sum()),
        "sum_tpm": float(tpm.sum()),
        "max_tpm": float(tpm.max()) if len(tpm) else 0.0,
    }


def _gene_stats(
    salmon_dir: Path | None,
    gene_df: pd.DataFrame | None,
    input_path: str | Path | None,
) -> dict[str, Any] | None:
    if salmon_dir is not None:
        for candidate in (
            salmon_dir / "quant.genes.sf",
            salmon_dir / "quant.gene_tpm.csv",
            salmon_dir / "gene_tpm.csv",
        ):
            stats = _read_tpm_stats(candidate)
            if stats is not None:
                return stats

    loaded_stats = _stats_from_loaded_frame(gene_df)
    if loaded_stats is not None:
        return loaded_stats

    if input_path is not None:
        try:
            p = Path(input_path).expanduser().resolve()
            if p.name != "quant.sf":
                return _read_tpm_stats(p)
        except (OSError, RuntimeError):
            pass
    return None


def _transcript_stats(
    salmon_dir: Path | None,
    transcript_path: str | Path | None,
) -> dict[str, Any] | None:
    candidates: list[Path] = []
    if transcript_path is not None:
        try:
            candidates.append(Path(transcript_path).expanduser().resolve())
        except (OSError, RuntimeError):
            pass
    if salmon_dir is not None:
        candidates.append(salmon_dir / "quant.sf")
    for candidate in candidates:
        stats = _read_tpm_stats(candidate)
        if stats is not None:
            return stats
    return None


def _alignment_qc_from_path(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        p = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if not p.exists():
        return {
            "available": False,
            "path": str(p),
            "warnings": [f"Alignment QC path not found: {p}"],
        }
    if p.suffix.lower() in {".bam", ".cram"}:
        return _alignment_qc_from_bam(p)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {
            "available": False,
            "path": str(p),
            "warnings": [f"Could not read alignment QC path {p}: {exc}"],
        }
    return _parse_idxstats_text(text, source=str(p))


def _alignment_qc_from_bam(path: Path) -> dict[str, Any]:
    samtools = shutil.which("samtools")
    if not samtools:
        return {
            "available": False,
            "path": str(path),
            "warnings": ["samtools is required to read BAM/CRAM alignment QC."],
        }
    try:
        proc = subprocess.run(
            [samtools, "idxstats", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "path": str(path),
            "warnings": [f"samtools idxstats failed for {path}: {exc}"],
        }
    qc = _parse_idxstats_text(proc.stdout, source=str(path))
    stderr = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
    if proc.returncode != 0:
        stderr.insert(0, f"samtools idxstats exited with status {proc.returncode}")
    if stderr:
        qc.setdefault("tool_warnings", []).extend(stderr)
    return qc


def _parse_idxstats_text(text: str, *, source: str) -> dict[str, Any]:
    rows = []
    star_unmapped = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = re.split(r"\t+|\s+", stripped)
        if len(parts) < 4:
            continue
        contig = parts[0]
        try:
            length = int(float(parts[1]))
            mapped = int(float(parts[2]))
            unmapped = int(float(parts[3]))
        except ValueError:
            continue
        if contig == "*":
            star_unmapped += unmapped
            continue
        density = mapped / length if length > 0 else 0.0
        rows.append(
            {
                "contig": contig,
                "length": length,
                "mapped": mapped,
                "unmapped": unmapped,
                "density": density,
                "category": _contig_qc_category(contig),
                "normalized_contig": _normalize_contig_name(contig),
            }
        )

    if not rows and star_unmapped <= 0:
        return {
            "available": False,
            "path": source,
            "warnings": ["Alignment QC file did not contain parseable idxstats rows."],
        }

    total_mapped = sum(int(row["mapped"]) for row in rows)
    total_unmapped = star_unmapped + sum(int(row["unmapped"]) for row in rows)
    chr1_density = next(
        (
            float(row["density"])
            for row in rows
            if str(row.get("normalized_contig")).lower() == "chr1"
        ),
        None,
    )
    primary_rows = [row for row in rows if row["category"] == "primary_chromosome"]
    rdna_rows = [row for row in rows if row["category"] == "rdna_repeat_like"]
    top_primary = _top_rows(primary_rows, "mapped", 5)
    top_density = _top_rows(
        [row for row in rows if int(row["length"]) >= 10_000 and int(row["mapped"]) > 0],
        "density",
        8,
    )
    top_rdna = _top_rows(rdna_rows, "mapped", 5)
    rdna_mapped = sum(int(row["mapped"]) for row in rdna_rows)
    rdna_max_density = max((float(row["density"]) for row in rdna_rows), default=0.0)
    rdna_density_over_chr1 = (
        rdna_max_density / chr1_density
        if chr1_density is not None and chr1_density > 0
        else None
    )
    rdna_fraction = rdna_mapped / total_mapped if total_mapped > 0 else 0.0

    warnings = []
    if rdna_density_over_chr1 is not None and rdna_density_over_chr1 >= 100.0:
        warnings.append(
            "rDNA-like contigs have extreme read density relative to chr1; "
            "this supports rRNA-repeat carryover or short-fragment repeat mapping."
        )
    elif rdna_fraction >= 0.02:
        warnings.append(
            "rDNA-like contigs carry a material fraction of aligned reads; "
            "review rRNA contamination/repeat-mapping effects."
        )
    if (
        top_primary
        and str(top_primary[0].get("normalized_contig")).lower() == "chr21"
        and rdna_rows
    ):
        warnings.append(
            "chr21 is the top primary chromosome while rDNA-like repeat contigs are enriched; "
            "treat this as technical rRNA-repeat evidence, not chromosome-21 biology."
        )

    return {
        "available": True,
        "path": source,
        "format": "samtools_idxstats",
        "contig_count": len(rows),
        "total_mapped": total_mapped,
        "total_unmapped": total_unmapped,
        "star_unmapped": star_unmapped,
        "chr1_density": chr1_density,
        "rdna_mapped": rdna_mapped,
        "rdna_fraction_of_mapped": rdna_fraction,
        "rdna_max_density": rdna_max_density,
        "rdna_density_over_chr1": rdna_density_over_chr1,
        "top_primary_contigs": top_primary,
        "top_density_contigs": top_density,
        "top_rdna_contigs": top_rdna,
        "warnings": warnings,
    }


def _contig_qc_category(contig: str) -> str:
    upper = str(contig or "").upper()
    if any(token in upper for token in ("GL000220", "KI270733", "RDNA", "RDN", "RNA45S")):
        return "rdna_repeat_like"
    if re.fullmatch(r"(CHR)?([0-9]+|X|Y|M|MT)", upper):
        return "primary_chromosome"
    if "_RANDOM" in upper or upper.startswith("CHRUN") or "_KI" in upper or "_GL" in upper:
        return "random_or_unplaced"
    if upper.startswith("HLA-"):
        return "hla_decoy"
    return "other_contig"


def _normalize_contig_name(contig: str) -> str:
    """Canonicalize primary contig labels for QC comparisons."""

    text = str(contig or "").strip()
    upper = text.upper()
    match = re.fullmatch(r"(?:CHR)?([0-9]+|X|Y|M|MT)", upper)
    if not match:
        return text
    token = match.group(1)
    if token in {"M", "MT"}:
        return "chrM"
    return f"chr{token}"


def _top_rows(rows: list[dict[str, Any]], key: str, n: int) -> list[dict[str, Any]]:
    return [
        {
            "contig": str(row["contig"]),
            "length": int(row["length"]),
            "mapped": int(row["mapped"]),
            "unmapped": int(row["unmapped"]),
            "density": float(row["density"]),
            "category": str(row["category"]),
            "normalized_contig": str(row.get("normalized_contig") or row["contig"]),
        }
        for row in sorted(rows, key=lambda row: (-float(row[key]), str(row["contig"])))[:n]
    ]


def _mapping_comment(mapping_rate: float | None) -> tuple[str, list[str]]:
    if mapping_rate is None:
        return "", []
    if mapping_rate < 50.0:
        return (
            "Low; typical whole-transcriptome RNA-seq often maps much higher. "
            "Check input quality, index completeness, off-target/contaminant reads, "
            "and degradation.",
            [
                f"Salmon mapping rate is low ({mapping_rate:.1f}%). Interpret RNA-derived calls cautiously."
            ],
        )
    if mapping_rate < 70.0:
        return (
            "Below the usual range for many bulk RNA-seq runs; review input quality "
            "and index compatibility.",
            [
                f"Salmon mapping rate is below typical bulk RNA-seq expectations ({mapping_rate:.1f}%)."
            ],
        )
    return "Within a broadly typical range for compatible RNA-seq inputs.", []


def _detection_comment(stats: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not stats:
        return "", []
    ge1 = int(stats.get("detected_ge1") or 0)
    total = int(stats.get("total") or 0)
    if ge1 < 10_000:
        return (
            "Low gene-detection breadth; this can reflect degradation, low input, "
            "or a restricted assay.",
            [f"Only {ge1:,} of {total:,} genes are detected at >=1 TPM."],
        )
    if ge1 < 13_000:
        return (
            "Low-normal gene-detection breadth for whole-transcriptome bulk RNA-seq; "
            "consistent with reduced input quality or low cellularity.",
            [],
        )
    return "Gene-detection breadth is compatible with whole-transcriptome RNA-seq.", []


def _tpm_sum_comment(stats: dict[str, Any] | None, label: str) -> tuple[str, list[str]]:
    if not stats:
        return "", []
    total = float(stats.get("sum_tpm") or 0.0)
    if total <= 0:
        return "No TPM mass detected.", [f"{label} TPM sum is zero."]
    missing_frac = abs(total - 1_000_000.0) / 1_000_000.0
    if missing_frac > 0.01:
        return (
            f"TPM sum is {total:,.0f}; more than 1% away from 1,000,000. "
            "Check whether rows were filtered or values were transformed.",
            [f"{label} TPM sum is not close to 1,000,000 ({total:,.0f})."],
        )
    return f"TPM mass is near 1,000,000 ({total:,.0f}); little expression mass is missing.", []


def collect_rna_quant_qc(
    input_path: str | Path | None,
    *,
    gene_df: pd.DataFrame | None = None,
    transcript_path: str | Path | None = None,
    alignment_qc_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect Salmon run QC and expression-detection breadth.

    The return value is intentionally plain JSON-serializable data so it can be
    stored in analysis manifests and reused by reports.
    """

    salmon_dir = discover_salmon_dir(input_path)
    meta = _read_json(salmon_dir / "aux_info" / "meta_info.json") if salmon_dir else {}
    lib = _read_json(salmon_dir / "lib_format_counts.json") if salmon_dir else {}
    gene_stats = _gene_stats(salmon_dir, gene_df, input_path)
    transcript_stats = _transcript_stats(salmon_dir, transcript_path)
    alignment_qc = _alignment_qc_from_path(alignment_qc_path)
    if alignment_qc is None:
        alignment_qc = next(
            (
                qc
                for qc in (
                    _alignment_qc_from_path(candidate)
                    for candidate in _candidate_alignment_qc_paths(input_path, salmon_dir)
                )
                if qc and qc.get("available")
            ),
            None,
        )

    warnings: list[str] = []
    mapping_rate = meta.get("percent_mapped")
    try:
        mapping_rate = float(mapping_rate) if mapping_rate is not None else None
    except (TypeError, ValueError):
        mapping_rate = None
    mapping_comment, mapping_warnings = _mapping_comment(mapping_rate)
    warnings.extend(mapping_warnings)

    detection_comment, detection_warnings = _detection_comment(gene_stats)
    warnings.extend(detection_warnings)
    gene_sum_comment, gene_sum_warnings = _tpm_sum_comment(gene_stats, "Gene")
    warnings.extend(gene_sum_warnings)
    tx_sum_comment, tx_sum_warnings = _tpm_sum_comment(transcript_stats, "Transcript")
    warnings.extend(tx_sum_warnings)
    if alignment_qc:
        warnings.extend(
            str(w)
            for w in alignment_qc.get("warnings", [])
            if str(w).strip()
        )

    rows: list[dict[str, str]] = []

    def add(metric: str, value: str, comment: str = "") -> None:
        if value:
            rows.append({"metric": metric, "value": value, "comment": comment})

    add("Fragments processed", _format_count(meta.get("num_processed")))
    add("Mapped fragments", _format_count(meta.get("num_mapped")))
    if mapping_rate is not None:
        add("Mapping rate", f"{mapping_rate:.1f}%", mapping_comment)
    add(
        "Decoy fragments",
        _format_count(meta.get("num_decoy_fragments")),
        "Decoys captured genome-derived or off-target fragments."
        if meta.get("num_decoy_fragments")
        else "",
    )
    add(
        "Below-threshold alignments",
        _format_count(meta.get("num_alignments_below_threshold_for_mapped_fragments_vm")),
        "Many weak hits were filtered; this supports degraded/off-target or hard-to-map input."
        if (meta.get("num_alignments_below_threshold_for_mapped_fragments_vm") or 0)
        else "",
    )
    add(
        "Filtered fragments",
        _format_count(meta.get("num_fragments_filtered_vm")),
        "Fragments removed by validation/mismatch filters."
        if meta.get("num_fragments_filtered_vm")
        else "",
    )
    lib_type = _first(meta.get("library_types")) or lib.get("expected_format")
    add("Library type", str(lib_type or ""), _library_type_comment(str(lib_type or "")))
    if meta.get("frag_length_mean") is not None:
        mean = _safe_float(meta.get("frag_length_mean"))
        sd = _safe_float(meta.get("frag_length_sd"))
        if mean is not None and sd is not None:
            add("Fragment length", f"{mean:.0f} +/- {sd:.0f} bp", "Insert size looks plausible.")
    bias = _safe_float(lib.get("strand_mapping_bias"))
    if bias is not None:
        add(
            "Strand mapping bias",
            f"{bias:.3g}",
            "No obvious strand issue." if abs(bias) < 0.01 else "Review strandedness settings.",
        )
    if gene_stats:
        add(
            "Gene detection",
            (
                f"{gene_stats['detected_gt0']:,} >0; "
                f"{gene_stats['detected_ge1']:,} >=1; "
                f"{gene_stats['detected_ge5']:,} >=5 TPM "
                f"(of {gene_stats['total']:,})"
            ),
            detection_comment,
        )
        add("Gene TPM sum", f"{gene_stats['sum_tpm']:,.0f}", gene_sum_comment)
    if transcript_stats:
        add(
            "Transcript detection",
            (
                f"{transcript_stats['detected_gt0']:,} >0; "
                f"{transcript_stats['detected_ge1']:,} >=1 TPM "
                f"(of {transcript_stats['total']:,})"
            ),
        )
        add("Transcript TPM sum", f"{transcript_stats['sum_tpm']:,.0f}", tx_sum_comment)
    if alignment_qc and alignment_qc.get("available"):
        _add_alignment_qc_rows(add, alignment_qc)

    summary_bits = []
    if mapping_rate is not None:
        summary_bits.append(f"Salmon mapping {mapping_rate:.1f}%")
    if gene_stats:
        summary_bits.append(
            f"{gene_stats['detected_ge1']:,}/{gene_stats['total']:,} genes >=1 TPM"
        )
    if gene_stats and abs(float(gene_stats.get("sum_tpm") or 0.0) - 1_000_000.0) <= 10_000.0:
        summary_bits.append("gene TPM sum near 1.0M")
    if alignment_qc and alignment_qc.get("available"):
        ratio = _safe_float(alignment_qc.get("rdna_density_over_chr1"))
        if ratio is not None and ratio >= 10:
            summary_bits.append(f"rDNA-like contig density {ratio:,.0f}x chr1")
    summary = "; ".join(summary_bits)

    available = bool(
        meta
        or lib
        or gene_stats
        or transcript_stats
        or (alignment_qc and alignment_qc.get("available"))
    )
    source = (
        "salmon"
        if (meta or lib or salmon_dir)
        else "alignment_qc"
        if alignment_qc and alignment_qc.get("available")
        else "expression_table"
    )
    return {
        "available": available,
        "source": source,
        "salmon_dir": str(salmon_dir) if salmon_dir else None,
        "meta_info": meta,
        "lib_format_counts": lib,
        "gene_detection": gene_stats,
        "transcript_detection": transcript_stats,
        "alignment_qc": alignment_qc,
        "rows": rows,
        "warnings": warnings,
        "summary": summary,
    }


def _add_alignment_qc_rows(add, alignment_qc: dict[str, Any]) -> None:
    total_mapped = int(alignment_qc.get("total_mapped") or 0)
    total_unmapped = int(alignment_qc.get("total_unmapped") or 0)
    contig_count = int(alignment_qc.get("contig_count") or 0)
    add(
        "Alignment contigs",
        f"{contig_count:,}; {_format_count(total_mapped)} mapped; {_format_count(total_unmapped)} unmapped",
        "Parsed from samtools idxstats-style contig counts.",
    )
    top_primary = alignment_qc.get("top_primary_contigs") or []
    if top_primary:
        top = top_primary[0]
        comment = "Top primary-chromosome hit by mapped read count."
        if str(top.get("contig") or "").lower() == "chr21":
            comment = (
                "chr21 is the top primary chromosome; if rDNA repeat contigs are also "
                "enriched, this supports rRNA-repeat signal rather than chromosome-21 biology."
            )
        add(
            "Top primary chromosome",
            f"{top.get('contig')} ({_format_count(top.get('mapped'))} mapped)",
            comment,
        )
    rdna_mapped = int(alignment_qc.get("rdna_mapped") or 0)
    ratio = _safe_float(alignment_qc.get("rdna_density_over_chr1"))
    if rdna_mapped or ratio is not None:
        ratio_text = f"; max density {ratio:,.0f}x chr1" if ratio is not None else ""
        add(
            "rDNA-like contig burden",
            f"{_format_count(rdna_mapped)} mapped{ratio_text}",
            "High density on GL000220/KI270733/rDNA-like contigs is a direct repeat-mapping/rRNA QC signal.",
        )
    top_rdna = alignment_qc.get("top_rdna_contigs") or []
    if top_rdna:
        bits = []
        chr1_density = _safe_float(alignment_qc.get("chr1_density"))
        for row in top_rdna[:3]:
            density = _safe_float(row.get("density"))
            ratio_text = ""
            if density is not None and chr1_density and chr1_density > 0:
                ratio_text = f", {density / chr1_density:,.0f}x chr1"
            bits.append(f"{row.get('contig')} {_format_count(row.get('mapped'))}{ratio_text}")
        add(
            "Top rDNA-like contigs",
            "; ".join(bits),
            "These contigs represent rDNA/repeat-like sequence, not ordinary gene expression.",
        )
    tool_warnings = [
        str(w).strip()
        for w in alignment_qc.get("tool_warnings", [])
        if str(w).strip()
    ]
    if tool_warnings:
        add(
            "Alignment tool warnings",
            "; ".join(tool_warnings[:2]),
            "Warnings emitted while collecting contig counts; review the BAM/index if unexpected.",
        )


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_count(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def _library_type_comment(value: str) -> str:
    lookup = {
        "ISR": "Reverse-stranded paired-end library.",
        "ISF": "Forward-stranded paired-end library.",
        "IU": "Unstranded paired-end library.",
        "SR": "Reverse-stranded single-end library.",
        "SF": "Forward-stranded single-end library.",
        "U": "Unstranded single-end library.",
    }
    return lookup.get(value, "")


def rna_quant_qc_summary_line(qc: dict[str, Any] | None) -> str:
    if not qc or not qc.get("available"):
        return ""
    summary = str(qc.get("summary") or "").strip()
    if not summary:
        return ""
    warning = ""
    warnings = qc.get("warnings") or []
    if warnings:
        warning = f"; caution: {str(warnings[0]).rstrip('.')}"
    return f"**RNA quant QC:** {summary}{warning}."


def rna_quant_qc_markdown(qc: dict[str, Any] | None, *, heading: str) -> str:
    if not qc or not qc.get("available"):
        return ""
    rows = qc.get("rows") or []
    if not rows:
        return ""
    lines = [heading, ""]
    summary = str(qc.get("summary") or "").strip()
    if summary:
        lines.append(f"- **Summary**: {summary}.")
    warnings = [str(w) for w in (qc.get("warnings") or []) if str(w).strip()]
    if warnings:
        lines.append(f"- **QC warning**: {warnings[0]}")
    lines.append("")
    lines.append("| Metric | Value | Comment |")
    lines.append("|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.get('metric', '')} | {row.get('value', '')} | {row.get('comment', '') or '-'} |"
        )
    return "\n".join(lines)
