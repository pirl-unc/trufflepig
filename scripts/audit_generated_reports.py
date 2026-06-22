#!/usr/bin/env python3
"""Audit generated trufflepig report trees for report-level coherence.

This complements ``scripts/analyze_reports.py``.  The scorer answers "did the
headline call stay compatible with the expected cancer type?"  This script also
looks for user-visible report problems: mechanical rendering errors, conflicting
headline/summary labels, unsupported metastatic-site claims, duplicated
fit-only met templates, and suspicious therapy/target prose.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from analyze_reports import (
    Compat,
    _calls_of,
    _expected_for,
    _iter_samples,
    _load_truth,
    _registry_parent_map,
    _broad_lineage_fn,
)

_SUMMARY_CALL = re.compile(r"\*\*Cancer call:\*\*\s*([A-Za-z][A-Za-z0-9_]*)\b")
_WORKING_CALL = re.compile(r"\*\*Working cancer call\*\*:\s*([^.\n]+)")
_CODE_BEFORE_PAREN = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*\(")
_MECHANICAL_PATTERNS = {
    "traceback": re.compile(r"\bTraceback\b"),
    "rendering_error": re.compile(r"rendering failed|I/O operation on closed file"),
    "nan_text": re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])|— nan \(", re.I),
    "format_error": re.compile(r"unsupported format string|Length of values"),
}
_FALLBACK_PHRASES = (
    "consistent with the broad reference context",
    "fallback expression reference",
    "RNA reference context",
)
_STALE_ROLE_PHRASES = (
    "leading broad expression context used for cohort-normalized downstream analyses",
)


def _read(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _sample_paths(analysis_md: Path, sample_id: str) -> dict[str, Path | None]:
    root = analysis_md.parent
    return {
        "analysis": analysis_md,
        "summary": _first_existing([root / f"{sample_id}-summary.md"]),
        "evidence": _first_existing([root / f"{sample_id}-evidence.md"]),
        "decomposition": _first_existing(
            [root / f"{sample_id}-decomposition-hypotheses.tsv"]
        ),
        "ranges": _first_existing([root / f"{sample_id}-tumor-expression-ranges.tsv"]),
    }


def _summary_call(summary_text: str) -> str:
    match = _SUMMARY_CALL.search(summary_text)
    return match.group(1) if match else ""


def _working_codes(analysis_text: str) -> list[str]:
    match = _WORKING_CALL.search(analysis_text)
    if not match:
        return []
    return _CODE_BEFORE_PAREN.findall(match.group(1))


def _decomposition_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError:
        return []


def _sample_issues(
    *,
    sample_id: str,
    expected: str,
    paths: dict[str, Path | None],
    compat: Compat,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    missing = [key for key, path in paths.items() if path is None]
    for key in missing:
        issues.append(
            {
                "sample": sample_id,
                "severity": "error",
                "category": f"missing_{key}",
                "detail": f"missing {key} artifact",
            }
        )

    analysis_text = _read(paths.get("analysis"))
    summary_text = _read(paths.get("summary"))
    evidence_text = _read(paths.get("evidence"))
    all_text = "\n".join([analysis_text, summary_text, evidence_text])

    for category, pattern in _MECHANICAL_PATTERNS.items():
        if pattern.search(all_text):
            issues.append(
                {
                    "sample": sample_id,
                    "severity": "error",
                    "category": category,
                    "detail": pattern.pattern,
                }
            )

    for phrase in _STALE_ROLE_PHRASES:
        if phrase in all_text:
            issues.append(
                {
                    "sample": sample_id,
                    "severity": "error",
                    "category": "stale_reference_role_phrase",
                    "detail": phrase,
                }
            )

    summary_call = _summary_call(summary_text)
    working_codes = _working_codes(analysis_text)
    if summary_call and working_codes and summary_call not in working_codes:
        issues.append(
            {
                "sample": sample_id,
                "severity": "error",
                "category": "summary_analysis_call_mismatch",
                "detail": f"summary={summary_call}; analysis={','.join(working_codes)}",
            }
        )
    if expected and working_codes and not any(compat(code, expected) for code in working_codes):
        issues.append(
            {
                "sample": sample_id,
                "severity": "error",
                "category": "headline_incompatible_with_expected",
                "detail": f"expected={expected}; analysis={','.join(working_codes)}",
            }
        )

    rows = _decomposition_rows(paths.get("decomposition"))
    if rows:
        top = rows[0]
        top_template = str(top.get("template") or "")
        top_warnings = str(top.get("warnings") or "")
        if top_template.startswith("met_") and (
            "below template-specific threshold" in top_warnings
            or "No non-tumor components" in top_warnings
        ):
            issues.append(
                {
                    "sample": sample_id,
                    "severity": "warning",
                    "category": "unsupported_met_template_ranked_first",
                    "detail": f"{top.get('cancer_type')}/{top_template}: {top_warnings}",
                }
            )
        for row in rows[:6]:
            template = str(row.get("template") or "")
            warnings = str(row.get("warnings") or "")
            if template.startswith("met_") and "No non-tumor components" in warnings:
                issues.append(
                    {
                        "sample": sample_id,
                        "severity": "warning",
                        "category": "fit_only_met_template_in_top_rows",
                        "detail": f"{row.get('cancer_type')}/{template}: {warnings}",
                    }
                )
                break

    if "Selected fallback-reference decomposition" in all_text and not any(
        phrase in all_text for phrase in _FALLBACK_PHRASES
    ):
        issues.append(
            {
                "sample": sample_id,
                "severity": "warning",
                "category": "fallback_decomposition_without_context",
                "detail": "fallback-reference decomposition lacks explanatory context",
            }
        )
    therapy_lines = [
        line
        for line in summary_text.splitlines()
        if line.startswith("- **") and ("Phase " in line or "Off-label" in line)
    ]
    for line in therapy_lines:
        lower = line.lower()
        if (
            any(token in lower for token in ("fusion", "mutation", "amplification"))
            and "confirm" not in lower
            and "verify" not in lower
        ):
            issues.append(
                {
                    "sample": sample_id,
                    "severity": "warning",
                    "category": "therapy_requires_molecular_confirmation",
                    "detail": line[:240],
                }
            )
            break
    return issues


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["severity", "category", "sample", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown(path: Path, rows: list[dict[str, str]], *, reports: Path) -> None:
    counts = Counter(row["category"] for row in rows)
    severities = Counter(row["severity"] for row in rows)
    lines = [
        f"# Generated Report Audit — {reports}",
        "",
        f"Audited issue count: {len(rows)}",
        f"Errors: {severities.get('error', 0)}; warnings: {severities.get('warning', 0)}",
        "",
        "## Categories",
        "",
    ]
    if counts:
        for category, count in counts.most_common():
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- No issues detected.")
    lines.extend(["", "## Findings", ""])
    for row in rows:
        lines.append(
            f"- **{row['severity']} / {row['category']} / {row['sample']}**: "
            f"{row['detail']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit generated report markdown/TSV artifacts for coherence.",
    )
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--infer-expected", action="store_true")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    truth = _load_truth(args.truth) if args.truth else {}
    compat = Compat(_registry_parent_map(), _broad_lineage_fn())
    issues: list[dict[str, str]] = []
    for group, sample_id, analysis_md in _iter_samples(args.reports):
        expected, _notes = _expected_for(group, sample_id, truth, args.infer_expected)
        paths = _sample_paths(analysis_md, sample_id)
        issues.extend(
            _sample_issues(
                sample_id=sample_id,
                expected=expected,
                paths=paths,
                compat=compat,
            )
        )

    issues.sort(key=lambda row: (row["severity"], row["category"], row["sample"]))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_csv, issues)
    _write_markdown(args.out_md, issues, reports=args.reports)
    counts = Counter(row["category"] for row in issues)
    print(f"audited {args.reports}: {len(issues)} issues")
    for category, count in counts.most_common(20):
        print(f"{category}: {count}")
    print(f"markdown -> {args.out_md}")
    print(f"csv -> {args.out_csv}")
    return 1 if any(row["severity"] == "error" for row in issues) else 0


if __name__ == "__main__":
    raise SystemExit(main())
