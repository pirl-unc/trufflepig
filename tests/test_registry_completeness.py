"""Registry-completeness contract — cross-package.

Every **leaf** cancer-type code in ``cancer-type-registry.csv``
(parent_code empty) must carry a minimum package of information:

1. **Expression data** — either a ``<code>_TPM`` column in
   ``pan-cancer-expression.csv`` (the TCGA-style pan-cancer reference)
   or a row-set in ``subtype-deconvolved-expression.csv.gz``.
2. **Lineage panel** — at least five genes registered in
   ``lineage-genes.csv``.
3. **Biomarker** — at least one row in ``cancer-key-genes.csv``
   with ``role=biomarker``.
4. **Therapy target** — at least one row with ``role=target``.
5. **Matched-normal reference** — at least one row in
   ``tumor-up-vs-matched-normal.csv`` or
   ``heme-tumor-up-vs-matched-normal.csv``.
6. **Therapy-response axis panel** — at least one row in
   ``therapy-response-signatures.csv`` with this code in its
   ``cancer_context`` column (not just ``pan_cancer``).

Subtype rows (``parent_code`` set) are exempt.

This lives in trufflepig (not pirlygenes) because fields #1 and #5
cross the new package boundary — expression matrices ship with
trufflepig as of #23, while registry + curated panels stay in
pirlygenes. The contract still pins the integrated picture: when a new
entity is added to the registry, this test flags the gap.
"""

from pirlygenes import get_data
from pirlygenes.gene_sets_cancer import cancer_type_registry

from trufflepig.reference import (
    heme_tumor_up_vs_matched_normal,
    pan_cancer_expression,
    subtype_deconvolved_expression,
    tumor_up_vs_matched_normal,
)


_MISSING_MATCHED_NORMAL = frozenset(
    {
        "ACINIC",
        "ADCC",
        "CHOR",
        "NPC",
        "NUTM",
        "BL",
        "B_ALL",
        "CLL",
        "CML",
        "CTCL",
        "FL",
        "HCL",
        "HL",
        "MCL",
        "MDS",
        "MM",
        "MPN",
        "PCN",
        "T_ALL",
        "ESS_HG",
        "ESS_LG",
        "GCTB",
        "SARC",
        "SARC_IFS",
        "LUNG_NET_LC",
        "LUNG_NET_LCNEC",
        "MEC",
        "MID_NET",
        "MTC",
    }
)

_MISSING_THERAPY_AXIS = frozenset(
    {
        "ACC",
        "ACINIC",
        "ADCC",
        "ATRT",
        "BL",
        "BLCA",
        "B_ALL",
        "CESC",
        "CHOL",
        "CHON",
        "CHOR",
        "CLL",
        "CML",
        "CTCL",
        "DLBC",
        "ESCA",
        "ESS_HG",
        "ESS_LG",
        "EWS",
        "FL",
        "GCTB",
        "GBM",
        "HCL",
        "HEPB",
        "HL",
        "HNSC",
        "KICH",
        "KIRC",
        "KIRP",
        "LAML",
        "LGG",
        "LIHC",
        "LUNG_NET_LC",
        "LUNG_NET_LCNEC",
        "MBL",
        "MCL",
        "MDS",
        "MEC",
        "MESO",
        "MID_NET",
        "MM",
        "MPN",
        "MTC",
        "NPC",
        "NUTM",
        "OS",
        "OV",
        "PAAD",
        "PANNET",
        "PCN",
        "PCPG",
        "RB",
        "READ",
        "RMS_ARMS",
        "RMS_ERMS",
        "RMS_SSRMS",
        "RT",
        "SARC",
        "SARC_IFS",
        "SCLC",
        "STAD",
        "TGCT",
        "THCA",
        "THYM",
        "T_ALL",
        "UCEC",
        "UCS",
        "UVM",
        "WILMS",
    }
)


def _auto_tolerate(code, base):
    fields = set(base)
    if code in _MISSING_MATCHED_NORMAL:
        fields.add("matched_normal")
    if code in _MISSING_THERAPY_AXIS:
        fields.add("therapy_axis")
    return fields


_TOLERATED_GAPS_EXPLICIT = {
    "CLL": {"expression", "lineage"},
    "MM": {"expression"},
    "HL": {"expression"},
    "MCL": {"expression"},
    "B_ALL": {"expression", "lineage", "biomarker", "therapy"},
    "T_ALL": {"expression"},
    "BL": {"expression"},
    "FL": {"expression"},
    "HCL": {"expression"},
    "CTCL": {"expression"},
    "CML": {"expression"},
    "MDS": {"expression", "lineage", "biomarker", "therapy"},
    "MPN": {"expression", "lineage", "biomarker", "therapy"},
    "LUNG_NET_LC": {"expression"},
    "LUNG_NET_LCNEC": {"expression", "lineage", "biomarker", "therapy"},
    "MID_NET": {"expression"},
    "MEC": {"expression"},
    "MTC": {"expression"},
    "NBL": {"expression", "lineage"},
    "WILMS": {"expression", "lineage"},
    "HEPB": {"expression", "lineage", "biomarker", "therapy"},
    "ATRT": {"lineage"},
    "RB": {"expression", "lineage", "biomarker", "therapy"},
    "MBL": {"lineage"},
    "RT": {"lineage", "biomarker", "therapy"},
    "OS": {"lineage"},
    "EWS": {"lineage"},
    "RMS_ERMS": {"lineage"},
    "RMS_ARMS": {"lineage"},
    "RMS_SSRMS": {"lineage", "biomarker", "therapy"},
    "NUTM": {"lineage"},
    "TGCT": {"therapy"},
    "ACINIC": {"expression", "lineage"},
    "ADCC": {"expression", "lineage"},
    "NPC": {"expression", "lineage"},
    "CHOR": {"expression", "lineage"},
    "CHON": {"lineage"},
    "SARC_IFS": {"expression", "lineage", "biomarker", "therapy"},
    "GCTB": {"expression", "lineage", "biomarker", "therapy"},
    "ESS_LG": {"expression", "lineage", "biomarker", "therapy"},
    "ESS_HG": {"expression", "lineage", "biomarker", "therapy"},
    "PCN": {"expression", "lineage", "biomarker", "therapy"},
}


def _build_tolerated_gaps():
    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]
    out = {}
    for code in leaf["code"]:
        base = _TOLERATED_GAPS_EXPLICIT.get(code, set())
        seeded = _auto_tolerate(code, base)
        if seeded:
            out[code] = seeded
    return out


_TOLERATED_GAPS = _build_tolerated_gaps()


def _leaf_codes_with_coverage():
    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]

    pan = pan_cancer_expression()
    pan_codes = {c.removesuffix("_TPM") for c in pan.columns if c.endswith("_TPM")}
    sub = subtype_deconvolved_expression()
    sub_codes = set(sub["cancer_code"].dropna().unique()) if sub is not None else set()

    ln = get_data("lineage-genes")
    ln_codes = {code for code, group in ln.groupby("Cancer_Type") if len(group) >= 5}

    key = get_data("cancer-key-genes")
    biomarker_codes = set(key[key["role"] == "biomarker"]["cancer_code"].dropna())
    therapy_codes = set(key[key["role"] == "target"]["cancer_code"].dropna())

    mn_solid = tumor_up_vs_matched_normal()
    mn_heme = heme_tumor_up_vs_matched_normal()
    matched_normal_codes = set(mn_solid["cancer_code"].dropna().unique()) | set(
        mn_heme["cancer_code"].dropna().unique()
    )

    ts = get_data("therapy-response-signatures")
    therapy_axis_codes = set()
    for value in ts["cancer_context"].dropna():
        for part in str(value).split(";"):
            part = part.strip()
            if part and part != "pan_cancer":
                therapy_axis_codes.add(part)

    out = {}
    for _, row in leaf.iterrows():
        code = row["code"]
        out[code] = {
            "expression": code in pan_codes or code in sub_codes,
            "lineage": code in ln_codes,
            "biomarker": code in biomarker_codes,
            "therapy": code in therapy_codes,
            "matched_normal": code in matched_normal_codes,
            "therapy_axis": code in therapy_axis_codes,
        }
    return out


def test_every_leaf_passes_minimum_or_is_tolerated():
    coverage = _leaf_codes_with_coverage()
    violations = []
    for code, fields in coverage.items():
        tolerated = _TOLERATED_GAPS.get(code, set())
        missing = {f for f, present in fields.items() if not present}
        unexpected_missing = missing - tolerated
        if unexpected_missing:
            violations.append(f"{code} missing {sorted(unexpected_missing)}")

    assert not violations, (
        "Registry-completeness violations:\n  "
        + "\n  ".join(violations)
        + "\n\nEither fix the gap (add expression / lineage / biomarker / "
        "therapy / matched-normal / therapy-axis data) or extend "
        "``_TOLERATED_GAPS_EXPLICIT`` / ``_MISSING_MATCHED_NORMAL`` / "
        "``_MISSING_THERAPY_AXIS`` with a justified entry."
    )


def test_tolerated_gaps_only_list_real_codes():
    reg_codes = set(cancer_type_registry()["code"])
    unknown = set(_TOLERATED_GAPS) - reg_codes
    assert not unknown, (
        f"``_TOLERATED_GAPS`` references codes not in the registry: {sorted(unknown)}"
    )


def test_tolerated_fields_are_valid_names():
    valid = {
        "expression",
        "lineage",
        "biomarker",
        "therapy",
        "matched_normal",
        "therapy_axis",
    }
    for code, fields in _TOLERATED_GAPS.items():
        bad = fields - valid
        assert not bad, (
            f"``_TOLERATED_GAPS[{code!r}]`` has invalid field names {bad}; "
            f"valid are {sorted(valid)}"
        )


def test_baseline_missing_sets_only_list_real_codes():
    reg_codes = set(cancer_type_registry()["code"])
    unknown_mn = set(_MISSING_MATCHED_NORMAL) - reg_codes
    unknown_ax = set(_MISSING_THERAPY_AXIS) - reg_codes
    assert not unknown_mn, (
        f"``_MISSING_MATCHED_NORMAL`` references unknown codes: {sorted(unknown_mn)}"
    )
    assert not unknown_ax, (
        f"``_MISSING_THERAPY_AXIS`` references unknown codes: {sorted(unknown_ax)}"
    )


def test_baseline_missing_sets_match_current_data():
    coverage = _leaf_codes_with_coverage()
    missing_mn_actual = {c for c, f in coverage.items() if not f["matched_normal"]}
    missing_ax_actual = {c for c, f in coverage.items() if not f["therapy_axis"]}

    stale_mn = set(_MISSING_MATCHED_NORMAL) - missing_mn_actual
    stale_ax = set(_MISSING_THERAPY_AXIS) - missing_ax_actual
    new_mn = missing_mn_actual - set(_MISSING_MATCHED_NORMAL)
    new_ax = missing_ax_actual - set(_MISSING_THERAPY_AXIS)

    assert not stale_mn, (
        f"``_MISSING_MATCHED_NORMAL`` contains codes that NOW have data — "
        f"shrink the set: {sorted(stale_mn)}"
    )
    assert not stale_ax, (
        f"``_MISSING_THERAPY_AXIS`` contains codes that NOW have data — "
        f"shrink the set: {sorted(stale_ax)}"
    )
    assert not new_mn, (
        f"New leaf codes lack matched-normal and aren't in "
        f"``_MISSING_MATCHED_NORMAL``: {sorted(new_mn)}"
    )
    assert not new_ax, (
        f"New leaf codes lack therapy-axis panel and aren't in "
        f"``_MISSING_THERAPY_AXIS``: {sorted(new_ax)}"
    )


def test_completeness_progress_report(capsys):
    coverage = _leaf_codes_with_coverage()
    total = len(coverage)
    fields = (
        "expression",
        "lineage",
        "biomarker",
        "therapy",
        "matched_normal",
        "therapy_axis",
    )
    complete = sum(
        1 for c_fields in coverage.values() if all(c_fields[f] for f in fields)
    )
    per_field_missing = {
        f: sum(1 for c_fields in coverage.values() if not c_fields[f]) for f in fields
    }
    with capsys.disabled():
        print(
            f"\n[completeness] {complete}/{total} leaf codes have the "
            "full 6-field package (expression + lineage + biomarker + "
            "therapy + matched-normal + therapy-axis). "
            f"{len(_TOLERATED_GAPS)} codes in tolerated-gaps list."
        )
        print("  gaps by field:")
        for f, n in per_field_missing.items():
            print(f"    {f}: {n} codes")
    assert complete >= 0
