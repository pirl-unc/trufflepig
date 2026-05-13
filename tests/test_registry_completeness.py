"""Registry-completeness contract.

Every **leaf** cancer-type code in ``cancer-type-registry.csv``
(parent_code empty) must carry a minimum package of information:

1. **Expression data** — either an ``FPKM_<code>`` column in
   ``pan-cancer-expression.csv`` (the TCGA-style pan-cancer reference)
   or a row-set in ``subtype-deconvolved-expression.csv.gz``.
2. **Lineage panel** — at least five genes registered in
   ``lineage-genes.csv`` (also enforced by #170's floor test).
3. **Biomarker** — at least one row in ``cancer-key-genes.csv``
   with ``role=biomarker``.
4. **Therapy target** — at least one row with ``role=target``.
5. **Matched-normal reference** — at least one row in
   ``tumor-up-vs-matched-normal.csv`` or
   ``heme-tumor-up-vs-matched-normal.csv``. Drives the
   ``sample-matched-normal-*.png`` figures and the
   over-predicted-by-matched-normal attribution in targets.md.
6. **Therapy-response axis panel** — at least one row in
   ``therapy-response-signatures.csv`` with this code in its
   ``cancer_context`` column (not just ``pan_cancer``). Drives the
   ``sample-subtype-signature.png`` therapy-axis plot and the
   disease-state synthesis line in summary.md.

Subtype rows (``parent_code`` set) are exempt — the parent carries the
minimum and the subtype inherits context via the mixture-cohort
mechanism (#171) or key-genes subtype_key lookup.

This test pins the contract. When a new entity is added to the
registry, this test flags the gap. The ``_TOLERATED_GAPS`` set below
enumerates codes that are allowed to miss one or more fields today —
the goal is to shrink it to empty over time. Each entry lists the
specific fields that code is allowed to be missing.

Fields #5 and #6 are a registry contract expansion (issue #199) —
every clinician-facing markdown should read the same way regardless
of which cancer type the sample landed in. Both got introduced with
a large baseline of tolerated gaps; individual follow-up PRs shrink
the list as matched-normal medians and cancer-specific axis panels
get curated per family.
"""

import pandas as pd

from trufflepig._data import DATA_DIR as _DATA_DIR
from pirlygenes.gene_sets_cancer import (
    cancer_type_registry,
    pan_cancer_expression,
    subtype_deconvolved_expression,
)


# ── Tolerated gaps ────────────────────────────────────────────────────
#
# Format: ``code: set of allowed-to-miss fields``. As gaps are closed,
# shrink these entries. A code with an empty set here passes every
# minimum — remove the entry entirely when that happens.
#
# Currently tolerated:
#   - Rare / pediatric / sarcoma entities that need a dedicated curation
#     pass for biomarker + therapy rows. Expression data may also be
#     missing for entities where no open cohort exists yet.
#   - Heme / NET entities whose expression cohort downloads are
#     deferred (MMRF CoMMpass, TARGET ALL, George 2018 lung NE, etc.).
#
# Re-run the audit after each gap-closure PR to see which codes can be
# removed from this allowlist.

# Codes currently lacking a matched-normal reference (tumor-up-vs-matched-
# normal data). Shrunk in two sweeps:
#   - v4.49.3 HPA-direct: LUAD, MESO, GBM, LGG, SKCM, UVM, UCEC, UCS,
#     TGCT, THYM, PCPG (11 TCGA-covered codes)
#   - v4.50.1 subtype-deconvolved: OS, EWS, ATRT, MBL, RB, RMS_ERMS/ARMS/
#     SSRMS, RT, HEPB, PANNET, CHON, NBL, WILMS (14 pediatric + NET codes
#     whose tumor reference lives in subtype-deconvolved-expression)
# See scripts/generate_matched_normal.py for both recipes.
_MISSING_MATCHED_NORMAL = frozenset(
    {
        # Closed v4.50.4: SCLC (subtype-deconvolved, lung; ASCL1 / INSM1 /
        # CHGA textbook NE panel), ACC (hand-curated IGF2 override;
        # PMID:23095921). Remaining codes are blocked on tumor-expression
        # data acquisition (#151 / #152 / #197) or need CTA / aberrantly-
        # derepressed markers per entity (heme, NET, rare sarcomas).
        # Rare-entity / head-neck not easily auto-generated
        "ACINIC",
        "ADCC",
        "CHOR",
        "NPC",
        "NUTM",
        # Heme — blocked on tumor expression data (#151 / #197)
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
        # Rare sarcoma subtypes without subtype-deconvolved data
        "ESS_HG",
        "ESS_LG",
        "GCTB",
        "SARC",
        "SARC_IFS",
        # NET axis — blocked on expression data (#152 / #197)
        "LUNG_NET_LC",
        "LUNG_NET_LCNEC",
        "MEC",
        "MID_NET",
        "MTC",
    }
)

# Codes currently lacking a cancer-specific therapy-response axis panel
# (rows in ``therapy-response-signatures.csv`` mentioning this code in
# ``cancer_context``, beyond the pan_cancer fallback). As of v4.48.1,
# only BRCA / COAD / LUAD / LUSC / NBL / PRAD / SKCM ship a curated
# cancer-specific panel. GBM / THCA still get the generalized pan-cancer
# MAPK activity score at runtime, but no longer have a cancer-specific
# therapy-axis row after the MPAS panel was generalized.
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
    """Seed ``matched_normal`` / ``therapy_axis`` into ``base`` based on
    the baseline-coverage frozen sets above. Any explicit override in
    ``_TOLERATED_GAPS_EXPLICIT`` below still wins."""
    fields = set(base)
    if code in _MISSING_MATCHED_NORMAL:
        fields.add("matched_normal")
    if code in _MISSING_THERAPY_AXIS:
        fields.add("therapy_axis")
    return fields


_TOLERATED_GAPS_EXPLICIT = {
    # Heme entries still awaiting expression-data integration
    "CLL": {"expression", "lineage"},
    "MM": {"expression"},  # MMRF CoMMpass deferred
    "HL": {"expression"},
    "MCL": {"expression"},
    "B_ALL": {"expression", "lineage", "biomarker", "therapy"},  # TARGET ALL deferred
    "T_ALL": {"expression"},
    "BL": {"expression"},
    "FL": {"expression"},
    "HCL": {"expression"},
    "CTCL": {"expression"},  # biomarker + therapy + lineage filled v4.48.0
    "CML": {"expression"},
    "MDS": {"expression", "lineage", "biomarker", "therapy"},
    "MPN": {"expression", "lineage", "biomarker", "therapy"},
    # NET axis — PANNET shipped v4.47.0; rest pending
    "LUNG_NET_LC": {"expression"},  # George 2018 deferred
    "LUNG_NET_LCNEC": {"expression", "lineage", "biomarker", "therapy"},
    "MID_NET": {"expression"},  # small GEO deferred
    "MEC": {"expression"},  # Merkel cohort deferred
    "MTC": {"expression"},  # small GEO deferred
    # Pediatric entities — expression in subtype-deconvolved already
    # for OS/EWS/NBL/RMS_*/ATRT/MBL/RT via Treehouse; lineage panels
    # still need curation.
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
    # TGCT is chemo-dominant (BEP); the ``cancer-key-genes`` curation
    # bar explicitly leaves its therapy panel empty because no
    # clinician-validated molecular-targeted therapy exists. Pinned
    # by ``test_tgct_is_biomarker_only``; mirrored here as a tolerated
    # gap so both contracts hold.
    "TGCT": {"therapy"},
    # Rare entities — need dedicated curation
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


# Final tolerated-gap lookup — explicit 4-field gaps from above, unioned
# with the matched-normal / therapy-axis baseline for every leaf code that
# lacks those. Codes not in ``_TOLERATED_GAPS_EXPLICIT`` may still have
# ``matched_normal`` or ``therapy_axis`` in their tolerated set (via
# ``_auto_tolerate``).
def _build_tolerated_gaps():
    from pirlygenes.gene_sets_cancer import cancer_type_registry

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
    """Return a dict ``{code: {field: bool}}`` for every leaf entry."""
    reg = cancer_type_registry()
    leaf = reg[reg["parent_code"].fillna("").astype(str).eq("")]

    pan = pan_cancer_expression()
    pan_codes = {c.replace("FPKM_", "") for c in pan.columns if c.startswith("FPKM_")}
    sub = subtype_deconvolved_expression()
    sub_codes = set(sub["cancer_code"].dropna().unique()) if sub is not None else set()

    # lineage
    ln = pd.read_csv(_DATA_DIR / "lineage-genes.csv")
    ln_codes = {code for code, group in ln.groupby("Cancer_Type") if len(group) >= 5}

    # key-genes
    key = pd.read_csv(_DATA_DIR / "cancer-key-genes.csv")
    biomarker_codes = set(key[key["role"] == "biomarker"]["cancer_code"].dropna())
    therapy_codes = set(key[key["role"] == "target"]["cancer_code"].dropna())

    # matched-normal — union of solid + heme reference files
    mn_solid = pd.read_csv(_DATA_DIR / "tumor-up-vs-matched-normal.csv")
    mn_heme = pd.read_csv(_DATA_DIR / "heme-tumor-up-vs-matched-normal.csv")
    matched_normal_codes = set(mn_solid["cancer_code"].dropna().unique()) | set(
        mn_heme["cancer_code"].dropna().unique()
    )

    # therapy-response axis panel — ``cancer_context`` can be a
    # semicolon-separated list; split and skip the ``pan_cancer``
    # fallback so only cancer-specific curation counts.
    ts = pd.read_csv(_DATA_DIR / "therapy-response-signatures.csv")
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
    """Every entry in ``_TOLERATED_GAPS`` must point at a real
    registry code. Prevents silent drift if a code is renamed."""
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
    """The baseline missing sets must reference real registry codes."""
    from pirlygenes.gene_sets_cancer import cancer_type_registry

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
    """If the baseline sets claim a code is missing matched-normal /
    therapy-axis but the data actually present covers that code, the
    set has drifted and should be shrunk. Conversely, if a code has
    the data missing but isn't in the baseline set, the contract will
    misfire. Both are drift signals."""
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
    """Informational — print the current gap distribution. Not a
    contract assertion; just a way to see progress in CI logs."""
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
    assert complete >= 0  # always passes; smoke
