"""Guard tests keeping trufflepig's hardcoded taxonomy in lockstep with the
live pirlygenes registry (#46).

pirlygenes' cancer-type taxonomy is under active restructure (Phase C):
- 5.12 already moved `family` to lineage-only (OS/EWS/RMS -> sarcoma,
  NBL -> neuroendocrine, ATRT/MBL -> cns, WILMS/RB/HEPB/RT -> embryonal,
  `net` -> neuroendocrine, `rare` dissolved).
- 5.13 renamed the sarcoma atoms (OS -> SARC_OS, ...); 5.18 landed the
  neuroendocrine/AML code-rename pass (MEC -> NEC_MERKEL, MID_NET -> NET_MIDGUT,
  PANNET -> NET_PANCREAS, NBL_MYCN_amp -> NBL_MYCNamp, LAML_ELN_Adv ->
  LAML_ELNadv, ...), each keeping a backward-compat alias in CANCER_TYPE_ALIASES.

These guards turn a *silent* desync — a trufflepig curated key that no longer
matches a registry code or family — into a loud failure that names the exact
code/family to update and (via resolve_cancer_type) its new canonical name. They
are the mechanism that keeps trufflepig's categories/names matching pirlygenes as
the curation lands.
"""

from pirlygenes.gene_sets_cancer import cancer_type_registry, resolve_cancer_type


# Families retired by the 5.12 lineage-only ontology but still kept as harmless
# back-compat keys in trufflepig maps (so an older pirlygenes still resolves).
_RETIRED_FAMILIES = {
    "net",
    "rare",
    "pediatric-bone",
    "pediatric-soft",
    "pediatric-cns",
    "pediatric-net",
    "pediatric-embryonal",
    "pediatric-eye",
    "pediatric-liver",
}


def _live_codes():
    return set(cancer_type_registry()["code"].dropna().astype(str))


def _live_families():
    return set(cancer_type_registry()["family"].dropna().astype(str))


def _rename_hint(code):
    try:
        resolved = resolve_cancer_type(code)
    except Exception:
        return " (no alias — code removed from registry + aliases)"
    return f" (resolve_cancer_type -> {resolved!r})" if resolved != code else ""


def test_curated_codes_are_live_registry_codes():
    """Every cancer code trufflepig hardcodes in curated taxonomy data must be a
    current registry code. When the Phase-C rename lands this fails per stale
    code, with a hint naming the new canonical code to migrate to."""
    from trufflepig.literature_signatures import _SIGNATURE_ROWS
    from trufflepig.tumor_type_ontology import _CURATED_HIGH, _CURATED_LOW

    live = _live_codes()
    codes = set()
    for row in _SIGNATURE_ROWS:
        codes.add(str(row.cancer_code))
        codes.update(str(c) for c in row.parent_context_codes)
    codes.update(_CURATED_HIGH)
    codes.update(_CURATED_LOW)

    stale = [f"{code}{_rename_hint(code)}" for code in sorted(codes) if code not in live]
    assert not stale, (
        "trufflepig curated taxonomy references codes no longer in the pirlygenes "
        "registry — the Phase-C rename likely landed; migrate these (hint shows "
        f"the new canonical name): {stale}"
    )


def test_family_map_keys_are_live_or_known_retired():
    """Every family string trufflepig keys maps on must be a live registry family
    (or a documented retired alias). Catches both stale keys to prune and a newly
    curated family name that a map hasn't adopted."""
    from trufflepig.tumor_type_ontology import _FAMILY_LOW_GENES
    from trufflepig.cancer_type_evidence import _LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY
    from trufflepig.analyze.cancer_type_context import _REFERENCE_FAMILY_FALLBACKS

    live = _live_families()
    live_roots = {f.split("-", 1)[0] for f in live}
    keys = (
        set(_FAMILY_LOW_GENES)
        | set(_LOCAL_REFERENCE_CONTEXT_CODES_BY_FAMILY)
        | set(_REFERENCE_FAMILY_FALLBACKS)
    )
    unknown = [
        key
        for key in sorted(keys)
        # exact live family, family-root key (e.g. "carcinoma"/"heme"), or retired
        if key not in live and key not in live_roots and key not in _RETIRED_FAMILIES
    ]
    assert not unknown, (
        f"family-map keys neither live in the registry nor known-retired: {unknown}"
    )


def test_sarc_reference_is_parental_not_a_single_histology_subset():
    """SARC must stand for the *parental* sarcoma category, not a small single-
    histology subset (e.g. a TCGA-LMS slice). GCTB, ESS_*, and the round-cell
    SARC_* codes all fall back to SARC, so a narrow SARC reference would
    misrepresent every one of them.

    Today SARC draws on the full multi-histology TCGA sarcoma cohort. The true
    grand-union aggregate over sarcoma_lineage_codes is pirlygenes Phase C.3
    (pending: "computed aggregates SARC / TCGA_SARC / SARC_<hist>"); when it
    lands SARC should pool across cohorts. This guard fails loudly if SARC ever
    collapses onto a single subtype's / single-study reference.
    """
    from trufflepig.analyze import effective_expression_reference
    from pirlygenes.gene_sets_cancer import sarcoma_lineage_codes

    ref = effective_expression_reference("SARC")
    assert ref is not None and ref.reference_code == "SARC"
    # SARC has its own parental reference; it never borrows a subtype's cohort.
    assert ref.direct, f"SARC fell back to {ref.reference_code} ({ref.fallback_reason})"
    # Not a single GEO study (those are single-histology, e.g. GSE30929 = LPS).
    assert not ref.source.upper().startswith("GSE"), (
        f"SARC reference {ref.source!r} is a single-study cohort — SARC must pool "
        "the parental sarcoma category across histologies"
    )
    # The SARC source must not be the dedicated cohort of exactly one subtype
    # (i.e. one histology wearing the parental label).
    serving = []
    for code in sarcoma_lineage_codes():
        if code == "SARC":
            continue
        sub = effective_expression_reference(code)
        if sub is not None and sub.direct and sub.source == ref.source:
            serving.append(code)
    assert len(serving) != 1, (
        f"SARC reference {ref.source!r} is the dedicated cohort of a single "
        f"subtype {serving}; SARC must represent the parental category"
    )


def test_report_code_compatibility_distinguishes_entity_from_context():
    from trufflepig.cancer_ontology import (
        cancer_codes_context_compatible,
        cancer_codes_entity_compatible,
        molecular_status_parent_code,
        subtype_display_parent_code,
    )

    assert molecular_status_parent_code("READ_MSI") == "READ"
    assert molecular_status_parent_code("HNSC_HPV_pos") == "HNSC"
    assert subtype_display_parent_code("SARC_ASPS") == "SARC"

    assert cancer_codes_entity_compatible("READ", "READ_MSI")
    assert cancer_codes_entity_compatible("SARC", "SARC_ASPS")
    assert not cancer_codes_entity_compatible("READ", "COAD")

    assert cancer_codes_context_compatible("READ", "COAD", context_code="CRC")
    assert not cancer_codes_context_compatible("GBM", "COAD", context_code="CRC")


def test_sarcoma_membership_matches_pirlygenes():
    """trufflepig's sarcoma-membership notion must equal pirlygenes'
    canonical ``sarcoma_lineage_codes()`` — for every registry code. The
    single trufflepig accessor (``tumor_type_ontology.is_sarcoma_code``)
    is the only thing any call site should consult; this guard turns a
    future pirlygenes addition (a new SARC_* subtype, a re-parented
    histology) into a loud failure instead of a silent desync."""
    from trufflepig.tumor_type_ontology import (
        is_sarcoma_code,
        sarcoma_lineage_codes as tp_sarcoma_codes,
    )
    from pirlygenes.gene_sets_cancer import (
        cancer_type_registry,
        sarcoma_lineage_codes as pg_sarcoma_codes,
    )

    # The trufflepig accessor must be a pure pass-through of the pirlygenes set.
    assert set(tp_sarcoma_codes()) == set(pg_sarcoma_codes())

    canonical = set(pg_sarcoma_codes())
    mismatched = [
        code
        for code in cancer_type_registry()["code"].dropna().astype(str)
        if is_sarcoma_code(code) != (code in canonical)
    ]
    assert not mismatched, (
        "is_sarcoma_code disagrees with pirlygenes.sarcoma_lineage_codes() "
        f"for: {sorted(mismatched)}"
    )


def test_clinical_group_routes_every_sarcoma_code_to_the_sarcoma_group():
    """The `cancers` CLI clinical-group classifier must place every canonical
    sarcoma-lineage code in the sarcoma/bone/soft-tissue group via the shared
    membership API — not a hardcoded SARC_* allowlist that drifts."""
    from pirlygenes.gene_sets_cancer import (
        cancer_type_registry,
        sarcoma_lineage_codes,
    )
    from trufflepig.main import clinical_group_for_row

    reg = cancer_type_registry().fillna("")
    rows = {str(r["code"]): dict(r) for _, r in reg.iterrows()}
    misrouted = []
    for code in sarcoma_lineage_codes():
        row = rows.get(code)
        if row is None:
            continue
        if clinical_group_for_row(row) != "Sarcoma, bone, and soft-tissue tumors":
            misrouted.append(code)
    assert not misrouted, (
        f"sarcoma-lineage codes not routed to the sarcoma group: {sorted(misrouted)}"
    )


def test_every_live_family_is_classified_by_the_cancers_cli(capsys):
    """The `cancers` CLI clinical-group classifier must give every live family a
    real group label — a newly curated family must not fall through to the raw
    family string."""
    from trufflepig.main import print_cancer_registry

    for family in sorted(_live_families()):
        print_cancer_registry(family=family)
        out = capsys.readouterr().out
        assert "Clinical group:" in out, f"family {family!r} produced no clinical group"
        # The raw lineage family string should have been mapped to a titled
        # group, not echoed verbatim as the group label.
        assert f"Clinical group: {family}\n" not in out, (
            f"family {family!r} fell through to its raw name as the clinical group"
        )
