import re

import pandas as pd

from pirlygenes.gene_sets_cancer import cancer_type_registry

from trufflepig.analyze import effective_expression_reference, expression_reference_options
from trufflepig.common import ensembl_id_to_symbol_map
from trufflepig.literature_signatures import (
    _SIGNATURE_ROWS,
    literature_signature,
    literature_signature_rules_df,
)
from trufflepig.rare_inference import (
    infer_rare_cancer_marker_hypotheses_from_rna,
    rare_cancer_rna_surrogate_rules_df,
)


# A ``source`` field is a ``;``-separated list of citation tokens, each of which
# must be a well-formed PubMed/PMC id or a curated-label. This is a *format*
# guard — it catches typos and stray formats (a bare number, ``PMID:PMC123``, a
# truncated id), the cheap class of error. It can't tell a wrong-but-well-formed
# PMID from a right one (that needs the offline NCBI audit), but it stops typos
# at CI time.
_CITATION_TOKEN = re.compile(r"^(PMID:\d+|PMCID:PMC\d+|curated_[a-z0-9_]+_literature)$")


def test_signature_source_citations_are_well_formed():
    bad = []
    for row in _SIGNATURE_ROWS:
        for token in str(row.source).split(";"):
            token = token.strip()
            if not _CITATION_TOKEN.fullmatch(token):
                bad.append(f"{row.cancer_code}: {token!r}")
    assert not bad, (
        "malformed citation token(s) in literature_signatures source fields "
        f"(expected PMID:<digits> / PMCID:PMC<digits> / curated_*_literature): {bad}"
    )


def _expression_frame(tpm_by_symbol):
    symbol_to_id = {symbol: ensg for ensg, symbol in ensembl_id_to_symbol_map().items()}
    rows = []
    for symbol, tpm in tpm_by_symbol.items():
        ensg = symbol_to_id.get(symbol)
        if not ensg:
            continue
        rows.append(
            {
                "ensembl_gene_id": ensg,
                "canonical_gene_name": symbol,
                "TPM": float(tpm),
            }
        )
    return pd.DataFrame(rows)


def test_all_codes_without_direct_expression_reference_have_literature_signature():
    missing = []
    mismatched_context = []
    reg = cancer_type_registry()
    # Abstract grouping nodes (a code that is the parent_code of other codes,
    # e.g. CRC -> COAD/READ) are not classifiable cohorts — the classifier scores
    # the child cohorts, and the ontology walk only stops at the parent to report
    # the tied children as a set. They don't need their own literature signature.
    parent_nodes = set(reg["parent_code"].dropna().astype(str)) - {"", "nan"}
    # Nor do aggregate/subtype/scope codes that aren't surrogate-inference targets:
    # ontology_level == "grouping" unions are scored via their constituent cohorts;
    # "molecular_subtype" codes classify as their parent (the molecular axis —
    # MSI/EBV/HER2/... — is orthogonal to the expression signature); and
    # "evidence_scope" buckets (NET_NONPANCREATIC, NEN_G3_EXTRAPULMONARY — oncoref
    # #326) are literature-pooling scopes with no own marker program, scored via
    # their members. An "_UNCLASSIFIED" bucket has no defining marker by definition.
    # Gate on oncoref's semantic ontology_level, not a hand-maintained code list.
    aggregate_or_subtype = (
        set(
            reg.loc[
                reg["ontology_level"].astype(str).isin(
                    ("grouping", "molecular_subtype", "evidence_scope")
                ),
                "code",
            ].astype(str)
        )
        if "ontology_level" in reg.columns
        else set()
    )
    for code in reg["code"].dropna().astype(str):
        if code in parent_nodes or code in aggregate_or_subtype:
            continue
        if code.endswith("_UNCLASSIFIED"):
            continue
        if expression_reference_options(code, include_fallback=False):
            continue
        signature = literature_signature(code)
        if signature is None:
            missing.append(code)
            continue
        reference = effective_expression_reference(code)
        if reference is None:
            mismatched_context.append(f"{code}: no effective reference")
            continue
        if reference.reference_code not in set(signature.parent_context_codes):
            mismatched_context.append(
                f"{code}: {reference.reference_code} not in "
                f"{signature.parent_context_codes}"
            )

    assert not missing
    assert not mismatched_context


def test_literature_signature_rules_are_rare_surrogate_compatible():
    rules = literature_signature_rules_df()

    assert not rules.empty
    assert rules["rule_id"].is_unique
    assert {"lit_sarc_gist", "lit_sarc_wdlps", "lit_mbl_wnt"} <= set(
        rules["rule_id"]
    )
    assert (rules["context_codes"].astype(str).str.len() > 0).all()
    assert (rules["primary_gene"].astype(str).str.len() > 0).all()
    assert (rules["required_support_genes"].astype(str).str.len() > 0).all()
    promoting = rules[rules["promote_report_scope"].astype(bool)]
    assert set(promoting["rule_id"]) == {"lit_bl"}
    bl = promoting.iloc[0]
    assert int(bl["min_support_genes"]) == 4


def test_combined_rare_surrogate_rules_include_literature_overlay():
    rules = rare_cancer_rna_surrogate_rules_df()

    assert "nutm_nutm1" in set(rules["rule_id"])
    assert "lit_sarc_gist" in set(rules["rule_id"])
    assert "lit_sclc_pou2f3" in set(rules["rule_id"])


def test_literature_signature_can_emit_non_promoting_marker_hypothesis():
    findings = infer_rare_cancer_marker_hypotheses_from_rna(
        _expression_frame(
            {
                "KIT": 25.0,
                "ANO1": 8.0,
                "PDGFRA": 4.0,
                "ETV1": 5.0,
            }
        ),
        {
            "cancer_type": "SARC",
            "candidate_trace": [{"code": "SARC", "support_fraction_of_top": 1.0}],
        },
    )

    gist = next(row for row in findings if row["cancer_type"] == "SARC_GIST")
    assert gist["rule_id"] == "lit_sarc_gist"
    assert gist["surrogate"] == "KIT"
    assert set(gist["support_genes"]) >= {"ANO1", "PDGFRA"}
    assert gist["promote_report_scope"] is False


def test_literature_signature_prompt_requires_active_expression_context():
    findings = infer_rare_cancer_marker_hypotheses_from_rna(
        _expression_frame(
            {
                "CTNNB1": 150.0,
                "KRT8": 40.0,
                "KRT18": 45.0,
                "BRAF": 6.0,
                "EPCAM": 20.0,
            }
        ),
        {
            "cancer_type": "ACC",
            "candidate_trace": [
                {"code": "ACC", "support_fraction_of_top": 1.0},
                {"code": "LGG", "support_fraction_of_top": 0.72},
            ],
        },
    )

    assert all(row["cancer_type"] != "CRANIO" for row in findings)


def test_literature_signature_prompt_emits_when_context_is_top():
    findings = infer_rare_cancer_marker_hypotheses_from_rna(
        _expression_frame(
            {
                "CTNNB1": 150.0,
                "KRT8": 40.0,
                "KRT18": 45.0,
                "BRAF": 6.0,
                "EPCAM": 20.0,
            }
        ),
        {
            "cancer_type": "LGG",
            "candidate_trace": [
                {"code": "LGG", "support_fraction_of_top": 1.0},
                {"code": "ACC", "support_fraction_of_top": 0.72},
            ],
        },
    )

    cranio = next(row for row in findings if row["cancer_type"] == "CRANIO")
    assert cranio["rule_id"] == "lit_cranio"
    assert cranio["context_match_reason"] == "top_context"


def test_promoting_rare_surrogate_can_use_strong_near_top_parent_context():
    findings = infer_rare_cancer_marker_hypotheses_from_rna(
        _expression_frame(
            {
                "MYB": 25.0,
                "KRT7": 15.0,
                "SOX10": 8.0,
                "KIT": 5.0,
            }
        ),
        {
            "cancer_type": "LUAD",
            "candidate_trace": [
                {"code": "LUAD", "support_fraction_of_top": 1.0},
                {"code": "HNSC", "support_fraction_of_top": 0.93},
            ],
        },
    )

    adcc = next(row for row in findings if row["rule_id"] == "adcc_myb")
    assert adcc["cancer_type"] == "ADCC"
    assert adcc["context_match_reason"] == "near_top_context"
