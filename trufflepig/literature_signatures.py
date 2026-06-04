"""Literature-backed marker signatures for cancer types without direct cohorts.

These are not expression reference cohorts. They are compact RNA marker
sets that help explain and test subtype hypotheses when the closest usable
cohort is a parent or related cancer type.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import pandas as pd


@dataclass(frozen=True)
class LiteratureSignature:
    cancer_code: str
    parent_context_codes: tuple[str, ...]
    marker_genes: tuple[str, ...]
    source: str
    rationale: str
    confidence: str = "moderate"
    promote_report_scope: bool = False
    min_primary_tpm: float = 5.0
    min_support_genes: int = 2
    support_min_tpm: float = 2.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SIGNATURE_ROWS: tuple[LiteratureSignature, ...] = (
    LiteratureSignature(
        "SARC_WDLPS",
        ("SARC",),
        ("MDM2", "CDK4", "HMGA2", "FRS2", "TSPAN31"),
        "PMCID:PMC8001728",
        "well-differentiated/dedifferentiated liposarcoma 12q13-15 amplicon",
    ),
    LiteratureSignature(
        "SARC_MYXLPS",
        ("SARC",),
        ("DDIT3", "FUS", "EWSR1", "CTAG1B", "PPARG"),
        "PMID:33465826;PMID:35273728",
        "myxoid liposarcoma DDIT3-rearranged/lipogenic program",
    ),
    LiteratureSignature(
        "SARC_DSRCT",
        ("SARC",),
        ("WT1", "EWSR1", "KRT8", "KRT18", "DES"),
        "PMID:1847340",
        "DSRCT EWSR1-WT1 and polyphenotypic epithelial/myogenic markers",
    ),
    LiteratureSignature(
        "SARC_GIST",
        ("SARC",),
        ("KIT", "ANO1", "PDGFRA", "ETV1", "SDHB"),
        "PMID:15215166",
        "GIST KIT/DOG1/PDGFRA expression program",
    ),
    LiteratureSignature(
        "SARC_MPNST",
        ("SARC",),
        ("SOX10", "S100B", "NGFR", "NF1", "SUZ12"),
        "curated_pathology_literature",
        "MPNST Schwann-lineage and NF1/PRC2-associated markers",
    ),
    LiteratureSignature(
        "SARC_ANGIO",
        ("SARC",),
        ("ERG", "PECAM1", "CD34", "VWF", "KDR"),
        "PMID:25352641",
        "angiosarcoma endothelial marker program",
    ),
    LiteratureSignature(
        "NBL",
        ("NBL_MYCN_nonamp", "NBL_MYCN_amp"),
        ("PHOX2B", "TH", "ALK", "MYCN", "B4GALNT1"),
        "curated_neuroblastoma_literature",
        "neuroblastoma sympathoadrenal/neuroblastic marker program",
    ),
    LiteratureSignature(
        "MBL_WNT",
        ("MBL",),
        ("WIF1", "CTNNB1", "DKK1", "LEF1", "TNC"),
        "PMID:22358457",
        "medulloblastoma WNT subgroup expression program",
    ),
    LiteratureSignature(
        "MBL_SHH",
        ("MBL",),
        ("GLI1", "PTCH1", "ATOH1", "SFRP1", "YAP1"),
        "PMID:22358457",
        "medulloblastoma SHH subgroup expression program",
    ),
    LiteratureSignature(
        "MBL_G3",
        ("MBL",),
        ("MYC", "OTX2", "GFI1B", "NPR3", "IMPG2"),
        "PMID:22358457",
        "medulloblastoma Group 3 expression program",
    ),
    LiteratureSignature(
        "MBL_G4",
        ("MBL",),
        ("KCNA1", "EOMES", "LMX1A", "GRM4", "OTX2"),
        "PMID:22358457",
        "medulloblastoma Group 4 expression program",
    ),
    LiteratureSignature(
        "MCL",
        ("DLBC",),
        ("CCND1", "SOX11", "CD5", "MS4A1", "CD79A"),
        "PMID:19016864;PMCID:PMC4650381",
        "mantle-cell lymphoma cyclin-D1/SOX11 B-cell program",
    ),
    LiteratureSignature(
        "FL",
        ("DLBC",),
        ("BCL2", "BCL6", "MME", "MS4A1", "CD79A"),
        "curated_lymphoma_literature",
        "follicular lymphoma germinal-center B-cell program",
    ),
    LiteratureSignature(
        "HL",
        ("DLBC",),
        ("TNFRSF8", "CD274", "PDCD1LG2", "PAX5", "FUT4"),
        "curated_lymphoma_literature",
        "classic Hodgkin/Reed-Sternberg marker program",
    ),
    LiteratureSignature(
        "BL",
        ("DLBC",),
        ("MYC", "BCL6", "MME", "CD79A", "MS4A1"),
        "curated_lymphoma_literature",
        "Burkitt lymphoma MYC/germinal-center B-cell program",
    ),
    LiteratureSignature(
        "MDS",
        ("LAML",),
        ("CD34", "KIT", "GATA2", "MECOM", "ANPEP"),
        "curated_myeloid_literature",
        "myelodysplastic immature myeloid/stem-progenitor marker program",
        confidence="low",
    ),
    LiteratureSignature(
        "MPN",
        ("LAML",),
        ("JAK2", "MPL", "CALR", "EPOR", "CSF3R"),
        "curated_myeloid_literature",
        "myeloproliferative driver/hematopoietic cytokine-axis marker set",
        confidence="low",
    ),
    LiteratureSignature(
        "CML",
        ("LAML",),
        ("BCR", "ABL1", "MPO", "ELANE", "LYZ"),
        "curated_myeloid_literature",
        "CML BCR-ABL1 plus granulocytic myeloid program",
    ),
    LiteratureSignature(
        "CTCL",
        ("T_ALL",),
        ("CCR4", "TOX", "KIR3DL2", "CD4", "IL2RA"),
        "curated_t_cell_lymphoma_literature",
        "cutaneous T-cell lymphoma helper/T-cell activation marker program",
    ),
    LiteratureSignature(
        "MID_NET",
        ("SCLC",),
        ("CHGA", "SYP", "INSM1", "SSTR2", "TPH1"),
        "PMID:26386079",
        "midgut neuroendocrine/carcinoid marker program",
    ),
    LiteratureSignature(
        "LUNG_NET_LC",
        ("SCLC",),
        ("CHGA", "SYP", "INSM1", "SSTR2", "NKX2-1"),
        "PMID:26386079",
        "pulmonary carcinoid neuroendocrine marker program",
    ),
    LiteratureSignature(
        "LUNG_NET_LCNEC",
        ("SCLC",),
        ("ASCL1", "NEUROD1", "INSM1", "CHGA", "SYP", "DLL3"),
        "PMID:33011388;PMID:26386079",
        "large-cell lung neuroendocrine carcinoma marker program",
    ),
    LiteratureSignature(
        "SCLC_ASCL1",
        ("SCLC",),
        ("ASCL1", "DLL3", "BCL2", "INSM1", "CHGA"),
        "PMID:33011388",
        "ASCL1-dominant SCLC subtype program",
    ),
    LiteratureSignature(
        "SCLC_NEUROD1",
        ("SCLC",),
        ("NEUROD1", "MYC", "INSM1", "CHGA", "SYP"),
        "PMID:33011388",
        "NEUROD1-dominant SCLC subtype program",
    ),
    LiteratureSignature(
        "SCLC_POU2F3",
        ("SCLC",),
        ("POU2F3", "TRPM5", "SOX9", "ASCL2", "AVIL"),
        "PMID:33011388",
        "POU2F3/tuft-cell-like SCLC subtype program",
    ),
    LiteratureSignature(
        "SCLC_YAP1",
        ("SCLC",),
        ("YAP1", "WWTR1", "VIM", "AXL", "CD44"),
        "PMID:33011388",
        "YAP1/mesenchymal-like SCLC subtype program",
        confidence="low",
    ),
    LiteratureSignature(
        "MEC",
        ("SCLC",),
        ("KRT20", "ATOH1", "CHGA", "SYP", "SOX2"),
        "curated_merkel_literature",
        "Merkel-cell carcinoma epithelial/neuroendocrine marker program",
    ),
    LiteratureSignature(
        "ACINIC",
        ("HNSC",),
        ("NR4A3", "ANO1", "AQP5", "SOX10", "BCL6"),
        "PMID:31094928;PMID:34535188",
        "acinic-cell carcinoma NR4A3/acinar-salivary marker program",
    ),
    LiteratureSignature(
        "ADCC",
        ("HNSC",),
        ("MYB", "MYBL1", "NFIB", "KIT", "SOX10"),
        "PMID:20702610;PMID:21572406",
        "adenoid-cystic carcinoma MYB/MYBL1 salivary marker program",
    ),
    LiteratureSignature(
        "MTC",
        ("THCA",),
        ("CALCA", "RET", "CEACAM5", "CHGA", "SYP"),
        "PMID:28929017;PMID:26494386",
        "medullary-thyroid C-cell/neuroendocrine marker program",
    ),
    LiteratureSignature(
        "NPC",
        ("HNSC",),
        ("KRT5", "KRT14", "TP63", "SOX2", "CD274"),
        "curated_head_neck_literature",
        "nasopharyngeal squamous/EBV-associated carcinoma marker program",
    ),
    LiteratureSignature(
        "HCL",
        ("DLBC",),
        ("ANXA1", "ITGAE", "ITGAX", "IL2RA", "MS4A1"),
        "curated_lymphoma_literature",
        "hairy-cell leukemia ANXA1/CD103/CD11c/CD25 B-cell program",
    ),
    LiteratureSignature(
        "PCN",
        ("MM",),
        ("MZB1", "XBP1", "SDC1", "TNFRSF17", "SLAMF7"),
        "curated_plasma_cell_literature",
        "plasmacytoma plasma-cell/myeloma marker program",
    ),
    LiteratureSignature(
        "SARC_EPITH",
        ("SARC",),
        ("SMARCB1", "KRT8", "KRT18", "EPCAM", "CD34"),
        "PMID:23060122;PMID:27145123",
        "epithelioid sarcoma epithelial/CD34 program with SMARCB1 loss caveat",
    ),
    LiteratureSignature(
        "SARC_DFSP",
        ("SARC",),
        ("COL1A1", "PDGFB", "CD34", "PDGFRB", "APOD"),
        "PMID:19805687",
        "DFSP COL1A1-PDGFB/CD34 fibroblastic marker program",
    ),
    LiteratureSignature(
        "SARC_ASPS",
        ("SARC",),
        ("TFE3", "ASPSCR1", "MET", "ANGPTL2", "KDR"),
        "PMID:12835757",
        "alveolar soft-part sarcoma ASPSCR1-TFE3/angiogenic marker program",
    ),
    LiteratureSignature(
        "SARC_CCS",
        ("SARC",),
        ("MITF", "SOX10", "MLANA", "PMEL", "TYR"),
        "PMID:14576204",
        "clear-cell sarcoma melanocytic/EWSR1-ATF1 marker program",
    ),
    LiteratureSignature(
        "SARC_IFS",
        ("SARC",),
        ("ETV6", "NTRK3", "COL1A1", "COL1A2", "VIM"),
        "curated_infantile_fibrosarcoma_literature",
        "infantile fibrosarcoma ETV6-NTRK3/fibroblastic marker program",
    ),
    LiteratureSignature(
        "SARC_EHE",
        ("SARC",),
        ("CAMTA1", "WWTR1", "ERG", "FLI1", "PECAM1"),
        "PMID:22265237",
        "EHE WWTR1-CAMTA1/YAP1-TFE3 vascular marker program",
    ),
    LiteratureSignature(
        "SARC_PEC",
        ("SARC",),
        ("PMEL", "MLANA", "MITF", "ACTA2", "TFE3"),
        "curated_pecoma_literature",
        "PEComa melanocytic/smooth-muscle marker program",
    ),
    LiteratureSignature(
        "SARC_KS",
        ("SARC",),
        ("KDR", "FLT4", "PROX1", "PDPN", "PECAM1"),
        "curated_kaposi_literature",
        "Kaposi sarcoma lymphatic/endothelial marker program",
    ),
    LiteratureSignature(
        "SARC_SFT",
        ("SARC",),
        ("STAT6", "NAB2", "CD34", "ALDH1A1", "IGF2"),
        "PMID:23313952;PMID:27039712",
        "solitary-fibrous tumor NAB2-STAT6/CD34 marker program",
    ),
    LiteratureSignature(
        "SARC_IMT",
        ("SARC",),
        ("ALK", "ROS1", "NTRK3", "PDGFRB", "ACTA2"),
        "PMID:26647767;PMCID:PMC4125481",
        "inflammatory myofibroblastic tumor kinase-fusion/myofibroblastic program",
    ),
    LiteratureSignature(
        "GCTB",
        ("SARC",),
        ("TNFSF11", "CSF1", "RUNX2", "ALPL", "SPP1"),
        "PMID:22102368;PMCID:PMC1603441",
        "giant-cell tumor of bone RANKL/osteoclastogenic stromal program",
    ),
    LiteratureSignature(
        "ESS_LG",
        ("SARC",),
        ("ESR1", "PGR", "MME", "JAZF1", "SUZ12"),
        "PMID:27142911",
        "low-grade endometrial stromal sarcoma ER/PR/CD10 fusion-associated program",
    ),
    LiteratureSignature(
        "ESS_HG",
        ("SARC",),
        ("BCOR", "CCND1", "YWHAE", "NUTM2A", "NUTM2B"),
        "PMID:28621321",
        "high-grade endometrial stromal sarcoma BCOR/YWHAE/cyclin-D1 program",
    ),
    LiteratureSignature(
        "SARC_EMC",
        ("SARC",),
        ("NR4A3", "EWSR1", "TAF15", "TCF12", "S100B"),
        "PMID:29327709",
        "extraskeletal myxoid chondrosarcoma NR4A3-rearranged marker program",
    ),
    LiteratureSignature(
        "SARC_CIC",
        ("SARC",),
        ("ETV4", "ETV1", "ETV5", "WT1", "CCND2"),
        "PMID:28765524",
        "CIC-rearranged sarcoma CIC-DUX4 PEA3 (ETV) + WT1 target program",
    ),
    LiteratureSignature(
        "SARC_BCOR",
        ("SARC",),
        ("BCOR", "CCNB3", "SATB2", "TLE1", "VIM"),
        "PMID:22387997",
        "BCOR-rearranged sarcoma BCOR/CCNB3 upregulation program",
    ),
    LiteratureSignature(
        "SARC_MYOEP",
        ("SARC",),
        ("S100B", "SOX10", "KRT8", "KRT18", "GFAP"),
        "curated_soft_tissue_myoepithelial_literature",
        "soft-tissue myoepithelial keratin/S100/SOX10 program (EWSR1-POU5F1/PBX1)",
    ),
    LiteratureSignature(
        "SARC_SMARCA4",
        ("SARC",),
        ("SOX2", "SALL4", "CD34", "MYC", "NES"),
        "curated_smarca4_thoracic_sarcoma_literature",
        "SMARCA4-deficient thoracic sarcoma SOX2/SALL4 dedifferentiation program (BRG1 loss)",
    ),
)


@lru_cache(maxsize=1)
def literature_signatures_by_code() -> dict[str, LiteratureSignature]:
    return {row.cancer_code: row for row in _SIGNATURE_ROWS}


def literature_signature(code: str | None) -> LiteratureSignature | None:
    if not code:
        return None
    return literature_signatures_by_code().get(str(code).strip())


def literature_signature_rules_df() -> pd.DataFrame:
    """Return signatures in the rare-RNA-surrogate rule schema."""
    rows: list[dict[str, object]] = []
    for signature in _SIGNATURE_ROWS:
        genes = tuple(dict.fromkeys(signature.marker_genes))
        if not genes:
            continue
        rows.append(
            {
                "rule_id": f"lit_{signature.cancer_code.lower()}",
                "cancer_code": signature.cancer_code,
                "primary_gene": genes[0],
                "min_tpm": signature.min_primary_tpm,
                "required_support_genes": ";".join(genes[1:]),
                "min_support_genes": min(signature.min_support_genes, len(genes) - 1),
                "support_min_tpm": signature.support_min_tpm,
                "expected_absent_genes": "",
                "absent_max_tpm": 0.0,
                "exclusion_genes": "",
                "exclusion_max_tpm": 0.0,
                "context_codes": ";".join(signature.parent_context_codes),
                "context_top_k": 5,
                "excluded_context_codes": "",
                "confidence": signature.confidence,
                "promote_report_scope": signature.promote_report_scope,
                "basis": signature.rationale,
                "confirmatory_tests": (
                    "orthogonal pathology review; IHC/FISH/fusion or mutation "
                    "testing as appropriate for this entity"
                ),
                "caveat": (
                    "Literature marker signature only; use as subtype evidence "
                    "inside the related expression context, not as a standalone "
                    "expression reference cohort."
                ),
                "source": signature.source,
            }
        )
    return pd.DataFrame(rows)
