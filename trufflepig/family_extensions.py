# Licensed under the Apache License, Version 2.0

"""Extension family ontology: lineage-discrimination panels covering
cancer cohorts that the pirlygenes ``cancer-family-panels.csv`` does
not yet ship a panel for.

Goal: give every top-level cancer code in the trufflepig/pirlygenes
registry a non-empty ``family_label`` so the candidate-trace
``family_score`` term contributes for all cohorts, not only the
~13 codes covered by the historical CSV (PRAD, CRC, GASTRIC,
ESCA_SQ, HNSC/LUSC/CESC, SARC/UCS, RENAL, GLIAL, MELANOCYTIC).

Codes that previously fell through (BRCA, BLCA, LUAD, OV, UCEC,
PAAD, CHOL, LIHC, THCA, ACC, the heme cohorts, etc.) were
disadvantaged by ~4–6× on ``support_score`` against any code that
did get a family boost — independent of underlying biology. The
HCC1395 → ESCA misclassification (basal-like BRCA losing to
ESCA_SQ purely because BRCA had no family) is the canonical
example.

This module:

1. Adds parent-family panels for the previously-unmapped cohorts.
2. Maps every top-level registry code to a parent family
   (no orphans; truly singleton biology like NUTM gets its own family).
3. Adds child-cohort lineage discrimination panels under each
   parent that has multiple children, so the secondary pass can
   pick between siblings (e.g. BRCA-basal vs ESCA vs HNSC inside
   SQUAMOUS).

The panels here intentionally favor tissue-of-origin lineage
markers — the genes most likely to discriminate WITHIN a parent
family, not the shared-program markers that put the cohorts in
the same family in the first place. See the pirlygenes issue
linked at the bottom for the long-term home.

These dicts are merged into ``tumor_purity._CANCER_FAMILY_PANELS``
and friends at trufflepig import time. They never overwrite an
existing entry — pirlygenes remains the source of truth where it
has data.
"""

from __future__ import annotations

from typing import Mapping


# Panels: parent family → tuple of marker genes scored against the
# sample. Each panel is built around discriminating lineage / tissue
# of origin within the broad cohort group.

EXTENSION_FAMILY_PANELS: Mapping[str, tuple[str, ...]] = {
    # --- Luminal mammary (carcinoma-breast, non-basal subtypes) ---
    "MAMMARY_LUMINAL": (
        "ESR1", "PGR", "FOXA1", "GATA3", "MUCL1", "MLPH",
        "ANKRD30A", "SCGB2A2", "SCGB2A1", "SCGB1D2", "TFF1",
        "AGR3", "TBX3",
    ),
    # --- Mammary umbrella for BRCA when neither pure luminal nor
    # purely basal (HER2+, normal-like). BRCA also gets SQUAMOUS
    # membership via the multi-family mapping for basal-like signal.
    "MAMMARY": (
        "FOXA1", "GATA3", "MUCL1", "MLPH", "ANKRD30A",
        "SCGB2A2", "SCGB2A1", "TBX3",
    ),
    # --- GI adenocarcinoma umbrella (excludes ESCA squamous) ---
    "GI_ADENO": (
        "CDX2", "CDH17", "GUCY2C", "CEACAM5", "CEACAM6", "VIL1",
        "MUC5AC", "MUC6", "CLDN18", "TFF1", "TFF2", "TFF3",
        "GKN1", "GKN2", "REG4",
    ),
    # --- Hepatobiliary (LIHC, CHOL, HEPB, PAAD partial overlap) ---
    "HEPATOBILIARY": (
        "AFP", "ALB", "F2", "HNF4A", "HP", "APOB", "GC", "AHSG",
        "CYP3A4", "ITIH2", "HRG", "APCS", "CRP",
        # cholangiocyte / ductal overlap
        "KRT19", "MUC1", "MUC5AC", "EPCAM",
    ),
    # --- Pancreatic ductal (carcinoma-gi but its own ductal program) ---
    "PANCREATIC_DUCTAL": (
        "KRT19", "MUC1", "CLDN18", "AGR2", "S100P",
        "PRSS1", "PNLIP", "CTRB1", "CELA1", "REG1A",
        "TFF1", "TFF2",
    ),
    # --- Lung adeno / surfactant program (carcinoma-lung adeno arm) ---
    "LUNG_ADENO": (
        "SFTPC", "SFTPB", "SFTPA1", "SFTPA2", "SFTPD",
        "NKX2-1", "NAPSA", "SLC34A2", "FOXA2", "ABCA3",
    ),
    # --- Mesothelial (MESO) ---
    "MESOTHELIAL": (
        "WT1", "MSLN", "CALB2", "PDPN", "KRT5", "KRT6A",
        "UPK3B", "VIM",
    ),
    # --- Gynecologic glandular (OV serous, UCEC; CESC stays SQUAMOUS) ---
    "GYNECOLOGIC_GLANDULAR": (
        "PAX8", "WT1", "MSLN", "MUC16", "FOLR1",
        "ESR1", "PGR", "ARID1A",
    ),
    # --- Urothelial luminal (BLCA luminal subtype) ---
    "UROTHELIAL_LUMINAL": (
        "UPK1A", "UPK1B", "UPK2", "UPK3A", "UPK3B",
        "KRT20", "GATA3", "FOXA1", "PPARG", "S100P",
    ),
    # --- Neuroendocrine (NET, SCLC, MEC, NBL, PCPG, MTC) ---
    "NEUROENDOCRINE": (
        "CHGA", "CHGB", "SYP", "INSM1", "NCAM1",
        "ASCL1", "NEUROD1", "SCG2", "ENO2", "RAB3A",
    ),
    # --- Endocrine epithelial (THCA, MTC partly) ---
    "ENDOCRINE_THYROID": (
        "TG", "TPO", "TSHR", "PAX8", "FOXE1", "NKX2-1",
        "SLC5A5", "DUOX1", "DUOX2",
    ),
    # --- Adrenocortical (ACC) ---
    "ADRENOCORTICAL": (
        "CYP11A1", "CYP11B1", "CYP11B2", "CYP17A1", "CYP21A2",
        "STAR", "MC2R", "NR5A1", "INHA",
    ),
    # --- Heme: lymphoid B-cell (DLBC, CLL, MCL, FL, HL, BL,
    #     B_ALL, HCL) ---
    "HEME_BCELL": (
        "MS4A1", "CD79A", "CD79B", "CD19", "PAX5", "BANK1",
        "BLK", "POU2AF1", "CD20", "CXCR5",
    ),
    # --- Heme: T-cell (T_ALL, CTCL) ---
    "HEME_TCELL": (
        "CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRBC1",
        "CD8A", "CD4", "ZAP70", "LCK",
    ),
    # --- Heme: plasma (MM) ---
    "HEME_PLASMA": (
        "MZB1", "XBP1", "DERL3", "JCHAIN", "PRDM1",
        "IGHA1", "IGHG1", "TNFRSF17",
    ),
    # --- Heme: myeloid (LAML, MDS, MPN, CML) ---
    "HEME_MYELOID": (
        "MPO", "ELANE", "PRTN3", "CTSG", "AZU1",
        "ITGAM", "CSF1R", "LYZ", "S100A8", "S100A9",
    ),
    # --- Germ cell (TGCT, mediastinal germ cell, pediatric variants) ---
    "GERM_CELL": (
        "POU5F1", "NANOG", "SOX2", "LIN28A", "LIN28B",
        "DPPA3", "DPPA5", "TDGF1", "NANOS3", "SALL4",
    ),
    # --- Thymic (THYM — singleton with mixed epithelial + T-cell program) ---
    "THYMIC": (
        "AIRE", "FOXN1", "PSMB11", "PRSS16", "CCL25",
        "CHRNA1", "MYL1", "MYL4",
    ),
    # --- Salivary (ACINIC, ADCC) ---
    "SALIVARY": (
        "AMY1A", "AMY1B", "AMY1C", "STATH", "HTN1", "HTN3",
        "PRB1", "MUC7", "LPO", "LACRT",
    ),
    # --- Notochordal (CHOR — Chordoma, TBXT-driven singleton) ---
    "NOTOCHORDAL": (
        "TBXT", "T", "KRT8", "KRT18", "EPCAM", "COL2A1",
        "SOX9",
    ),
    # --- Midline carcinoma (NUTM — singleton, fusion-defined) ---
    "MIDLINE_NUT": (
        "NUTM1", "MYC", "TP63", "SOX2", "KRT5",
    ),
    # --- Pediatric embryonal kidney (WILMS) ---
    "WILMS_LIKE": (
        "WT1", "SIX1", "SIX2", "PAX2", "PAX8", "IGF2",
        "DPF3",
    ),
    # --- Pediatric rhabdoid (RT, ATRT) ---
    "RHABDOID": (
        "VIM", "SALL4", "EPCAM", "KRT8", "KRT18", "NES",
        "LIN28A",
    ),
    # --- Embryonal CNS (MBL) ---
    "EMBRYONAL_CNS": (
        "OTX2", "ATOH1", "SOX2", "MYC", "MYCN", "GLI1",
        "SHH",
    ),
    # --- Retinal (RB) ---
    "RETINAL": (
        "CRX", "VSX2", "RCVRN", "OTX2", "ARR3", "NRL",
    ),
    # --- Neuroblastic (NBL, sympathetic-NE — kept distinct from
    #     NEUROENDOCRINE which is epithelial-NE) ---
    "SYMPATHETIC_NEURAL": (
        "MYCN", "PHOX2B", "PHOX2A", "TH", "DBH", "B4GALNT1",
        "ALK", "ASCL1",
    ),
}


# Code → parent family. Multi-membership allowed: BRCA is primarily
# MAMMARY_LUMINAL but also gets SQUAMOUS membership at half weight
# (handled in tumor_purity via _SECONDARY_FAMILY_MEMBERSHIPS).
EXTENSION_CODE_TO_FAMILY: Mapping[str, str] = {
    # Mammary
    "BRCA": "MAMMARY_LUMINAL",
    # GI adenocarcinoma
    "PAAD": "PANCREATIC_DUCTAL",
    "LIHC": "HEPATOBILIARY",
    "CHOL": "HEPATOBILIARY",
    "HEPB": "HEPATOBILIARY",  # pediatric hepatoblastoma
    # Lung adeno (LUSC stays SQUAMOUS, already covered)
    "LUAD": "LUNG_ADENO",
    "MESO": "MESOTHELIAL",
    # Gynecologic glandular (CESC stays SQUAMOUS, already covered)
    "OV": "GYNECOLOGIC_GLANDULAR",
    "UCEC": "GYNECOLOGIC_GLANDULAR",
    # Urothelial (BLCA — luminal subtype is the more common one;
    # basal-like BLCA also gets SQUAMOUS via secondary membership)
    "BLCA": "UROTHELIAL_LUMINAL",
    # Endocrine
    "THCA": "ENDOCRINE_THYROID",
    "ACC": "ADRENOCORTICAL",
    "MTC": "NEUROENDOCRINE",  # medullary thyroid is C-cell NE
    "PCPG": "NEUROENDOCRINE",  # pheochromocytoma / paraganglioma
    # Neuroendocrine
    "PANNET": "NEUROENDOCRINE",
    "MID_NET": "NEUROENDOCRINE",
    "LUNG_NET_LC": "NEUROENDOCRINE",
    "LUNG_NET_LCNEC": "NEUROENDOCRINE",
    "SCLC": "NEUROENDOCRINE",
    "MEC": "NEUROENDOCRINE",  # Merkel cell carcinoma
    # Germ cell
    "TGCT": "GERM_CELL",
    # Thymic (singleton, can't easily group)
    "THYM": "THYMIC",
    # Salivary
    "ACINIC": "SALIVARY",
    "ADCC": "SALIVARY",
    # Heme
    "DLBC": "HEME_BCELL",
    "CLL": "HEME_BCELL",
    "MCL": "HEME_BCELL",
    "FL": "HEME_BCELL",
    "HL": "HEME_BCELL",
    "BL": "HEME_BCELL",
    "B_ALL": "HEME_BCELL",
    "HCL": "HEME_BCELL",
    "T_ALL": "HEME_TCELL",
    "CTCL": "HEME_TCELL",
    "MM": "HEME_PLASMA",
    "LAML": "HEME_MYELOID",
    "MDS": "HEME_MYELOID",
    "MPN": "HEME_MYELOID",
    "CML": "HEME_MYELOID",
    # Singletons / rare
    "CHOR": "NOTOCHORDAL",
    "NUTM": "MIDLINE_NUT",
    # Pediatric (most have their own biology)
    "WILMS": "WILMS_LIKE",
    "RT": "RHABDOID",
    "ATRT": "RHABDOID",
    "MBL": "EMBRYONAL_CNS",
    "RB": "RETINAL",
    "NBL": "SYMPATHETIC_NEURAL",
    # NPC, HNSC variant — stays SQUAMOUS
    "NPC": "SQUAMOUS",
    # Sarcoma extensions to MESENCHYMAL
    "GCTB": "MESENCHYMAL",
    "ESS_LG": "MESENCHYMAL",
    "ESS_HG": "MESENCHYMAL",
    "CHON": "MESENCHYMAL",
    "OS": "MESENCHYMAL",
    "EWS": "MESENCHYMAL",
    "RMS_ERMS": "MESENCHYMAL",
    "RMS_ARMS": "MESENCHYMAL",
    "RMS_SSRMS": "MESENCHYMAL",
}


# Secondary memberships: a code gets a *partial* family-score boost
# from another family in addition to its primary. Used when biology
# spans two lineages (BRCA-basal is mammary AND squamous-like).
# Value is the weight (0.0–1.0) applied to that family's score.
SECONDARY_FAMILY_MEMBERSHIPS: Mapping[str, tuple[tuple[str, float], ...]] = {
    "BRCA": (("SQUAMOUS", 0.5),),  # basal/TNBC subtype is squamous-like
    "BLCA": (("SQUAMOUS", 0.4),),  # basal BLCA is squamous-like
    "ESCA": (("GI_ADENO", 0.3),),  # Western ESCA has substantial adenocarcinoma fraction
    "LIHC": (("HEPATOBILIARY", 1.0),),  # already primary
    "MTC": (("ENDOCRINE_THYROID", 0.4),),  # arises in thyroid, but NE biology dominant
}


# Lineage discrimination panels: when a parent family wins, score
# each child cohort against its lineage-specific panel to pick
# between siblings. This is the OS-within-SARC pattern applied at
# the family level.
LINEAGE_DISCRIMINATION_PANELS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "SQUAMOUS": {
        "BRCA": (
            "SCGB2A2", "SCGB2A1", "SCGB1D2", "MUCL1",
            "ANKRD30A", "FOXA1", "GATA3", "MLPH",
        ),
        "ESCA": ("AGR2", "AGR3", "TFF1", "TFF3", "VSIG1", "MAL", "EVPL"),
        "HNSC": ("KLK10", "KLK11", "MUC21", "BPIFB1", "SPRR2A", "SPRR2B"),
        "LUSC": ("SOX2", "TP63", "SPRR2A", "KRT5", "KRT6A"),
        "CESC": ("KRT4", "KRT13", "PAX8", "MMP10"),
        "NPC": ("LMP1", "EBNA1", "EBER1"),  # EBV-associated, rare in transcript
        "BLCA": ("UPK1A", "UPK1B", "UPK2", "S100P"),
    },
    "GI_ADENO": {
        "COAD": ("CDX2", "CDH17", "GUCY2C", "CEACAM5", "VIL1"),
        "READ": ("CDX2", "CDH17", "GUCY2C", "CEACAM5", "VIL1"),
        "STAD": ("MUC5AC", "MUC6", "CLDN18", "TFF1", "GKN1", "GKN2"),
    },
    "HEPATOBILIARY": {
        "LIHC": ("AFP", "ALB", "F2", "HNF4A", "HP", "APOB"),
        "CHOL": ("KRT19", "MUC1", "MUC5AC", "EPCAM"),
        "HEPB": ("AFP", "DLK1", "GPC3"),
    },
    "NEUROENDOCRINE": {
        "SCLC": ("ASCL1", "INSM1", "CHGA", "SYP"),
        "MEC": ("KRT20", "NEFM", "SOX2"),
        "PANNET": ("PDX1", "NEUROD1", "INS", "GCG"),
        "MID_NET": ("CDX2", "TPH1", "GCG"),
        "LUNG_NET_LC": ("CHGA", "SYP", "NKX2-1"),
        "LUNG_NET_LCNEC": ("ASCL1", "INSM1", "CHGA"),
        "MTC": ("CALCA", "CEACAM5", "ASCL1"),
        "PCPG": ("TH", "DBH", "PNMT", "CHGA"),
    },
    "MESENCHYMAL": {
        "OS": ("RUNX2", "COL1A1", "ALPL", "SPP1", "IBSP"),
        "EWS": ("CD99", "NKX2-2", "CAV1", "FLI1"),
        "CHON": ("COL2A1", "SOX9", "ACAN", "COMP"),
        "RMS_ERMS": ("MYOD1", "MYOG", "DES", "MYF5"),
        "RMS_ARMS": ("PAX3", "PAX7", "FOXO1", "MYOG"),
        "RMS_SSRMS": ("MYOD1", "MYOG", "DES", "MYF5"),
        "SARC": ("VIM", "COL1A1", "COL1A2", "CD44"),
        "UCS": ("PAX8", "ESR1", "WT1"),  # carcinosarcoma with epithelial
        "GCTB": ("CSF1", "ACP5", "MMP9"),
        "ESS_LG": ("CD10", "ESR1", "PGR"),
        "ESS_HG": ("YWHAE", "BCOR"),
    },
    "GYNECOLOGIC_GLANDULAR": {
        "OV": ("PAX8", "WT1", "MSLN", "MUC16", "FOLR1"),
        "UCEC": ("PAX8", "ESR1", "PGR", "MUC16", "ARID1A"),
    },
    "RENAL": {
        "KIRC": ("CA9", "NDUFA4L2", "AQP1", "VEGFA"),
        "KIRP": ("MET", "AQP1"),
        "KICH": ("RHCG", "KIT", "TFCP2L1", "FOXI1", "PARM1"),
    },
}
