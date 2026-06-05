"""Degenerate-subtype disambiguation (#198).

Several within-family subtypes share gene-expression signatures that
the classifier can't distinguish on expression alone — e.g. osteosarcoma
vs dedifferentiated liposarcoma both carry the 12q13-15 amplicon
(MDM2 + CDK4 + FRS2); Ewing / DSRCT / ARMS all carry a CD99+
small-blue-round-cell signature. The canonical tiebreaker is either
anatomic site (bone vs retroperitoneum) or a deterministic
fusion-surrogate expression pattern (FATE1/NR0B1 for EWS-FLI1;
NUTM1 mRNA for NUT carcinoma; CCND1 for MCL).

Two declarative catalogs drive the resolution:

- ``degenerate-subtype-pairs.csv`` — pairs/triples of subtypes that
  share a signature, plus the tiebreaker rule, mapping, and an
  ``activation_signature`` that gates when the pair applies.
- ``fusion-surrogate-expression.csv`` — genes whose expression is a
  deterministic surrogate for a specific fusion/translocation class
  (ectopic expression from fusion-driven derepression, or promoter-
  swap hyperexpression).

Activation gating
-----------------
Each pair declares an ``activation_signature`` (e.g. ``MDM2:100`` or
``CD99:20``). The resolver evaluates the signature against the
observed tumor-attributed TPMs before consulting the tiebreaker; if
the shared signature isn't present in the sample, the pair is
**inactive** and the upstream classifier's call is kept unchanged.
This prevents routine high-confidence calls (e.g. LUSC with clear
squamous signature but no NUTM1 expression) from being pulled into
irrelevant degenerate-pair resolutions.

Information flow
----------------
The resolver consumes two contextual inputs beyond the raw winning
subtype: the decomposition's top-ranked site template (bone /
retroperitoneum / lung / …) and the full tumor-attributed TPM dict.
These come from earlier pipeline stages — the resolver reasons over
the full context, not any single gene in isolation.
"""

import logging
from functools import lru_cache

from pirlygenes.gene_sets_cancer import resolve_cancer_type
from pirlygenes.load_dataset import get_data

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def degenerate_subtype_pairs():
    """Load ``degenerate-subtype-pairs.csv``.

    Returns a DataFrame with ``pair_id``, ``members`` (list of subtype
    codes), ``shared_signature``, ``activation_signature`` (dict
    ``{gene: min_tpm}``), ``tiebreaker_rule``, ``tiebreaker_mapping``
    (dict keyed by context value → subtype), and free-text ``notes`` /
    ``refs``.
    """
    df = get_data("degenerate-subtype-pairs")
    df["members"] = (
        df["members"]
        .fillna("")
        .astype(str)
        .apply(lambda s: [m.strip() for m in s.split(";") if m.strip()])
    )

    def _parse_mapping(value):
        if not isinstance(value, str) or not value.strip():
            return {}
        out = {}
        for part in value.split("|"):
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            out[key.strip()] = val.strip()
        return out

    def _parse_activation(value):
        """``GENE:min_tpm|GENE:min_tpm|...`` → dict of {gene: float}."""
        raw = _parse_mapping(value)
        parsed = {}
        for gene, tpm_str in raw.items():
            try:
                parsed[gene] = float(tpm_str)
            except (TypeError, ValueError):
                continue
        return parsed

    df["tiebreaker_mapping"] = df["tiebreaker_mapping"].apply(_parse_mapping)
    if "activation_signature" in df.columns:
        df["activation_signature"] = df["activation_signature"].apply(_parse_activation)
    else:
        df["activation_signature"] = [{} for _ in range(len(df))]
    return df


@lru_cache(maxsize=1)
def fusion_surrogate_expression():
    """Load ``fusion-surrogate-expression.csv``."""
    return get_data("fusion-surrogate-expression")


def _pair_applicable(pair_row, site_template, tumor_tpm_by_symbol):
    """A pair is *applicable* when its ``tiebreaker_rule`` has the
    context inputs it needs. ``site_template`` rules need a template;
    ``fusion_surrogate`` / ``marker_combo`` rules need TPMs. Used to
    prefer applicable pairs over inapplicable ones when a subtype sits
    in multiple pairs (HNSC ∈ {LUSC_vs_HNSC_vs_CESC, NUTM_vs_squamous}).
    """
    rule = pair_row["tiebreaker_rule"]
    if rule == "site_template":
        return bool(site_template)
    if rule in ("fusion_surrogate", "marker_combo"):
        return bool(tumor_tpm_by_symbol)
    return False


def _find_pair_for_subtype(subtype_code, site_template=None, tumor_tpm_by_symbol=None):
    """Return the degenerate-pair row whose ``members`` contain this
    subtype. When the subtype is listed in multiple pairs, prefer the
    first *applicable* pair (rule's context is available). Falls back
    to first-match if none are applicable. Returns ``None`` when no
    pair lists the subtype.
    """
    pairs = degenerate_subtype_pairs()
    matches = [row for _, row in pairs.iterrows() if subtype_code in row["members"]]
    if not matches:
        return None
    applicable = [
        row
        for row in matches
        if _pair_applicable(row, site_template, tumor_tpm_by_symbol)
    ]
    if applicable:
        return applicable[0]
    return matches[0]


def _activation_met(pair_row, tumor_tpm_by_symbol):
    """Return ``True`` when the pair's ``activation_signature`` is
    satisfied by the observed TPMs (at least one gene meets its
    threshold). Empty activation signature is always satisfied —
    the pair applies unconditionally. Missing TPM dict with a
    populated signature means we cannot confirm activation and the
    pair is treated as inactive (don't second-guess a classifier pick
    when context isn't available).
    """
    activation = pair_row.get("activation_signature") or {}
    if not activation:
        return True
    if not tumor_tpm_by_symbol:
        return False
    for gene, min_tpm in activation.items():
        observed = tumor_tpm_by_symbol.get(gene)
        if observed is None:
            continue
        try:
            if float(observed) >= float(min_tpm):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _resolve_site_template(pair_row, site_template):
    """Apply a ``site_template`` tiebreaker. Returns the mapped
    subtype or ``None`` when the site doesn't appear in the mapping."""
    if not site_template:
        return None
    mapping = pair_row["tiebreaker_mapping"] or {}
    return mapping.get(site_template)


def _resolve_fusion_surrogate(pair_row, tumor_tpm_by_symbol, min_tpm=1.0):
    """Apply a ``fusion_surrogate`` tiebreaker. Count votes by
    subtype — each mapping gene whose TPM is ≥ ``min_tpm`` adds one
    vote for its mapped subtype. Returns the subtype with the most
    votes when (a) it's the only subtype with votes, or (b) its
    margin over the runner-up is ≥ 2. Ties or narrow margins return
    ``None`` so the resolver can surface the ambiguity."""
    mapping = pair_row["tiebreaker_mapping"] or {}
    if not mapping or not tumor_tpm_by_symbol:
        return None
    votes = {}
    for gene, subtype in mapping.items():
        tpm = tumor_tpm_by_symbol.get(gene)
        if tpm is None:
            continue
        try:
            if float(tpm) >= min_tpm:
                votes[subtype] = votes.get(subtype, 0) + 1
        except (TypeError, ValueError):
            continue
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] - ranked[1][1] >= 2:
        return ranked[0][0]
    return None


def _resolve_marker_combo(pair_row, tumor_tpm_by_symbol, min_tpm=1.0):
    """Same mechanism as ``fusion_surrogate`` but used for pairs whose
    disambiguator is a marker panel (not a fusion) — e.g. CCND1 for
    MCL vs CLL. Kept as a distinct rule name so the markdown reason
    text stays accurate."""
    return _resolve_fusion_surrogate(
        pair_row,
        tumor_tpm_by_symbol,
        min_tpm=min_tpm,
    )


def resolve_degenerate_subtype(
    winning_subtype,
    site_template=None,
    tumor_tpm_by_symbol=None,
):
    """Return the final subtype call + provenance.

    Parameters
    ----------
    winning_subtype : str or None
        The subtype the upstream classifier picked. ``None`` skips
        resolution (nothing to disambiguate).
    site_template : str or None
        Top decomposition template (e.g. ``met_bone``,
        ``primary_retroperitoneum``). Drives ``site_template``
        tiebreaker rules.
    tumor_tpm_by_symbol : dict[str, float] or None
        Tumor-attributed TPM per gene symbol, from the attribution
        stage. Drives activation-signature gating plus
        ``fusion_surrogate`` and ``marker_combo`` tiebreakers.

    Returns
    -------
    dict with keys:
        ``final_subtype`` — the resolved subtype, may equal
            ``winning_subtype`` when the catalog confirms or can't
            rule it out.
        ``status`` — one of:
            * ``'no_pair'`` — subtype not in any degenerate pair.
            * ``'pair_inactive'`` — subtype is in a pair but the
              activation signature isn't met in the sample; the
              classifier's pick stands unchanged.
            * ``'confirmed'`` — tiebreaker agrees with classifier.
            * ``'corrected'`` — tiebreaker swapped the call for a
              pair member.
            * ``'degenerate'`` — pair active but tiebreaker
              inconclusive; render ambiguity explicitly rather than
              committing.
        ``reason`` — short human-readable note for the markdown layer.
        ``pair_id`` — identifier of the matched pair (or ``None``).
        ``alternatives`` — list of the other members of the pair.
        ``rule`` — the tiebreaker family used (or ``None``).
        ``shared_signature`` — short label for the shared ambiguous signal.
    """
    if not winning_subtype:
        return {
            "final_subtype": winning_subtype,
            "status": "no_pair",
            "reason": "",
            "pair_id": None,
            "alternatives": [],
            "rule": None,
            "shared_signature": None,
        }

    pair_row = _find_pair_for_subtype(
        winning_subtype,
        site_template=site_template,
        tumor_tpm_by_symbol=tumor_tpm_by_symbol,
    )
    if pair_row is None:
        return {
            "final_subtype": winning_subtype,
            "status": "no_pair",
            "reason": "",
            "pair_id": None,
            "alternatives": [],
            "rule": None,
            "shared_signature": None,
        }

    members = pair_row["members"]
    alternatives = [m for m in members if m != winning_subtype]
    rule = pair_row["tiebreaker_rule"]
    pair_id = pair_row["pair_id"]

    # Activation gate: the pair's shared signature must be present in
    # the sample for the ambiguity to be real. Without it, the
    # classifier's pick stands — we don't pull LUSC into NUTM_vs_squamous
    # just because LUSC happens to be a member of the pair.
    if not _activation_met(pair_row, tumor_tpm_by_symbol):
        activation_desc = ", ".join(
            f"{g}>={int(t)}"
            for g, t in (pair_row.get("activation_signature") or {}).items()
        )
        return {
            "final_subtype": winning_subtype,
            "status": "pair_inactive",
            "reason": (
                f"{pair_id} pair skipped — activation signature "
                f"({activation_desc}) not present in sample"
            ),
            "pair_id": pair_id,
            "alternatives": alternatives,
            "rule": rule,
            "shared_signature": pair_row["shared_signature"],
        }

    resolved = None
    if rule == "site_template":
        resolved = _resolve_site_template(pair_row, site_template)
    elif rule == "fusion_surrogate":
        resolved = _resolve_fusion_surrogate(pair_row, tumor_tpm_by_symbol)
    elif rule == "marker_combo":
        resolved = _resolve_marker_combo(pair_row, tumor_tpm_by_symbol)

    # The tiebreaker_mapping values can lag the canonical registry codes
    # (pirlygenes #287 shipped renamed `members` but left stale codes like
    # `PANNET`/`MID_NET` in the mapping). Canonicalize so a stale upstream
    # mapping can't leak a dead code into the final call.
    if resolved is not None:
        resolved = resolve_cancer_type(resolved)

    if resolved is None:
        return {
            "final_subtype": winning_subtype,
            "status": "degenerate",
            "reason": (
                f"degenerate between {sorted(members)} — "
                f"{rule} tiebreaker inconclusive "
                f"(shared signature: {pair_row['shared_signature']})"
            ),
            "pair_id": pair_id,
            "alternatives": alternatives,
            "rule": rule,
            "shared_signature": pair_row["shared_signature"],
        }

    if resolved == winning_subtype:
        return {
            "final_subtype": winning_subtype,
            "status": "confirmed",
            "reason": (
                f"{rule} tiebreaker confirms {winning_subtype} "
                f"(shared signature: {pair_row['shared_signature']})"
            ),
            "pair_id": pair_id,
            "alternatives": alternatives,
            "rule": rule,
            "shared_signature": pair_row["shared_signature"],
        }

    return {
        "final_subtype": resolved,
        "status": "corrected",
        "reason": (
            f"{rule} tiebreaker swapped {winning_subtype} → {resolved} "
            f"(shared signature: {pair_row['shared_signature']})"
        ),
        "pair_id": pair_id,
        "alternatives": [m for m in members if m != resolved],
        "rule": rule,
        "shared_signature": pair_row["shared_signature"],
    }


def fusion_surrogate_genes_for(cancer_code):
    """Return the list of surrogate gene symbols associated with the
    given cancer code (either via the ``cancer_code`` column match or
    via a ``pan_cancer`` / semicolon-separated multi-code row)."""
    df = fusion_surrogate_expression()
    hits = []
    for _, row in df.iterrows():
        cc = str(row.get("cancer_code") or "")
        codes = {c.strip() for c in cc.split(";") if c.strip()}
        if cancer_code in codes or "pan_cancer" in codes:
            hits.append(
                {
                    "gene": row["surrogate_gene"],
                    "fusion_class": row["fusion_class"],
                    "role": row["surrogate_role"],
                    "rationale": row.get("rationale", ""),
                }
            )
    return hits
