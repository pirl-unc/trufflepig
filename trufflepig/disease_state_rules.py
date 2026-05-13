"""Data-driven disease-state narrative rules (#202).

Replaces the hardcoded ``if cancer_code == "PRAD"`` / ``elif
cancer_code in ("BRCA",)`` branches in ``cli._synthesize_disease_state``
with a declarative rule engine. Per-cancer narratives live in
``disease-state-rules.csv``; named gene sets referenced by the rules
live in ``narrative-gene-sets.csv``.

The uniformity principle: every cancer type runs through the same
synthesis loop. Differences between cancers live in data, not in
code branches. Adding a new cancer narrative means adding a CSV row,
not touching Python.

Rule record
-----------
- ``rule_id`` — stable identifier for the rule
- ``cancer_code`` — specific code (e.g. ``PRAD``) or ``pan_cancer``
  (applies to every sample)
- ``priority`` — lower = more specific / evaluated earlier
- ``claims`` — optional concept name; when a rule fires, any
  lower-priority rule that claims the same concept is skipped. This
  expresses mutual-exclusion: ``prad_crpc_with_ne``, ``prad_crpc``,
  and ``prad_ar_suppressed`` all claim ``AR_axis`` so the most-
  specific matching rule wins.
- ``conditions`` — pipe-separated AND expression. Tokens:
    * ``axis:<axis_name>=<state>`` — the axis's state must equal
      ``<state>`` (e.g. ``axis:AR_signaling=down``)
    * ``retained:<gene_or_set>`` — a single gene or any member of a
      named gene set is in the ``retained`` set
    * ``retained_all:<set>`` — every member of a named set is
      retained
    * ``collapsed:<gene_or_set>`` — single gene or any member of a
      named set is collapsed
    * ``collapsed_ge:<N>=<set>`` — at least N members of a named set
      are collapsed
- ``narrative`` — emit string with optional ``{collapsed:<set>}`` /
  ``{retained:<set>}`` placeholders rendered as comma-joined lists

The engine is pure and stateless: ``synthesize_disease_state`` takes
the three inputs a narrative rule could possibly consult (axis
states, retained genes, collapsed genes) plus the cancer code, and
returns a single concatenated string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from pirlygenes.load_dataset import get_data


# ── Data loaders ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def narrative_gene_sets() -> dict[str, tuple[str, ...]]:
    """Return ``{set_name: tuple[str]}`` from ``narrative-gene-sets.csv``."""
    df = get_data("narrative-gene-sets")
    out: dict[str, tuple[str, ...]] = {}
    for _, row in df.iterrows():
        name = str(row.get("set_name") or "").strip()
        members = str(row.get("members") or "")
        parsed = tuple(m.strip() for m in members.split(";") if m.strip())
        if name and parsed:
            out[name] = parsed
    return out


@dataclass(frozen=True)
class Rule:
    rule_id: str
    cancer_code: str
    priority: int
    claims: str | None
    conditions_raw: str
    narrative: str


@lru_cache(maxsize=1)
def _all_rules() -> tuple[Rule, ...]:
    df = get_data("disease-state-rules")
    rules: list[Rule] = []
    for _, row in df.iterrows():
        rule_id = str(row.get("rule_id") or "").strip()
        if not rule_id:
            continue
        cancer_code = str(row.get("cancer_code") or "").strip()
        priority = int(row.get("priority") or 100)
        claims_val = str(row.get("claims") or "").strip() or None
        conditions = str(row.get("conditions") or "").strip()
        narrative = str(row.get("narrative") or "").strip()
        rules.append(
            Rule(
                rule_id=rule_id,
                cancer_code=cancer_code,
                priority=priority,
                claims=claims_val,
                conditions_raw=conditions,
                narrative=narrative,
            )
        )
    return tuple(rules)


# ── Condition DSL ─────────────────────────────────────────────────────


_COND_RE = re.compile(
    r"^(?P<op>[a-z_]+)(?::(?P<arg1>[^=]+))?(?:=(?P<arg2>.+))?$",
)


def _resolve_gene_or_set(token: str) -> tuple[str, ...]:
    """Token is either a literal gene symbol or a named gene set.
    Returns the tuple of gene symbols to check."""
    sets = narrative_gene_sets()
    if token in sets:
        return sets[token]
    return (token,)


def _evaluate_condition(
    condition_str: str,
    axis_states: dict[str, str | None],
    retained: set[str],
    collapsed: set[str],
) -> bool:
    """Evaluate a single condition token against the context.

    Token grammar (see module docstring for full list)::

        axis:<name>=<state>
        retained:<gene_or_set>
        retained_all:<set>
        collapsed:<gene_or_set>
        collapsed_ge:<N>=<set>
    """
    cond = condition_str.strip()
    if not cond:
        return True

    # axis:NAME=STATE
    if cond.startswith("axis:"):
        payload = cond[len("axis:") :]
        if "=" not in payload:
            return False
        axis_name, state = payload.split("=", 1)
        observed = axis_states.get(axis_name.strip())
        return observed == state.strip()

    # retained:TOKEN  (any-of if TOKEN is a set)
    if cond.startswith("retained:"):
        token = cond[len("retained:") :].strip()
        genes = _resolve_gene_or_set(token)
        return any(g in retained for g in genes)

    # retained_all:SET
    if cond.startswith("retained_all:"):
        token = cond[len("retained_all:") :].strip()
        genes = _resolve_gene_or_set(token)
        return all(g in retained for g in genes)

    # collapsed:TOKEN  (any-of if TOKEN is a set)
    if cond.startswith("collapsed:"):
        token = cond[len("collapsed:") :].strip()
        genes = _resolve_gene_or_set(token)
        return any(g in collapsed for g in genes)

    # collapsed_ge:N=SET
    if cond.startswith("collapsed_ge:"):
        payload = cond[len("collapsed_ge:") :].strip()
        if "=" not in payload:
            return False
        n_str, set_name = payload.split("=", 1)
        try:
            n = int(n_str)
        except ValueError:
            return False
        genes = _resolve_gene_or_set(set_name.strip())
        return sum(1 for g in genes if g in collapsed) >= n

    # Unknown token — fail closed so bad CSVs surface in tests.
    return False


def _conditions_match(
    conditions_raw: str,
    axis_states: dict[str, str | None],
    retained: set[str],
    collapsed: set[str],
) -> bool:
    """Pipe-separated AND. Every subcondition must match."""
    for part in conditions_raw.split("|"):
        if not _evaluate_condition(part, axis_states, retained, collapsed):
            return False
    return True


# ── Template rendering ────────────────────────────────────────────────


_TEMPLATE_RE = re.compile(r"\{(?P<op>retained|collapsed):(?P<set_name>[^}]+)\}")


def _render_narrative(
    narrative: str,
    retained: set[str],
    collapsed: set[str],
) -> str:
    """Expand ``{retained:SET}`` / ``{collapsed:SET}`` placeholders.

    Renders the intersection of the named set's members with the
    corresponding bucket, in the order declared in the gene-set CSV.
    """
    sets = narrative_gene_sets()

    def replace(match: re.Match) -> str:
        op = match.group("op")
        set_name = match.group("set_name").strip()
        members = sets.get(set_name, ())
        if op == "retained":
            matched = [g for g in members if g in retained]
        else:  # collapsed
            matched = [g for g in members if g in collapsed]
        return ", ".join(matched)

    return _TEMPLATE_RE.sub(replace, narrative)


# ── Public API ────────────────────────────────────────────────────────


def _rules_for_cancer(cancer_code: str | None) -> list[Rule]:
    """Return rules applicable to this cancer, sorted by priority
    ascending. Includes ``pan_cancer`` rules regardless of code."""
    rules = _all_rules()
    out = []
    for r in rules:
        if r.cancer_code == "pan_cancer":
            out.append(r)
        elif cancer_code and r.cancer_code == cancer_code:
            out.append(r)
    return sorted(out, key=lambda r: r.priority)


def synthesize_disease_state(
    cancer_code: str | None,
    axis_states: dict[str, str | None],
    retained: Iterable[str],
    collapsed: Iterable[str],
) -> str:
    """Return the composed disease-state narrative.

    Parameters
    ----------
    cancer_code : str or None
        TCGA code (PRAD, BRCA, SARC, ...). Determines which
        cancer-specific rules are in scope in addition to pan-cancer
        rules.
    axis_states : dict[str, str | None]
        Map from therapy-axis name (``AR_signaling``, ``EMT``,
        ``hypoxia``, ...) to the axis's state string (``"up"``,
        ``"down"``, ``"intact"``, ``None``).
    retained : iterable[str]
        Gene symbols classified as lineage-retained (purity ≥ 0.30).
    collapsed : iterable[str]
        Gene symbols classified as lineage-collapsed (purity < 0.05).

    Returns
    -------
    str
        Space-joined narrative. Empty string when no rule matches.
    """
    retained_set = set(retained)
    collapsed_set = set(collapsed)
    rules = _rules_for_cancer(cancer_code)

    claimed: set[str] = set()
    parts: list[str] = []
    for rule in rules:
        if rule.claims and rule.claims in claimed:
            continue
        if _conditions_match(
            rule.conditions_raw,
            axis_states,
            retained_set,
            collapsed_set,
        ):
            rendered = _render_narrative(
                rule.narrative,
                retained_set,
                collapsed_set,
            )
            if rendered:
                parts.append(rendered)
            if rule.claims:
                claimed.add(rule.claims)
    return " ".join(parts)
