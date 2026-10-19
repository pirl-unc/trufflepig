"""HLA nomenclature and auditable class-I therapy requirements.

mhcgnomes owns parsing and canonical display. Protein-group membership comes
from the unmodified IPD-IMGT/HLA WMDA reference bundled with its attribution.
Normalization and matching do not establish clinical assay validity.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from mhcgnomes import Allele, parse


# Locate candidate tokens in prose; mhcgnomes interprets each token.
_HLA_TEXT_TOKEN_RE = re.compile(
    r"(?<![\w:])(?:HLA[-_\s]?)?[ABC]\*?\d{2,8}"
    r"(?::\d{2,3}){0,3}[NLSCAQPG]?\+?(?![\w:])",
    re.IGNORECASE,
)


@lru_cache(maxsize=8192)
def hla_allele(value: str) -> Allele | None:
    """Parse one human class-I allele, retaining every field and annotation."""
    text = str(value).strip().rstrip(",;.")
    text = re.sub(r"(?:\s+positive|\s+pos|\+)$", "", text, flags=re.I)
    result = parse(text, species="Homo sapiens", only_class1=True, raise_on_error=False)
    if not isinstance(result, Allele) or result.gene_name not in {"A", "B", "C"}:
        return None
    return result


def normalize_hla_type(value: object) -> str:
    """Canonical display of one allele, retaining full resolution and suffix."""
    allele = hla_allele(str(value or ""))
    return allele.to_string(include_species=False) if allele else ""


def parse_hla_types(value: object) -> list[str]:
    """Normalize a comma/semicolon/whitespace-separated list of supplied alleles.

    Invalid or ambiguous values raise a diagnostic instead of disappearing.
    Use a list or commas for co-present alleles; slash ambiguity is not a genotype.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = re.sub(r"\bHLA\s+(?=[ABC]\*?\d)", "HLA-", value, flags=re.I)
        pieces = [piece for piece in re.split(r"[,;\s]+", value.strip()) if piece]
    elif isinstance(value, Iterable):
        return sorted({item for part in value for item in parse_hla_types(part)})
    else:
        pieces = [str(value)]
    normalized = []
    for piece in pieces:
        canonical = normalize_hla_type(piece)
        if not canonical:
            raise ValueError(
                f"Invalid or ambiguous HLA type {piece!r}; supply human class-I "
                "alleles such as A*02:01, retaining typing resolution and suffixes. "
                "Separate co-present alleles with commas."
            )
        normalized.append(canonical)
    return sorted(set(normalized))


def extract_hla_types_from_text(text: object) -> list[str]:
    """Locate restrictions in prose and normalize each with mhcgnomes."""
    return sorted({
        canonical for match in _HLA_TEXT_TOKEN_RE.finditer(str(text or ""))
        if (canonical := normalize_hla_type(match.group(0)))
    })


def is_hla_asterisk(text: str, position: int) -> bool:
    """Whether a character belongs to a parsed allele rather than markup."""
    return text[position:position + 1] == "*" and any(
        match.start() <= position < match.end() and hla_allele(match.group(0)) is not None
        for match in _HLA_TEXT_TOKEN_RE.finditer(text)
    )


@lru_cache(maxsize=1)
def protein_group_reference() -> tuple[dict, str]:
    """Known class-I allele memberships and release of the IPD-IMGT/HLA source."""
    entries = {}
    version = ""
    path = Path(__file__).parent / "data" / "hla_nom_p.txt.gz"
    with gzip.open(path, "rt") as source:
        for line in source:
            if line.startswith("# version:"):
                version = line.partition(":")[2].strip()
            if not line.startswith(("A*;", "B*;", "C*;")):
                continue
            locus, members, group = line.strip().split(";")
            canonical_group = normalize_hla_type(locus + group) if group else ""
            for member in members.split("/"):
                allele = hla_allele(locus + member)
                if allele is None:
                    raise ValueError(f"Unparseable allele in HLA reference: {locus}{member}")
                # Prefixes retain all possible memberships at supplied resolution.
                for depth in range(1, len(allele.allele_fields) + 1):
                    key = (allele.gene_name, allele.allele_fields[:depth], allele.annotations)
                    entries.setdefault(key, set()).add(canonical_group)
    return entries, version


def hla_requirement_match(supplied: Allele, required: Allele) -> str:
    """Return matched, mismatched, or unresolved for one allele requirement."""
    if supplied.gene != required.gene:
        return "mismatched"
    if supplied.mutations or required.mutations:
        return "unresolved"
    # Null, secreted, and cytoplasmic products cannot supply ordinary surface HLA.
    if set(supplied.annotations) & {"N", "S", "C"}:
        return "mismatched"
    if supplied == required:
        return "matched"
    if "G" in supplied.annotations or "G" in required.annotations:
        return "unresolved"
    if required.annotations == ("P",):
        if supplied.annotations == ("P",):
            return "mismatched"
        groups, _ = protein_group_reference()
        observed = groups.get((supplied.gene_name, supplied.allele_fields, supplied.annotations))
        if observed is None:
            return "unresolved"
        desired = required.to_string(include_species=False)
        if desired not in observed:
            return "mismatched"
        return "matched" if observed == {desired} else "unresolved"
    if supplied.annotations or required.annotations:
        return "unresolved"
    required_fields, supplied_fields = required.allele_fields, supplied.allele_fields
    if supplied_fields[:len(required_fields)] == required_fields:
        return "matched"
    if required_fields[:len(supplied_fields)] == supplied_fields:
        return "unresolved"
    return "mismatched"


@dataclass(frozen=True)
class HlaEligibility:
    """One decision shared by shortlist selection and report interpretation."""

    status: str
    required: tuple[str, ...]
    supplied: tuple[str, ...]
    excluded: tuple[str, ...] = ()
    matched_supplied: str | None = None
    matched_required: str | None = None
    reason: str = ""
    nomenclature_version: str = ""
    assays: tuple[dict, ...] = ()

    def public_dict(self) -> dict:
        result = asdict(self)
        for key in ("required", "supplied", "excluded", "assays"):
            result[key] = list(result[key])
        return result


def hla_typings_conflict(
    left: Iterable[str] | str, right: Iterable[str] | str, *,
    left_complete: Iterable[str] = (), right_complete: Iterable[str] = (),
) -> bool:
    """Detect incompatible reported alleles within an explicitly complete locus.

    Partial typing is not a complete genotype. More fields may refine a coarse
    call; a P-group assertion is compared through the published membership
    matcher. Nomenclature comes exclusively from mhcgnomes.
    """
    left = [hla_allele(a) for a in parse_hla_types(left)]
    right = [hla_allele(a) for a in parse_hla_types(right)]

    def compatible(a, b):
        if a == b:
            return True
        if a.gene != b.gene:
            return False
        if set(a.annotations + b.annotations) & {"P", "G"}:
            matches = {hla_requirement_match(x, y) for x, y in ((a, b), (b, a))}
            # A known non-member conflicts even when the reverse comparison
            # cannot expand a group into one specific allele.
            return "matched" in matches or "mismatched" not in matches
        depth = min(len(a.allele_fields), len(b.allele_fields))
        return a.annotations == b.annotations and a.allele_fields[:depth] == b.allele_fields[:depth]

    for complete, other, loci in ((left, right, left_complete), (right, left, right_complete)):
        for allele in other:
            if allele.gene_name in loci and not any(compatible(allele, a) for a in complete):
                return True
    return False


def evaluate_hla_eligibility(supplied, required, *, excluded=()) -> HlaEligibility:
    """Evaluate positive requirements and exclusions, with exclusions first."""
    supplied = tuple(parse_hla_types(supplied))
    required = tuple(parse_hla_types(required))
    excluded = tuple(parse_hla_types(excluded))
    version = protein_group_reference()[1] if any(
        hla_allele(name).annotations == ("P",) for name in required + excluded
    ) else ""

    def decision(status, observed=None, expected=None, reason=""):
        return HlaEligibility(status, required, supplied, excluded, observed, expected, reason, version)

    if not required and not excluded:
        return decision("not_hla_restricted")
    if not supplied:
        return decision("unknown", reason="HLA typing was not supplied")
    unresolved_exclusion = None
    for observed in supplied:
        for blocked in excluded:
            match = hla_requirement_match(hla_allele(observed), hla_allele(blocked))
            if match == "matched":
                return decision("excluded", observed, blocked, "Supplied typing matches an excluded HLA group")
            if match == "unresolved":
                unresolved_exclusion = unresolved_exclusion or (observed, blocked)
    if unresolved_exclusion:
        return decision("insufficient_resolution", *unresolved_exclusion,
                        reason="Supplied typing cannot resolve the HLA exclusion")
    if not required:
        return decision("matched", reason="Supplied typing does not match an excluded group")
    unresolved = None
    for observed in supplied:
        for expected in required:
            match = hla_requirement_match(hla_allele(observed), hla_allele(expected))
            if match == "matched":
                return decision("matched", observed, expected)
            if match == "unresolved":
                unresolved = unresolved or (observed, expected)
    if unresolved:
        return decision("insufficient_resolution", *unresolved,
                        reason="Typing resolution, group membership, or expression annotation remains unresolved")
    return decision("mismatched", reason="Supplied typing does not satisfy the required HLA")
