"""Small helpers for class-I HLA eligibility strings.

The analyze report uses RNA expression, but TCR-T / pMHC-style therapies
also require a germline HLA match. Keep this parser deliberately narrow:
it normalizes common clinical shorthand and leaves full HLA interpretation
to a clinical-grade typing tool.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_HLA_TOKEN_RE = re.compile(
    r"""
    (?:
        HLA[-_\s]?
    )?
    (?P<locus>[ABC])
    \*?
    (?P<field1>\d{2})
    (?:
        :?
        (?P<field2>\d{2})
    )?
    (?P<suffix>[A-Z])?
    \+?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_HLA_TEXT_TOKEN_RE = re.compile(
    r"""
    \b
    (?:
        HLA[-_\s]?[ABC]\*?\d{2}(?::?\d{2})?[A-Z]?\+?
        |
        [ABC]\*\d{2}(?::?\d{2})?[A-Z]?\+?
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_hla_type(value: object) -> str:
    """Return normalized ``A*02:01`` / ``A*02`` text when parseable."""
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.replace("HLA_", "HLA-").replace(" ", "")
    text = text.rstrip(",;.")
    text = re.sub(r"(POSITIVE|POS|\+)$", "", text)
    # FDA labels sometimes use protein-expression suffixes such as
    # A*02:01P. For eligibility matching, the allele fields are enough.
    text = re.sub(r"([ABC]\*?\d{2}:?\d{0,2})[A-Z]$", r"\1", text)
    match = _HLA_TOKEN_RE.fullmatch(text)
    if not match:
        return ""
    locus = match.group("locus").upper()
    field1 = match.group("field1")
    field2 = match.group("field2")
    if field2:
        return f"{locus}*{field1}:{field2}"
    return f"{locus}*{field1}"


def parse_hla_types(value: object) -> list[str]:
    """Parse comma/semicolon/whitespace-separated HLA class-I types."""
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[,;\s]+", value)
    elif isinstance(value, Iterable):
        pieces = []
        for item in value:
            pieces.extend(parse_hla_types(item))
        return sorted(set(pieces))
    else:
        pieces = [str(value)]
    normalized = [normalize_hla_type(piece) for piece in pieces]
    return sorted({item for item in normalized if item})


def extract_hla_types_from_text(text: object) -> list[str]:
    """Extract HLA-like restrictions embedded in free text."""
    found: list[str] = []
    for match in _HLA_TEXT_TOKEN_RE.finditer(str(text or "")):
        normalized = normalize_hla_type(match.group(0))
        if normalized:
            found.append(normalized)
    return sorted(set(found))


def hla_types_compatible(
    supplied_hla_types: Iterable[str],
    required_hla_types: Iterable[str],
) -> tuple[bool, str | None, str | None]:
    """Return whether any supplied type satisfies any required type.

    Broad required restrictions such as ``A*02`` are satisfied by concrete
    supplied alleles such as ``A*02:01``. The inverse is not true: a broad
    supplied type does not prove eligibility for an exact required allele.
    """
    status, matched_supplied, matched_required = hla_types_compatibility_status(
        supplied_hla_types,
        required_hla_types,
    )
    return status == "matched", matched_supplied, matched_required


def hla_types_compatibility_status(
    supplied_hla_types: Iterable[str],
    required_hla_types: Iterable[str],
) -> tuple[str, str | None, str | None]:
    """Return matched / insufficient_resolution / mismatched HLA status."""
    supplied = parse_hla_types(supplied_hla_types)
    required = parse_hla_types(required_hla_types)
    insufficient: tuple[str, str] | None = None
    for supplied_type in supplied:
        for required_type in required:
            if supplied_type == required_type:
                return "matched", supplied_type, required_type
            if ":" not in required_type and supplied_type.startswith(
                required_type + ":"
            ):
                return "matched", supplied_type, required_type
            if ":" not in supplied_type and required_type.startswith(
                supplied_type + ":"
            ):
                insufficient = insufficient or (supplied_type, required_type)
    if insufficient:
        return "insufficient_resolution", insufficient[0], insufficient[1]
    return "mismatched", None, None
