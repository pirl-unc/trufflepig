"""Named report paragraphs over explicit evidence facts.

Templates own phrasing. The caller supplies decisions; templates never select
therapies or reinterpret assays. Missing facts fail visibly under StrictUndefined.
The same resulting paragraphs feed the structured report and its renderers.
"""

from __future__ import annotations

import re

from jinja2 import Environment, PackageLoader, StrictUndefined


def report_literal(value: object) -> str:
    """Quote supplied text as literal inline Markdown, including table pipes."""
    text = " ".join(str(value).split())
    return re.sub(r"([\\`*_{}\[\]()#+!|<>])", r"\\\1", text)


_ENVIRONMENT = Environment(
    loader=PackageLoader("trufflepig", "report_templates"),
    undefined=StrictUndefined,
    autoescape=False,  # Plain Markdown; HTML escaping belongs to the HTML renderer.
    trim_blocks=False,
    lstrip_blocks=True,
)
_ENVIRONMENT.filters["literal"] = report_literal


def render_report_template(name: str, **facts: object) -> str:
    """Render a packaged report or section without discarding paragraph breaks."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Invalid report paragraph name: {name!r}")
    text = _ENVIRONMENT.get_template(f"{name}.md.j2").render(**facts).strip()
    return re.sub(r"\n{3,}", "\n\n", text) + "\n"


def render_report_paragraph(name: str, **facts: object) -> str:
    """Render one complete paragraph from a packaged, named template."""
    return " ".join(render_report_template(name, **facts).split())


from markdown_it import MarkdownIt
from .hla import is_hla_asterisk


def _literal_hla_asterisk(state, silent: bool) -> bool:
    if not is_hla_asterisk(state.src, state.pos):
        return False
    if not silent:
        state.pending += "*"
    state.pos += 1
    return True


_INLINE_MARKDOWN = MarkdownIt("commonmark", {"html": False})
_INLINE_MARKDOWN.inline.ruler.before("emphasis", "hla_allele", _literal_hla_asterisk)


def report_plain_text(line: str) -> str:
    """Render inline Markdown as text, preserving literal biomedical identifiers."""
    tokens = report_inline_tokens(line)
    line = "".join(
        " " if token.type in {"softbreak", "hardbreak"} else token.content
        for token in tokens
        if token.type in {"text", "code_inline", "image", "softbreak", "hardbreak"}
    )
    line = line.replace("—", "-").replace("–", "-")
    return line.strip()


def report_inline_tokens(text: str):
    """Tokenize presentation markup without changing HLA nomenclature."""
    return _INLINE_MARKDOWN.parseInline(str(text))[0].children or []
