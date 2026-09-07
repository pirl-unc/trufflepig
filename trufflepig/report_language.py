"""Named report paragraphs over explicit evidence facts.

Templates own phrasing. The caller supplies decisions; templates never select
therapies or reinterpret assays. Missing facts fail visibly under StrictUndefined.
The same resulting paragraphs feed the structured report and its renderers.
"""

from __future__ import annotations

import re

from jinja2 import Environment, PackageLoader, StrictUndefined


_ENVIRONMENT = Environment(
    loader=PackageLoader("trufflepig", "report_templates"),
    undefined=StrictUndefined,
    autoescape=False,  # Plain Markdown; HTML escaping belongs to the HTML renderer.
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_report_paragraph(name: str, **facts: object) -> str:
    """Render one complete paragraph from a packaged, named template."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"Invalid report paragraph name: {name!r}")
    text = _ENVIRONMENT.get_template(f"{name}.md.j2").render(**facts)
    return " ".join(text.split())
