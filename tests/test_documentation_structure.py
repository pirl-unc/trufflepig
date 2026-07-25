from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "docs/CALIBRATION.md",
        "docs/CANCER_CALL_DECISION_FLOW.md",
        "docs/SERVICE_PERFORMANCE.md",
        "docs/cancer-type-hierarchical-classifier.md",
        "docs/cancer-type-ontology.md",
        "docs/cancer-type-residual-matching-findings.md",
        "docs/report-belief-consistency-and-friendliness-plan.md",
        "docs/rnaseq-cancer-call-redesign.md",
    ],
)
def test_long_form_documentation_puts_summary_before_details(relative_path):
    headings = [
        line
        for line in (ROOT / relative_path).read_text().splitlines()
        if line.startswith("## ")
    ]
    assert headings[0] == "## At a glance"


def test_documentation_map_starts_with_reader_navigation():
    headings = [
        line
        for line in (ROOT / "docs/README.md").read_text().splitlines()
        if line.startswith("## ")
    ]
    assert headings[0] == "## Start here"
