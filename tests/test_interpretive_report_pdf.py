"""Guards for the reader-facing interpretive-report PDF manifest.

The reader PDF captions each figure with an *interpretation sentence* (what the
figure means for the decision), not its PNG filename, and its figure manifest must
stay in sync with the figures the pipeline actually emits — a stale manifest
silently omitted the therapy figure (``treatments.png``) and shipped names that no
longer exist. These are cheap structural checks; the rendering itself is validated
by eye against a real report.
"""

import importlib.util
import inspect
from pathlib import Path


def _load_pdf_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_interpretive_report_pdf.py"
    spec = importlib.util.spec_from_file_location("build_interpretive_report_pdf", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_figure_specs_are_suffix_title_interpretation_triples():
    b = _load_pdf_builder()
    for entry in b.FIGURE_SPECS:
        assert len(entry) == 3, f"expected (suffix, title, interpretation): {entry!r}"
        suffix, title, interpretation = entry
        assert suffix.endswith(".png")
        assert title and not title.endswith(".png")  # a title, never a filename
        assert interpretation and interpretation[0].isupper() and interpretation.rstrip().endswith(".")


def test_reader_manifest_includes_therapy_and_excludes_near_duplicates():
    b = _load_pdf_builder()
    suffixes = {suffix for suffix, _, _ in b.FIGURE_SPECS}
    # The ranked-therapy figure must be present (was silently dropped by the old
    # manifest, which looked only for priority-targets/actionable-targets).
    assert "treatments.png" in suffixes
    # The near-duplicate log-TPM dumbbell plots belong to the audit PDF only.
    assert "priority-target-context.png" not in suffixes
    assert "actionable-targets.png" not in suffixes


def test_figure_page_captions_with_interpretation_not_filename():
    b = _load_pdf_builder()
    params = list(inspect.signature(b._figure_page).parameters)
    assert params[:3] == ["path", "title", "interpretation"]
