from pathlib import Path

from types import SimpleNamespace

from trufflepig.main import (
    _clean_prefix_outputs,
    _derive_sample_display_id,
    _matched_normal_split_summary,
    _summarize_sample_call,
)


def _touch(p: Path, text: str = "") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_clean_prefix_removes_prefixed_files_only(tmp_path):
    # Prior-run files matching prefix base
    _touch(tmp_path / "sample-summary.md", "old")
    _touch(tmp_path / "sample-decomposition.png", "old")
    _touch(tmp_path / "figures" / "sample-targets.png", "old")
    # Files that must NOT be deleted
    _touch(tmp_path / "README.md", "keep")
    _touch(tmp_path / "other-user-file.txt", "keep")
    _touch(tmp_path / "figures" / "not-ours.png", "keep")

    prefix = str(tmp_path / "sample")
    removed = _clean_prefix_outputs(tmp_path, prefix)

    assert removed == 3
    assert not (tmp_path / "sample-summary.md").exists()
    assert not (tmp_path / "sample-decomposition.png").exists()
    assert not (tmp_path / "figures" / "sample-targets.png").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "other-user-file.txt").exists()
    assert (tmp_path / "figures" / "not-ours.png").exists()


def test_clean_prefix_removes_scatter_subdir(tmp_path):
    scatter = tmp_path / "sample-vs-cancer"
    _touch(scatter / "PRAD.png", "old")
    _touch(scatter / "LUAD.png", "old")
    _touch(tmp_path / "sample-summary.md", "old")

    removed = _clean_prefix_outputs(tmp_path, str(tmp_path / "sample"))

    assert removed == 3
    assert not scatter.exists()
    assert not (tmp_path / "sample-summary.md").exists()


def test_clean_prefix_custom_image_prefix(tmp_path):
    # When user supplies output_image_prefix="bg002", only bg002-* should be cleaned
    _touch(tmp_path / "bg002-summary.md", "old")
    _touch(tmp_path / "sample-summary.md", "from-different-sample")

    removed = _clean_prefix_outputs(tmp_path, str(tmp_path / "bg002"))
    assert removed == 1
    assert not (tmp_path / "bg002-summary.md").exists()
    assert (tmp_path / "sample-summary.md").exists()


def test_clean_prefix_empty_or_missing_dir(tmp_path):
    # Calling on an empty / nonexistent dir should be a no-op
    assert _clean_prefix_outputs(tmp_path, str(tmp_path / "sample")) == 0
    assert (
        _clean_prefix_outputs(
            tmp_path / "doesnotexist", str(tmp_path / "doesnotexist" / "sample")
        )
        == 0
    )


def test_derive_sample_display_id_prefers_case_like_path_tokens():
    assert (
        _derive_sample_display_id(
            "/Users/me/data/TL-12-ABCD/gene_expression_salmon.tsv"
        )
        == "TL-12-ABCD"
    )
    assert (
        _derive_sample_display_id(
            "/Users/me/data/study/BG12345/WashU/run/rna_stringtie_gene_expression.tsv"
        )
        == "BG12345"
    )


def test_matched_normal_split_summary_omits_zero_fraction():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "matched_normal_tissue": "breast",
                "matched_normal_fraction": 0.0,
            }
        ]
    )
    assert _matched_normal_split_summary(df) is None


def test_summarize_sample_call_humanizes_hypotheses():
    top = SimpleNamespace(
        cancer_type="BRCA",
        template="solid_primary",
        score=1.0,
        warnings=[],
        template_site_factor=1.0,
        template_tissue_score=1.0,
    )
    runner = SimpleNamespace(
        cancer_type="READ",
        template="met_peritoneal",
        score=0.95,
        warnings=[],
        template_site_factor=1.0,
        template_tissue_score=1.0,
    )
    analysis = {
        "cancer_type": "READ",
        "candidate_trace": [{"code": "READ"}],
        "fit_quality": {"label": "ambiguous"},
    }
    summary = _summarize_sample_call(analysis, [top, runner], sample_mode="solid")
    assert summary["hypothesis_display"] == [
        "BRCA (Breast Invasive Carcinoma)-like primary site pattern",
        "READ (Rectum Adenocarcinoma) peritoneal-associated host context",
    ]
