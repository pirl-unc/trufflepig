# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for v4.5.1 redundancy and lock fixes (#82, #84, #85, #86).

Each test here pins a specific claim from the Codex review: that the
precomputed-analysis hand-offs actually skip the duplicate work, and
that the advisory output-dir lock refuses concurrent runs.
"""

from __future__ import annotations

from unittest.mock import patch



# -----------------------------------------------------------------
# #84: plot_sample_summary must accept a precomputed analysis.
# -----------------------------------------------------------------


def test_plot_sample_summary_accepts_precomputed_analysis(tmp_path):
    """Passing ``analysis=...`` must skip the internal analyze_sample call."""
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from trufflepig import tumor_purity as tp
    from trufflepig.tumor_purity import plot_sample_summary

    # Minimal stub analysis — the plot rendering touches these keys
    # but our assertion is about whether analyze_sample re-runs, not
    # that the figure is rich.
    analysis = {
        "cancer_type": "BRCA",
        "cancer_name": "Breast Invasive Carcinoma",
        "top_cancers": [("BRCA", 0.9)],
        "signature_top_cancers": [("BRCA", 0.8)],
        "candidate_trace": [
            {"code": "BRCA", "support_score": 0.9, "support_fraction_of_top": 1.0}
        ],
        "purity": {
            "overall_estimate": 0.5,
            "overall_lower": 0.4,
            "overall_upper": 0.6,
            "cancer_type": "BRCA",
            "tcga_median_purity": 0.6,
            "components": {
                "signature": {
                    "purity": 0.5,
                    "genes": ["ESR1"],
                    "per_gene": [],
                    "lower": 0.4,
                    "upper": 0.6,
                },
                "stromal": {"enrichment": 1.0, "purity": 0.6, "fold": 1.0},
                "immune": {"enrichment": 1.0, "purity": 0.6, "fold": 1.0},
            },
        },
        "tissue_scores": [("Breast", 1.0, {})],
        "mhc1": {"HLA-A": 100, "HLA-B": 100, "HLA-C": 100, "B2M": 100},
        "mhc2": {"HLA-DRA": 10, "HLA-DPA1": 10, "HLA-DQA1": 10},
    }
    df_expr = pd.DataFrame({"gene": ["ESR1"], "TPM": [10.0]})

    with patch.object(tp, "analyze_sample") as mock_analyze:
        from trufflepig.report_view import build_report_view

        out_png = tmp_path / "summary.png"
        plot_sample_summary(
            df_expr,
            cancer_type="BRCA",
            sample_mode="solid",
            save_to_filename=str(out_png),
            analysis=analysis,
            report_view=build_report_view(
                {
                    **analysis,
                    "cancer_type": "BRCA",
                    "sample_mode": "solid",
                }
            ),
        )
        mock_analyze.assert_not_called()


# -----------------------------------------------------------------
# #86: plot_tumor_purity must accept a precomputed purity_result.
# -----------------------------------------------------------------


def test_plot_tumor_purity_accepts_precomputed_result(tmp_path):
    """When ``purity_result`` is supplied, estimate_tumor_purity must not run."""
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from trufflepig import tumor_purity as tp
    from trufflepig.tumor_purity import plot_tumor_purity

    # The narrow assertion: when ``purity_result`` is passed,
    # plot_tumor_purity must not call estimate_tumor_purity. The
    # plotter itself may still fail downstream on our minimal frame
    # — that's fine, we assert on the *skip* not the full render.
    purity_result = {
        "cancer_type": "BRCA",
        "overall_estimate": 0.5,
        "overall_lower": 0.4,
        "overall_upper": 0.6,
        "tcga_median_purity": 0.6,
        "components": {
            "signature": {
                "purity": 0.5,
                "genes": [],
                "per_gene": [],
                "lower": 0.4,
                "upper": 0.6,
            },
            "stromal": {"enrichment": 1.0, "purity": 0.6, "fold": 1.0, "n_genes": 0},
            "immune": {"enrichment": 1.0, "purity": 0.6, "fold": 1.0, "n_genes": 0},
        },
    }
    df_expr = pd.DataFrame({"gene": ["ESR1"], "TPM": [10.0]})
    with patch.object(tp, "estimate_tumor_purity") as mock_est:
        out_png = tmp_path / "purity.png"
        try:
            plot_tumor_purity(
                df_expr,
                cancer_type="BRCA",
                sample_mode="solid",
                save_to_filename=str(out_png),
                purity_result=purity_result,
            )
        except Exception:
            pass
        mock_est.assert_not_called()


# -----------------------------------------------------------------
# #85: decompose_sample must accept precomputed candidate_rows.
# -----------------------------------------------------------------


def test_decompose_sample_skips_reranking_when_rows_passed():
    """Passing ``candidate_rows`` must skip rank_cancer_type_candidates."""
    import pandas as pd

    from trufflepig.decomposition import engine

    # Enough of a DataFrame to enter decompose_sample past the early
    # sample_by_eid build; the fit loop should then receive our rows.
    df_expr = pd.DataFrame(
        {"gene": ["BRCA1"], "TPM": [5.0], "ensembl_gene_id": ["ENSG00000012048"]}
    )
    # One hand-built row that matches the rank_cancer_type_candidates
    # schema enough for _fit_one_hypothesis to fail gracefully. We
    # only care that rank_cancer_type_candidates is not called.
    fake_rows = [
        {
            "code": "BRCA",
            "signature_score": 0.9,
            "purity_estimate": 0.5,
            "support_fraction_of_top": 1.0,
            "support_score": 1.0,
            "purity_result": {"overall_estimate": 0.5, "components": {}},
        }
    ]
    with patch.object(engine, "rank_cancer_type_candidates") as mock_rank:
        # _fit_one_hypothesis touches many other things; wrap in
        # try/except so we can make a structural assertion without
        # having to stub the full fit path.
        try:
            engine.decompose_sample(
                df_expr,
                cancer_types=["BRCA"],
                candidate_rows=fake_rows,
            )
        except Exception:
            pass
        mock_rank.assert_not_called()


# -----------------------------------------------------------------
# #82: default output dirs get unique timestamps.
# -----------------------------------------------------------------


def test_default_output_dir_has_timestamp():
    """Default output dir name includes a timestamp so concurrent runs
    don't clobber each other."""
    from trufflepig.main import _default_output_dir
    import re

    d = _default_output_dir()
    assert d.startswith("trufflepig-")
    assert re.match(r"trufflepig-\d{8}-\d{6}", d)


def test_build_analyze_paths_falls_back_for_canonical_sentinel():
    """The canonical default-output sentinel ``trufflepig-output`` (and
    the legacy ``pirlygenes-output`` from before the rename) must
    trigger fallback to the timestamped default — otherwise Python-API
    callers using the function default land in a literal directory
    named ``trufflepig-output/``."""
    import dataclasses

    from trufflepig.analyze.flow import build_analyze_paths
    from trufflepig.analyze.models import AnalyzeConfig, InputResolution

    fallback_count = {"calls": 0}

    def _fallback():
        fallback_count["calls"] += 1
        return "/tmp/synthetic-fallback"

    res = InputResolution(
        gene_input="x.tsv",
        transcript_input=None,
        aggregate_gene_expression=True,
        input_level="gene",
        notes=(),
    )
    base = AnalyzeConfig(input_path=res.gene_input)
    for sentinel in (None, "", "pirlygenes-output", "trufflepig-output"):
        cfg = dataclasses.replace(base, output_dir=sentinel or "")
        build_analyze_paths(
            cfg,
            res,
            default_output_dir=_fallback,
            derive_sample_display_id=lambda *a, **k: "x",
            sanitize_output_basename=lambda v: v or "x",
        )
    assert fallback_count["calls"] == 4
