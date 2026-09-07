"""Subtype figure branches describe RNA without inventing clinical states."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import trufflepig.plot_subtype_signature as plotting


@pytest.mark.parametrize("code", ["PRAD", "BRCA", "LUAD"])
@pytest.mark.parametrize("a,b", [(30.0, 1.0), (1.0, 30.0), (1.0, 1.0), (30.0, 30.0)])
def test_subtype_figure_keeps_rna_context_and_separates_tpm_labels(
    monkeypatch, tmp_path, code, a, b
):
    contrast = plotting.SUBTYPE_CONTRASTS[code][0]
    signatures = {
        contrast["axis_a"]: {"up": [{"symbol": "AR"}]},
        contrast["axis_b"]: {"up": [{"symbol": "ASCL1"}]},
    }
    monkeypatch.setattr(plotting, "load_therapy_signatures", lambda: {})
    monkeypatch.setattr(plotting, "_sigs_for_cancer", lambda *args: signatures)
    monkeypatch.setattr(
        plotting,
        "pan_cancer_expression",
        lambda **kwargs: pd.DataFrame({"Symbol": ["AR", "ASCL1"], f"{code}_TPM": [10.0, 10.0]}),
    )
    frame = pd.DataFrame({"gene_symbol": ["AR", "ASCL1"], "TPM": [a, b]})
    output = tmp_path / "subtype.png"
    figure = plotting.plot_subtype_signature(frame, code, save_to_filename=output, save_dpi=100)
    try:
        caption = " ".join(text.get_text() for text in figure.texts)
        assert "RNA" in caption
        for claim in [
            "NEPC — AR collapsed",
            "post-ADT) without NE emergence — CRPC",
            "Triple-negative pattern — checkpoint",
            "TKI resistance; checkpoint",
        ]:
            assert claim not in caption
        assert output.stat().st_size > 0
        for axis, value in zip(figure.axes, (a, b)):
            assert any(f"{value:.1f} TPM" == t.get_text() for t in axis.texts)
            for t in axis.texts:
                assert t.get_ha() == "right"
                assert 0 < t.get_position()[0] < 1
    finally:
        plt.close(figure)
