"""Save report plots as both raster previews and zoomable vector PDFs."""

from pathlib import Path


def save_report_figure(figure, filename, *, dpi=300):
    path = Path(filename)
    # TrueType preserves searchable labels as well as vector geometry.
    import matplotlib as mpl

    with mpl.rc_context({"pdf.fonttype": 42}):
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        if path.suffix.lower() == ".png":
            figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
