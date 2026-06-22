import importlib.util
from pathlib import Path


def _load_analyze_reports():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_reports.py"
    spec = importlib.util.spec_from_file_location("analyze_reports_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infer_expected_strips_rep_suffix_for_flat_report_roots():
    ar = _load_analyze_reports()

    assert ar._expected_for("KIRC_rep02", "KIRC_rep02", {}, True) == ("KIRC", "")
    assert ar._expected_for("COAD_MSI_rep05", "COAD_MSI_rep05", {}, True) == (
        "COAD_MSI",
        "",
    )


def test_infer_expected_preserves_nested_group_code():
    ar = _load_analyze_reports()

    assert ar._expected_for("KIRC", "KIRC_rep02", {}, True) == ("KIRC", "")


def test_compat_reports_match_level():
    ar = _load_analyze_reports()
    compat = ar.Compat(
        {"COAD": "CRC", "READ": "CRC", "COAD_MSI": "COAD"},
        lambda code: {"LUSC": "squamous", "HNSC": "squamous"}.get(code, ""),
    )

    assert compat.match_level("COAD_MSI", "COAD_MSI") == "exact"
    assert compat.match_level("COAD_MSI", "COAD") == "subtype_prefix"
    assert compat.match_level("READ", "COAD") == "registry_root"
    assert compat.match_level("SARC_GIST", "SARC_ANGIO") == "base_token"
    assert compat.match_level("LUSC", "HNSC") == "broad_lineage"
    assert compat.match_level("LUSC", "MBL") == "none"
