import importlib.util
import logging
from pathlib import Path


def _load_regen_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "regenerate_local_reports.py"
    spec = importlib.util.spec_from_file_location("regenerate_local_reports", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_translate_command_can_inject_manifest_cancer_type(tmp_path):
    regen = _load_regen_module()
    run = {
        "input": "/tmp/sample.tsv",
        "cancer_type": "NBL",
        "command": [
            "python",
            "-m",
            "trufflepig.cli",
            "run",
            "--sample",
            "/tmp/sample.tsv",
            "--no-figures",
        ],
    }

    cmd = regen._translate_command(
        "NBL_rep01",
        run,
        tmp_path / "ws",
        use_manifest_cancer_type=True,
    )

    assert "--cancer-type" in cmd
    assert cmd[cmd.index("--cancer-type") + 1] == "NBL"


def test_translate_command_keeps_blind_mode_blind(tmp_path):
    regen = _load_regen_module()
    run = {
        "input": "/tmp/sample.tsv",
        "cancer_type": "NBL",
        "command": [
            "python",
            "-m",
            "trufflepig.cli",
            "run",
            "--sample",
            "/tmp/sample.tsv",
            "--cancer-type",
            "NBL",
            "--no-figures",
        ],
    }

    cmd = regen._translate_command("NBL_rep01", run, tmp_path / "ws", blind=True)

    assert "--cancer-type" not in cmd


def test_translate_command_can_request_full_figures(tmp_path):
    regen = _load_regen_module()
    run = {
        "input": "/tmp/sample.tsv",
        "command": [
            "python",
            "-m",
            "trufflepig.cli",
            "run",
            "--sample",
            "/tmp/sample.tsv",
            "--no-figures",
        ],
    }

    cmd = regen._translate_command(
        "sample",
        run,
        tmp_path / "ws",
        with_figures=True,
    )

    assert "--no-figures" not in cmd


def test_remove_logging_handlers_for_per_run_stream(tmp_path):
    regen = _load_regen_module()
    logger = logging.getLogger("trufflepig-test-regenerate-local-reports")
    original_handlers = list(logger.handlers)
    logger.handlers[:] = []
    log_file = (tmp_path / "sample.log").open("w")
    handler = logging.StreamHandler(log_file)
    logger.addHandler(handler)

    try:
        regen._remove_logging_handlers_for_stream(log_file)

        assert handler not in logger.handlers
    finally:
        logger.handlers[:] = original_handlers
        log_file.close()
