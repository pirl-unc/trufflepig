import importlib.util
import json
import logging
from pathlib import Path
import sys


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


def test_main_reports_skipped_inputs_separately(monkeypatch, tmp_path, capsys):
    regen = _load_regen_module()
    sample = tmp_path / "sample.tsv"
    sample.write_text("gene\tTPM\nEGFR\t1\n")
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "name": "present",
                        "input": str(sample),
                        "command": ["python", "-m", "trufflepig.cli", "run", "--sample", str(sample)],
                    },
                    {
                        "name": "missing",
                        "input": str(tmp_path / "missing.tsv"),
                        "command": ["python", "-m", "trufflepig.cli", "run"],
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(regen, "_run", lambda *_args: (0, 1.0))
    monkeypatch.setattr(regen, "_collect_signal_matrix_artifacts", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "regenerate_local_reports.py",
            "--source",
            str(source),
            "--root",
            str(tmp_path / "reports"),
            "--skip-comparisons",
        ],
    )

    assert regen.main() == 0
    assert "ok=1  skipped=1  failed=0" in capsys.readouterr().out
