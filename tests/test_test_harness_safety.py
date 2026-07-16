"""Regression tests for the test-suite OOM safeguards."""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pytest_probe(*args, env_updates=None):
    env = os.environ.copy()
    for name in (
        "PYTEST_ADDOPTS",
        "TRUFFLEPIG_XDIST_APPROVED_WORKERS",
        "TRUFFLEPIG_TEST_LOCK_OWNER",
        "TRUFFLEPIG_TEST_LOCK_BYPASS",
    ):
        env.pop(name, None)
    env.update(env_updates or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_pyproject_does_not_enable_xdist_by_default():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pytest_section = text.split("[tool.pytest.ini_options]", 1)[1]
    pytest_section = pytest_section.split("\n[", 1)[0]
    assert not re.search(r"^\s*addopts\s*=.*(?:-n|--numprocesses)", pytest_section, re.MULTILINE)


def test_unbudgeted_xdist_pool_is_rejected_before_collection():
    result = _pytest_probe("-n", "2", "--collect-only", str(__file__))
    assert result.returncode == 4
    assert "refusing unsafe pytest-xdist worker request" in result.stderr


def test_direct_pytest_respects_live_suite_lock(tmp_path):
    lock_dir = tmp_path / "suite.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    result = _pytest_probe(
        "--collect-only",
        str(__file__),
        env_updates={"TRUFFLEPIG_TEST_LOCK_DIR": str(lock_dir)},
    )
    assert result.returncode == 4
    assert "refusing concurrent test run" in result.stderr
