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


def test_direct_pytest_fails_closed_for_ownerless_lock(tmp_path):
    """A competing controller may have mkdir'd but not published its PID yet."""
    lock_dir = tmp_path / "suite.lock"
    lock_dir.mkdir()

    result = _pytest_probe(
        "--collect-only",
        str(__file__),
        env_updates={"TRUFFLEPIG_TEST_LOCK_DIR": str(lock_dir)},
    )

    assert result.returncode == 4
    assert "owner metadata is absent or incomplete" in result.stderr
    assert lock_dir.is_dir(), "a contender must never reclaim an ownerless lock"
    assert not (lock_dir / "pid").exists()


def test_direct_pytest_does_not_reclaim_a_stale_lock(tmp_path):
    lock_dir = tmp_path / "suite.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text("999999\n", encoding="utf-8")

    result = _pytest_probe(
        "--collect-only",
        str(__file__),
        env_updates={"TRUFFLEPIG_TEST_LOCK_DIR": str(lock_dir)},
    )

    assert result.returncode == 4
    assert "recorded owner pid=999999" in result.stderr
    assert (lock_dir / "pid").read_text(encoding="utf-8").strip() == "999999"


def test_wrapper_lock_authorizes_only_its_direct_pytest_child(tmp_path):
    lock_dir = tmp_path / "suite.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    env_updates = {
        "TRUFFLEPIG_TEST_LOCK_DIR": str(lock_dir),
        "TRUFFLEPIG_TEST_LOCK_OWNER": str(os.getpid()),
        "TRUFFLEPIG_XDIST_APPROVED_WORKERS": "1",
    }

    direct = _pytest_probe(
        "-n",
        "1",
        "--collect-only",
        str(__file__),
        env_updates=env_updates,
    )

    assert direct.returncode == 0, direct.stderr

    env = os.environ.copy()
    for name in ("PYTEST_ADDOPTS", "TRUFFLEPIG_TEST_LOCK_BYPASS"):
        env.pop(name, None)
    env.update(env_updates)
    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        "1",
        "--collect-only",
        str(__file__),
    ]
    intermediary = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys; "
                "raise SystemExit(subprocess.run(sys.argv[1:]).returncode)"
            ),
            *pytest_command,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert intermediary.returncode == 4
    assert "refusing concurrent test run" in intermediary.stderr
