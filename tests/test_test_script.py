import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SCRIPT = REPO_ROOT / "test.sh"


def _mock_pytest(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_path = tmp_path / "pytest-args.txt"
    executable = bin_dir / "pytest"
    executable.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$MOCK_PYTEST_ARGS\"\n"
    )
    executable.chmod(0o755)
    return bin_dir, args_path


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir, args_path = _mock_pytest(tmp_path)
    lock_dir = tmp_path / "test.lock"
    env = os.environ.copy()
    env.update(
        {
            "MOCK_PYTEST_ARGS": str(args_path),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PER_WORKER_GB": "9999",
            "RAM_RESERVE_GB": "8",
            "TEST_SH_LOCK_DIR": str(lock_dir),
        }
    )
    return env, lock_dir, args_path


def test_test_script_caps_workers_and_releases_lock(tmp_path):
    env, lock_dir, args_path = _environment(tmp_path)

    result = subprocess.run(
        ["bash", str(TEST_SCRIPT), "-q", "-k", "focused"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert args_path.read_text().strip() == "-n 1 tests -q -k focused"
    assert not lock_dir.exists()


def test_test_script_refuses_a_live_cross_worktree_lock(tmp_path):
    env, lock_dir, args_path = _environment(tmp_path)
    lock_dir.mkdir()
    (lock_dir / "pid").write_text(f"{os.getpid()}\n")

    result = subprocess.run(
        ["bash", str(TEST_SCRIPT), "-q"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 75
    assert "refusing concurrent run" in result.stderr
    assert not args_path.exists()


def test_test_script_recovers_a_stale_lock(tmp_path):
    env, lock_dir, args_path = _environment(tmp_path)
    lock_dir.mkdir()
    (lock_dir / "pid").write_text("999999\n")

    result = subprocess.run(
        ["bash", str(TEST_SCRIPT), "-q"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert args_path.read_text().strip() == "-n 1 tests -q"
    assert not lock_dir.exists()
