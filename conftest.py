"""Fail-safe pytest guards for this repository's memory-heavy test suite.

Each xdist worker can retain roughly 10 GB after loading the expression
references.  Keep this guard in the repository root so pytest loads it before
xdist starts workers, including for IDE and direct ``pytest`` invocations.
"""

import os
import tempfile
from pathlib import Path

import pytest

_UNWRAPPED_WORKER_LIMIT = 1
_LOCK_ENV = "TRUFFLEPIG_TEST_LOCK_DIR"
_LOCK_OWNER_ENV = "TRUFFLEPIG_TEST_LOCK_OWNER"
_LOCK_BYPASS_ENV = "TRUFFLEPIG_TEST_LOCK_BYPASS"
_APPROVED_WORKERS_ENV = "TRUFFLEPIG_XDIST_APPROVED_WORKERS"


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _requested_workers(config):
    value = getattr(config.option, "numprocesses", None)
    if value in (None, 0, "0"):
        return 0
    return _positive_int(value) or value


def _enforce_worker_limit(config):
    requested = _requested_workers(config)
    if requested == 0:
        return

    approved_raw = os.environ.get(_APPROVED_WORKERS_ENV)
    approved = _positive_int(approved_raw)
    if approved_raw is not None and approved is None:
        raise pytest.UsageError(
            f"invalid {_APPROVED_WORKERS_ENV}={approved_raw!r}; run tests with ./test.sh"
        )

    if approved is None:
        if not isinstance(requested, int) or requested > _UNWRAPPED_WORKER_LIMIT:
            raise pytest.UsageError(
                f"refusing unsafe pytest-xdist worker request -n {requested}; "
                "run tests with ./test.sh so workers are RAM-budgeted"
            )
        return

    if not isinstance(requested, int) or requested > approved:
        raise pytest.UsageError(
            f"refusing pytest-xdist worker request -n {requested}; "
            f"./test.sh approved at most {approved} worker(s) for current RAM"
        )


def _default_lock_dir():
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    return Path(tempfile.gettempdir()) / f"trufflepig-test-{uid}.lock"


def _read_owner(lock_dir):
    owner_file = lock_dir / "pid"
    try:
        return _positive_int(owner_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError):
        return None


def _remove_owned_lock(lock_dir):
    try:
        (lock_dir / "pid").unlink(missing_ok=True)
        lock_dir.rmdir()
    except (FileNotFoundError, OSError):
        return False
    return True


def _enforce_suite_lock(config):
    if os.environ.get(_LOCK_BYPASS_ENV) == "1":
        return

    lock_dir = Path(os.environ.get(_LOCK_ENV, str(_default_lock_dir())))
    expected_owner = _positive_int(os.environ.get(_LOCK_OWNER_ENV))

    try:
        lock_dir.mkdir()
    except FileExistsError:
        owner = _read_owner(lock_dir)
        # test.sh owns the lock only for its DIRECT pytest controller.
        # Environment variables are inherited by arbitrary descendants; an
        # inner pytest launched by a test must not reuse this authorization
        # while the outer worker retains its multi-GB reference frames.
        if (
            expected_owner is not None
            and owner == expected_owner
            and os.getppid() == expected_owner
        ):
            return
        if owner is None:
            detail = "owner metadata is absent or incomplete"
        else:
            detail = f"recorded owner pid={owner}"
        # Fail closed for every existing lock.  In particular, never reclaim an
        # ownerless directory: another controller may be between atomic mkdir
        # and publishing its PID.  Avoiding all automatic reclamation also
        # removes the stale-lock ABA race where a reclaimer can delete a newly
        # acquired lock after checking the previous owner.
        raise pytest.UsageError(
            f"refusing concurrent test run: existing test-suite lock "
            f"{lock_dir}: {detail}; "
            "wait for the active run, or remove the lock only after confirming "
            "that no pytest/test.sh process is active"
        )
    except OSError as exc:
        raise pytest.UsageError(f"cannot create test-suite lock {lock_dir}: {exc}") from exc

    try:
        (lock_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError as exc:
        _remove_owned_lock(lock_dir)
        raise pytest.UsageError(f"cannot write test-suite lock {lock_dir}: {exc}") from exc
    config._trufflepig_test_lock = lock_dir


def pytest_configure(config):
    """Reject unsafe concurrency before xdist's session-start worker spawn."""
    if hasattr(config, "workerinput"):
        return
    _enforce_worker_limit(config)
    _enforce_suite_lock(config)


def pytest_unconfigure(config):
    """Release only locks acquired by direct pytest, never test.sh's lock."""
    lock_dir = getattr(config, "_trufflepig_test_lock", None)
    if lock_dir is None:
        return
    owner = _read_owner(lock_dir)
    if owner == os.getpid():
        _remove_owned_lock(lock_dir)
