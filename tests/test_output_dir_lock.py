"""Tests for the output-directory lockfile (#82)."""

import json
import os

import pytest

from trufflepig.main import (
    _acquire_output_dir_lock,
    _pid_is_alive,
    _LOCKFILE_NAME,
)


def test_pid_is_alive_self():
    """The current process pid must register as alive."""
    assert _pid_is_alive(os.getpid())


def test_pid_is_alive_nonexistent():
    """A very high pid that doesn't exist should register as dead."""
    # 2^22 - 1 is above typical pid_max on macOS/Linux; if by luck
    # this is an alive pid the test is flaky, but extremely unlikely.
    assert not _pid_is_alive(4_194_303)


def test_pid_is_alive_invalid_returns_false():
    """Negative / zero pids are treated as not-alive."""
    assert not _pid_is_alive(0)
    assert not _pid_is_alive(-1)


def test_acquire_lock_writes_pid_and_timestamp(tmp_path):
    lock_path = _acquire_output_dir_lock(tmp_path, force=False)
    assert lock_path.exists()
    assert lock_path.name == _LOCKFILE_NAME
    payload = json.loads(lock_path.read_text())
    assert payload["pid"] == os.getpid()
    assert "started_at" in payload


def test_acquire_lock_refuses_live_holder(tmp_path):
    """A second acquire against a lockfile held by a live pid must
    raise with a clear error. Simulate by writing a live-pid lockfile
    (our own pid) and then trying to re-acquire."""
    # Pre-seed lock with our own pid (which is definitely alive).
    lock_path = tmp_path / _LOCKFILE_NAME
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started_at": "test"}))

    with pytest.raises(RuntimeError) as exc:
        _acquire_output_dir_lock(tmp_path, force=False)
    msg = str(exc.value).lower()
    assert "already" in msg or "lock" in msg
    assert "--force" in str(exc.value)
    assert str(os.getpid()) in str(exc.value)


def test_acquire_lock_force_overrides_live_holder(tmp_path, capsys):
    """``force=True`` claims the lock even when a live pid holds it.

    Also pins the user-facing notice: regression for a missing ``f``
    prefix that printed ``{holder_started}`` literally.
    """
    lock_path = tmp_path / _LOCKFILE_NAME
    lock_path.write_text(json.dumps({"pid": os.getpid(), "started_at": "2026-05-13T01:23:45"}))

    new_lock = _acquire_output_dir_lock(tmp_path, force=True)
    # Should have re-written the lock with this process's pid.
    payload = json.loads(new_lock.read_text())
    assert payload["pid"] == os.getpid()

    out = capsys.readouterr().out
    # Both interpolations must have rendered — no literal "{holder_started}".
    assert "2026-05-13T01:23:45" in out
    assert "{holder_started}" not in out
    assert "{holder_pid}" not in out


def test_acquire_lock_reclaims_stale_lockfile(tmp_path):
    """A lockfile held by a dead pid is reclaimed silently without
    requiring --force — crashed runs shouldn't leave a zombie lock
    that blocks subsequent invocations."""
    stale_pid = 4_194_303  # not a live pid
    lock_path = tmp_path / _LOCKFILE_NAME
    lock_path.write_text(json.dumps({"pid": stale_pid, "started_at": "stale"}))

    # Must not raise — stale locks are auto-reclaimed.
    new_lock = _acquire_output_dir_lock(tmp_path, force=False)
    payload = json.loads(new_lock.read_text())
    assert payload["pid"] == os.getpid()


def test_acquire_lock_handles_corrupt_lockfile(tmp_path):
    """A malformed JSON lockfile shouldn't crash acquire — treat it
    as stale and reclaim."""
    lock_path = tmp_path / _LOCKFILE_NAME
    lock_path.write_text("not-json-at-all")

    new_lock = _acquire_output_dir_lock(tmp_path, force=False)
    assert new_lock.exists()
    payload = json.loads(new_lock.read_text())
    assert payload["pid"] == os.getpid()
