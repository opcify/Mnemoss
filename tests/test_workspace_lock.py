"""Tests for the cross-process workspace lock.

Exercises both the ``WorkspaceLock`` primitive and its integration
with ``SQLiteBackend.open()``. End-to-end multi-process is stubbed by
simulating the second "process" with a second ``WorkspaceLock`` /
``SQLiteBackend`` instance in the same interpreter — fcntl and
msvcrt locks are per-file-descriptor, so two FDs on the same path
are indistinguishable from two processes as far as the OS is
concerned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.store._workspace_lock import WorkspaceLock, WorkspaceLockError
from mnemoss.store.sqlite_backend import SchemaMismatchError, SQLiteBackend

# ─── WorkspaceLock primitive ──────────────────────────────────────


def test_acquire_then_release(tmp_path: Path) -> None:
    lock = WorkspaceLock(tmp_path / ".lock")
    assert not lock.held
    lock.acquire()
    assert lock.held
    lock.release()
    assert not lock.held


def test_double_acquire_is_idempotent(tmp_path: Path) -> None:
    lock = WorkspaceLock(tmp_path / ".lock")
    lock.acquire()
    lock.acquire()  # no-op
    assert lock.held
    lock.release()


def test_second_lock_on_same_path_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / ".lock"
    first = WorkspaceLock(lock_path)
    first.acquire()
    second = WorkspaceLock(lock_path)
    try:
        with pytest.raises(WorkspaceLockError, match="already open"):
            second.acquire()
        assert not second.held  # failed acquires leave no FD behind
    finally:
        first.release()


def test_lock_can_be_reacquired_after_release(tmp_path: Path) -> None:
    lock_path = tmp_path / ".lock"
    first = WorkspaceLock(lock_path)
    first.acquire()
    first.release()

    second = WorkspaceLock(lock_path)
    second.acquire()  # succeeds now that first released
    second.release()


def test_release_without_acquire_is_safe(tmp_path: Path) -> None:
    lock = WorkspaceLock(tmp_path / ".lock")
    lock.release()  # no-op


# ─── SQLiteBackend integration ────────────────────────────────────


def _memory(mid: str, ws: str = "ws") -> Memory:
    from datetime import datetime, timezone

    return Memory(
        id=mid,
        workspace_id=ws,
        agent_id=None,
        session_id="s",
        created_at=datetime.now(timezone.utc),
        content="hello",
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[datetime.now(timezone.utc)],
        index_tier=IndexTier.HOT,
    )


async def test_second_backend_on_same_workspace_raises(tmp_path: Path) -> None:
    first = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake",
    )
    await first.open()

    second = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake",
    )
    try:
        with pytest.raises(WorkspaceLockError, match="already open"):
            await second.open()
    finally:
        await first.close()


async def test_backend_reopen_after_close_succeeds(tmp_path: Path) -> None:
    first = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake",
    )
    await first.open()
    await first.close()

    # Same path, fresh backend, should acquire cleanly.
    second = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake",
    )
    await second.open()
    await second.close()


async def test_failed_open_does_not_leak_lock(tmp_path: Path) -> None:
    """If backend.open() raises partway through (e.g. schema mismatch),
    the workspace lock should still be released so a corrected
    reopen can succeed."""

    # Create a valid workspace with one embedder_id.
    first = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake-a",
    )
    await first.open()
    await first.close()

    # Open with a mismatched embedder_id — should raise.
    wrong = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake-b",
    )
    with pytest.raises(SchemaMismatchError):
        await wrong.open()

    # Now the correct backend should still be able to open.
    correct = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=4,
        embedder_id="fake-a",
    )
    try:
        await correct.open()
    finally:
        await correct.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fcntl-specific assertion; msvcrt semantics differ slightly",
)
def test_lock_file_persists_after_release(tmp_path: Path) -> None:
    """The sentinel file is left on disk after release — the OS lock
    is what matters, not the file's presence. Recreating the file on
    every acquire would race with concurrent processes."""

    lock_path = tmp_path / ".lock"
    lock = WorkspaceLock(lock_path)
    lock.acquire()
    assert lock_path.exists()
    lock.release()
    assert lock_path.exists()  # sentinel stays
