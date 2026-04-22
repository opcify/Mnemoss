"""Cross-process advisory lock for single-writer-per-workspace.

Mnemoss's invariant is "one writer process per workspace" — within a
process, ``asyncio.Lock`` + WAL serialize writes. This module adds
the process-level half: a second process trying to open the same
workspace fails fast instead of racing with the first and corrupting
data.

Implementation uses OS-level advisory locks on a sentinel file:

- Unix (macOS, Linux, BSD): ``fcntl.flock(LOCK_EX | LOCK_NB)``.
- Windows: ``msvcrt.locking(LK_NBLCK)``.

Both are stdlib, both auto-release when the process holding the lock
exits (even on SIGKILL). No third-party dependency, no stale-lock
bookkeeping.

A readers-only process would also fail today because we don't
distinguish read-only opens. That's intentional for v1 — the concern
is corruption, and a reader-with-writer scenario on SQLite WAL is
itself subtle enough to deserve explicit design later.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import IO, Any


class WorkspaceLockError(RuntimeError):
    """Raised when a workspace is already open in another process.

    The error message names the lock path so operators can trace back
    to the workspace + machine. The typical cause is two processes
    pointed at the same workspace directory — fix the deployment
    topology, don't force-release the lock.
    """


class WorkspaceLock:
    """One-writer-per-workspace advisory lock.

    Usage::

        lock = WorkspaceLock(workspace_dir / ".mnemoss.lock")
        lock.acquire()              # raises WorkspaceLockError if held
        try:
            ...
        finally:
            lock.release()

    Locks are reentrant in the sense that re-calling ``acquire`` on a
    lock we already hold is a no-op. The lock file itself is left on
    disk after release — the OS-level lock is what matters, not the
    sentinel's existence.
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fh: IO[Any] | None = None

    def acquire(self) -> None:
        if self._fh is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``a+`` mode creates if missing, doesn't truncate existing
        # sentinels, gives us an fd we can flock on. We intentionally
        # keep the fd open for the lifetime of the lock rather than
        # using a ``with`` block — releasing the fd releases the flock.
        fh = open(self._path, "a+")  # noqa: SIM115 — lifetime managed by release()
        try:
            if sys.platform == "win32":
                self._lock_windows(fh)
            else:
                self._lock_unix(fh)
        except WorkspaceLockError:
            fh.close()
            raise
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                self._unlock_windows(self._fh)
            else:
                self._unlock_unix(self._fh)
        finally:
            self._fh.close()
            self._fh = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    # ─── unix ───────────────────────────────────────────────────────

    def _lock_unix(self, fh: IO[Any]) -> None:
        import fcntl

        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise WorkspaceLockError(
                f"Workspace is already open in another process "
                f"(lock file: {self._path}). Mnemoss enforces one "
                "writer per workspace.\n"
                "To fix:\n"
                "  • Stop the other process holding the workspace, or\n"
                "  • Point this process at a different workspace.\n"
                "If you're sure no other process is alive and the lock "
                "is stale (e.g. a previous crash), it's safe to delete "
                f"{self._path} — the OS would have already released the "
                "flock when the holder died, so the file's mere existence "
                "is not what's blocking you."
            ) from e

    def _unlock_unix(self, fh: IO[Any]) -> None:
        import fcntl

        with contextlib.suppress(OSError):  # pragma: no cover - release best-effort
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # ─── windows ────────────────────────────────────────────────────

    def _lock_windows(self, fh: IO[Any]) -> None:  # pragma: no cover - windows-only
        import msvcrt

        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        except OSError as e:
            raise WorkspaceLockError(
                f"Workspace is already open in another process "
                f"(lock file: {self._path}). Mnemoss enforces one "
                "writer per workspace. Stop the other process or point "
                "this one at a different workspace. If you're sure "
                "the lock is stale (previous crash), it's safe to "
                f"delete {self._path}."
            ) from e

    def _unlock_windows(self, fh: IO[Any]) -> None:  # pragma: no cover - windows-only
        import msvcrt

        with contextlib.suppress(OSError):
            # msvcrt requires seeking to the locked byte before unlocking.
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
