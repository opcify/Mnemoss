"""Per-server pool of ``Mnemoss`` workspace instances.

The server multiplexes many callers over a small set of workspaces.
We lazily create a ``Mnemoss`` on first touch and cache it; the lifespan
hook calls ``close_all`` on shutdown so WAL files are checkpointed and
SQLite connections released cleanly.

Thread-safety: a single ``asyncio.Lock`` around the registry is enough
because ``Mnemoss`` instances themselves serialize internal writes.
"""

from __future__ import annotations

import asyncio
import contextlib

from mnemoss.client import Mnemoss
from mnemoss.core.config import StorageParams
from mnemoss.encoder import Embedder
from mnemoss.scheduler import DreamScheduler
from mnemoss.server.config import ServerConfig


class WorkspaceNotAllowedError(Exception):
    """Caller requested a workspace not in ``allowed_workspaces``."""


class WorkspacePool:
    """Owns every ``Mnemoss`` instance the server has opened."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._instances: dict[str, Mnemoss] = {}
        self._schedulers: dict[str, DreamScheduler] = {}
        self._lock = asyncio.Lock()

    async def get(self, workspace_id: str) -> Mnemoss:
        """Return the cached instance, creating it if necessary.

        Raises ``WorkspaceNotAllowedError`` if the config restricts
        workspace IDs and this one is not in the allow-list.
        """

        if (
            self._config.allowed_workspaces is not None
            and workspace_id not in self._config.allowed_workspaces
        ):
            raise WorkspaceNotAllowedError(workspace_id)
        async with self._lock:
            existing = self._instances.get(workspace_id)
            if existing is not None:
                return existing
            instance = self._build(workspace_id)
            self._instances[workspace_id] = instance
            if self._config.scheduler is not None:
                sched = DreamScheduler(instance, self._config.scheduler)
                await sched.start()
                self._schedulers[workspace_id] = sched
            return instance

    async def close_all(self) -> None:
        async with self._lock:
            instances = list(self._instances.values())
            schedulers = list(self._schedulers.values())
            self._instances.clear()
            self._schedulers.clear()
        # Stop schedulers first so they don't race with shutting-down
        # workspaces. Close outside the lock — close() takes the
        # instance's own lock; holding the pool lock here would
        # serialize shutdown. A single wedged workspace should not
        # block the others from closing; surface via logs in a real
        # deployment.
        for sched in schedulers:
            with contextlib.suppress(Exception):
                await sched.stop()
        for mem in instances:
            with contextlib.suppress(Exception):
                await mem.close()

    # ─── internal ─────────────────────────────────────────────────

    def _build(self, workspace_id: str) -> Mnemoss:
        embedding_model: str | Embedder
        if self._config.embedder_override is not None:
            embedding_model = self._config.embedder_override
        else:
            embedding_model = self._config.embedding_model
        storage = (
            StorageParams(root=self._config.storage_root)
            if self._config.storage_root is not None
            else StorageParams()
        )
        return Mnemoss(
            workspace=workspace_id,
            embedding_model=embedding_model,
            storage=storage,
            llm=self._config.llm,
        )
