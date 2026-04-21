"""Backend abstraction for the MCP wrapper.

Two concrete backends: a local ``Mnemoss`` (embedded) and an HTTP
``WorkspaceHandle`` (via the Python SDK). Both conform structurally
to the ``MCPBackend`` protocol because ``Mnemoss`` and
``WorkspaceHandle`` expose the same public method signatures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mnemoss.client import Mnemoss
from mnemoss.mcp.config import MCPConfig
from mnemoss.sdk import MnemossClient


@asynccontextmanager
async def open_backend(config: MCPConfig) -> AsyncIterator[Any]:
    """Yield a backend (``Mnemoss`` or ``WorkspaceHandle``) and clean up.

    HTTP mode: opens an ``MnemossClient``, returns ``.workspace(id)``,
    closes the client on exit.

    Embedded mode: opens a ``Mnemoss`` directly, closes it on exit.
    """

    if config.api_url:
        client = MnemossClient(config.api_url, api_key=config.api_key)
        try:
            yield client.workspace(config.workspace)
        finally:
            await client.close()
    else:
        mem = Mnemoss(workspace=config.workspace)
        try:
            yield mem
        finally:
            await mem.close()
