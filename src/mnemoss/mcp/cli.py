"""``mnemoss-mcp`` entry point.

Reads ``MCPConfig`` from environment, opens a backend (embedded or
HTTP), builds a FastMCP server, and runs it over stdio.
"""

from __future__ import annotations

import asyncio

from mnemoss.mcp.backend import open_backend
from mnemoss.mcp.config import MCPConfig
from mnemoss.mcp.server import create_mcp_server


async def _run() -> None:
    config = MCPConfig.from_env()
    async with open_backend(config) as backend:
        mcp = create_mcp_server(backend, config=config)
        # run_stdio_async is the standard transport for local MCP
        # integrations (Claude Code, Cursor, etc.); remote SSE/
        # StreamableHTTP variants can be added if demand appears.
        await mcp.run_stdio_async()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
