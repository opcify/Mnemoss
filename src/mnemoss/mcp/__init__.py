"""MCP server wrapper for Mnemoss.

Exposes the Mnemoss surface (observe / recall / pin / explain / dream /
export / rebalance / dispose / tombstones / tiers / flush) as MCP
tools. Two backend modes:

- **Embedded** (default): opens a local ``Mnemoss`` in this process.
  Zero setup — ``pip install mnemoss[mcp]`` and point your MCP client
  at ``mnemoss-mcp``.
- **HTTP**: when ``MNEMOSS_API_URL`` is set, routes tool calls through
  the Python SDK to a remote ``mnemoss-server``. Lets one Mnemoss
  deployment serve many MCP-enabled agents.

Both modes expose an identical tool surface — the choice is operational.
"""

from mnemoss.mcp.config import MCPConfig
from mnemoss.mcp.server import create_mcp_server

__all__ = ["MCPConfig", "create_mcp_server"]
