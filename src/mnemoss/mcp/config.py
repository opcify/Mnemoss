"""Configuration for the MCP wrapper.

Env-var-driven so the MCP launcher (e.g. the entry in a Claude Code
settings file) can fully parameterize the server without flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class MCPConfig:
    """All tunables for the MCP server.

    ``api_url`` being set (non-empty) is what selects HTTP mode over
    embedded; the default is embedded for the zero-setup use case.
    """

    workspace: str = "default"
    agent_id: str | None = None

    # HTTP-mode knobs. Ignored in embedded mode.
    api_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> MCPConfig:
        """Read configuration from environment variables.

        Recognized:

        - ``MNEMOSS_WORKSPACE``  — workspace id (default ``"default"``)
        - ``MNEMOSS_AGENT_ID``   — if set, every tool call scopes to this agent
        - ``MNEMOSS_API_URL``    — if set, use HTTP mode via SDK
        - ``MNEMOSS_API_KEY``    — bearer token for HTTP mode
        """

        return cls(
            workspace=os.environ.get("MNEMOSS_WORKSPACE", "default"),
            agent_id=os.environ.get("MNEMOSS_AGENT_ID") or None,
            api_url=os.environ.get("MNEMOSS_API_URL") or None,
            api_key=os.environ.get("MNEMOSS_API_KEY") or None,
        )
