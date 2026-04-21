"""HTTP server for Mnemoss (Stage 6).

Exposes the public ``Mnemoss`` API over JSON/HTTP so non-Python callers
(and the MCP wrapper) can consume it uniformly. The server is the
authoritative integration surface — every SDK is a thin wrapper over
these endpoints.
"""

from mnemoss.server.app import create_app
from mnemoss.server.config import ServerConfig
from mnemoss.server.pool import WorkspacePool

__all__ = ["ServerConfig", "WorkspacePool", "create_app"]
