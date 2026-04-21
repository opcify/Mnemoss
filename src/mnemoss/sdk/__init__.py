"""Python SDK for the Mnemoss REST API (Stage 6).

``MnemossClient`` holds the HTTP connection; ``WorkspaceHandle`` binds
a workspace id; ``AgentHandle`` further binds an agent id. The three
shapes mirror the in-process ``Mnemoss`` / ``AgentHandle`` pair so
framework plugins can swap one for the other without rewriting call
sites.
"""

from mnemoss.sdk.client import AgentHandle, MnemossClient, WorkspaceHandle

__all__ = ["AgentHandle", "MnemossClient", "WorkspaceHandle"]
