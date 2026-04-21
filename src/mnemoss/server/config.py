"""Server-side configuration.

Read from environment variables by default so the ``mnemoss-server``
CLI is zero-config. Tests and embedded usage construct a ``ServerConfig``
directly and may inject a custom ``Embedder`` / ``LLMClient`` rather
than selecting by string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from mnemoss.encoder import Embedder
from mnemoss.llm.client import LLMClient


@dataclass
class ServerConfig:
    """All tunables for a running server.

    Only the first three fields are typically set via env vars; the
    embedder / llm overrides are for programmatic use (tests, embedded
    deployments) where constructing the real adapters in the server
    process is preferable to string-based factory dispatch.
    """

    # Bearer-token auth. ``None`` means auth is disabled — appropriate
    # for local dev where the server is reachable only on 127.0.0.1.
    api_key: str | None = None

    # Default embedder for every workspace. Options: ``"local"``,
    # ``"openai"``, ``"openai:<model>"``. See ``make_embedder``.
    embedding_model: str = "local"

    # Storage root. ``None`` → ``~/.mnemoss``.
    storage_root: Path | None = None

    # Programmatic override for the embedder. If set, takes precedence
    # over ``embedding_model``. Shared across every workspace the server
    # lazily opens; fine because Embedder instances are stateless in
    # practice.
    embedder_override: Embedder | None = None

    # LLM client shared across every workspace. ``None`` means dream
    # LLM-dependent phases (P3 Extract, P4 Refine, P6 Generalize) will
    # record ``status="skipped"`` — the rest still run.
    llm: LLMClient | None = None

    # Workspace names the server is willing to open. ``None`` means any
    # caller-provided workspace id is allowed; a non-empty set restricts
    # to those IDs. Useful for multi-tenant deployments where the caller
    # should not be able to probe other customers' workspace names.
    allowed_workspaces: set[str] | None = field(default=None)

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Construct from environment variables.

        Recognized:

        - ``MNEMOSS_API_KEY`` — bearer token; unset/empty = no auth
        - ``MNEMOSS_EMBEDDING_MODEL`` — default ``"local"``
        - ``MNEMOSS_STORAGE_ROOT`` — default ``~/.mnemoss``
        - ``MNEMOSS_ALLOWED_WORKSPACES`` — comma-separated list
        """

        api_key = os.environ.get("MNEMOSS_API_KEY") or None
        embedding_model = os.environ.get("MNEMOSS_EMBEDDING_MODEL", "local")
        storage_root_raw = os.environ.get("MNEMOSS_STORAGE_ROOT")
        storage_root = Path(storage_root_raw) if storage_root_raw else None
        allowed_raw = os.environ.get("MNEMOSS_ALLOWED_WORKSPACES")
        allowed = (
            {s.strip() for s in allowed_raw.split(",") if s.strip()}
            if allowed_raw
            else None
        )
        return cls(
            api_key=api_key,
            embedding_model=embedding_model,
            storage_root=storage_root,
            allowed_workspaces=allowed,
        )
