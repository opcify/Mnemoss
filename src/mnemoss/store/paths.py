"""Workspace path resolution.

A workspace directory now contains two SQLite files (schema v6+):

- ``memory.sqlite`` — Memory Store, vectors, FTS, relations, pins,
  tombstones. Bounded in size; read- and write-heavy.
- ``raw_log.sqlite`` — Raw Log only. Pure append, grows linearly with
  conversation volume, may be rotated/archived independently.

The split realizes Principle 3 (Raw Log and Memory Store are separate
layers) as a physical file boundary rather than only a table boundary,
so each file can have its own retention, backup, and tuning profile.
"""

from __future__ import annotations

from pathlib import Path


def workspace_root(root: Path | None, workspace: str) -> Path:
    """Return the directory for a workspace, creating none of the tree."""

    base = root if root is not None else Path.home() / ".mnemoss"
    return base / "workspaces" / workspace


def workspace_db_path(root: Path | None, workspace: str) -> Path:
    """Return the Memory Store SQLite file path for a workspace."""

    return workspace_root(root, workspace) / "memory.sqlite"


def raw_log_db_path(root: Path | None, workspace: str) -> Path:
    """Return the Raw Log SQLite file path for a workspace."""

    return workspace_root(root, workspace) / "raw_log.sqlite"
