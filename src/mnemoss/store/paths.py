"""Workspace path resolution."""

from __future__ import annotations

from pathlib import Path


def workspace_root(root: Path | None, workspace: str) -> Path:
    """Return the directory for a workspace, creating none of the tree."""

    base = root if root is not None else Path.home() / ".mnemoss"
    return base / "workspaces" / workspace


def workspace_db_path(root: Path | None, workspace: str) -> Path:
    """Return the SQLite file path for a workspace."""

    return workspace_root(root, workspace) / "memory.sqlite"
