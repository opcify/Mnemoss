"""Tests for ``mnemoss inspect``.

Exercises the three surfaces of the CLI:

- Human-readable output on a seeded workspace.
- ``--json`` output is valid JSON and contains the expected keys.
- ``--tombstones`` includes a tombstones section.

We don't test the ``argparse`` wrapper directly; we import the
coroutine and call it via the same ``Namespace`` the CLI would
produce. That keeps tests fast (no subprocess) and deterministic.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    StorageParams,
)
from mnemoss.cli.inspect import _run


async def _seed_workspace(tmp_path: Path) -> None:
    """Create + populate a workspace ready for inspection."""

    mem = Mnemoss(
        workspace="ws",
        embedding_model=FakeEmbedder(dim=16),
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )
    try:
        for i in range(5):
            await mem.observe(role="user", content=f"note {i}")
    finally:
        await mem.close()


async def test_human_readable_output_shows_sections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    await _seed_workspace(tmp_path)

    # The CLI constructs its own Mnemoss internally with FakeEmbedder(dim=1)
    # which won't match our seeded workspace (dim=16). That path is
    # exercised by test_falls_back_when_embedder_mismatches below; to
    # exercise the happy path we pass a matching dim via env isn't
    # supported, so we seed at dim=1 instead.
    mem = Mnemoss(
        workspace="ws_match",
        embedding_model=FakeEmbedder(dim=1),
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await mem.observe(role="user", content="inspect me")
    finally:
        await mem.close()

    args = Namespace(
        workspace="ws_match",
        root=str(tmp_path),
        json=False,
        tombstones=False,
    )
    rc = await _run(args)
    assert rc == 0
    out = capsys.readouterr().out
    # Section headers present.
    for section in ("workspace", "memories", "llm cost", "dreams", "timestamps"):
        assert section in out, f"section {section!r} missing in output"
    # Memory count renders.
    assert "total" in out


async def test_json_output_is_valid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mem = Mnemoss(
        workspace="ws_json",
        embedding_model=FakeEmbedder(dim=1),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await mem.observe(role="user", content="json me")
    finally:
        await mem.close()

    args = Namespace(
        workspace="ws_json",
        root=str(tmp_path),
        json=True,
        tombstones=False,
    )
    rc = await _run(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Canonical keys are all present.
    for key in ("workspace", "schema_version", "memory_count", "llm_cost", "dreams"):
        assert key in payload


async def test_falls_back_when_embedder_mismatches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the CLI's FakeEmbedder(dim=1) doesn't match the workspace's
    pinned dim, status() raises SchemaMismatchError. The CLI should
    catch that and print a filesystem-only peek rather than crashing."""

    await _seed_workspace(tmp_path)  # seeds at dim=16

    args = Namespace(
        workspace="ws",
        root=str(tmp_path),
        json=False,
        tombstones=False,
    )
    rc = await _run(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "cannot open workspace under this embedder" in out
    assert "memory.sqlite" in out


async def test_tombstones_flag_includes_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--tombstones`` lists recent drops. On a fresh workspace with
    no disposals, the section should still appear if at least the
    JSON view includes an empty list."""

    mem = Mnemoss(
        workspace="ws_tomb",
        embedding_model=FakeEmbedder(dim=1),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await mem.observe(role="user", content="alive")
    finally:
        await mem.close()

    args = Namespace(
        workspace="ws_tomb",
        root=str(tmp_path),
        json=True,
        tombstones=True,
    )
    rc = await _run(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "tombstones" in payload
    assert payload["tombstones"] == []
