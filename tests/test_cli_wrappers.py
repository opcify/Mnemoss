"""Smoke tests for the three CLI entry points.

These are thin wrappers (``uvicorn.run`` / ``asyncio.run`` /
``asyncio.run`` again) around code tested more thoroughly elsewhere.
We cover them here so coverage doesn't show 0% and so an accidental
rename of an upstream function surfaces in CI.

Nothing here actually starts a network listener — we patch the
underlying ``uvicorn.run`` / ``asyncio.run`` to return immediately
and assert the wiring passed the expected arguments.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_server_cli_main_wires_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """``mnemoss-server`` reads host / port from env and calls
    ``uvicorn.run`` with the FastAPI factory. We fake ``uvicorn`` so
    the test doesn't actually bind a port."""

    # Stub a fake uvicorn module so the lazy import inside main() is
    # satisfied without needing the real package.
    fake_uvicorn = MagicMock()
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("MNEMOSS_HOST", "0.0.0.0")
    monkeypatch.setenv("MNEMOSS_PORT", "9999")

    from mnemoss.server import cli

    cli.main()

    fake_uvicorn.run.assert_called_once()
    args, kwargs = fake_uvicorn.run.call_args
    # Factory path, forwarded env vars.
    assert args[0] == "mnemoss.server.app:create_app"
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9999
    assert kwargs["factory"] is True


def test_mcp_cli_main_runs_stdio() -> None:
    """``mnemoss-mcp`` reads env config, opens a backend, creates an
    MCP server, and serves over stdio. We patch the ``asyncio.run``
    call inside the module so we don't actually spin up the stdio
    transport (which blocks)."""

    from mnemoss.mcp import cli

    # Drain the returned coroutine so Python doesn't warn about it
    # never being awaited.
    def _drain(coro: object) -> None:
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[attr-defined]

    with patch.object(cli.asyncio, "run", side_effect=_drain) as mock_run:
        cli.main()
        mock_run.assert_called_once()
