"""Smoke tests for ``docs/figures.py``.

Each figure function should produce a parseable SVG with minimal
structural content. We don't assert on pixel-level rendering — this
is a smoke check for "did matplotlib refuse to draw."
"""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from docs.figures import fig_a_decay_curve, fig_b_spreading_graph, fig_c_breakdown_bar  # noqa: E402


def _assert_is_valid_svg(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 1000, "SVG suspiciously small"
    body = path.read_text()
    assert body.startswith("<?xml") or body.startswith("<svg")
    assert "</svg>" in body


# ─── Fig A ────────────────────────────────────────────────────────


def test_fig_a_decay_curve_smoke(tmp_path: Path) -> None:
    out = tmp_path / "fig-a.svg"
    fig_a_decay_curve(out)
    _assert_is_valid_svg(out)
    body = out.read_text()
    assert "24h" in body or "activation" in body.lower()


def test_fig_a_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "fig-a.svg"
    fig_a_decay_curve(out)
    _assert_is_valid_svg(out)


# ─── Fig B ────────────────────────────────────────────────────────


def test_fig_b_spreading_graph_smoke(tmp_path: Path) -> None:
    out = tmp_path / "fig-b.svg"
    fig_b_spreading_graph(out)
    _assert_is_valid_svg(out)
    body = out.read_text()
    # Spreading is the visible story — verify the title text landed.
    assert "Spreading" in body or "spreading" in body


# ─── Fig C ────────────────────────────────────────────────────────


def test_fig_c_with_canned_breakdowns(tmp_path: Path) -> None:
    """Supply hand-canned breakdowns so the test doesn't need to run
    Mnemoss live — isolates the plotting code from the formula code."""

    breakdowns = [
        {"base_level": 1.2, "spreading": 0.4, "matching": 0.8, "noise": 0.0, "total": 2.4},
        {"base_level": 1.0, "spreading": 0.3, "matching": 0.6, "noise": 0.05, "total": 1.95},
        {"base_level": 0.9, "spreading": 0.2, "matching": 0.5, "noise": 0.0, "total": 1.6},
        {"base_level": 0.7, "spreading": 0.1, "matching": 0.4, "noise": 0.02, "total": 1.22},
        {"base_level": 0.5, "spreading": 0.05, "matching": 0.3, "noise": 0.0, "total": 0.85},
    ]
    out = tmp_path / "fig-c.svg"
    fig_c_breakdown_bar(out, breakdowns=breakdowns)
    _assert_is_valid_svg(out)


def test_fig_c_raises_when_no_breakdowns_available(tmp_path: Path) -> None:
    out = tmp_path / "fig-c.svg"
    with pytest.raises(ValueError, match="no breakdowns"):
        fig_c_breakdown_bar(out, breakdowns=[])


def test_fig_c_live_capture_produces_output(tmp_path: Path) -> None:
    """Run the default Mnemoss-backed capture path end-to-end. Slower
    than the canned-input test but catches schema drift on
    ``ActivationBreakdown.to_dict()``."""

    out = tmp_path / "fig-c-live.svg"
    fig_c_breakdown_bar(out)
    _assert_is_valid_svg(out)


# ─── CLI ──────────────────────────────────────────────────────────


def test_cli_renders_all_three_figures(tmp_path: Path) -> None:
    from docs.figures import main as cli_main

    rc = cli_main(["--out-dir", str(tmp_path)])
    assert rc == 0
    for name in ("fig-a-decay-curve.svg", "fig-b-spreading-graph.svg", "fig-c-breakdown-bar.svg"):
        _assert_is_valid_svg(tmp_path / name)


def test_cli_renders_subset(tmp_path: Path) -> None:
    from docs.figures import main as cli_main

    rc = cli_main(["--out-dir", str(tmp_path), "--figures", "a", "b"])
    assert rc == 0
    assert (tmp_path / "fig-a-decay-curve.svg").exists()
    assert (tmp_path / "fig-b-spreading-graph.svg").exists()
    assert not (tmp_path / "fig-c-breakdown-bar.svg").exists()
