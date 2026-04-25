"""Smoke tests for ``bench/plots.py``.

Each plot function should produce a non-empty, parseable SVG. The
tests check structure, not pixels — the visual quality lives in the
blog post, not CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from bench.plots import BACKEND_COLORS, plot_chart_1  # noqa: E402


def _minimal_chart1_result(
    backend: str,
    per_conv: list[tuple[str, float]],
    aggregate: float,
) -> dict:
    """Build the minimal JSON payload ``plot_chart_1`` needs."""

    return {
        "chart": 1,
        "backend": backend,
        "corpus": "locomo",
        "k": 10,
        "params": {},
        "per_conversation": [
            {
                "conversation_id": cid,
                "n_memories": 100,
                "n_queries_scored": 20,
                "n_queries_skipped": 0,
                "mean_recall_at_k": val,
            }
            for cid, val in per_conv
        ],
        "aggregate": {
            "mean_recall_at_k": aggregate,
            "n_conversations": len(per_conv),
            "n_queries": sum(20 for _ in per_conv),
        },
        "timestamp": "2026-04-23T00:00:00Z",
        "duration_seconds": 1.0,
    }


# ─── happy path ───────────────────────────────────────────────────


def test_plot_chart_1_writes_svg_file(tmp_path: Path) -> None:
    mnemoss = _minimal_chart1_result(
        "mnemoss",
        [("conv-1", 0.72), ("conv-2", 0.68), ("conv-3", 0.75)],
        aggregate=0.72,
    )
    static_file = _minimal_chart1_result(
        "static_file",
        [("conv-1", 0.31), ("conv-2", 0.29), ("conv-3", 0.34)],
        aggregate=0.31,
    )
    p1 = tmp_path / "mnemoss.json"
    p2 = tmp_path / "static.json"
    p1.write_text(json.dumps(mnemoss))
    p2.write_text(json.dumps(static_file))

    out = tmp_path / "chart1.svg"
    plot_chart_1([p1, p2], out, title="Recall@10 on LoCoMo 2024")

    assert out.exists()
    body = out.read_text()
    assert body.startswith("<?xml") or body.startswith("<svg")
    assert "</svg>" in body
    # Title text lands in the SVG.
    assert "Recall@10 on LoCoMo 2024" in body
    # Both backend labels show up in the legend.
    assert "mnemoss" in body
    assert "static_file" in body
    # Our ACCENT-ish colors are embedded — a simple sanity proof that
    # the palette made it through.
    assert BACKEND_COLORS["mnemoss"].lower() in body.lower()


# ─── validation ───────────────────────────────────────────────────


def test_plot_chart_1_raises_on_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_chart_1([], tmp_path / "oops.svg")


def test_plot_chart_1_rejects_mixed_corpus(tmp_path: Path) -> None:
    a = _minimal_chart1_result("mnemoss", [("c1", 0.5)], 0.5)
    b = _minimal_chart1_result("static_file", [("c1", 0.3)], 0.3)
    b["corpus"] = "synthetic"
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(a))
    p2.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="corpus"):
        plot_chart_1([p1, p2], tmp_path / "out.svg")


def test_plot_chart_1_rejects_mixed_k(tmp_path: Path) -> None:
    a = _minimal_chart1_result("mnemoss", [("c1", 0.5)], 0.5)
    b = _minimal_chart1_result("static_file", [("c1", 0.3)], 0.3)
    b["k"] = 5
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(a))
    p2.write_text(json.dumps(b))
    with pytest.raises(ValueError, match="k"):
        plot_chart_1([p1, p2], tmp_path / "out.svg")


def test_plot_chart_1_rejects_non_chart1_input(tmp_path: Path) -> None:
    a = _minimal_chart1_result("mnemoss", [("c1", 0.5)], 0.5)
    a["chart"] = 2
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(a))
    with pytest.raises(ValueError, match="non-Chart-1"):
        plot_chart_1([p1], tmp_path / "out.svg")


# ─── conversation ordering is stable ──────────────────────────────


def test_plot_chart_1_preserves_conversation_order_from_first_run(
    tmp_path: Path,
) -> None:
    """X-axis order should follow the first run's conversation list
    (plus any new ones from later runs, appended). If this drifts, the
    blog post's side-by-side comparison reads wrong."""

    a = _minimal_chart1_result(
        "mnemoss",
        [("conv-a", 0.5), ("conv-b", 0.6), ("conv-c", 0.7)],
        0.6,
    )
    # Second run has conversations in different order + one extra.
    b = _minimal_chart1_result(
        "static_file",
        [("conv-c", 0.3), ("conv-d", 0.2), ("conv-a", 0.4)],
        0.3,
    )
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(a))
    p2.write_text(json.dumps(b))

    out = tmp_path / "chart1.svg"
    plot_chart_1([p1, p2], out)
    body = out.read_text()
    # conv-a, conv-b, conv-c from the first run, then conv-d from the second.
    pos_a = body.find("conv-a")
    pos_b = body.find("conv-b")
    pos_c = body.find("conv-c")
    pos_d = body.find("conv-d")
    assert pos_a < pos_b < pos_c < pos_d
