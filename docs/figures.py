"""Static explainer figures for the launch blog post.

Three figures illustrating the core mechanics of Mnemoss's formula.
All are built from ``mnemoss.formula`` directly (not from benchmark
data), so they stay accurate even if we swap corpora or backends.

- :func:`fig_a_decay_curve` — B_i over 24h for a memory accessed
  once at creation. Shows the η (grace) term dominating for the
  first hour and the power-law history term taking over past that.
- :func:`fig_b_spreading_graph` — A small directed graph showing a
  query memory and the three strongest 1-hop neighbors via
  ``S_ji`` edges, plus one 2-hop neighbor.
- :func:`fig_c_breakdown_bar` — Stacked horizontal bars for the
  top-5 hits of one recall, with segments colored by component
  (``B_i`` / spreading / matching / noise).

Each function takes ``out_path`` and writes an SVG. Intentionally
short; these are slides, not dashboards.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from bench.plots import _apply_minimal_style  # noqa: E402

# ─── shared palette ────────────────────────────────────────────────

COLOR_BASE = "#4f46e5"  # indigo
COLOR_SPREAD = "#06b6d4"  # cyan
COLOR_MATCH = "#10b981"  # emerald
COLOR_NOISE = "#6b7280"  # grey
COLOR_ACCENT = "#f59e0b"  # amber
COLOR_BG_GREY = "#f3f4f6"


# ─── Fig A: base-level decay curve ────────────────────────────────


def fig_a_decay_curve(out_path: Path, *, hours_max: float = 24.0) -> None:
    """Plot B_i for a memory accessed once at creation, over 0..``hours_max``.

    Uses ``FormulaParams`` defaults. The left segment (0..30 min) is
    dominated by the η(t) grace term; past that, the log(age^-d)
    history term is the only contributor.
    """

    from mnemoss.core.config import FormulaParams
    from mnemoss.formula.base_level import compute_base_level

    params = FormulaParams()
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # Sample densely in the first hour (where grace matters), sparser after.
    seconds: list[float] = []
    s = 1.0
    while s < 60:
        seconds.append(s)
        s *= 1.5
    while s < 3600:
        seconds.append(s)
        s *= 1.3
    while s < hours_max * 3600:
        seconds.append(s)
        s *= 1.5
    # Always include the endpoints.
    if seconds[-1] < hours_max * 3600:
        seconds.append(hours_max * 3600)

    b_values: list[float] = []
    grace_values: list[float] = []
    history_values: list[float] = []
    for sec in seconds:
        now = created_at + timedelta(seconds=sec)
        b = compute_base_level(
            access_history=[created_at],
            now=now,
            created_at=created_at,
            params=params,
        )
        b_values.append(b)
        grace_values.append(params.eta_0 * math.exp(-sec / params.eta_tau_seconds))
        history_values.append(-params.d * math.log(max(sec, params.t_floor_seconds)))

    hours = [s / 3600 for s in seconds]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    ax.plot(hours, b_values, color=COLOR_BASE, linewidth=2.2, label="B_i (total)")
    ax.plot(
        hours,
        grace_values,
        color=COLOR_ACCENT,
        linewidth=1.4,
        linestyle="--",
        label="η(t) grace term",
    )
    ax.plot(
        hours,
        history_values,
        color=COLOR_NOISE,
        linewidth=1.4,
        linestyle=":",
        label="ln(age⁻ᵈ) history term",
    )
    ax.axhline(0, color="#d1d5db", linewidth=0.8)

    ax.set_xlabel("hours since creation")
    ax.set_ylabel("activation contribution")
    ax.set_xlim(0, hours_max)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.5)
    _apply_minimal_style(ax)

    fig.suptitle(
        "Base-level activation over 24h",
        fontsize=12,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.98,
    )
    ax.set_title(
        "FormulaParams defaults: d=0.5, η₀=1.0, τ_η=1h",
        fontsize=9,
        color="#6b7280",
        loc="left",
        pad=8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── Fig B: spreading activation graph ────────────────────────────


def fig_b_spreading_graph(out_path: Path) -> None:
    """Illustrate ``S_ji`` with a small hand-laid graph.

    One query node, three 1-hop neighbors with ``S_ji`` weights
    annotated on the edges, and one 2-hop neighbor to show that
    spreading chains. Intentionally static — this is an explainer,
    not a visualization of a live Mnemoss workspace.
    """

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=100)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Node positions. Query in the center; 1-hop at 120° apart;
    # 2-hop past the right-hand 1-hop.
    nodes = {
        "query": (0.0, 0.0, "query", COLOR_ACCENT),
        "a": (0.65, 0.45, "m_a", COLOR_BASE),
        "b": (0.65, -0.45, "m_b", COLOR_BASE),
        "c": (-0.75, 0.0, "m_c", COLOR_BASE),
        "d": (0.95, 0.85, "m_d", COLOR_NOISE),  # 2-hop
    }

    edges = [
        # (src, dst, Sji_label, curvature)
        ("query", "a", "0.82", 0.1),
        ("query", "b", "0.64", -0.1),
        ("query", "c", "0.41", 0.05),
        ("a", "d", "0.55", 0.15),
    ]

    # Draw edges first (underneath nodes).
    for src, dst, label, curvature in edges:
        x0, y0, _, _ = nodes[src]
        x1, y1, _, _ = nodes[dst]
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="-|>",
            mutation_scale=12,
            color="#9ca3af",
            linewidth=1.3,
            zorder=1,
        )
        ax.add_patch(arrow)
        # Label midpoint (slight offset from the curve).
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.annotate(
            f"S = {label}",
            xy=(mx, my),
            fontsize=8.5,
            color="#4b5563",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8),
            zorder=2,
        )

    # Draw nodes on top.
    for x, y, label, color in nodes.values():
        ax.scatter([x], [y], s=900, color=color, edgecolors="white", linewidths=2.0, zorder=3)
        ax.annotate(
            label,
            xy=(x, y),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
            zorder=4,
        )

    # Legend (light text block in the lower-left).
    ax.text(
        -1.05,
        -0.93,
        "amber: query    indigo: 1-hop    grey: 2-hop",
        fontsize=9,
        color="#6b7280",
    )

    fig.suptitle(
        "Spreading activation: Σ W_j · S_ji",
        fontsize=12,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.97,
    )
    ax.set_title(
        "Query activates 1-hop neighbors; activation chains to 2-hop",
        fontsize=9,
        color="#6b7280",
        loc="left",
        pad=6,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── Fig C: ActivationBreakdown stacked bars ──────────────────────


async def _capture_breakdowns(num_hits: int = 5) -> list[dict[str, float]]:
    """Run a tiny Mnemoss scenario and grab the top-k breakdowns from
    one recall. Used by fig C when no pre-canned breakdown is supplied.

    The numbers are illustrative, not benchmark-grade — the fixture
    scenario is a 9-memory workspace with overlapping topics so recall
    returns a spread of component values.
    """

    from bench.backends.mnemoss_backend import MnemossBackend
    from mnemoss import FakeEmbedder

    corpus = [
        "alice scheduled the kickoff meeting for friday",
        "bob mentioned the deployment timeline",
        "carol asked about the Q3 roadmap",
        "the kickoff went well, alice sent notes",
        "deployment is blocked on bob's review",
        "Q3 roadmap has three confirmed deliverables",
        "friday slot is free on alice's calendar",
        "bob approved the staging release",
        "alice followed up on the kickoff action items",
    ]
    query = "when is alice's kickoff"

    breakdowns: list[dict[str, float]] = []
    async with MnemossBackend(embedding_model=FakeEmbedder(dim=32)) as be:
        for c in corpus:
            await be.observe(c, ts=0.0)
        hits = await be.recall(query, k=num_hits)
        for h in hits:
            b = await be.explain(query, h.memory_id)
            if b is not None:
                breakdowns.append(b)
    return breakdowns


def fig_c_breakdown_bar(
    out_path: Path,
    *,
    breakdowns: list[dict[str, float]] | None = None,
) -> None:
    """Stacked horizontal bar of ``ActivationBreakdown`` components for
    the top-k hits of one recall.

    ``breakdowns`` is optional — pass in hand-picked data to pin the
    figure, or omit to run a tiny Mnemoss scenario and capture live
    values. The live capture uses ``FakeEmbedder`` so no network calls
    and no API keys.
    """

    if breakdowns is None:
        breakdowns = asyncio.run(_capture_breakdowns(num_hits=5))
    if not breakdowns:
        raise ValueError("fig_c_breakdown_bar: no breakdowns to plot")

    # Components we visualize (matches ActivationBreakdown.to_dict keys).
    components = [
        ("base_level", COLOR_BASE, "B_i"),
        ("spreading", COLOR_SPREAD, "Σ W_j·S_ji"),
        ("matching", COLOR_MATCH, "MP·sim"),
        ("noise", COLOR_NOISE, "ε noise"),
    ]

    # Sort by total descending so the strongest hit is at the top
    # (matplotlib draws bar 0 at the bottom).
    breakdowns = sorted(
        breakdowns,
        key=lambda b: b.get("total", sum(b.get(c, 0.0) for c, _, _ in components)),
        reverse=False,
    )
    labels = [f"rank #{len(breakdowns) - i}" for i in range(len(breakdowns))]

    fig, ax = plt.subplots(figsize=(6.4, 0.7 * len(breakdowns) + 1.4), dpi=100)

    # Positive contributions start at 0 and stack right; noise can be
    # negative, so we render it as its own segment with sign preserved.
    for row_idx, b in enumerate(breakdowns):
        running = 0.0
        for key, color, legend_label in components:
            val = float(b.get(key, 0.0))
            width = abs(val)
            if width == 0:
                continue
            ax.barh(
                row_idx,
                width,
                left=running,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                label=legend_label if row_idx == 0 else None,
            )
            running += width

    ax.set_yticks(list(range(len(breakdowns))))
    ax.set_yticklabels(labels)
    ax.set_xlabel("|component magnitude| (sum)")
    ax.xaxis.grid(True, color="#e5e7eb", linewidth=0.5)
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=9,
        ncols=4,
    )
    _apply_minimal_style(ax)

    fig.suptitle(
        "Per-hit activation breakdown",
        fontsize=12,
        fontweight="bold",
        x=0.08,
        ha="left",
        y=0.98,
    )
    ax.set_title(
        "A_i = B_i + Σ W_j·S_ji + MP·sim + ε",
        fontsize=9,
        color="#6b7280",
        loc="left",
        pad=8,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── CLI ───────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the three static explainer figures used in the launch blog post."
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures_out"),
        help="Directory where figure SVGs land (default: docs/figures_out/).",
    )
    p.add_argument(
        "--figures",
        nargs="+",
        choices=["a", "b", "c"],
        default=["a", "b", "c"],
        help="Which figures to render (default: all three).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if "a" in args.figures:
        fig_a_decay_curve(args.out_dir / "fig-a-decay-curve.svg")
        print(f"wrote {args.out_dir / 'fig-a-decay-curve.svg'}")
    if "b" in args.figures:
        fig_b_spreading_graph(args.out_dir / "fig-b-spreading-graph.svg")
        print(f"wrote {args.out_dir / 'fig-b-spreading-graph.svg'}")
    if "c" in args.figures:
        fig_c_breakdown_bar(args.out_dir / "fig-c-breakdown-bar.svg")
        print(f"wrote {args.out_dir / 'fig-c-breakdown-bar.svg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
