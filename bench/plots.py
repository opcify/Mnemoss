"""Chart renderers for the launch post.

Each ``plot_chart_*`` consumes one or more JSON artifacts produced by
``bench.launch_comparison`` and writes a vector-format (SVG) figure
ready to embed in the blog post. PNG rendering is a fallback; the
blog post uses SVG for crisp zoom.

Style
-----

- White background, black text — matches a blog post's body copy.
- One accent per chart (`ACCENT`). Backend bars use distinct hues
  from a small palette (`BACKEND_COLORS`); any unknown backend falls
  back to a grey fill.
- No top / right spines. No gridlines by default — only a soft
  horizontal grid where it aids comparison.
- Titles short, subtitle carries the context (corpus, k, date).
- Font falls back to system defaults so the SVG renders consistently
  on any machine without a Mnemoss install.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# Force a non-interactive backend so the CLI works in CI / containers
# without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# ─── style constants ───────────────────────────────────────────────

ACCENT = "#f59e0b"  # matches the player's accent (amber)

BACKEND_COLORS = {
    "mnemoss": "#4f46e5",  # indigo — the hero (default params)
    "mnemoss_semantic": "#8b5cf6",  # violet — Mnemoss with recency bias off
    "mnemoss_fast": "#06b6d4",  # cyan — ANN + skip knobs, default formula
    "mnemoss_prod": "#4f46e5",  # indigo — the launch config (treat as hero)
    "mnemoss_rocket": "#4f46e5",  # indigo — Phase 2 fast-index launch hero
    "raw_stack": "#10b981",  # emerald — the "built it myself" baseline
    "static_file": "#6b7280",  # cool grey — the grep-only floor
}

_DEFAULT_COLOR = "#9ca3af"


def _apply_minimal_style(ax: plt.Axes) -> None:
    """Mnemoss's chart conventions: cut chrome, keep signal."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.set_axisbelow(True)


# ─── Chart 1: recall@k per conversation, grouped by backend ────────


def plot_chart_1(
    result_paths: list[Path],
    out_path: Path,
    *,
    title: str = "Recall@k on LoCoMo 2024",
    subtitle: str | None = None,
) -> None:
    """Render Chart 1: per-conversation + aggregate recall@k, grouped by backend.

    Parameters
    ----------
    result_paths:
        One path per backend-run. Each file is the JSON artifact
        produced by ``bench.launch_comparison``. All runs should share
        the same ``corpus`` and ``k`` for the comparison to be
        meaningful; ``plot_chart_1`` raises if they diverge.
    out_path:
        Where the SVG (or PNG, inferred from extension) lands. Parent
        directory is created if missing.
    title:
        Chart title. Default cites LoCoMo 2024 as the corpus.
    subtitle:
        If ``None``, one is derived from the common ``corpus`` / ``k`` /
        number of conversations.
    """

    if not result_paths:
        raise ValueError("plot_chart_1 needs at least one result path")

    runs = [json.loads(p.read_text()) for p in result_paths]

    # Sanity: same chart number, corpus, k across all runs.
    charts = {r["chart"] for r in runs}
    corpora = {r["corpus"] for r in runs}
    ks = {r["k"] for r in runs}
    if charts != {1}:
        raise ValueError(f"Chart 1 renderer got non-Chart-1 runs: {charts}")
    if len(corpora) != 1 or len(ks) != 1:
        raise ValueError(f"Chart 1 runs must share corpus and k (got corpora={corpora}, ks={ks})")
    corpus = corpora.pop()
    k = ks.pop()

    # Collect union of conversation_ids, preserving the order from
    # the first run so the x-axis is stable.
    seen: list[str] = []
    for run in runs:
        for row in run["per_conversation"]:
            cid = row["conversation_id"]
            if cid not in seen:
                seen.append(cid)
    conv_ids = seen
    # Add a trailing "ALL" slot for the aggregate.
    categories = conv_ids + ["ALL"]

    # Organize data: {backend_name: [recall per category]}
    by_backend: dict[str, list[float]] = {}
    for run in runs:
        backend = run["backend"]
        per_conv = {
            row["conversation_id"]: row["mean_recall_at_k"] for row in run["per_conversation"]
        }
        values = [per_conv.get(cid, 0.0) for cid in conv_ids]
        values.append(run["aggregate"]["mean_recall_at_k"])
        by_backend[backend] = values

    # ── plot ──────────────────────────────────────────────────────
    n_groups = len(categories)
    n_bars = len(by_backend)
    # Total width per group = 0.8; split into n bars with a small gap.
    group_width = 0.8
    bar_width = group_width / max(n_bars, 1)
    x_base = range(n_groups)

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 0.9), 4.5), dpi=100)

    for i, (backend, values) in enumerate(by_backend.items()):
        color = BACKEND_COLORS.get(backend, _DEFAULT_COLOR)
        offsets = [x + (i - (n_bars - 1) / 2) * bar_width for x in x_base]
        ax.bar(
            offsets,
            values,
            bar_width * 0.9,
            color=color,
            label=backend,
            edgecolor="none",
        )

    # Emphasize the aggregate column with a subtle background band.
    # Slot index of "ALL" is n_groups - 1.
    agg_x = n_groups - 1
    ax.axvspan(
        agg_x - 0.5,
        agg_x + 0.5,
        color="#f3f4f6",
        zorder=-1,
    )

    ax.set_xticks(list(x_base))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(f"mean recall@{k}")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.5)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        ncols=min(n_bars, 4),
    )

    _apply_minimal_style(ax)

    if subtitle is None:
        n_conv = len(conv_ids)
        subtitle = f"corpus: {corpus} · k={k} · {n_conv} conversations"
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.08, ha="left", y=0.99)
    ax.set_title(subtitle, fontsize=9, color="#6b7280", loc="left", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── Chart 2: recall@k vs corpus size, per backend ────────────────


def plot_chart_2(
    sweep_path: Path,
    out_path: Path,
    *,
    title: str = "Recall@k as memory grows",
    subtitle: str | None = None,
) -> None:
    """Render Chart 2: recall@k vs N (corpus size), one line per backend.

    Consumes a ``scale_sweep`` collated JSON (produced by
    ``bench.scale_sweep``). Expected shape:

        {"chart": "scale", "embedder": "...", "sizes": [...],
         "backends": [...], "results": [{"backend":..., "scale_n":...,
         "recall_at_k":...}, ...]}

    Draws one line per backend. The "parity at small N, divergence at
    large N" story is carried by line slope: a horizontal line means
    "robust to corpus growth"; a declining line means "losing signal
    as distractors accumulate."
    """

    payload = json.loads(sweep_path.read_text())
    if payload.get("chart") != "scale":
        raise ValueError(f"plot_chart_2 expects a 'scale' sweep JSON, got {payload.get('chart')!r}")

    results = payload["results"]
    by_backend: dict[str, list[tuple[int, float]]] = {}
    for row in results:
        by_backend.setdefault(row["backend"], []).append((row["scale_n"], row["recall_at_k"]))
    for backend in by_backend:
        by_backend[backend].sort()

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=100)

    for backend, xy in by_backend.items():
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        color = BACKEND_COLORS.get(backend, _DEFAULT_COLOR)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=backend,
        )

    ax.set_xlabel("memories in workspace (N)")
    ax.set_ylabel("mean recall@10")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    _apply_minimal_style(ax)

    if subtitle is None:
        embedder = payload.get("embedder", "openai")
        gold = payload.get("gold_conversation", "?")
        subtitle = (
            f"embedder: {embedder} · gold conversation: {gold} · "
            f"padding: distractors from other LoCoMo conversations"
        )
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.08, ha="left", y=0.99)
    ax.set_title(subtitle, fontsize=9, color="#6b7280", loc="left", pad=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── Chart 3: the launch chart (recall + latency side-by-side) ────


def plot_chart_launch(
    sweep_paths: list[Path],
    latency_paths: list[Path],
    out_path: Path,
    *,
    title: str = "Mnemoss vs cosine-only: scaling on LoCoMo 2024",
    subtitle: str | None = None,
    backends: list[str] | None = None,
) -> None:
    """Render the two-panel launch figure: recall + latency vs N.

    Parameters
    ----------
    sweep_paths:
        One or more ``bench.scale_sweep`` collated JSONs. Their
        ``results`` rows are merged into one ``{backend: [(N, recall)]}``
        map for the left panel.
    latency_paths:
        One or more ``bench.scale_latency`` JSONs. Merged the same way
        for the right panel.
    backends:
        Optional whitelist — only these backend names are plotted. Order
        controls legend order. When ``None`` every backend found in the
        artifacts is drawn.
    """

    def _merge(paths: list[Path], value_key: str) -> dict[str, list[tuple[int, float]]]:
        by_backend: dict[str, list[tuple[int, float]]] = {}
        for p in paths:
            data = json.loads(p.read_text())
            for row in data["results"]:
                b = row["backend"]
                n = int(row["scale_n"])
                v = float(row[value_key])
                by_backend.setdefault(b, []).append((n, v))
        for b in by_backend:
            by_backend[b].sort()
        return by_backend

    recall_by_backend = _merge(sweep_paths, "recall_at_k")
    latency_by_backend = _merge(latency_paths, "p50_ms")

    all_backends = list({*recall_by_backend.keys(), *latency_by_backend.keys()})
    if backends is None:
        # Deterministic ordering — mnemoss_prod first if present, then the rest.
        backends = sorted(all_backends, key=lambda b: (0 if b.startswith("mnemoss") else 1, b))
    else:
        backends = [b for b in backends if b in all_backends]

    fig, (ax_recall, ax_latency) = plt.subplots(
        1, 2, figsize=(11.5, 4.5), dpi=100, gridspec_kw={"wspace": 0.25}
    )

    for backend in backends:
        color = BACKEND_COLORS.get(backend, _DEFAULT_COLOR)

        if backend in recall_by_backend:
            xs = [p[0] for p in recall_by_backend[backend]]
            ys = [p[1] for p in recall_by_backend[backend]]
            ax_recall.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=backend,
            )

        if backend in latency_by_backend:
            xs = [p[0] for p in latency_by_backend[backend]]
            ys = [p[1] for p in latency_by_backend[backend]]
            ax_latency.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=backend,
            )

    ax_recall.set_xscale("log")
    ax_recall.set_xlabel("memories in workspace (N)")
    ax_recall.set_ylabel("mean recall@10")
    ax_recall.set_ylim(0, 1.0)
    ax_recall.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_recall.yaxis.grid(True, color="#e5e7eb", linewidth=0.5)
    ax_recall.legend(loc="upper right", frameon=False, fontsize=9)
    ax_recall.set_title("Recall@10 — parity", fontsize=10, loc="left", pad=8)
    _apply_minimal_style(ax_recall)

    ax_latency.set_xscale("log")
    ax_latency.set_xlabel("memories in workspace (N)")
    ax_latency.set_ylabel("p50 recall latency (ms)")
    ax_latency.yaxis.grid(True, color="#e5e7eb", linewidth=0.5)
    ax_latency.legend(loc="upper left", frameon=False, fontsize=9)
    ax_latency.set_title("p50 latency — Mnemoss pulls away", fontsize=10, loc="left", pad=8)
    _apply_minimal_style(ax_latency)

    if subtitle is None:
        # Pull embedder / gold conversation from the first sweep artifact.
        first = json.loads(sweep_paths[0].read_text())
        embedder = first.get("embedder", "local")
        gold = first.get("gold_conversation", "?")
        subtitle = (
            f"embedder: {embedder} · gold conversation: {gold} · "
            f"padding: distractors from other LoCoMo conversations"
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.06, ha="left", y=1.00)
    fig.text(0.06, 0.93, subtitle, fontsize=9, color="#6b7280", ha="left")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─── CLI ───────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render launch-comparison JSON as SVG charts.")
    p.add_argument(
        "--chart",
        choices=["1", "2", "launch"],
        required=True,
        help=(
            "Which chart to render: "
            "1 = Chart 1 per-conversation recall grouped by backend; "
            "2 = scale-sweep recall curves; "
            "launch = two-panel launch chart (recall + latency vs N)."
        ),
    )
    p.add_argument(
        "--results",
        type=Path,
        nargs="+",
        help=(
            "One or more JSON result files (for chart 1 / 2). Chart "
            "``launch`` uses --sweep and --latency instead."
        ),
    )
    p.add_argument(
        "--sweep",
        type=Path,
        nargs="+",
        help="One or more scale-sweep JSONs (chart ``launch``).",
    )
    p.add_argument(
        "--latency",
        type=Path,
        nargs="+",
        help="One or more scale-latency JSONs (chart ``launch``).",
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Chart ``launch`` only: whitelist + ordering of backends to plot.",
    )
    p.add_argument("--out", type=Path, required=True, help="Output SVG/PNG path.")
    p.add_argument("--title", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.chart == "1":
        if not args.results:
            raise SystemExit("Chart 1 needs --results")
        title = args.title or "Recall@k on LoCoMo 2024"
        plot_chart_1(args.results, args.out, title=title)
    elif args.chart == "2":
        if not args.results or len(args.results) != 1:
            raise SystemExit("Chart 2 takes exactly one --results file (the scale-sweep JSON).")
        title = args.title or "Recall@k as memory grows"
        plot_chart_2(args.results[0], args.out, title=title)
    elif args.chart == "launch":
        if not args.sweep or not args.latency:
            raise SystemExit("Chart `launch` needs both --sweep and --latency paths.")
        title = args.title or "Mnemoss vs cosine-only: scaling on LoCoMo 2024"
        plot_chart_launch(
            args.sweep,
            args.latency,
            args.out,
            title=title,
            backends=args.backends,
        )
    else:  # pragma: no cover — choices above constrain this
        raise SystemExit(f"chart {args.chart} not yet implemented")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
