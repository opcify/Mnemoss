"""Pareto chart for the dreaming-validation harness.

Reads ``bench/results/topology_results.jsonl`` (or the path passed
on the CLI) and emits a scatter:

  x = LLM calls spent during dream (from CostLedger)
  y = mean recall@10 across labeled queries
  one point per ablation, labeled by the ablation's name

Pareto frontier emerges visually: dreaming_off anchors bottom-left
(0 calls, baseline recall), full anchors top-right (max calls, max
recall). Per-phase masks land in between; phases that earn their
LLM cost lie above the diagonal.

No interactive components, no external state — this is a plot
generator the harness invokes once at the end of the run.

Usage::

    python -m bench.plot_pareto
    python -m bench.plot_pareto --results PATH --out PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display; we write PNG.
import matplotlib.pyplot as plt


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"results file is empty: {path}")
    return rows


def _plot(rows: list[dict[str, Any]], out: Path) -> None:
    xs = [r["llm_calls_during_dream"] for r in rows]
    ys = [r["mean_recall_at_k"] for r in rows]
    labels = [r["label"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color: full = green, dreaming_off = red, others = blue.
    colors = []
    for label in labels:
        if label == "full":
            colors.append("#2ca02c")
        elif label == "dreaming_off":
            colors.append("#d62728")
        else:
            colors.append("#1f77b4")

    ax.scatter(xs, ys, c=colors, s=120, alpha=0.85, edgecolor="black", linewidth=0.6)

    # Annotate each point.
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.annotate(
            label,
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("LLM calls spent during dream", fontsize=11)
    ax.set_ylabel("Mean recall@10 (labeled queries only)", fontsize=11)
    ax.set_title(
        "Mnemoss dreaming-validation — per-phase Pareto",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=-0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the per-phase Pareto chart from ablate_dreaming results."
    )
    parser.add_argument(
        "--results",
        default="bench/results/topology_results.jsonl",
        help="Path to the harness's results.jsonl.",
    )
    parser.add_argument(
        "--out",
        default="bench/results/pareto.png",
        help="Where to write the PNG.",
    )
    args = parser.parse_args()

    try:
        rows = _load_rows(Path(args.results))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    _plot(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
