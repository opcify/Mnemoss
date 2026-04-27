"""Gist-quality bar chart with bootstrap confidence intervals.

Reads ``bench/results/gist_quality.jsonl`` (one row per pairwise
comparison + a ``_summary`` row at the end) and renders a single bar
showing the post-Consolidate win rate with a 95% CI error bar.
A horizontal line at 0.5 marks the null (level-1 indistinguishable).

Usage::

    python -m bench.plot_gist
    python -m bench.plot_gist --results PATH --out PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"results file not found: {path}")
    summary: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "_summary" in row:
            summary = row["_summary"]
    if summary is None:
        raise ValueError(f"results file missing _summary row: {path}")
    return summary


def _plot(summary: dict[str, Any], out: Path) -> None:
    win = summary["win_rate"]
    lo = summary["ci_lower"]
    hi = summary["ci_upper"]
    n = summary["num_comparisons"]
    verdict = summary["verdict"]

    fig, ax = plt.subplots(figsize=(7, 6))

    color = {"KEEP": "#2ca02c", "CUT": "#d62728", "REBUILD": "#ff7f0e"}.get(verdict, "#1f77b4")
    ax.bar(
        ["post-Consolidate"],
        [win],
        yerr=[[win - lo], [hi - win]],
        color=color,
        edgecolor="black",
        linewidth=0.6,
        capsize=10,
    )
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=1, label="null (50%)")
    ax.axhline(0.55, linestyle=":", color="#999", linewidth=0.8, label="CUT threshold (≤55%)")
    ax.axhline(0.65, linestyle=":", color="#666", linewidth=0.8, label="KEEP threshold (≥65%)")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Win rate (post-Consolidate vs level-1 gist)", fontsize=11)
    ax.set_title(
        f"Gist quality — pairwise judge\n"
        f"n={n}  verdict={verdict}  "
        f"win={win:.3f}  CI95=[{lo:.3f}, {hi:.3f}]",
        fontsize=11,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the gist-quality bar chart.")
    parser.add_argument(
        "--results",
        default="bench/results/gist_quality.jsonl",
    )
    parser.add_argument(
        "--out",
        default="bench/results/gist_quality.png",
    )
    args = parser.parse_args()

    try:
        summary = _load_summary(Path(args.results))
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    _plot(summary, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
