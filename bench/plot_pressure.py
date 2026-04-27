"""Pressure-effect chart: recall@10 + top-K cleanliness per ablation.

Reads ``bench/results/pressure_results.jsonl`` (or the path passed
on the CLI) and emits a two-panel bar chart:

  top    — mean recall@10 across labeled queries, per ablation
  bottom — mean top-K cleanliness (fraction of queries with zero
           junk-utility memories in top-10), per ablation

Used together with ``bench/plot_pareto.py`` (which is corpus-agnostic
and shows LLM-cost vs recall@10) for the weekend-2 verdict on Dispose
and Rebalance.

Usage::

    python -m bench.plot_pressure
    python -m bench.plot_pressure --results PATH --out PATH
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
    if any(r.get("corpus_kind") != "pressure" for r in rows):
        raise ValueError(
            f"results at {path} are not from a pressure-corpus run "
            "(use bench/plot_pareto.py for topology results)"
        )
    return rows


def _color_for(label: str) -> str:
    if label == "full":
        return "#2ca02c"
    if label == "dreaming_off":
        return "#d62728"
    if label.startswith("no_") or label.endswith("_only"):
        return "#1f77b4"
    return "#7f7f7f"


def _plot(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [r["label"] for r in rows]
    recalls = [r["mean_recall_at_k"] for r in rows]
    cleans = [r["mean_cleanliness"] if r["mean_cleanliness"] is not None else 0.0 for r in rows]
    colors = [_color_for(label) for label in labels]

    fig, (ax_r, ax_c) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    bar_kwargs = dict(color=colors, edgecolor="black", linewidth=0.5)

    ax_r.bar(labels, recalls, **bar_kwargs)
    ax_r.set_ylabel("Mean recall@10", fontsize=11)
    ax_r.set_title("Pressure corpus — recall@10 per ablation", fontsize=12)
    ax_r.set_ylim(bottom=0.0)
    ax_r.grid(True, axis="y", linestyle="--", alpha=0.35)
    for x, y in zip(labels, recalls, strict=True):
        ax_r.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=8)

    ax_c.bar(labels, cleans, **bar_kwargs)
    ax_c.set_ylabel("Mean top-K cleanliness", fontsize=11)
    ax_c.set_xlabel("Ablation condition", fontsize=11)
    ax_c.set_title(
        "Pressure corpus — top-K cleanliness per ablation "
        "(fraction of queries with zero junk in top-10)",
        fontsize=12,
    )
    ax_c.set_ylim(bottom=0.0, top=1.0)
    ax_c.grid(True, axis="y", linestyle="--", alpha=0.35)
    for x, y in zip(labels, cleans, strict=True):
        ax_c.text(x, y + 0.01, f"{y:.3f}", ha="center", fontsize=8)
    ax_c.tick_params(axis="x", rotation=30)
    plt.setp(ax_c.get_xticklabels(), ha="right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the pressure-effect bar chart from ablate_dreaming results."
    )
    parser.add_argument(
        "--results",
        default="bench/results/pressure_results.jsonl",
    )
    parser.add_argument(
        "--out",
        default="bench/results/pressure.png",
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
