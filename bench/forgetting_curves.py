"""Forgetting-curves panel for the dreaming-validation study.

Snapshots base-level activation ``B_i`` for every memory at a fixed
end-of-simulation time, then plots B_i vs memory age bucketed by
ground-truth utility (high / medium / low from the pressure corpus's
``utility`` field).

Two visual questions this answers:

  1. **Does the formula's B_i actually decay as a power law?**
     Plotting ``B_i(t) vs log(age)`` should be roughly linear with
     slope -d, where d is the decay parameter from FormulaParams
     (default 0.5). Curvature off that line means the formula is
     producing a different shape than ACT-R's published power law.

  2. **Do high-utility memories sit higher on the curve than low-
     utility ones at the same age?** With this corpus (single-access
     memories, no rehearsal), B_i depends ONLY on age — utility-by-
     B_i differences would have to come from Consolidate's refinement
     touching ``access_history`` (it doesn't, per source) or from
     Dispose tombstoning low-utility memories preferentially.

If the panel shows utility buckets cleanly separated, ACT-R's
intended "rehearsal lifts useful memories" behavior is visible. If
they're stacked, the corpus / pipeline isn't generating the
differential rehearsal the formula is designed to amplify — which
is itself a useful finding.

Network: requires ``OPENAI_API_KEY`` (embedder).

Usage::

    python -m bench.forgetting_curves
    python -m bench.forgetting_curves --ablation dreaming_off
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
import tomllib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from freezegun import freeze_time

from mnemoss import (
    DreamerParams,
    FormulaParams,
    Mnemoss,
    OpenAIEmbedder,
    StorageParams,
)
from mnemoss.formula.base_level import compute_base_level


def _load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


# ─── data collection ──────────────────────────────────────────────


async def _collect_snapshots(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    ablation: str,
) -> list[dict[str, Any]]:
    """Seed corpus, optionally run dream, then snapshot B_i per memory.

    Returns one dict per memory: {corpus_id, utility, age_seconds, b_i}.
    """

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    formula = FormulaParams(**cfg["formula"])
    dreamer = DreamerParams(**cfg["dreamer"])

    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )
    last_offset = max(int(m["ts_offset_seconds"]) for m in corpus["memories"])
    end_time = anchor + timedelta(seconds=last_offset + 60)

    with tempfile.TemporaryDirectory(prefix="forgetting-") as td:
        # No LLM client — even with --ablation full we skip Consolidate
        # so this run is fast and free. B_i depends only on observe
        # times + access_history, not on Consolidate output. Dispose's
        # tombstoning is the only ablation effect we'd see here, and
        # earlier matrix runs showed Dispose is a no-op on this corpus.
        mem = Mnemoss(
            workspace="forgetting",
            embedding_model=embedder,
            formula=formula,
            dreamer=dreamer,
            storage=StorageParams(root=Path(td)),
        )
        try:
            id_map: dict[str, str] = {}
            print(f"  seeding {len(corpus['memories'])} memories...", flush=True)
            for m in corpus["memories"]:
                target = anchor + timedelta(seconds=int(m["ts_offset_seconds"]))
                with freeze_time(target, tick=True):
                    mid = await mem.observe(role="user", content=m["content"])
                    if mid is None:
                        continue
                id_map[m["id"]] = mid

            # If ablation requests it, run Dispose-only at end_time so
            # we can see whether tombstoning preferentially removes
            # low-utility memories. With no LLM, Consolidate skips
            # naturally; Cluster/Relations need replay candidates;
            # Rebalance + Dispose run independently.
            if ablation == "dispose_rebalance":
                with freeze_time(end_time, tick=True):
                    await mem.dream(trigger="nightly", phases={"replay", "rebalance", "dispose"})

            # Snapshot.
            await mem._ensure_open()
            store = mem._store
            assert store is not None
            mnemoss_ids = list(id_map.values())
            materialized = await store.materialize_memories(mnemoss_ids)
            mem_by_mid = {m.id: m for m in materialized}
            utility_by_corpus = {m["id"]: m["utility"] for m in corpus["memories"]}

            snapshots: list[dict[str, Any]] = []
            for corpus_id, mnemoss_id in id_map.items():
                row = mem_by_mid.get(mnemoss_id)
                if row is None:
                    # Tombstoned by Dispose.
                    snapshots.append(
                        {
                            "corpus_id": corpus_id,
                            "utility": utility_by_corpus[corpus_id],
                            "age_seconds": None,
                            "b_i": None,
                            "tombstoned": True,
                        }
                    )
                    continue
                age = (end_time - row.created_at).total_seconds()
                # Use end_time as the snapshot "now" so the formula's
                # eta(t) (encoding-grace bonus) decays consistently.
                b_i = compute_base_level(row.access_history, end_time, row.created_at, formula)
                snapshots.append(
                    {
                        "corpus_id": corpus_id,
                        "utility": utility_by_corpus[corpus_id],
                        "age_seconds": age,
                        "b_i": b_i,
                        "tombstoned": False,
                    }
                )
            return snapshots
        finally:
            await mem.close()


# ─── plotting ──────────────────────────────────────────────────────


_UTILITY_COLOR = {"high": "#2ca02c", "medium": "#1f77b4", "low": "#d62728"}
_UTILITY_LABEL = {
    "high": "high (10% — Project Phoenix)",
    "medium": "medium (20% — backbone)",
    "low": "low (70% — junk)",
}


def _plot(snapshots: list[dict[str, Any]], ablation: str, out: Path) -> None:
    live = [s for s in snapshots if not s["tombstoned"]]
    tombs = [s for s in snapshots if s["tombstoned"]]

    by_utility: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in live:
        # Convert age to days for plotting.
        age_days = s["age_seconds"] / 86400.0
        by_utility[s["utility"]].append((age_days, s["b_i"]))

    fig, ax = plt.subplots(figsize=(10, 7))

    for utility in ("high", "medium", "low"):
        pts = by_utility.get(utility, [])
        if not pts:
            continue
        ages = np.array([p[0] for p in pts])
        b_is = np.array([p[1] for p in pts])
        # Scatter all live memories.
        ax.scatter(
            ages,
            b_is,
            s=24,
            alpha=0.55,
            color=_UTILITY_COLOR[utility],
            label=f"{_UTILITY_LABEL[utility]} (n={len(pts)})",
            edgecolor="black",
            linewidth=0.3,
        )
        # Overlay a binned mean line so the trend is visible through
        # the noise.
        if len(ages) >= 6:
            order = np.argsort(ages)
            sorted_ages = ages[order]
            sorted_bis = b_is[order]
            n_bins = 6
            bin_edges = np.linspace(sorted_ages.min(), sorted_ages.max(), n_bins + 1)
            bin_centers = []
            bin_means = []
            for i in range(n_bins):
                in_bin = (sorted_ages >= bin_edges[i]) & (sorted_ages < bin_edges[i + 1])
                if i == n_bins - 1:
                    in_bin |= sorted_ages == bin_edges[i + 1]
                if in_bin.sum() > 0:
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                    bin_means.append(sorted_bis[in_bin].mean())
            ax.plot(
                bin_centers,
                bin_means,
                color=_UTILITY_COLOR[utility],
                linewidth=2.0,
                alpha=0.9,
            )

    ax.set_xlabel("Memory age at end_time (days)", fontsize=11)
    ax.set_ylabel("Base-level activation $B_i$", fontsize=11)
    ax.set_title(
        f"Forgetting curves — pressure corpus, ablation = {ablation}\n"
        f"{len(live)} live memories, {len(tombs)} tombstoned",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# ─── orchestrator ─────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render forgetting-curves panel for the pressure corpus."
    )
    parser.add_argument(
        "--corpus",
        default="bench/fixtures/pressure_corpus_seed42.jsonl",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
    )
    parser.add_argument(
        "--ablation",
        choices=["dreaming_off", "dispose_rebalance"],
        default="dreaming_off",
        help=(
            "Which condition to snapshot. dreaming_off = no dream phases. "
            "dispose_rebalance = run Dispose + Rebalance only (no LLM). "
            "Tombstoned memories are excluded from the live scatter."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=("Where to write the PNG. Default: bench/results/forgetting_curves_{ablation}.png"),
    )
    parser.add_argument(
        "--save-snapshots",
        default=None,
        help="Optional path to write the raw per-memory snapshots as JSON.",
    )
    args = parser.parse_args()

    corpus = _load_corpus(Path(args.corpus))
    cfg = _load_config(Path(args.config))

    if args.out is None:
        args.out = f"bench/results/forgetting_curves_{args.ablation}.png"

    print(f"collecting snapshots for ablation={args.ablation!r}...", flush=True)
    snapshots = asyncio.run(_collect_snapshots(corpus, cfg, ablation=args.ablation))
    if args.save_snapshots:
        Path(args.save_snapshots).write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
        print(f"wrote {args.save_snapshots}", flush=True)

    _plot(snapshots, args.ablation, Path(args.out))

    # Print a small summary for quick read.
    by_util: dict[str, list[float]] = defaultdict(list)
    for s in snapshots:
        if s["b_i"] is not None:
            by_util[s["utility"]].append(s["b_i"])
    print()
    print("summary (mean B_i per utility bucket):", flush=True)
    for u in ("high", "medium", "low"):
        if u in by_util:
            print(
                f"  {u:<6}  n={len(by_util[u]):>3}  "
                f"mean B_i = {np.mean(by_util[u]):>7.3f}  "
                f"(min={min(by_util[u]):.3f}  max={max(by_util[u]):.3f})",
                flush=True,
            )
    n_tomb = sum(1 for s in snapshots if s["tombstoned"])
    print(f"  tombstoned: {n_tomb}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
