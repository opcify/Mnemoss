"""Cluster REBUILD: sweep ``cluster_min_size`` to find a value that
pushes ARI past the pre-registered KEEP threshold (≥0.7).

Topology corpus only — ARI requires per-memory ``topic`` labels which
the pressure corpus doesn't carry. No LLM, no Consolidate (the LLM
phase is irrelevant to ARI; Cluster runs over embeddings + min_size).

Usage::

    python -m bench.sweep_cluster
    python -m bench.sweep_cluster --sizes 2 3 5 7

Output: a tabular summary and ``bench/results/cluster_sweep.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import tomllib

from bench._metrics import noise_aware_ari
from mnemoss import (
    DreamerParams,
    FormulaParams,
    Mnemoss,
    OpenAIEmbedder,
    StorageParams,
)


async def _measure_ari(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cluster_min_size: int,
) -> dict[str, Any]:
    """Seed corpus, run dream({"replay","cluster"}), return ARI."""

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    formula = FormulaParams(**cfg["formula"])
    dreamer = DreamerParams(
        cluster_min_size=cluster_min_size,
        replay_limit=int(cfg["dreamer"]["replay_limit"]),
    )

    with tempfile.TemporaryDirectory(prefix="cluster-sweep-") as td:
        mem = Mnemoss(
            workspace=f"sweep-{cluster_min_size}",
            embedding_model=embedder,
            formula=formula,
            dreamer=dreamer,
            storage=StorageParams(root=Path(td)),
        )
        try:
            id_map: dict[str, str] = {}
            for m in corpus["memories"]:
                mid = await mem.observe(role="user", content=m["content"])
                if mid is None:
                    continue
                id_map[m["id"]] = mid

            # Run only Replay + Cluster — that's all we need for ARI.
            # No Consolidate (no LLM cost), no Relations / Rebalance /
            # Dispose (irrelevant to cluster_id assignments).
            await mem.dream(trigger="nightly", phases={"replay", "cluster"})

            await mem._ensure_open()
            store = mem._store
            assert store is not None
            mnemoss_ids = list(id_map.values())
            materialized = await store.materialize_memories(mnemoss_ids)
            cluster_by_mid = {m.id: m.cluster_id for m in materialized}

            predicted: dict[str, Any] = {}
            gold: dict[str, Any] = {}
            topic_by_corpus = {m["id"]: m["topic"] for m in corpus["memories"]}
            for corpus_id, mnemoss_id in id_map.items():
                predicted[corpus_id] = cluster_by_mid.get(mnemoss_id)
                gold[corpus_id] = topic_by_corpus[corpus_id]

            ari, scored, dropped = noise_aware_ari(predicted, gold)

            distinct_clusters = len(
                {c for c in cluster_by_mid.values() if c is not None and c != -1}
            )
            return {
                "cluster_min_size": cluster_min_size,
                "ari": round(ari, 4),
                "scored": scored,
                "dropped": dropped,
                "distinct_clusters_found": distinct_clusters,
            }
        finally:
            await mem.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep cluster_min_size to find an ARI-maximizing value.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 7],
        help="cluster_min_size values to try (default: 2 3 4 5 7).",
    )
    parser.add_argument(
        "--corpus",
        default="bench/fixtures/topology_corpus.json",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
    )
    parser.add_argument(
        "--out",
        default="bench/results/cluster_sweep.json",
    )
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    with Path(args.config).open("rb") as fh:
        cfg = tomllib.load(fh)

    if "topic" not in corpus["memories"][0]:
        print(
            f"error: corpus at {args.corpus} has no 'topic' labels — ARI undefined.",
            file=sys.stderr,
        )
        return 2

    print(f"sweeping cluster_min_size across {args.sizes} on {args.corpus}", flush=True)
    rows: list[dict[str, Any]] = []
    for size in args.sizes:
        row = asyncio.run(_measure_ari(corpus, cfg, cluster_min_size=size))
        rows.append(row)
        print(
            f"  cluster_min_size={size:>2}: "
            f"ari={row['ari']:.4f}  "
            f"scored={row['scored']:>2}  "
            f"dropped={row['dropped']:>2}  "
            f"clusters_found={row['distinct_clusters_found']}",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)

    best = max(rows, key=lambda r: r["ari"])
    print(
        f"\nbest: cluster_min_size={best['cluster_min_size']} with ari={best['ari']:.4f}",
        flush=True,
    )
    if best["ari"] >= 0.7:
        print("VERDICT: KEEP threshold (≥0.7) cleared.", flush=True)
    elif best["ari"] >= 0.5:
        print("VERDICT: still REBUILD (0.5 ≤ ari < 0.7).", flush=True)
    else:
        print("VERDICT: CUT (ari < 0.5).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
