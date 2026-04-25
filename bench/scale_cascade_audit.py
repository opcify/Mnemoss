"""Cascade-audit benchmark — Phase 1.1.

Runs Mnemoss's gold queries against a scaled LoCoMo corpus using the
engine's ``recall_with_stats`` hook, and aggregates:

- **Stopped-at distribution** — how often the cascade actually short-
  circuits at HOT, WARM, or COLD versus scanning all tiers.
- **Candidates scored per query** — the pool size the ACT-R formula
  evaluates on (one activation compute per unique id across tiers).
- **Gold-memory tier placement** — where the known-relevant memories
  for each query actually live at recall time.
- **Workspace tier counts** — the tier histogram of the full padded
  corpus at benchmark time.

The whole point: find out why the tier cascade doesn't save us any
latency at scale. Hypothesis going in (from reading the code):

- Fresh ``observe()`` sets ``idx_priority = σ(η_0) ≈ 0.731`` which is
  just above the 0.7 HOT threshold → every freshly-ingested memory
  lands in HOT. With no rebalance between ingest and recall, the
  cascade scans N candidates in HOT regardless of how many tiers we
  define. That would make the cascade dead code for bulk-ingest
  workloads like LoCoMo.

Run it and see.

Usage::

    python -m bench.scale_cascade_audit \
        --embedder local \
        --sizes 500 1500 3000 5000 \
        --gold-conversation conv-26 \
        --out bench/results/cascade_audit_local.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from bench.backends.mnemoss_backend import MnemossBackend
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)


async def _audit_one(
    *,
    scale_n: int,
    embedder: str,
    gold_conversation: str,
    k: int,
) -> dict:
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories,
        queries,
        gold_conversation_id=gold_conversation,
        scale_n=scale_n,
    )

    backend = MnemossBackend(embedding_model=_resolve_embedder(embedder))
    try:
        # Ingest: dia_id → mnemoss memory id mapping so we can look up
        # where the gold memories ended up tier-wise.
        dia_to_mem_id: dict[str, str] = {}
        for m in padded_mems:
            mid = await backend.observe(m["text"], ts=m["ts"])
            dia_to_mem_id[m["dia_id"]] = mid

        store = backend._mem._store  # internal handle — benchmark-only
        engine = backend._mem._engine
        assert engine is not None

        # Workspace-wide tier histogram.
        workspace_tier_counts: dict[str, int] = {
            tier.value: count for tier, count in (await store.tier_counts()).items()
        }

        stopped_at_counter: Counter[str] = Counter()
        tiers_scanned_counter: Counter[int] = Counter()
        candidates_scored: list[int] = []
        gold_tier_counter: Counter[str] = Counter()
        gold_in_returned: list[int] = []  # 0 or 1 per gold memory
        n_gold_total = 0

        for q in gold_queries:
            results, stats = await engine.recall_with_stats(
                q["question"],
                agent_id=None,
                k=k,
                auto_expand=False,
                include_deep=True,
                reconsolidate=False,
            )
            stopped_key = stats.stopped_at.value if stats.stopped_at else "exhausted"
            stopped_at_counter[stopped_key] += 1
            tiers_scanned_counter[len(stats.tiers_scanned)] += 1
            candidates_scored.append(stats.candidates_scored)

            returned_ids = {r.memory.id for r in results}

            # Tier placement of gold memories for this query.
            for dia in q["relevant_dia_ids"]:
                mid = dia_to_mem_id.get(dia)
                if mid is None:
                    continue
                mem = await store.get_memory(mid)
                if mem is None:
                    continue
                n_gold_total += 1
                gold_tier_counter[mem.index_tier.value] += 1
                gold_in_returned.append(1 if mid in returned_ids else 0)

    finally:
        await backend.close()

    return {
        "scale_n": scale_n,
        "embedder": embedder,
        "k": k,
        "n_queries": len(gold_queries),
        "n_gold_total": n_gold_total,
        "workspace_tier_counts": workspace_tier_counts,
        "stopped_at_distribution": dict(stopped_at_counter),
        "tiers_scanned_distribution": {str(k): v for k, v in tiers_scanned_counter.items()},
        "candidates_scored": {
            "mean": round(statistics.mean(candidates_scored), 1) if candidates_scored else 0.0,
            "median": statistics.median(candidates_scored) if candidates_scored else 0,
            "max": max(candidates_scored) if candidates_scored else 0,
            "min": min(candidates_scored) if candidates_scored else 0,
        },
        "gold_tier_distribution": dict(gold_tier_counter),
        "gold_recall_rate": (
            round(sum(gold_in_returned) / n_gold_total, 4) if n_gold_total else 0.0
        ),
    }


async def _audit_all(
    *,
    sizes: list[int],
    embedder: str,
    gold_conversation: str,
    k: int,
) -> list[dict]:
    out: list[dict] = []
    for n in sizes:
        print(f"[audit] N={n}  embedder={embedder}  gold={gold_conversation}", flush=True)
        row = await _audit_one(
            scale_n=n,
            embedder=embedder,
            gold_conversation=gold_conversation,
            k=k,
        )
        out.append(row)
        tier_counts = row["workspace_tier_counts"]
        stopped = row["stopped_at_distribution"]
        gold_tiers = row["gold_tier_distribution"]
        print(
            f"  workspace: HOT={tier_counts.get('hot', 0)}  "
            f"WARM={tier_counts.get('warm', 0)}  "
            f"COLD={tier_counts.get('cold', 0)}  "
            f"DEEP={tier_counts.get('deep', 0)}",
            flush=True,
        )
        print(
            f"  stopped_at: {dict(stopped)}  "
            f"cands_scored mean={row['candidates_scored']['mean']}  "
            f"max={row['candidates_scored']['max']}",
            flush=True,
        )
        print(
            f"  gold_in_tier: {dict(gold_tiers)}  "
            f"gold_recall_rate={row['gold_recall_rate']}",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", choices=["openai", "local", "fake"], default="local")
    p.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 3000, 5000])
    p.add_argument("--gold-conversation", default="conv-26")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _audit_all(
            sizes=args.sizes,
            embedder=args.embedder,
            gold_conversation=args.gold_conversation,
            k=args.k,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "cascade_audit",
                "embedder": args.embedder,
                "sizes": args.sizes,
                "gold_conversation": args.gold_conversation,
                "k": args.k,
                "timestamp": started.isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")

    # Terminal summary.
    print()
    print(f"{'N':>6}  {'HOT':>6} {'WARM':>6} {'COLD':>6} {'DEEP':>6}  "
          f"{'stop=HOT':>10} {'stop=WARM':>10} {'stop=COLD':>10} {'exhaust':>9}  {'cands':>6}")
    print("─" * 110)
    def _pct(st_dist: dict, key: str, n_q: int) -> str:
        return f"{st_dist.get(key, 0) / max(n_q, 1):.0%}"

    for r in results:
        wc = r["workspace_tier_counts"]
        st = r["stopped_at_distribution"]
        n_q = r["n_queries"]
        print(
            f"{r['scale_n']:>6}  "
            f"{wc.get('hot', 0):>6} {wc.get('warm', 0):>6} "
            f"{wc.get('cold', 0):>6} {wc.get('deep', 0):>6}  "
            f"{_pct(st, 'hot', n_q):>10} {_pct(st, 'warm', n_q):>10} "
            f"{_pct(st, 'cold', n_q):>10} {_pct(st, 'exhausted', n_q):>9}  "
            f"{r['candidates_scored']['mean']:>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
