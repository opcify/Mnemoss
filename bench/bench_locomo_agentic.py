"""LoCoMo-Agentic: realistic multi-domain recall benchmark.

Standard LoCoMo recall@10 measures only one slice of what a memory
layer has to do: paraphrased historical retrieval. Real agents'
workspaces contain both:

1. **Historical observations** — things the user said / did at some
   point. Questions about them want "closest cosine match to the
   evidence memory" — LoCoMo measures this.
2. **Evolving state facts** — things that used to be true but have
   since changed. Questions about them want the *current* version,
   not the stale earlier version. LoCoMo does NOT measure this.

A memory system shipping to agents has to handle both. Pure cosine
handles (1) well and (2) badly. Time-aware systems handle both.

This bench scores the union:
- 5,000 LoCoMo conv-26 memories (padded with distractors) —
  historical observations
- 50 supersession memories (25 handcrafted old/new pairs) —
  evolving state

Ingestion protocol puts a wall-clock gap between the LoCoMo bulk
(oldest) and the supersession batches (newest), which mirrors how
evolving state actually accumulates in production: the user's job
changes AFTER their historical conversation memories have been
sitting in the store for a while.

Per-subset metrics report cleanly so the reader can see the
per-class tradeoff, not just an averaged number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.raw_stack_backend import RawStackBackend
from bench.bench_stale_fact import PAIRS_PATH as SUPERSEDE_PAIRS_PATH
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)


def _load_supersession_pairs(path: Path | None = None) -> list[dict]:
    p = path if path is not None else SUPERSEDE_PAIRS_PATH
    return [json.loads(line) for line in p.open()]


async def _run_arm(
    *,
    backend,
    backend_name: str,
    scale_n: int,
    gap_seconds: float,
    k: int,
    locomo_question_sample: int | None = None,
    sample_seed: int = 42,
    supersession_file: Path | None = None,
    old_repeats: int = 1,
) -> dict:
    # ─── Ingestion ─────────────────────────────────────────────────
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )
    supersession_pairs = _load_supersession_pairs(supersession_file)

    # Phase 1: LoCoMo historical observations (oldest).
    print(
        f"[{backend_name}] phase 1: ingesting {len(padded_mems)} LoCoMo memories...",
        flush=True,
    )
    t0 = time.perf_counter()
    locomo_dia_to_mid: dict[str, str] = {}
    for m in padded_mems:
        mid = await backend.observe(m["text"], ts=time.time())
        if mid is not None:
            locomo_dia_to_mid[m["dia_id"]] = mid
    ingest_locomo_s = time.perf_counter() - t0
    print(f"[{backend_name}]   done in {ingest_locomo_s:.1f}s", flush=True)

    # Phase 2: Ingest "old" facts ``old_repeats`` times, with a wall-clock
    # gap between each round (and before the first). Repeated ingest builds
    # up base-level activation on the stale version — a stress test for
    # whether the system still surfaces the freshly-ingested "new" fact
    # over a thrice-reinforced "old" prior.
    supersede_old_ids: dict[int, list[str]] = {i: [] for i in range(len(supersession_pairs))}
    for round_idx in range(old_repeats):
        if gap_seconds > 0:
            print(
                f"[{backend_name}] gap: sleeping {gap_seconds}s before old "
                f"round {round_idx + 1}/{old_repeats}...",
                flush=True,
            )
            await asyncio.sleep(gap_seconds)
        print(
            f"[{backend_name}] phase 2.{round_idx + 1}: ingesting "
            f"{len(supersession_pairs)} 'old' facts (round {round_idx + 1}/"
            f"{old_repeats})...",
            flush=True,
        )
        for i, p in enumerate(supersession_pairs):
            mid = await backend.observe(p["old"], ts=time.time())
            if mid is not None:
                supersede_old_ids[i].append(mid)

    # Phase 3: Final wall-clock gap, then ingest the "new" version once.
    if gap_seconds > 0:
        print(
            f"[{backend_name}] gap: sleeping {gap_seconds}s before new ingest...",
            flush=True,
        )
        await asyncio.sleep(gap_seconds)
    print(
        f"[{backend_name}] phase 3: ingesting {len(supersession_pairs)} 'new' "
        "supersession facts...",
        flush=True,
    )
    supersede_new_ids: dict[int, str] = {}
    for i, p in enumerate(supersession_pairs):
        mid = await backend.observe(p["new"], ts=time.time())
        if mid is not None:
            supersede_new_ids[i] = mid

    # Warm-up recall to prime caches before timed queries.
    for q in ["warmup one", "warmup two", "warmup three"]:
        await backend.recall(q, k=k)

    # ─── Scoring: LoCoMo questions ────────────────────────────────
    print(
        f"[{backend_name}] scoring {len(gold_queries)} LoCoMo + "
        f"{len(supersession_pairs)} supersession questions...",
        flush=True,
    )
    # Optionally sub-sample LoCoMo questions (seeded) to change the
    # blended ratio of historical:supersession questions. Full mix
    # is ~197:25 (11% supersession). 70/30 wants ~58 LoCoMo, 50/50
    # wants 25. Caller passes the target LoCoMo count.
    locomo_queries_to_score = gold_queries
    if locomo_question_sample is not None and locomo_question_sample < len(gold_queries):
        rng = random.Random(sample_seed)
        locomo_queries_to_score = rng.sample(gold_queries, locomo_question_sample)

    locomo_recall_hits: list[float] = []
    locomo_latencies: list[float] = []
    locomo_scored = 0
    for q in locomo_queries_to_score:
        gold_ids = {
            locomo_dia_to_mid[d] for d in q["relevant_dia_ids"] if d in locomo_dia_to_mid
        }
        if not gold_ids:
            continue
        t0 = time.perf_counter()
        hits = await backend.recall(q["question"], k=k)
        locomo_latencies.append((time.perf_counter() - t0) * 1000)
        returned = {h.memory_id for h in hits}
        locomo_recall_hits.append(len(returned & gold_ids) / len(gold_ids))
        locomo_scored += 1

    # ─── Scoring: supersession questions ──────────────────────────
    # A "correct" answer here is: new version in top-k. When old_repeats > 1
    # there are multiple "old" memory rows per pair (one per round); we treat
    # rank-1 of any of them as old@1.
    supersede_new_in_topk = 0
    supersede_new_at_1 = 0
    supersede_old_at_1 = 0
    supersede_latencies: list[float] = []
    for i, p in enumerate(supersession_pairs):
        new_id = supersede_new_ids.get(i)
        old_ids = set(supersede_old_ids.get(i, []))
        if new_id is None or not old_ids:
            continue
        t0 = time.perf_counter()
        hits = await backend.recall(p["question"], k=k)
        supersede_latencies.append((time.perf_counter() - t0) * 1000)
        returned = [h.memory_id for h in hits]
        if new_id in returned:
            supersede_new_in_topk += 1
        if returned and returned[0] == new_id:
            supersede_new_at_1 += 1
        elif returned and returned[0] in old_ids:
            supersede_old_at_1 += 1

    n_s = len(supersession_pairs)

    # ─── Aggregate ─────────────────────────────────────────────────
    all_latencies = locomo_latencies + supersede_latencies
    lat_sorted = sorted(all_latencies)

    def pct(p: float) -> float:
        if not lat_sorted:
            return 0.0
        idx = min(int(p * len(lat_sorted)), len(lat_sorted) - 1)
        return lat_sorted[idx]

    locomo_recall = statistics.mean(locomo_recall_hits) if locomo_recall_hits else 0.0
    supersede_recall = supersede_new_in_topk / n_s if n_s else 0.0
    # Blended: questions are weighted equally (each question = 1 vote).
    # Each LoCoMo question contributes its recall@k (0..1); each
    # supersede question contributes 1 if new in top-k else 0.
    blended_sum = sum(locomo_recall_hits) + supersede_new_in_topk
    blended_n = locomo_scored + n_s
    blended_recall = blended_sum / blended_n if blended_n else 0.0

    return {
        "backend": backend_name,
        "scale_n": scale_n,
        "gap_seconds": gap_seconds,
        # ingest
        "ingest_locomo_seconds": round(ingest_locomo_s, 2),
        # subsets
        "locomo_n_scored": locomo_scored,
        "locomo_recall_at_k": round(locomo_recall, 4),
        "supersede_n": n_s,
        "supersede_new_in_topk_rate": round(supersede_recall, 4),
        "supersede_new_at_1_rate": round(supersede_new_at_1 / n_s, 4),
        "supersede_old_at_1_rate": round(supersede_old_at_1 / n_s, 4),
        # blended
        "blended_recall_at_k": round(blended_recall, 4),
        # latency (all queries together)
        "latency_ms_mean": round(statistics.mean(all_latencies), 2),
        "latency_ms_p50": round(pct(0.50), 2),
        "latency_ms_p95": round(pct(0.95), 2),
        "latency_ms_p99": round(pct(0.99), 2),
    }


async def _main(
    *,
    embedder_choice: str,
    scale_n: int,
    gap_seconds: float,
    k: int,
    backend_choice: str,
    locomo_question_sample: int | None = None,
    supersession_file: Path | None = None,
    old_repeats: int = 1,
) -> dict:
    embedder = _resolve_embedder(embedder_choice)
    if backend_choice == "raw_stack":
        backend = RawStackBackend(embedding_model=embedder)
    elif backend_choice == "mnemoss_default":
        backend = MnemossBackend(embedding_model=embedder)
    else:
        raise ValueError(f"unknown backend {backend_choice!r}")
    try:
        return await _run_arm(
            backend=backend,
            backend_name=backend_choice,
            scale_n=scale_n,
            gap_seconds=gap_seconds,
            k=k,
            locomo_question_sample=locomo_question_sample,
            supersession_file=supersession_file,
            old_repeats=old_repeats,
        )
    finally:
        await backend.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["raw_stack", "mnemoss_default"],
        required=True,
    )
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "gemma", "nomic", "fake"],
        default="openai",
    )
    p.add_argument("--scale-n", type=int, default=5000)
    p.add_argument(
        "--gap-seconds",
        type=float,
        default=60.0,
        help="Wall-clock sleep between phases (LoCoMo → old → new).",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--locomo-question-sample",
        type=int,
        default=None,
        help="Sub-sample N LoCoMo questions instead of all 197. Use to "
        "control the historical:supersession ratio. 58 → 70/30, "
        "25 → 50/50.",
    )
    p.add_argument(
        "--supersession-file",
        type=Path,
        default=None,
        help="Override the default supersession dataset "
        "(``bench/data/supersession_pairs.jsonl``). Use to swap in a "
        "larger variant like ``supersession_pairs_100.jsonl``.",
    )
    p.add_argument(
        "--old-repeats",
        type=int,
        default=1,
        help="Number of times to ingest each 'old' fact, with a wall-clock "
        "gap before each round. >1 stress-tests reinforced stale facts "
        "(builds up base-level activation on the old version before the "
        "single 'new' ingest).",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    row = asyncio.run(
        _main(
            embedder_choice=args.embedder,
            scale_n=args.scale_n,
            gap_seconds=args.gap_seconds,
            k=args.k,
            backend_choice=args.backend,
            locomo_question_sample=args.locomo_question_sample,
            supersession_file=args.supersession_file,
            old_repeats=args.old_repeats,
        )
    )
    row["embedder"] = args.embedder
    row["timestamp"] = started.isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print()
    print(f"=== {args.backend} / {args.embedder} / N={args.scale_n} + 50 supersession ===")
    print(
        f"  LoCoMo subset     : recall@{args.k} = {row['locomo_recall_at_k']:.4f} "
        f"over {row['locomo_n_scored']} questions"
    )
    print(
        f"  Supersede subset  : new_in_top{args.k} = {row['supersede_new_in_topk_rate']:.2%} "
        f"| new@1 = {row['supersede_new_at_1_rate']:.2%} "
        f"| old@1 = {row['supersede_old_at_1_rate']:.2%}  "
        f"over {row['supersede_n']} questions"
    )
    print(f"  BLENDED           : recall@{args.k} = {row['blended_recall_at_k']:.4f}")
    print(
        f"  Latency (all)     : mean={row['latency_ms_mean']}ms  "
        f"p50={row['latency_ms_p50']}ms  p95={row['latency_ms_p95']}ms  "
        f"p99={row['latency_ms_p99']}ms"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
