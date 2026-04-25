"""Final head-to-head: raw_stack vs mnemoss_default at N=5K OpenAI.

Measures both recall@10 and per-query latency in the same run so the
two axes are directly comparable. Ingest time is tracked separately
from query time — what matters for production is the recall-call
latency on a warm workspace, not the one-time ingest cost.

Protocol:
1. Build the N=5,000 LoCoMo corpus (conv-26 + 4,581 distractors).
2. Ingest all 5,000 memories into the backend; record total ingest wall time.
3. Warm-up: issue 3 throwaway queries to prime any caches / ANN state.
4. For each of 196 scorable gold queries:
   - time.perf_counter() → recall(q, k=10) → time.perf_counter()
   - compute recall@10 = |returned ∩ gold| / |gold|
   - record per-query latency in milliseconds
5. Report:
   - mean recall@10 (hit rate)
   - mean / p50 / p95 / p99 per-query latency
   - total ingest time
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)


async def _run_arm(
    *,
    backend,
    backend_name: str,
    scale_n: int,
    k: int,
) -> dict:
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    # 1. Ingest and time it.
    print(f"[{backend_name}] ingesting {len(padded_mems)} memories...", flush=True)
    ingest_t0 = time.perf_counter()
    dia_to_mid: dict[str, str] = {}
    for m in padded_mems:
        mid = await backend.observe(m["text"], ts=m.get("ts", time.time()))
        if mid is not None:
            dia_to_mid[m["dia_id"]] = mid
    ingest_seconds = time.perf_counter() - ingest_t0
    print(f"[{backend_name}] ingest complete in {ingest_seconds:.1f}s", flush=True)

    # 2. Warm-up queries — prime caches / ANN state. Not timed.
    for warm_q in [
        "a warm-up query for cache priming",
        "a second warm-up query",
        "a third warm-up query",
    ]:
        await backend.recall(warm_q, k=k)

    # 3. Scored queries — time each one.
    print(f"[{backend_name}] timing {len(gold_queries)} scored queries...", flush=True)
    latencies_ms: list[float] = []
    recall_hits: list[float] = []
    scored = 0

    for q in gold_queries:
        gold_ids = {dia_to_mid[d] for d in q["relevant_dia_ids"] if d in dia_to_mid}
        if not gold_ids:
            continue
        t0 = time.perf_counter()
        hits = await backend.recall(q["question"], k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        returned = {h.memory_id for h in hits}
        recall_hits.append(len(returned & gold_ids) / len(gold_ids))
        scored += 1

    # 4. Aggregate.
    lat_sorted = sorted(latencies_ms)

    def pct(p: float) -> float:
        if not lat_sorted:
            return 0.0
        idx = min(int(p * len(lat_sorted)), len(lat_sorted) - 1)
        return lat_sorted[idx]

    return {
        "backend": backend_name,
        "scale_n": scale_n,
        "n_scored": scored,
        "ingest_seconds": round(ingest_seconds, 2),
        "recall_at_10": round(statistics.mean(recall_hits), 4),
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2),
        "latency_ms_p50": round(pct(0.50), 2),
        "latency_ms_p90": round(pct(0.90), 2),
        "latency_ms_p95": round(pct(0.95), 2),
        "latency_ms_p99": round(pct(0.99), 2),
        "latency_ms_max": round(max(latencies_ms), 2) if latencies_ms else 0.0,
    }


async def _main(
    *,
    embedder_choice: str,
    scale_n: int,
    k: int,
    backend_choice: str,
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
            k=k,
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
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    row = asyncio.run(
        _main(
            embedder_choice=args.embedder,
            scale_n=args.scale_n,
            k=args.k,
            backend_choice=args.backend,
        )
    )
    row["embedder"] = args.embedder
    row["timestamp"] = started.isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print()
    for label, key in [
        ("ingest (s)", "ingest_seconds"),
        ("queries scored", "n_scored"),
        ("recall@10", "recall_at_10"),
        ("latency mean (ms)", "latency_ms_mean"),
        ("latency p50 (ms)", "latency_ms_p50"),
        ("latency p90 (ms)", "latency_ms_p90"),
        ("latency p95 (ms)", "latency_ms_p95"),
        ("latency p99 (ms)", "latency_ms_p99"),
        ("latency max (ms)", "latency_ms_max"),
    ]:
        print(f"  {label:>18}: {row[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
