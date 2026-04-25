"""Scale-axis latency benchmark.

Measures per-query recall latency (p50 / p99 / mean) for each
(backend, N) pair. Shares the scale-corpus builder with
``bench.launch_comparison`` so quality numbers (``scale_sweep``) and
speed numbers (this file) describe the same corpus.

Usage::

    python -m bench.scale_latency \
        --embedder local \
        --sizes 500 1500 3000 5000 10000 \
        --backends mnemoss raw_stack static_file \
        --gold-conversation conv-26 \
        --out bench/results/scale_latency_local.json

Each run ingests the corpus once (not timed), then runs every gold
query once for warmup (not timed), then runs each query ``--repeats``
times (timed). Per-query latency is the median across repeats — so
we measure steady-state recall speed, not first-call model-load
overhead.
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

from bench.backends.base import MemoryBackend
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_backend,
    _build_scale_corpus,
    _load_jsonl,
)


async def _ingest(backend: MemoryBackend, memories: list[dict]) -> None:
    for m in memories:
        await backend.observe(m["text"], ts=m["ts"])


async def _time_queries(
    backend: MemoryBackend,
    queries: list[dict],
    *,
    k: int,
    repeats: int,
) -> list[float]:
    """Return per-query median latency (ms) across ``repeats`` runs."""

    # Warmup — first call typically pays a one-time cost (lazy index
    # load, thread pool spin-up). Measure steady state.
    if queries:
        await backend.recall(queries[0]["question"], k=k)

    latencies: list[float] = []
    for q in queries:
        reps: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            await backend.recall(q["question"], k=k)
            reps.append((time.perf_counter() - t0) * 1000.0)
        latencies.append(statistics.median(reps))
    return latencies


async def _run_one(
    *,
    backend_name: str,
    scale_n: int,
    embedder: str,
    gold_conversation: str,
    k: int,
    repeats: int,
) -> dict:
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories,
        queries,
        gold_conversation_id=gold_conversation,
        scale_n=scale_n,
    )

    backend = _build_backend(backend_name, fake_embedder=False, embedder=embedder)
    try:
        ingest_start = time.perf_counter()
        await _ingest(backend, padded_mems)
        ingest_seconds = time.perf_counter() - ingest_start

        latencies = await _time_queries(backend, gold_queries, k=k, repeats=repeats)
    finally:
        await backend.close()

    latencies.sort()
    def _percentile(sorted_xs: list[float], pct: float) -> float:
        if not sorted_xs:
            return 0.0
        idx = min(len(sorted_xs) - 1, max(0, int(pct * len(sorted_xs)) - 1))
        return sorted_xs[idx]

    return {
        "backend": backend_name,
        "scale_n": scale_n,
        "embedder": embedder,
        "n_queries": len(latencies),
        "repeats": repeats,
        "p50_ms": round(_percentile(latencies, 0.50), 3),
        "p90_ms": round(_percentile(latencies, 0.90), 3),
        "p99_ms": round(_percentile(latencies, 0.99), 3),
        "mean_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "min_ms": round(latencies[0], 3) if latencies else 0.0,
        "max_ms": round(latencies[-1], 3) if latencies else 0.0,
        "ingest_seconds": round(ingest_seconds, 2),
    }


async def _run_all(
    *,
    backends: list[str],
    sizes: list[int],
    embedder: str,
    gold_conversation: str,
    k: int,
    repeats: int,
) -> list[dict]:
    out: list[dict] = []
    for backend in backends:
        for n in sizes:
            print(f"[latency] backend={backend} N={n} k={k} repeats={repeats}", flush=True)
            row = await _run_one(
                backend_name=backend,
                scale_n=n,
                embedder=embedder,
                gold_conversation=gold_conversation,
                k=k,
                repeats=repeats,
            )
            out.append(row)
            print(
                f"  → p50={row['p50_ms']:.2f}ms  p99={row['p99_ms']:.2f}ms  "
                f"mean={row['mean_ms']:.2f}ms  (ingest {row['ingest_seconds']:.1f}s)",
                flush=True,
            )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder", choices=["openai", "local", "nomic", "fake"], default="local"
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 3000, 5000, 10000])
    p.add_argument(
        "--backends",
        nargs="+",
        default=["mnemoss", "raw_stack", "static_file"],
        choices=[
            "mnemoss",
            "mnemoss_semantic",
            "mnemoss_fast",
            "mnemoss_prod",
            "mnemoss_rocket",
            "raw_stack",
            "static_file",
        ],
    )
    p.add_argument("--gold-conversation", default="conv-26")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3, help="Median-of-N per query for stability.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _run_all(
            backends=args.backends,
            sizes=args.sizes,
            embedder=args.embedder,
            gold_conversation=args.gold_conversation,
            k=args.k,
            repeats=args.repeats,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "scale_latency",
                "embedder": args.embedder,
                "sizes": args.sizes,
                "backends": args.backends,
                "gold_conversation": args.gold_conversation,
                "k": args.k,
                "repeats": args.repeats,
                "timestamp": started.isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")

    # Terminal table.
    print()
    header = f"{'backend':<16} {'N':>6} {'p50 ms':>9} {'p99 ms':>9} {'mean ms':>9}"
    print(header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r['backend']:<16} {r['scale_n']:>6} "
            f"{r['p50_ms']:>9.2f} {r['p99_ms']:>9.2f} {r['mean_ms']:>9.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
