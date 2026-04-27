"""Embedder speed micro-benchmark — MiniLM vs Nomic vs EmbeddingGemma.

The repeated-query and launch-comparison benches fold embedding into
larger wall-clock numbers (ingest + ANN build + scoring). This script
isolates the three embedder costs that actually matter to an operator:

1. **Model load** — cold-start seconds (sentence-transformers init +
   weight load). Paid once per process lifetime.
2. **Ingest throughput** — docs/sec when embedding a batch the size
   of a realistic observe wave (100 at a time). This is what caps
   the ``mem.observe()`` rate in practice; bigger is better.
3. **Query latency** — ms for a single-string embed on the query
   side. This is the embedder's contribution to every ``recall()``
   call; smaller is better, and asymmetric embedders can have a
   different number here vs the document side.

All three embedders are loaded and timed sequentially to avoid CPU
contention. Output is a small JSON blob + a terminal table.

Usage::

    python -m bench.bench_embedder_speed \\
        --embedders local nomic gemma \\
        --ingest-batch 100 \\
        --ingest-trials 3 \\
        --query-trials 50 \\
        --out bench/results/embedder_speed.json
"""

from __future__ import annotations

import argparse
import gc
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

from bench.launch_comparison import (
    MEMORIES_PATH,
    _load_jsonl,
    _resolve_embedder,
)


def _load_corpus_texts(n: int) -> list[str]:
    """Pull ``n`` real conversational utterances from the LoCoMo corpus.

    Using real text matters — random strings, Lorem Ipsum, or repeated
    short tokens tokenize differently from natural conversation and
    skew embedder throughput numbers. This picks the first ``n``
    memories in the JSONL so every embedder sees identical input.
    """

    rows = _load_jsonl(MEMORIES_PATH)
    return [m["text"] for m in rows[:n]]


def _time_load(choice: str) -> tuple[object, float]:
    """Instantiate the embedder and force it to warm up. Returns
    ``(embedder, load_seconds)``.

    LocalEmbedder is lazy — ``model_name`` sets the target but weights
    don't load until the first embed call. We force one with a
    throwaway query so ``load_seconds`` reflects real cold-start cost.
    """

    gc.collect()
    t0 = time.perf_counter()
    emb = _resolve_embedder(choice)
    # Force the lazy load + first forward pass.
    _ = emb.embed(["warm up"])
    elapsed = time.perf_counter() - t0
    return emb, elapsed


def _time_ingest(emb, texts: list[str], trials: int) -> dict:
    """Time ``trials`` full-batch embed calls; report throughput."""

    per_trial: list[float] = []
    for _ in range(trials):
        t0 = time.perf_counter()
        emb.embed(texts)
        per_trial.append(time.perf_counter() - t0)
    n = len(texts)
    return {
        "batch_size": n,
        "trials": trials,
        "seconds_per_batch_mean": round(statistics.mean(per_trial), 4),
        "seconds_per_batch_median": round(statistics.median(per_trial), 4),
        "docs_per_sec_mean": round(n / statistics.mean(per_trial), 1),
        "docs_per_sec_best": round(n / min(per_trial), 1),
    }


def _time_queries(emb, query_texts: list[str]) -> dict:
    """Time per-query single-string embeds. Uses ``embed_query`` when
    available (asymmetric embedders); falls back to ``embed``.
    """

    from mnemoss.encoder.embedder import embed_query_or_embed

    latencies_ms: list[float] = []
    for q in query_texts:
        t0 = time.perf_counter()
        embed_query_or_embed(emb, [q])
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "trials": len(query_texts),
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2),
        "latency_ms_median": round(statistics.median(latencies_ms), 2),
        "latency_ms_p95": round(
            statistics.quantiles(latencies_ms, n=20)[-1]
            if len(latencies_ms) >= 20
            else max(latencies_ms),
            2,
        ),
    }


def _run_one(
    choice: str, ingest_texts: list[str], query_texts: list[str], ingest_trials: int
) -> dict:
    print(f"\n[{choice}] loading…", flush=True)
    emb, load_s = _time_load(choice)
    print(f"[{choice}] load={load_s:.2f}s  dim={emb.dim}  id={emb.embedder_id}", flush=True)

    print(
        f"[{choice}] timing ingest ({len(ingest_texts)} docs × {ingest_trials} trials)…",
        flush=True,
    )
    ingest = _time_ingest(emb, ingest_texts, ingest_trials)
    print(
        f"[{choice}] ingest: {ingest['docs_per_sec_mean']:.1f} docs/sec "
        f"(best {ingest['docs_per_sec_best']:.1f})",
        flush=True,
    )

    print(f"[{choice}] timing queries ({len(query_texts)} singles)…", flush=True)
    queries = _time_queries(emb, query_texts)
    print(
        f"[{choice}] query latency: {queries['latency_ms_mean']:.2f}ms mean, "
        f"{queries['latency_ms_median']:.2f}ms p50, {queries['latency_ms_p95']:.2f}ms p95",
        flush=True,
    )

    # Release the model before loading the next embedder so we don't
    # measure under memory pressure from a prior model still in RAM.
    del emb
    gc.collect()

    return {
        "embedder": choice,
        "load_seconds": round(load_s, 3),
        "ingest": ingest,
        "query": queries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedders",
        nargs="+",
        default=["local", "nomic", "gemma"],
        choices=["openai", "local", "nomic", "gemma", "fake"],
    )
    p.add_argument("--ingest-batch", type=int, default=100)
    p.add_argument("--ingest-trials", type=int, default=3)
    p.add_argument("--query-trials", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    ingest_texts = _load_corpus_texts(args.ingest_batch)
    # Query texts reuse the corpus but treat each as a single-string
    # query. Real queries are shorter on average, but using utterances
    # keeps input distribution apples-to-apples with ingest.
    query_texts = _load_corpus_texts(args.query_trials + args.ingest_batch)[
        args.ingest_batch : args.ingest_batch + args.query_trials
    ]

    started = datetime.now(timezone.utc)
    results: list[dict] = []
    for choice in args.embedders:
        results.append(_run_one(choice, ingest_texts, query_texts, args.ingest_trials))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "embedder_speed",
                "ingest_batch": args.ingest_batch,
                "ingest_trials": args.ingest_trials,
                "query_trials": args.query_trials,
                "timestamp": started.isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )

    # Terminal summary table.
    print()
    print(
        f"{'embedder':>10}  {'load (s)':>10}  {'dim':>5}  "
        f"{'docs/sec':>10}  {'query p50 (ms)':>15}  {'query p95 (ms)':>15}"
    )
    print("─" * 75)
    for r in results:
        print(
            f"{r['embedder']:>10}  "
            f"{r['load_seconds']:>10.2f}  "
            f"{'—':>5}  "
            f"{r['ingest']['docs_per_sec_mean']:>10.1f}  "
            f"{r['query']['latency_ms_median']:>15.2f}  "
            f"{r['query']['latency_ms_p95']:>15.2f}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
