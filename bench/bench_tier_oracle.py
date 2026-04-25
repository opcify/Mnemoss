"""Tier oracle ceiling test — is mnemoss bottlenecked by classification or by ranking?

The earlier benches showed mnemoss recall@10 = 0.3737 vs raw_stack
0.4205 on the warm-cache aged corpus — a 4.7pp gap. Two possible
causes:

1. **Tier classification is imperfect.** Some gold-answer memories
   end up in COLD or DEEP, so the cascade short-circuits at HOT/WARM
   and never reaches them.
2. **Within-tier ranking is imperfect.** Even when gold answers are
   in HOT/WARM, cosine ranking doesn't always put them in top-K.

This bench separates the two by **oracle-placing all gold-answer
memories into HOT** before scoring. With perfect classification, the
cascade always reaches the gold answers on the first scan. If recall
matches raw_stack under oracle placement, the architecture's only
weakness is classification quality — fixable by better Rebalance
signal. If recall is *still* short of raw_stack, the bottleneck is
ranking — needs different recall logic.

Methodology
-----------

Phase 1: ingest 20K LoCoMo memories.

Phase 2: compute the union of all gold memory IDs across the 197
gold queries (≈ 133 unique IDs).

Phase 3: direct SQL UPDATE — set ``index_tier = 'hot'``,
``idx_priority = 0.9`` for gold IDs; ``index_tier = 'deep'``,
``idx_priority = 0.01`` for the rest. This bypasses Rebalance
entirely; we're not measuring the formula's classification ability,
only the cascade's reachability+ranking ability.

Phase 4: score the 197 gold queries. Two configurations:

- ``include_deep=True``: cascade scans HOT → WARM → COLD → DEEP,
  reaches everything regardless. Should match raw_stack since
  effectively no tier-side filtering happens.
- ``include_deep=False``: cascade stops at COLD. Gold IDs are in
  HOT under oracle placement, so they're reachable; everything else
  in DEEP is excluded. Tests whether "trust HOT" works when HOT
  contains exactly the right memories.

Phase 5: compare both modes against the raw_stack baseline.

Expected outcomes:

- include_deep=True ≈ raw_stack — confirms the cascade isn't
  introducing filtering artifacts when DEEP is reachable.
- include_deep=False ≈ raw_stack — confirms HOT cosine ranking can
  match flat ANN over the full corpus when the right memories are
  in HOT. Validates the "trust the cache" architectural bet.

If either falls short, the gap reveals where the pure-cosine cascade
loses information vs flat-cosine raw_stack. Most likely candidates:

- HNSW imprecision under tier_filter (different ANN graph traversal
  when restricted to a subset of vectors)
- vec_search ``over_scan`` budget being too small for HOT-only
  searches (200 candidates with k=10 might miss occasionally)
- supersede_filter or other store-side filtering quietly dropping
  candidates

Usage::

    python -m bench.bench_tier_oracle \\
        --embedder local --scale-n 20000 \\
        --out bench/results/tier_oracle_20k_local.json
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

import apsw
import numpy as np

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.raw_stack_backend import RawStackBackend
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)


class _BatchCachedEmbedder:
    """Wraps an embedder with a precomputed text→vector cache.

    The bench precomputes all ingest-side vectors in big batches (taking
    advantage of the underlying embedder's native batching, which can be
    10×+ faster per item than batch_size=1 calls). At observe-time, the
    cache resolves each text in O(1). For texts not in the cache (e.g.
    queries), we fall through to the inner embedder.

    Critically, ``embedder_id`` and ``dim`` pass through transparently
    so the workspace's schema pin is identical to using the inner
    embedder directly — recall results are unchanged.
    """

    def __init__(self, inner, cache: dict[str, np.ndarray]):
        self._inner = inner
        self._cache = cache
        self.dim = inner.dim
        self.embedder_id = inner.embedder_id

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        miss_idx: list[int] = []
        miss_text: list[str] = []
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is None:
                out.append(None)  # placeholder
                miss_idx.append(i)
                miss_text.append(t)
            else:
                out.append(v)
        if miss_text:
            miss_vecs = self._inner.embed(miss_text)
            for j, idx in enumerate(miss_idx):
                out[idx] = miss_vecs[j]
        return np.stack(out)

    # Some embedders expose an asymmetric query path (Nomic v2 MoE,
    # BGE-M3, etc.). The bench's recall side calls
    # ``embed_query_or_embed`` from ``mnemoss.encoder.embedder`` which
    # dispatches based on whether the underlying object exposes
    # ``embed_query``. We forward that attribute when present so the
    # query path stays correct.
    def __getattr__(self, name):
        return getattr(self._inner, name)


def _precompute_vectors(embedder, texts: list[str], batch_size: int = 100, label: str = "") -> dict[str, np.ndarray]:
    """Embed ``texts`` in batches and return a text → vec dict.

    Uses the embedder's native ``embed`` (which sentence-transformers
    backends batch internally — batch_size=100 is roughly the per-batch
    GPU sweet spot). De-duplicates texts before embedding so identical
    facts are not re-embedded.
    """

    unique = list(dict.fromkeys(texts))  # de-dup, preserve order
    print(
        f"[{label}] pre-embedding {len(unique)} unique texts "
        f"(of {len(texts)} total) in batches of {batch_size}...",
        flush=True,
    )
    t0 = time.perf_counter()
    cache: dict[str, np.ndarray] = {}
    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        vecs = embedder.embed(batch)
        for t, v in zip(batch, vecs, strict=False):
            cache[t] = v
    elapsed = time.perf_counter() - t0
    print(
        f"[{label}]   pre-embed done in {elapsed:.1f}s "
        f"({len(unique)/elapsed:.0f} texts/s)",
        flush=True,
    )
    return cache


def _snapshot_tier_distribution(db_path: Path) -> dict[str, int]:
    conn = apsw.Connection(str(db_path))
    try:
        rows = conn.cursor().execute(
            "SELECT index_tier, COUNT(*) FROM memory GROUP BY index_tier"
        ).fetchall()
        return {tier: count for tier, count in rows}
    finally:
        conn.close()


def _oracle_place_gold(db_path: Path, gold_ids: set[str]) -> dict:
    """Direct SQL: put all ``gold_ids`` in HOT with high idx_priority,
    everyone else in DEEP. Bypasses Rebalance — pure classification oracle."""

    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("BEGIN")
        try:
            # Everything → DEEP first
            cur.execute(
                "UPDATE memory SET index_tier = 'deep', idx_priority = 0.01"
            )
            # Then bump gold IDs to HOT
            placeholders = ",".join("?" for _ in gold_ids)
            if gold_ids:
                cur.execute(
                    f"UPDATE memory SET index_tier = 'hot', idx_priority = 0.9 "
                    f"WHERE id IN ({placeholders})",
                    list(gold_ids),
                )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        # Confirm placement
        rows = cur.execute(
            "SELECT index_tier, COUNT(*) FROM memory GROUP BY index_tier"
        ).fetchall()
        return {tier: count for tier, count in rows}
    finally:
        conn.close()


async def _ingest(
    backend, padded_mems: list[dict], dia_to_mid: dict[str, str], label: str
) -> float:
    print(f"[{label}] ingesting {len(padded_mems)} memories...", flush=True)
    t0 = time.perf_counter()
    for m in padded_mems:
        mid = await backend.observe(m["text"], ts=time.time())
        if mid is not None:
            dia_to_mid[m["dia_id"]] = mid
    elapsed = time.perf_counter() - t0
    print(f"[{label}]   ingest done in {elapsed:.1f}s", flush=True)
    return elapsed


async def _score(
    backend, gold_queries: list[dict], dia_to_mid: dict[str, str],
    *, k: int, label: str,
) -> dict:
    print(f"[{label}] scoring {len(gold_queries)} queries...", flush=True)
    latencies_ms: list[float] = []
    recall_hits: list[float] = []
    n_scored = 0
    for q in gold_queries:
        gold_ids = {dia_to_mid[d] for d in q["relevant_dia_ids"] if d in dia_to_mid}
        if not gold_ids:
            continue
        t0 = time.perf_counter()
        hits = await backend.recall(q["question"], k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        returned = {h.memory_id for h in hits}
        recall_hits.append(len(returned & gold_ids) / len(gold_ids))
        n_scored += 1

    lat_sorted = sorted(latencies_ms)

    def pct(p: float) -> float:
        if not lat_sorted:
            return 0.0
        idx = min(int(p * len(lat_sorted)), len(lat_sorted) - 1)
        return lat_sorted[idx]

    return {
        "n_scored": n_scored,
        "recall_at_k": (
            round(statistics.mean(recall_hits), 4) if recall_hits else 0.0
        ),
        "latency_ms_mean": (
            round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0
        ),
        "latency_ms_p50": round(pct(0.50), 2),
        "latency_ms_p95": round(pct(0.95), 2),
        "latency_ms_p99": round(pct(0.99), 2),
    }


async def _run_mnemoss_arm(
    *, embedder_choice: str, scale_n: int, k: int,
    skip_oracle_placement: bool = False,
) -> dict:
    embedder = _resolve_embedder(embedder_choice)
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    # Pre-embed all ingest texts in big batches, then wrap the embedder
    # with a cache. observe() calls embed([text]) per memory; without
    # this the embedder runs batch_size=1 forwards which are 10×+ slower
    # per item than batched calls on most sentence-transformer backends.
    cache = _precompute_vectors(
        embedder, [m["text"] for m in padded_mems], label="mnemoss"
    )
    cached = _BatchCachedEmbedder(embedder, cache)

    backend = MnemossBackend(embedding_model=cached, include_deep=True)
    dia_to_mid: dict[str, str] = {}
    try:
        ingest_seconds = await _ingest(backend, padded_mems, dia_to_mid, "mnemoss")
        db_path = backend._tempdir / "workspaces" / "bench" / "memory.sqlite"

        gold_ids: set[str] = set()
        for q in gold_queries:
            for d in q["relevant_dia_ids"]:
                if d in dia_to_mid:
                    gold_ids.add(dia_to_mid[d])
        print(f"[mnemoss] gold IDs across all queries: {len(gold_ids)}", flush=True)

        # Oracle placement: put all gold IDs in HOT.
        # When ``skip_oracle_placement=True``, we leave the workspace in
        # its natural post-observe state (every memory in HOT, no
        # Rebalance has run). Lets the user measure the "vanilla"
        # cascade behavior — with the 200 cap not yet enforced, HOT
        # contains the full corpus; cascade with HOT filter is then
        # equivalent to flat ANN over everything (raw_stack-equivalent).
        if skip_oracle_placement:
            post_oracle = _snapshot_tier_distribution(db_path)
            print(
                f"[mnemoss] skip_oracle_placement=True; tier dist: {post_oracle}",
                flush=True,
            )
        else:
            post_oracle = _oracle_place_gold(db_path, gold_ids)
            print(f"[mnemoss] post-oracle tier dist: {post_oracle}", flush=True)

        # include_deep=True (cascade reaches everything regardless)
        backend._include_deep = True
        score_with_deep = await _score(
            backend, gold_queries, dia_to_mid,
            k=k, label="mnemoss/include_deep=True",
        )

        # include_deep=False (cascade stops at COLD; tests "trust HOT")
        backend._include_deep = False
        score_no_deep = await _score(
            backend, gold_queries, dia_to_mid,
            k=k, label="mnemoss/include_deep=False",
        )
    finally:
        await backend.close()

    return {
        "ingest_seconds": round(ingest_seconds, 2),
        "n_gold_ids": len(gold_ids),
        "post_oracle_tier_distribution": post_oracle,
        "score_with_include_deep": score_with_deep,
        "score_without_include_deep": score_no_deep,
    }


async def _run_raw_stack_arm(
    *, embedder_choice: str, scale_n: int, k: int,
) -> dict:
    embedder = _resolve_embedder(embedder_choice)
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    cache = _precompute_vectors(
        embedder, [m["text"] for m in padded_mems], label="raw_stack"
    )
    cached = _BatchCachedEmbedder(embedder, cache)

    backend = RawStackBackend(embedding_model=cached)
    dia_to_mid: dict[str, str] = {}
    try:
        ingest_seconds = await _ingest(
            backend, padded_mems, dia_to_mid, "raw_stack"
        )
        score = await _score(
            backend, gold_queries, dia_to_mid, k=k, label="raw_stack",
        )
    finally:
        await backend.close()

    return {"ingest_seconds": round(ingest_seconds, 2), "score": score}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "gemma", "nomic", "fake"],
        default="local",
    )
    p.add_argument("--scale-n", type=int, default=20_000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--skip-raw-stack", action="store_true",
        help="Skip the raw_stack baseline (saves ingest time).",
    )
    p.add_argument(
        "--skip-oracle-placement", action="store_true",
        help="Skip the SQL UPDATE that puts gold IDs in HOT and the rest "
        "in DEEP. Leaves the workspace in its natural post-observe state "
        "(every memory in HOT). Use to measure 'vanilla' cascade behavior "
        "without classification assistance.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    print(
        f"\n=== Tier oracle bench: N={args.scale_n}, embedder={args.embedder} ===",
        flush=True,
    )

    mnemoss_result = asyncio.run(_run_mnemoss_arm(
        embedder_choice=args.embedder, scale_n=args.scale_n, k=args.k,
        skip_oracle_placement=args.skip_oracle_placement,
    ))
    raw_stack_result: dict | None = None
    if not args.skip_raw_stack:
        raw_stack_result = asyncio.run(_run_raw_stack_arm(
            embedder_choice=args.embedder, scale_n=args.scale_n, k=args.k,
        ))

    out = {
        "chart": "tier_oracle",
        "embedder": args.embedder,
        "scale_n": args.scale_n,
        "k": args.k,
        "timestamp": started.isoformat(),
        "mnemoss": mnemoss_result,
        "raw_stack": raw_stack_result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)

    print()
    print("─" * 72)
    print("ORACLE-PLACED gold (all in HOT) vs raw_stack baseline")
    print("─" * 72)
    rows = [
        ("mnemoss / include_deep=True",
         mnemoss_result["score_with_include_deep"]),
        ("mnemoss / include_deep=False",
         mnemoss_result["score_without_include_deep"]),
    ]
    if raw_stack_result is not None:
        rows.append(("raw_stack (no tiers)", raw_stack_result["score"]))
    print(f"  {'arm':>32}  {'recall@k':>9}  {'p50':>7}  {'p95':>7}  {'p99':>7}")
    for label, score in rows:
        print(
            f"  {label:>32}  "
            f"{score['recall_at_k']:>9.4f}  "
            f"{score['latency_ms_p50']:>6.1f}m  "
            f"{score['latency_ms_p95']:>6.1f}m  "
            f"{score['latency_ms_p99']:>6.1f}m"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
