"""Tier lifecycle bench — does the 4-tier index actually work in practice?

Our prior benches all created fresh workspaces, ingested in minutes, and
queried immediately. Result: 100% of memories landed in HOT tier (every
fresh memory's ``idx_priority = sigmoid(η_0) ≈ 0.731 > 0.7``), Rebalance
never ran, the cascade never short-circuited. We never measured the
architectural payoff the 4-tier design was built for.

This bench simulates ~6 months of real usage in seconds via direct SQL
backdating, then checks three load-bearing claims:

1. **Distribution health**: after Rebalance against a realistic spread
   of access ages, do HOT/WARM/COLD/DEEP carry meaningful share, or
   does HOT swallow everything?
2. **Cascade payoff**: does ``include_deep=False`` (cascade short-
   circuits at HOT/WARM) materially reduce latency vs full-scan?
3. **Recall safety**: does the cascade short-circuit lose hits the
   full scan would have caught? (the safety check on the speedup)

Methodology
-----------

Phase 1: bulk-ingest 20K LoCoMo memories into a Mnemoss workspace.

Phase 2: direct SQL UPDATE backdating ``created_at`` per memory using
a Zipfian-like distribution over the past 180 days::

    5%  in last 24h
    15% in last 7 days
    30% in last 30 days
    30% in last 90 days
    20% in last 180 days

Phase 3: pick 20% of memory IDs at random ("active subset") and append
1-5 synthetic recall timestamps to each one's ``access_history`` —
representing past reactivations. The other 80% only have their backdated
``created_at`` and never get re-recalled.

Phase 4: call ``mem.rebalance()``. The recompute uses real ``now``
against the backdated timestamps, redistributing memories across tiers
according to their decayed ``B_i`` + salience + emotional + pin terms.

Phase 5: snapshot the tier histogram. Sanity-check that HOT shrunk and
COLD/DEEP got populated — if they didn't, the formula's defaults are
wrong for any workload that isn't "everything's brand new."

Phase 6: score the 197 LoCoMo gold queries in three configurations:

- ``mnemoss_default`` with ``include_deep=True`` (current bench setting)
- ``mnemoss_default`` with ``include_deep=False`` (cascade short-circuits)
- ``raw_stack`` (no tier system — flat scan baseline)

Compare per-query latency (mean / p50 / p95) and recall@10. If the
cascade architecture works as designed, ``include_deep=False`` should
win on latency without giving up much recall.

Usage::

    python -m bench.bench_tier_lifecycle \\
        --embedder openai \\
        --scale-n 20000 \\
        --active-fraction 0.20 \\
        --out bench/results/tier_lifecycle_20k.json
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

import apsw

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.raw_stack_backend import RawStackBackend
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)

# Backdate distribution: (fraction, age_lo_seconds, age_hi_seconds).
#
# Canonical "realistic agent" workload: a long-running agent with
# 6 months of accumulated history. Daily ingest spreads memories
# roughly uniformly across the past 180 days, with a slight skew
# toward more recent. This is the distribution where time-decay
# meaningfully differentiates active from dormant — the regime the
# architecture is designed for.
_BACKDATE_BANDS: list[tuple[float, int, int]] = [
    (0.05, 0, 86_400),                 # 5% in last 24h
    (0.15, 86_400, 7 * 86_400),        # 15% in 1-7 days
    (0.30, 7 * 86_400, 30 * 86_400),   # 30% in 7-30 days
    (0.30, 30 * 86_400, 90 * 86_400),  # 30% in 30-90 days
    (0.20, 90 * 86_400, 180 * 86_400), # 20% in 90-180 days
]

# Active-subset access_history: each active memory gets N recent recalls.
_ACTIVE_RECALLS_LO = 1
_ACTIVE_RECALLS_HI = 5
# Recalls are within the last week (active memories are recently-touched).
_ACTIVE_RECALL_RECENCY_SECONDS = 7 * 86_400


def _draw_ages(rng: random.Random, n: int) -> list[float]:
    """Sample n age-in-seconds values from the backdate distribution.

    Returns a shuffled list — caller assigns to memories in-order so the
    ingest order vs age order are uncorrelated.
    """

    ages: list[float] = []
    for frac, lo, hi in _BACKDATE_BANDS:
        count = int(round(frac * n))
        for _ in range(count):
            ages.append(rng.uniform(lo, hi))
    # Top-up if rounding lost any.
    while len(ages) < n:
        ages.append(rng.uniform(*_BACKDATE_BANDS[-1][1:]))
    rng.shuffle(ages)
    return ages[:n]


def _gen_active_recalls(
    rng: random.Random, created_age_seconds: float, now_ts: float
) -> list[float]:
    """Generate synthetic past-recall timestamps for one active memory.

    Recalls happen *after* creation but bias toward the recent week.
    Returns Unix timestamps (newest first), suitable for JSON encoding.
    """

    n = rng.randint(_ACTIVE_RECALLS_LO, _ACTIVE_RECALLS_HI)
    out: list[float] = []
    # Each recall is between (now - min(created_age, _ACTIVE_RECALL_RECENCY))
    # and now. So a memory created 90 days ago that's active has its
    # recalls all in the last 7 days; a memory created 3 days ago has
    # recalls in the last 3 days.
    max_age = min(created_age_seconds, float(_ACTIVE_RECALL_RECENCY_SECONDS))
    for _ in range(n):
        age = rng.uniform(0.0, max_age)
        out.append(now_ts - age)
    return sorted(out, reverse=True)


def _backdate_workspace(
    db_path: Path,
    *,
    rng: random.Random,
    active_fraction: float,
    now_ts: float,
    correlated_active_ids: set[str] | None = None,
) -> dict:
    """Direct SQL: rewrite created_at and access_history on the workspace.

    Returns a stats dict. Does NOT touch ``idx_priority`` or
    ``index_tier`` — those get rewritten by the subsequent ``rebalance()``
    call against the backdated timestamps.

    ``correlated_active_ids`` (optional): if given, the active subset is
    seeded from this set (typically the gold-query memories). The
    remainder is filled by random sampling up to ``active_fraction × N``.
    This simulates the realistic case where users tend to recall what
    they recently observed (the active set correlates with the query
    distribution). When ``None``, active subset is fully random — the
    adversarial case where activation and query-relevance decouple.
    """

    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        ids = [row[0] for row in cur.execute("SELECT id FROM memory").fetchall()]
        n = len(ids)

        ages = _draw_ages(rng, n)
        active_count = int(round(active_fraction * n))

        if correlated_active_ids:
            # Start with the correlated seed (clamped to ids actually in
            # the workspace), then top up with random non-seed ids until
            # we hit the target active_count.
            id_set = set(ids)
            seed = correlated_active_ids & id_set
            active_ids = set(seed)
            if len(active_ids) < active_count:
                non_seed_pool = [mid for mid in ids if mid not in active_ids]
                top_up = active_count - len(active_ids)
                if top_up <= len(non_seed_pool):
                    active_ids.update(rng.sample(non_seed_pool, top_up))
                else:
                    active_ids.update(non_seed_pool)
            elif len(active_ids) > active_count:
                # Trim down — keep a random subset of the seed.
                active_ids = set(rng.sample(list(active_ids), active_count))
        else:
            active_ids = set(rng.sample(ids, active_count))

        # Backdate created_at and rewrite access_history per row.
        # access_history MUST always include created_at as its first entry
        # (per ``compute_base_level`` contract). For active memories we
        # also append synthetic recall timestamps.
        cur.execute("BEGIN")
        try:
            for mid, age in zip(ids, ages, strict=True):
                created_ts = now_ts - age
                if mid in active_ids:
                    extra = _gen_active_recalls(rng, age, now_ts)
                    history = [created_ts, *extra]
                else:
                    history = [created_ts]
                history_json = json.dumps(history)
                last_accessed = max(history)
                cur.execute(
                    "UPDATE memory SET created_at = ?, access_history = ?, "
                    "last_accessed_at = ? WHERE id = ?",
                    (created_ts, history_json, last_accessed, mid),
                )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        return {
            "n_memories": n,
            "n_active": len(active_ids),
            "n_dormant": n - len(active_ids),
            "active_fraction": active_fraction,
        }
    finally:
        conn.close()


def _snapshot_tier_distribution(db_path: Path) -> dict[str, int]:
    """Return ``{tier_name: count}`` from the memory.index_tier column."""

    conn = apsw.Connection(str(db_path))
    try:
        rows = conn.cursor().execute(
            "SELECT index_tier, COUNT(*) FROM memory GROUP BY index_tier"
        ).fetchall()
        return {tier: count for tier, count in rows}
    finally:
        conn.close()


def _snapshot_idx_priority_histogram(db_path: Path) -> dict[str, int]:
    """Return a coarse histogram of idx_priority values for sanity-checking.

    Buckets line up with the tier thresholds so a mismatch between this
    histogram and the tier counts indicates a stale ``index_tier`` cache.
    """

    conn = apsw.Connection(str(db_path))
    try:
        cur = conn.cursor()
        buckets = {
            ">0.7 (HOT-eligible)": 0,
            "0.3-0.7 (WARM)": 0,
            "0.1-0.3 (COLD)": 0,
            "<=0.1 (DEEP)": 0,
        }
        for row in cur.execute("SELECT idx_priority FROM memory").fetchall():
            ip = row[0]
            if ip > 0.7:
                buckets[">0.7 (HOT-eligible)"] += 1
            elif ip > 0.3:
                buckets["0.3-0.7 (WARM)"] += 1
            elif ip > 0.1:
                buckets["0.1-0.3 (COLD)"] += 1
            else:
                buckets["<=0.1 (DEEP)"] += 1
        return buckets
    finally:
        conn.close()


async def _ingest_corpus(
    backend, padded_mems: list[dict], dia_to_mid: dict[str, str], label: str
) -> float:
    """Ingest memories and record dia_id → memory_id mapping. Returns ingest seconds."""

    print(f"[{label}] ingesting {len(padded_mems)} memories...", flush=True)
    t0 = time.perf_counter()
    for m in padded_mems:
        mid = await backend.observe(m["text"], ts=time.time())
        if mid is not None:
            dia_to_mid[m["dia_id"]] = mid
    elapsed = time.perf_counter() - t0
    print(f"[{label}]   ingest done in {elapsed:.1f}s", flush=True)
    return elapsed


async def _score_queries(
    backend,
    gold_queries: list[dict],
    dia_to_mid: dict[str, str],
    *,
    k: int,
    label: str,
) -> dict:
    """Run gold queries, collect recall@k and per-query latency."""

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


async def _run_mnemoss_arms(
    *,
    embedder_choice: str,
    scale_n: int,
    active_fraction: float,
    seed: int,
    k: int,
    correlated_active: bool = False,
) -> dict:
    """Build a Mnemoss workspace, backdate, rebalance, then score in two
    cascade modes (include_deep True/False) using the same workspace.

    ``correlated_active``: when True, the active subset is seeded from
    the gold-query memories — simulating "users keep asking about what
    they recently observed." When False (default), the active subset is
    a random 20% of the corpus, decorrelated from the query distribution
    — the adversarial case that exposes whether the cascade short-stop
    relies on correlation between activation-priority and query-relevance.
    """

    embedder = _resolve_embedder(embedder_choice)
    rng = random.Random(seed)
    now_ts = time.time()

    # Build corpus.
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    # ── ingest with include_deep=True (default; we'll vary at recall) ──
    backend = MnemossBackend(embedding_model=embedder, include_deep=True)
    dia_to_mid: dict[str, str] = {}
    try:
        ingest_seconds = await _ingest_corpus(
            backend, padded_mems, dia_to_mid, "mnemoss"
        )
        # Find the workspace DB so we can backdate via direct SQL.
        # MnemossBackend stores workspace under self._tempdir / "workspaces" / "bench".
        db_path = backend._tempdir / "workspaces" / "bench" / "memory.sqlite"
        assert db_path.exists(), f"Expected workspace DB at {db_path}"

        # ── Phase 2+3: backdate timestamps + active access_history ──
        # Build the correlated-active seed (gold-query memories) when requested.
        correlated_seed: set[str] | None = None
        if correlated_active:
            correlated_seed = set()
            for q in gold_queries:
                for d in q["relevant_dia_ids"]:
                    if d in dia_to_mid:
                        correlated_seed.add(dia_to_mid[d])
            print(
                f"[mnemoss] correlated active subset seeded from "
                f"{len(correlated_seed)} gold-query memories",
                flush=True,
            )
        print("[mnemoss] backdating timestamps + seeding access_history...", flush=True)
        backdate_stats = _backdate_workspace(
            db_path,
            rng=rng,
            active_fraction=active_fraction,
            now_ts=now_ts,
            correlated_active_ids=correlated_seed,
        )
        print(f"[mnemoss]   {backdate_stats}", flush=True)

        # Snapshot pre-rebalance (everything still in HOT from observe).
        pre_rebalance_tiers = _snapshot_tier_distribution(db_path)
        print(f"[mnemoss] pre-rebalance tier dist: {pre_rebalance_tiers}", flush=True)

        # ── Phase 4: rebalance ──
        print("[mnemoss] running rebalance()...", flush=True)
        t0 = time.perf_counter()
        rebalance_stats = await backend._mem.rebalance()
        rebalance_seconds = time.perf_counter() - t0
        print(
            f"[mnemoss]   rebalance done in {rebalance_seconds:.1f}s, "
            f"scanned={rebalance_stats.scanned}, migrated={rebalance_stats.migrated}",
            flush=True,
        )

        # ── Phase 5: snapshot ──
        post_rebalance_tiers = _snapshot_tier_distribution(db_path)
        idx_priority_hist = _snapshot_idx_priority_histogram(db_path)
        print(
            f"[mnemoss] post-rebalance tier dist: {post_rebalance_tiers}",
            flush=True,
        )
        print(
            f"[mnemoss] idx_priority histogram:    {idx_priority_hist}",
            flush=True,
        )

        # ── Phase 6: score with include_deep=True (full sweep) ──
        backend._include_deep = True
        score_with_deep = await _score_queries(
            backend, gold_queries, dia_to_mid,
            k=k, label="mnemoss/include_deep=True",
        )

        # ── Phase 6': score with include_deep=False (cascade) ──
        backend._include_deep = False
        score_no_deep = await _score_queries(
            backend, gold_queries, dia_to_mid,
            k=k, label="mnemoss/include_deep=False",
        )
    finally:
        await backend.close()

    return {
        "ingest_seconds": round(ingest_seconds, 2),
        "rebalance_seconds": round(rebalance_seconds, 2),
        "rebalance_stats": {
            "scanned": rebalance_stats.scanned,
            "migrated": rebalance_stats.migrated,
        },
        "backdate_stats": backdate_stats,
        "pre_rebalance_tier_distribution": pre_rebalance_tiers,
        "post_rebalance_tier_distribution": post_rebalance_tiers,
        "idx_priority_histogram_post": idx_priority_hist,
        "score_with_include_deep": score_with_deep,
        "score_without_include_deep": score_no_deep,
    }


async def _run_raw_stack(
    *,
    embedder_choice: str,
    scale_n: int,
    k: int,
) -> dict:
    """Baseline: pure cosine, no tier system. Shares the corpus."""

    embedder = _resolve_embedder(embedder_choice)
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    backend = RawStackBackend(embedding_model=embedder)
    dia_to_mid: dict[str, str] = {}
    try:
        ingest_seconds = await _ingest_corpus(
            backend, padded_mems, dia_to_mid, "raw_stack"
        )
        score = await _score_queries(
            backend, gold_queries, dia_to_mid, k=k, label="raw_stack",
        )
    finally:
        await backend.close()

    return {
        "ingest_seconds": round(ingest_seconds, 2),
        "score": score,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "gemma", "nomic", "fake"],
        default="openai",
    )
    p.add_argument("--scale-n", type=int, default=20_000)
    p.add_argument(
        "--active-fraction", type=float, default=0.20,
        help="Fraction of memories that get synthetic recall reactivations.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--skip-raw-stack",
        action="store_true",
        help="Skip the raw_stack baseline arm (saves ~10 min at scale-n=20K).",
    )
    p.add_argument(
        "--correlated-active",
        action="store_true",
        help="Seed the active subset from gold-query memories (realistic: "
        "users tend to recall what they recently observed). Default off — "
        "active subset is fully random, the adversarial decorrelated case.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)

    print(
        f"\n=== Tier lifecycle bench: N={args.scale_n}, "
        f"active_fraction={args.active_fraction}, "
        f"correlated_active={args.correlated_active}, "
        f"embedder={args.embedder} ===",
        flush=True,
    )

    mnemoss_result = asyncio.run(_run_mnemoss_arms(
        embedder_choice=args.embedder,
        scale_n=args.scale_n,
        active_fraction=args.active_fraction,
        seed=args.seed,
        k=args.k,
        correlated_active=args.correlated_active,
    ))

    raw_stack_result: dict | None = None
    if not args.skip_raw_stack:
        raw_stack_result = asyncio.run(_run_raw_stack(
            embedder_choice=args.embedder,
            scale_n=args.scale_n,
            k=args.k,
        ))

    out = {
        "chart": "tier_lifecycle",
        "embedder": args.embedder,
        "scale_n": args.scale_n,
        "active_fraction": args.active_fraction,
        "correlated_active": args.correlated_active,
        "k": args.k,
        "seed": args.seed,
        "timestamp": started.isoformat(),
        "mnemoss": mnemoss_result,
        "raw_stack": raw_stack_result,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)

    # Summary
    print()
    print("─" * 70)
    print("TIER DISTRIBUTION (post-rebalance)")
    print("─" * 70)
    for tier, count in sorted(
        mnemoss_result["post_rebalance_tier_distribution"].items()
    ):
        pct = 100 * count / mnemoss_result["backdate_stats"]["n_memories"]
        print(f"  {tier:>6}: {count:>6}  ({pct:5.1f}%)")
    print()
    print("─" * 70)
    print("RECALL & LATENCY")
    print("─" * 70)
    rows = [
        ("mnemoss/include_deep=True",
         mnemoss_result["score_with_include_deep"]),
        ("mnemoss/include_deep=False",
         mnemoss_result["score_without_include_deep"]),
    ]
    if raw_stack_result is not None:
        rows.append(("raw_stack (no tiers)", raw_stack_result["score"]))
    print(f"  {'arm':>30}  {'recall@k':>9}  {'p50':>7}  {'p95':>7}  {'p99':>7}")
    for label, score in rows:
        print(
            f"  {label:>30}  "
            f"{score['recall_at_k']:>9.4f}  "
            f"{score['latency_ms_p50']:>6.1f}m  "
            f"{score['latency_ms_p95']:>6.1f}m  "
            f"{score['latency_ms_p99']:>6.1f}m"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
