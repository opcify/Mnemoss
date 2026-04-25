"""Rebalance lift bench — does running Rebalance actually speed up subsequent recall?

Mnemoss's async-cognition architecture: Dream/Rebalance recomputes
``idx_priority`` for every memory and re-buckets by capacity. The
hypothesis is that **after** Rebalance, frequently-recalled memories
sit in HOT/WARM and the cascade short-circuits at the top tier for
subsequent queries on the same workload — improving latency and
maintaining (or improving) recall.

This bench measures that lift end-to-end. Per arm:

1. **Ingest** N=20K LoCoMo memories (every memory lands in HOT by
   observe-time default ``idx_priority ≈ sigmoid(η_0)``).
2. **Warm-up queries** — score 197 LoCoMo gold queries with
   ``reconsolidate=True`` so each retrieved memory's
   ``access_history`` gets bumped. Records the *pre-rebalance*
   baseline (recall, latency, cascade telemetry).
3. **Rebalance** — recompute ``idx_priority`` using the bumped
   ``access_history``; capacity-bucketed re-classification. Memories
   that were retrieved in phase 2 rise; everything else drifts down.
4. **Test queries** — score the same 197 queries again, this time
   with ``reconsolidate=False`` so we measure pure post-rebalance
   behavior. Records the *post-rebalance* numbers.

Compare warm-up to test phase to measure the architectural lift.

The expectation under the new tier-cascade-pure-cosine recall path:

- Warm-up phase: cascade scans HOT (200 of 20K) but HOT is "all
  memories have observe-time priority ≈ 0.731" → cascade may not
  find gold answers in HOT alone, falls through to WARM/COLD.
- Test phase: cascade scans HOT (200 most-recently-recalled
  memories — which now intersects gold IDs by construction).
  Cascade short-circuits at HOT for most queries → faster.

If this works, the architecture is paying out exactly as documented:
expensive cognition runs async (in Rebalance), recall hits a cheap
warm cache.

Compare against ``raw_stack`` baseline (same query workload, no tier
system, no rebalance) to confirm we beat or match the flat-cosine
floor on both axes.

Usage::

    python -m bench.bench_rebalance_lift \\
        --embedder local \\
        --scale-n 20000 \\
        --out bench/results/rebalance_lift_20k_local.json
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

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.raw_stack_backend import RawStackBackend
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)
from mnemoss import FormulaParams, TierCapacityParams


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


async def _score_pass(
    backend,
    gold_queries: list[dict],
    dia_to_mid: dict[str, str],
    *,
    k: int,
    label: str,
    reconsolidate_hint: bool,
) -> dict:
    """One scoring pass. The ``reconsolidate_hint`` is informational —
    Mnemoss's MnemossBackend always passes ``reconsolidate=False`` to
    its own ``recall()`` call to keep the bench harness deterministic
    across hundreds of queries (see the docstring on
    ``MnemossBackend.recall``). To force reconsolidation we use
    ``mem.recall(..., reconsolidate=True)`` directly via the
    backend's underlying client when needed.
    """

    print(f"[{label}] scoring {len(gold_queries)} queries...", flush=True)
    latencies_ms: list[float] = []
    recall_hits: list[float] = []
    n_scored = 0
    for q in gold_queries:
        gold_ids = {dia_to_mid[d] for d in q["relevant_dia_ids"] if d in dia_to_mid}
        if not gold_ids:
            continue
        t0 = time.perf_counter()
        # Direct recall — bypassing MnemossBackend.recall's
        # reconsolidate=False default when we need to bump
        # access_history during the warm-up pass.
        if reconsolidate_hint and isinstance(backend, MnemossBackend):
            results = await backend._mem.recall(
                q["question"], k=k,
                include_deep=backend._include_deep,
                auto_expand=False,
                reconsolidate=True,
            )
            returned = {r.memory.id for r in results}
        else:
            hits = await backend.recall(q["question"], k=k)
            returned = {h.memory_id for h in hits}
        latencies_ms.append((time.perf_counter() - t0) * 1000)
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
    *,
    embedder_choice: str,
    scale_n: int,
    k: int,
    tier_capacity: TierCapacityParams,
    reconsolidate_min_cosine: float,
    cascade_min_cosine: float,
) -> dict:
    """Three-phase mnemoss arm: ingest → warm-up queries → rebalance → test queries."""

    embedder = _resolve_embedder(embedder_choice)
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    backend = MnemossBackend(
        embedding_model=embedder,
        include_deep=True,
        tier_capacity=tier_capacity,
        formula=FormulaParams(
            noise_scale=0.0,
            reconsolidate_min_cosine=reconsolidate_min_cosine,
            cascade_min_cosine=cascade_min_cosine,
        ),
    )
    dia_to_mid: dict[str, str] = {}
    try:
        # Phase 1: Ingest.
        ingest_seconds = await _ingest(backend, padded_mems, dia_to_mid, "mnemoss")
        db_path = backend._tempdir / "workspaces" / "bench" / "memory.sqlite"
        assert db_path.exists()

        pre_tiers = _snapshot_tier_distribution(db_path)
        print(f"[mnemoss] pre-warmup tier dist: {pre_tiers}", flush=True)

        # Phase 2: Warm-up queries (reconsolidation on, bumps access_history).
        warm_score = await _score_pass(
            backend, gold_queries, dia_to_mid,
            k=k, label="mnemoss/warmup", reconsolidate_hint=True,
        )

        warm_tiers = _snapshot_tier_distribution(db_path)
        print(f"[mnemoss] post-warmup tier dist: {warm_tiers}", flush=True)

        # Phase 3: Rebalance — recompute idx_priority from bumped histories.
        print("[mnemoss] running rebalance()...", flush=True)
        t0 = time.perf_counter()
        rebalance_stats = await backend._mem.rebalance()
        rebalance_seconds = time.perf_counter() - t0
        print(
            f"[mnemoss]   rebalance done in {rebalance_seconds:.1f}s, "
            f"scanned={rebalance_stats.scanned}, migrated={rebalance_stats.migrated}",
            flush=True,
        )

        post_tiers = _snapshot_tier_distribution(db_path)
        print(f"[mnemoss] post-rebalance tier dist: {post_tiers}", flush=True)

        # Phase 4: Test queries (reconsolidation off, pure post-rebalance perf).
        test_score = await _score_pass(
            backend, gold_queries, dia_to_mid,
            k=k, label="mnemoss/test", reconsolidate_hint=False,
        )
    finally:
        await backend.close()

    return {
        "ingest_seconds": round(ingest_seconds, 2),
        "rebalance_seconds": round(rebalance_seconds, 2),
        "rebalance_scanned": rebalance_stats.scanned,
        "rebalance_migrated": rebalance_stats.migrated,
        "tier_distribution": {
            "pre_warmup": pre_tiers,
            "post_warmup": warm_tiers,
            "post_rebalance": post_tiers,
        },
        "warmup_phase": warm_score,
        "test_phase": test_score,
        "lift": {
            "recall_delta": round(
                test_score["recall_at_k"] - warm_score["recall_at_k"], 4
            ),
            "latency_p50_delta_ms": round(
                test_score["latency_ms_p50"] - warm_score["latency_ms_p50"], 2
            ),
            "latency_p95_delta_ms": round(
                test_score["latency_ms_p95"] - warm_score["latency_ms_p95"], 2
            ),
        },
    }


async def _run_raw_stack_arm(
    *,
    embedder_choice: str,
    scale_n: int,
    k: int,
) -> dict:
    """Baseline: raw_stack runs both phases identically — no tier
    machinery, no rebalance — so the two phases should produce the
    same numbers (up to noise). Captures the flat-cosine floor."""

    embedder = _resolve_embedder(embedder_choice)
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories, queries, gold_conversation_id="conv-26", scale_n=scale_n
    )

    backend = RawStackBackend(embedding_model=embedder)
    dia_to_mid: dict[str, str] = {}
    try:
        ingest_seconds = await _ingest(
            backend, padded_mems, dia_to_mid, "raw_stack"
        )
        # raw_stack doesn't do reconsolidation, so both passes are identical.
        warm = await _score_pass(
            backend, gold_queries, dia_to_mid,
            k=k, label="raw_stack/warmup", reconsolidate_hint=False,
        )
        test = await _score_pass(
            backend, gold_queries, dia_to_mid,
            k=k, label="raw_stack/test", reconsolidate_hint=False,
        )
    finally:
        await backend.close()

    return {
        "ingest_seconds": round(ingest_seconds, 2),
        "warmup_phase": warm,
        "test_phase": test,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "gemma", "nomic", "fake"],
        default="local",
    )
    p.add_argument("--scale-n", type=int, default=20_000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--hot-cap", type=int, default=200,
                   help="Tier capacity for HOT (default 200).")
    p.add_argument("--warm-cap", type=int, default=2_000,
                   help="Tier capacity for WARM (default 2000).")
    p.add_argument("--cold-cap", type=int, default=20_000,
                   help="Tier capacity for COLD (default 20000).")
    p.add_argument(
        "--reconsolidate-min-cosine", type=float, default=0.5,
        help="Cosine threshold for the reconsolidation gate (default 0.5). "
        "Set to -1.0 to disable.",
    )
    p.add_argument(
        "--cascade-min-cosine", type=float, default=0.5,
        help="Cosine threshold for cascade early-stop (default 0.5). "
        "Set above 1.0 to effectively disable early-stop — cascade then "
        "scans every populated tier regardless of HOT confidence.",
    )
    p.add_argument(
        "--skip-raw-stack", action="store_true",
        help="Skip the raw_stack baseline (saves ingest time).",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    tier_capacity = TierCapacityParams(
        hot_cap=args.hot_cap, warm_cap=args.warm_cap, cold_cap=args.cold_cap,
    )

    started = datetime.now(timezone.utc)
    print(
        f"\n=== Rebalance lift bench: N={args.scale_n}, "
        f"caps={args.hot_cap}/{args.warm_cap}/{args.cold_cap}, "
        f"embedder={args.embedder} ===",
        flush=True,
    )

    mnemoss_result = asyncio.run(_run_mnemoss_arm(
        embedder_choice=args.embedder,
        scale_n=args.scale_n,
        k=args.k,
        tier_capacity=tier_capacity,
        reconsolidate_min_cosine=args.reconsolidate_min_cosine,
        cascade_min_cosine=args.cascade_min_cosine,
    ))

    raw_stack_result: dict | None = None
    if not args.skip_raw_stack:
        raw_stack_result = asyncio.run(_run_raw_stack_arm(
            embedder_choice=args.embedder,
            scale_n=args.scale_n,
            k=args.k,
        ))

    out = {
        "chart": "rebalance_lift",
        "embedder": args.embedder,
        "scale_n": args.scale_n,
        "k": args.k,
        "tier_capacity": {
            "hot_cap": args.hot_cap,
            "warm_cap": args.warm_cap,
            "cold_cap": args.cold_cap,
        },
        "timestamp": started.isoformat(),
        "mnemoss": mnemoss_result,
        "raw_stack": raw_stack_result,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)

    # Summary table.
    print()
    print("─" * 72)
    print("MNEMOSS — warm-up vs post-rebalance")
    print("─" * 72)
    warm = mnemoss_result["warmup_phase"]
    test = mnemoss_result["test_phase"]
    print(f"  {'phase':>20}  {'recall@k':>9}  {'p50':>7}  {'p95':>7}  {'p99':>7}")
    print(f"  {'warmup (pre-rebal)':>20}  "
          f"{warm['recall_at_k']:>9.4f}  "
          f"{warm['latency_ms_p50']:>6.1f}m  "
          f"{warm['latency_ms_p95']:>6.1f}m  "
          f"{warm['latency_ms_p99']:>6.1f}m")
    print(f"  {'test (post-rebal)':>20}  "
          f"{test['recall_at_k']:>9.4f}  "
          f"{test['latency_ms_p50']:>6.1f}m  "
          f"{test['latency_ms_p95']:>6.1f}m  "
          f"{test['latency_ms_p99']:>6.1f}m")
    lift = mnemoss_result["lift"]
    print(f"  {'lift (test - warm)':>20}  "
          f"{lift['recall_delta']:+9.4f}  "
          f"{lift['latency_p50_delta_ms']:+6.1f}m  "
          f"{lift['latency_p95_delta_ms']:+6.1f}m")

    if raw_stack_result is not None:
        print()
        print("─" * 72)
        print("RAW_STACK baseline (no tiers, no rebalance)")
        print("─" * 72)
        rwarm = raw_stack_result["warmup_phase"]
        rtest = raw_stack_result["test_phase"]
        print(f"  {'warmup':>20}  "
              f"{rwarm['recall_at_k']:>9.4f}  "
              f"{rwarm['latency_ms_p50']:>6.1f}m  "
              f"{rwarm['latency_ms_p95']:>6.1f}m")
        print(f"  {'test':>20}  "
              f"{rtest['recall_at_k']:>9.4f}  "
              f"{rtest['latency_ms_p50']:>6.1f}m  "
              f"{rtest['latency_ms_p95']:>6.1f}m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
