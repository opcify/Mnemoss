"""Recall latency benchmark harness.

Measures ``Mnemoss.recall()`` wall time across a sweep of workspace
sizes. Uses ``FakeEmbedder`` so results are reproducible and never
touch the network — what this benchmark surfaces is the storage +
cascade + scoring path, not model inference.

Usage::

    python -m bench.bench_recall                      # default sizes
    python -m bench.bench_recall --sizes 100 1000 5000
    python -m bench.bench_recall --queries 50 --sizes 10000

Last line of stdout is a single JSON blob so external dashboards can
parse the results without re-invoking Python.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    StorageParams,
)

# Query terms that intentionally overlap with the seeded content so
# the cascade actually has work to do (not a "zero FTS hits" trivial case).
_QUERIES = [
    "alice meeting today",
    "bob coffee order",
    "carol deadline project",
    "deployment schedule friday",
    "kickoff review next week",
]


@dataclass
class BenchResult:
    """One ``(size, tier_regime)`` row of the output table."""

    workspace_size: int
    regime: str
    queries: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float


async def _seed_workspace(mem: Mnemoss, size: int) -> None:
    """Populate a workspace with ``size`` memories that will land
    across all four tiers once rebalanced.

    The content deliberately rotates through names + nouns that
    overlap with ``_QUERIES`` so every recall hits at least some
    candidates — we want to measure scoring + cascade, not a
    degenerate zero-hit fast path.
    """

    names = ("alice", "bob", "carol", "dave", "eve")
    topics = ("meeting", "coffee", "deadline", "project", "review", "kickoff")
    verbs = ("scheduled", "mentioned", "asked about", "followed up on")
    for i in range(size):
        name = names[i % len(names)]
        topic = topics[i % len(topics)]
        verb = verbs[i % len(verbs)]
        # Index number at the end gives FTS enough tokens to
        # discriminate between memories of the same shape.
        await mem.observe(
            role="user",
            content=f"{name} {verb} the {topic} item {i}",
        )


async def _measure_recall(
    mem: Mnemoss, queries: int, *, include_deep: bool
) -> list[float]:
    """Run ``queries`` recalls, return per-call latencies in ms."""

    latencies: list[float] = []
    for i in range(queries):
        q = _QUERIES[i % len(_QUERIES)]
        t0 = time.perf_counter()
        await mem.recall(q, k=10, include_deep=include_deep)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return latencies


def _summarize(
    size: int, regime: str, latencies: list[float]
) -> BenchResult:
    latencies = sorted(latencies)
    n = len(latencies)

    def pct(p: float) -> float:
        if n == 0:
            return 0.0
        idx = max(0, min(n - 1, int(round(p * n)) - 1))
        return latencies[idx]

    return BenchResult(
        workspace_size=size,
        regime=regime,
        queries=n,
        p50_ms=pct(0.50),
        p90_ms=pct(0.90),
        p99_ms=pct(0.99),
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
    )


async def _run_one(size: int, queries: int) -> list[BenchResult]:
    """Run the full sweep for one workspace size and return a result
    row per tier regime."""

    results: list[BenchResult] = []
    with tempfile.TemporaryDirectory() as td:
        mem = Mnemoss(
            workspace=f"bench_{size}",
            embedding_model=FakeEmbedder(dim=64),
            formula=FormulaParams(noise_scale=0.0),
            storage=StorageParams(root=Path(td)),
        )
        try:
            # Seed + rebalance so tiers are distributed (otherwise
            # everything stays HOT and the cascade degenerates to one
            # tier, which isn't representative of a real workspace).
            await _seed_workspace(mem, size)
            await mem.rebalance()

            # Warm up once so JIT-ish costs (compiled SQL prep,
            # thread-pool spin-up) don't skew the first measurement.
            await mem.recall(_QUERIES[0], k=10)

            # Default cascade (HOT → WARM → COLD, no DEEP).
            lat_default = await _measure_recall(
                mem, queries, include_deep=False
            )
            results.append(_summarize(size, "cascade (no DEEP)", lat_default))

            # DEEP-inclusive cascade — what you get when a query has a
            # temporal "long ago" cue or the caller flags include_deep.
            lat_deep = await _measure_recall(
                mem, queries, include_deep=True
            )
            results.append(_summarize(size, "cascade (+DEEP)", lat_deep))
        finally:
            await mem.close()
    return results


def _print_table(rows: list[BenchResult]) -> None:
    """Pretty-print results as a fixed-width ASCII table."""

    header = (
        f"{'size':>8}  {'regime':<20}  {'queries':>7}  "
        f"{'p50 ms':>9}  {'p90 ms':>9}  {'p99 ms':>9}  {'mean ms':>9}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r.workspace_size:>8}  {r.regime:<20}  {r.queries:>7}  "
            f"{r.p50_ms:>9.2f}  {r.p90_ms:>9.2f}  "
            f"{r.p99_ms:>9.2f}  {r.mean_ms:>9.2f}"
        )
    print(sep)


async def _run_all(sizes: list[int], queries: int) -> list[BenchResult]:
    all_results: list[BenchResult] = []
    for size in sizes:
        all_results.extend(await _run_one(size, queries))
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mnemoss recall latency benchmark harness."
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 500, 2000],
        help="Workspace sizes to sweep. Larger sizes take longer to seed.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=30,
        help="Number of recall queries to measure per (size, regime).",
    )
    args = parser.parse_args()

    results = asyncio.run(_run_all(args.sizes, args.queries))

    _print_table(results)
    # JSON dump on the last line for machine consumers.
    payload = [
        {
            "workspace_size": r.workspace_size,
            "regime": r.regime,
            "queries": r.queries,
            "p50_ms": round(r.p50_ms, 3),
            "p90_ms": round(r.p90_ms, 3),
            "p99_ms": round(r.p99_ms, 3),
            "mean_ms": round(r.mean_ms, 3),
        }
        for r in results
    ]
    print("JSON_RESULTS=" + json.dumps(payload))


if __name__ == "__main__":
    main()
