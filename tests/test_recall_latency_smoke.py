"""Recall latency smoke test.

A quick, CI-friendly subset of the full benchmark harness in
``bench/bench_recall.py``. Seeds a 200-memory workspace, runs 30
recalls, and asserts latency stays under a generous budget. Not a
perf tuning tool — its job is to catch order-of-magnitude
regressions (e.g. someone accidentally dropping an index, or an
``O(n)`` Python loop landing on the scoring path).

Full benchmarking happens in ``bench/bench_recall.py``:

    python -m bench.bench_recall --sizes 100 1000 5000 --queries 50

That sweeps multiple workspace sizes and tier regimes. The smoke
test only runs one configuration so it stays under 2 seconds.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import pytest

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    StorageParams,
)

_SMOKE_WORKSPACE_SIZE = 200
_SMOKE_QUERIES = 30

# Generous budgets — this is a "wrong order of magnitude" gate, not a
# tuning benchmark. Local dev with FakeEmbedder hits <10ms p99 easily.
# Coverage instrumentation adds ~3-5× overhead per call, so widen the
# budget when coverage is active. Running `pytest` without coverage
# still enforces the tight gate. GitHub-hosted runners are shared
# (noisy-neighbor) hardware — widen there too so jitter doesn't trip
# the gate while still catching real order-of-magnitude regressions.
_UNDER_COVERAGE = (
    "coverage" in sys.modules or os.environ.get("COVERAGE_RUN") is not None
)
_ON_CI = os.environ.get("CI") is not None
if _UNDER_COVERAGE:
    _P99_BUDGET_MS = 500.0
    _MEDIAN_BUDGET_MS = 150.0
elif _ON_CI:
    _P99_BUDGET_MS = 500.0
    _MEDIAN_BUDGET_MS = 100.0
else:
    _P99_BUDGET_MS = 150.0
    _MEDIAN_BUDGET_MS = 40.0


_QUERIES = [
    "alice meeting today",
    "bob coffee order",
    "carol deadline project",
    "deployment schedule friday",
    "kickoff review next week",
]


async def _seed(mem: Mnemoss, n: int) -> None:
    names = ("alice", "bob", "carol", "dave", "eve")
    topics = ("meeting", "coffee", "deadline", "project", "review")
    for i in range(n):
        await mem.observe(
            role="user",
            content=f"{names[i % len(names)]} discussed "
            f"{topics[i % len(topics)]} item {i}",
        )


@pytest.mark.parametrize("include_deep", [False, True])
async def test_recall_latency_smoke(
    tmp_path: Path, include_deep: bool
) -> None:
    mem = Mnemoss(
        workspace="smoke",
        embedding_model=FakeEmbedder(dim=32),
        formula=FormulaParams(noise_scale=0.0, use_tier_cascade_recall=False),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await _seed(mem, _SMOKE_WORKSPACE_SIZE)
        await mem.rebalance()

        # Warmup call so compiled SQL / thread-pool startup don't
        # skew the first measured recall.
        await mem.recall(_QUERIES[0], k=10, include_deep=include_deep)

        latencies: list[float] = []
        for i in range(_SMOKE_QUERIES):
            q = _QUERIES[i % len(_QUERIES)]
            t0 = time.perf_counter()
            await mem.recall(q, k=10, include_deep=include_deep)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        latencies.sort()
        median = statistics.median(latencies)
        p99 = latencies[int(len(latencies) * 0.99) - 1]

        assert median < _MEDIAN_BUDGET_MS, (
            f"recall median latency {median:.2f}ms (include_deep={include_deep}) "
            f"exceeds smoke budget {_MEDIAN_BUDGET_MS}ms"
        )
        assert p99 < _P99_BUDGET_MS, (
            f"recall p99 latency {p99:.2f}ms (include_deep={include_deep}) "
            f"exceeds smoke budget {_P99_BUDGET_MS}ms"
        )
    finally:
        await mem.close()
