"""Hot-path budget enforcement.

Mnemoss pins two invariants on ``observe()``:

1. **Zero LLM calls.** Encoding is rule-based only — the LLM belongs
   to Dream's cold path. This is principle #4 in CLAUDE.md. A
   regression here (e.g. someone accidentally wiring an LLM-driven
   NER pass into event segmentation) would blow up latency AND cost.

2. **<50ms per call.** Documented target; the test pins a looser
   budget so CI jitter doesn't flake us, but still small enough to
   catch obvious regressions (e.g. accidental full-table scans,
   synchronous I/O on a loop, or embedder batching gone wrong).

Tests here use ``FakeEmbedder`` so we measure the hot-path itself
rather than model inference. Real deployments with
``LocalEmbedder`` will be slower per call (~5-20ms on CPU for
MiniLM) but the shape of the budget still holds.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    MockLLMClient,
    StorageParams,
)


def _make_mem(
    tmp_path: Path, *, llm: MockLLMClient | None = None
) -> Mnemoss:
    return Mnemoss(
        workspace="hp",
        embedding_model=FakeEmbedder(dim=16),
        llm=llm,
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )


# ─── zero LLM calls in the encode path ───────────────────────────


async def test_observe_makes_zero_llm_calls(tmp_path: Path) -> None:
    """Principle 4: the encoding hot path MUST NOT call the LLM.

    Mnemoss is constructed with a MockLLMClient so that, if any
    hot-path code path accidentally reaches for the LLM, the mock
    will record it on ``client.calls`` and we'll see it here.
    """

    llm = MockLLMClient(responses=[])  # no canned responses
    mem = _make_mem(tmp_path, llm=llm)
    try:
        for i in range(50):
            await mem.observe(role="user", content=f"routine observation {i}")

        # The LLM mock raises on any call, so reaching this point
        # already proves zero calls. The explicit assert documents it.
        assert llm.calls == [], (
            f"observe() made {len(llm.calls)} LLM call(s) — hot path "
            "must be LLM-free. Calls: "
            f"{[method for method, _ in llm.calls[:5]]}..."
        )
    finally:
        await mem.close()


async def test_observe_works_when_no_llm_configured(tmp_path: Path) -> None:
    """Zero LLM calls also means: observe() should work fine when no
    LLM client is configured at all. This is the default for users
    who don't plan to dream."""

    mem = _make_mem(tmp_path, llm=None)
    try:
        mid = await mem.observe(role="user", content="observed without llm")
        assert mid is not None
        results = await mem.recall("observed", k=3)
        assert any(r.memory.id == mid for r in results)
    finally:
        await mem.close()


# ─── latency budget ───────────────────────────────────────────────


# Budget is generous to absorb CI machine variance while still
# catching real regressions. Local dev machines comfortably hit
# single-digit ms with FakeEmbedder.
_HOT_PATH_P99_BUDGET_MS = 50.0
_HOT_PATH_MEDIAN_BUDGET_MS = 10.0


async def test_observe_p99_under_budget(tmp_path: Path) -> None:
    """200 observes, measure per-call wall time, assert p99 under the
    documented budget. Median is checked separately because p99 in a
    200-sample run is noisy — median is the stable signal."""

    mem = _make_mem(tmp_path)
    try:
        # Warm up: the first observe pays one-time costs (schema
        # pragma check, thread-pool spin-up) that we don't want to
        # include in the budget accounting.
        await mem.observe(role="user", content="warmup message")

        latencies_ms: list[float] = []
        for i in range(200):
            t0 = time.perf_counter()
            await mem.observe(role="user", content=f"measurable observation {i}")
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        latencies_ms.sort()
        median = statistics.median(latencies_ms)
        p99 = latencies_ms[int(len(latencies_ms) * 0.99) - 1]

        assert median < _HOT_PATH_MEDIAN_BUDGET_MS, (
            f"observe() median latency {median:.2f}ms exceeds budget "
            f"{_HOT_PATH_MEDIAN_BUDGET_MS}ms — hot path regressed?"
        )
        assert p99 < _HOT_PATH_P99_BUDGET_MS, (
            f"observe() p99 latency {p99:.2f}ms exceeds budget "
            f"{_HOT_PATH_P99_BUDGET_MS}ms — hot path regressed?"
        )
    finally:
        await mem.close()


async def test_observe_latency_does_not_grow_with_workspace_size(
    tmp_path: Path,
) -> None:
    """The hot path is ``O(1)`` with respect to workspace size — it
    appends a Memory row and updates indexes incrementally. An
    accidental ``SELECT * FROM memory`` would turn this into ``O(n)``
    and that's exactly what we want to catch early."""

    mem = _make_mem(tmp_path)
    try:
        # Warm up.
        for i in range(20):
            await mem.observe(role="user", content=f"warmup {i}")

        # Measure an "early" batch of 50 observes.
        early: list[float] = []
        for i in range(50):
            t0 = time.perf_counter()
            await mem.observe(role="user", content=f"early batch {i}")
            early.append((time.perf_counter() - t0) * 1000.0)

        # Fill the workspace with another 400 memories to grow n.
        for i in range(400):
            await mem.observe(role="user", content=f"filler {i}")

        # Measure a "late" batch of 50 observes — same shape, bigger
        # workspace behind them.
        late: list[float] = []
        for i in range(50):
            t0 = time.perf_counter()
            await mem.observe(role="user", content=f"late batch {i}")
            late.append((time.perf_counter() - t0) * 1000.0)

        early_median = statistics.median(early)
        late_median = statistics.median(late)

        # A 3× growth is very generous; a true O(n) regression would
        # show much worse.  If someone genuinely makes observes
        # depend on table size, this will trip well before prod pain.
        assert late_median < max(early_median * 3.0, 5.0), (
            f"observe() latency grew too much with workspace size "
            f"(early median {early_median:.2f}ms → late {late_median:.2f}ms)"
        )
    finally:
        await mem.close()


# ─── defensive: no accidental network dep ────────────────────────


@pytest.mark.parametrize("content", ["simple", "with Alice at 4:20", "一句中文"])
async def test_observe_is_sync_bound(tmp_path: Path, content: str) -> None:
    """Even under varied content (ASCII, mixed, CJK), observe() must
    stay under the per-call budget. A regression that made segmentation
    or extraction do synchronous I/O on certain inputs would show up
    here as an outlier."""

    mem = _make_mem(tmp_path)
    try:
        # Warmup.
        await mem.observe(role="user", content="warmup")

        t0 = time.perf_counter()
        await mem.observe(role="user", content=content)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < _HOT_PATH_P99_BUDGET_MS, (
            f"observe({content!r}) took {elapsed_ms:.2f}ms > "
            f"{_HOT_PATH_P99_BUDGET_MS}ms budget"
        )
    finally:
        await mem.close()
