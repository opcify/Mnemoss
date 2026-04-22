"""Dream cost-governor tests.

Covers the ``CostLedger`` persistence, the ``CostLimits`` budget check,
and the ``DreamRunner`` integration that stops consolidating when a
cap would be breached.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import apsw

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.dream.runner import DreamRunner
from mnemoss.encoder.embedder import FakeEmbedder
from mnemoss.llm.mock import MockLLMClient
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


# ─── ledger primitives ──────────────────────────────────────────────


def _fresh_ledger_conn() -> apsw.Connection:
    """Bare conn with just the keyspace ``CostLedger`` needs."""

    conn = apsw.Connection(":memory:")
    conn.execute(
        "CREATE TABLE workspace_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
    )
    return conn


def test_record_call_increments_today_and_total() -> None:
    ledger = CostLedger(_fresh_ledger_conn())
    now = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

    ledger.record_call(now=now)
    ledger.record_call(now=now)
    ledger.record_call(now=now)

    snap = ledger.snapshot(now=now)
    assert snap.today_calls == 3
    assert snap.month_calls == 3
    assert snap.total_calls == 3


def test_calls_are_bucketed_per_day() -> None:
    ledger = CostLedger(_fresh_ledger_conn())
    day1 = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)
    day2 = datetime(2026, 4, 23, 0, 30, tzinfo=UTC)

    ledger.record_call(now=day1)
    ledger.record_call(now=day1)
    ledger.record_call(now=day2)

    # Today (from day2's perspective) is only day2's one call.
    snap = ledger.snapshot(now=day2)
    assert snap.today_calls == 1
    assert snap.month_calls == 3
    assert snap.total_calls == 3


def test_month_aggregates_across_days() -> None:
    ledger = CostLedger(_fresh_ledger_conn())
    for day in range(1, 6):
        ts = datetime(2026, 4, day, 9, 0, tzinfo=UTC)
        ledger.record_call(now=ts)

    snap = ledger.snapshot(now=datetime(2026, 4, 10, 0, 0, tzinfo=UTC))
    assert snap.month_calls == 5
    # A call in a different month doesn't count for April.
    ledger.record_call(now=datetime(2026, 5, 1, 9, 0, tzinfo=UTC))
    april = ledger.snapshot(now=datetime(2026, 4, 30, 0, 0, tzinfo=UTC))
    assert april.month_calls == 5


# ─── CostLimits gating ───────────────────────────────────────────────


def test_unlimited_budget_never_fires() -> None:
    ledger = CostLedger(_fresh_ledger_conn())
    limits = CostLimits()
    assert limits.is_unlimited
    assert ledger.check_budget(limits, run_calls=1_000_000) is None


def test_run_cap_kicks_in_before_persistence() -> None:
    ledger = CostLedger(_fresh_ledger_conn())
    limits = CostLimits(max_llm_calls_per_run=2)
    now = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)

    assert ledger.check_budget(limits, run_calls=0, now=now) is None
    assert ledger.check_budget(limits, run_calls=1, now=now) is None
    # Third candidate call — run_calls is already 2, cap hit.
    reason = ledger.check_budget(limits, run_calls=2, now=now)
    assert reason is not None and "run cap" in reason


def test_daily_cap_spans_runs() -> None:
    ledger = CostLedger(_fresh_ledger_conn())
    limits = CostLimits(max_llm_calls_per_day=3)
    now = datetime(2026, 4, 22, 9, 0, tzinfo=UTC)

    # Previous run burned through the whole day's budget.
    ledger.record_call(now=now)
    ledger.record_call(now=now)
    ledger.record_call(now=now)

    # This run starts fresh (run_calls=0) but daily cap is already hit.
    reason = ledger.check_budget(limits, run_calls=0, now=now)
    assert reason is not None and "daily cap" in reason


def test_monthly_cap_fires_before_daily() -> None:
    """Precedence: run > daily > monthly. The earliest cap listed in
    ``check_budget`` wins when multiple apply."""

    ledger = CostLedger(_fresh_ledger_conn())
    limits = CostLimits(max_llm_calls_per_day=1000, max_llm_calls_per_month=2)
    now = datetime(2026, 4, 22, 9, 0, tzinfo=UTC)

    ledger.record_call(now=now)
    ledger.record_call(now=now)

    reason = ledger.check_budget(limits, run_calls=0, now=now)
    assert reason is not None and "monthly cap" in reason


# ─── DreamRunner integration ────────────────────────────────────────


async def _setup_backend(tmp_path: Path) -> tuple[SQLiteBackend, FakeEmbedder]:
    embedder = FakeEmbedder(dim=16)
    store = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=embedder.dim,
        embedder_id=embedder.embedder_id,
    )
    await store.open()
    return store, embedder


def _mem(mid: str, content: str, cluster_id: str) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=mid,
        workspace_id="ws",
        agent_id=None,
        session_id="s",
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        index_tier=IndexTier.HOT,
        cluster_id=cluster_id,
        cluster_similarity=0.9,
        is_cluster_representative=False,
    )


def _empty_response() -> dict:
    return {"summary": None, "refinements": [], "patterns": []}


async def _run_consolidate(
    runner: DreamRunner, clusters: list[list[Memory]]
) -> dict:
    """Directly drive ``_phase_consolidate`` with a hand-built state.

    Bypasses REPLAY / CLUSTER so the test controls exactly how many
    clusters reach the LLM loop. Returns the outcome details dict.
    """

    from mnemoss.dream.cluster import ClusterAssignment
    from mnemoss.dream.runner import _DreamState

    state = _DreamState()
    # Drop a flat list of cluster members into replay_set + build
    # cluster_assignments that group them.
    state.replay_set = [m for c in clusters for m in c]
    state.cluster_assignments = {}
    for ci, cluster in enumerate(clusters):
        for m in cluster:
            state.cluster_assignments[m.id] = ClusterAssignment(
                cluster_id=f"c{ci}", similarity=0.9, is_representative=False
            )

    outcome = await runner._phase_consolidate(state, datetime.now(UTC))
    return outcome.details


async def test_runner_stops_after_run_cap(tmp_path: Path) -> None:
    store, embedder = await _setup_backend(tmp_path)
    try:
        import numpy as np

        # Three clusters, two members each — gives the loop 3 candidate
        # LLM calls.
        clusters: list[list[Memory]] = []
        for i in range(3):
            a = _mem(f"m{i}a", f"content-{i}-a", cluster_id=f"c{i}")
            b = _mem(f"m{i}b", f"content-{i}-b", cluster_id=f"c{i}")
            await store.write_memory(
                a, np.array([1] + [0] * 15, dtype=np.float32)
            )
            await store.write_memory(
                b, np.array([1] + [0] * 15, dtype=np.float32)
            )
            clusters.append([a, b])

        llm = MockLLMClient(responses=[_empty_response()] * 10)
        ledger = CostLedger(store._require_conn())
        limits = CostLimits(max_llm_calls_per_run=2)
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=llm,
            embedder=embedder,
            cluster_min_size=2,
            cost_limits=limits,
            cost_ledger=ledger,
        )

        details = await _run_consolidate(runner, clusters)
        assert details["llm_calls_made"] == 2
        assert len(details["budget_skips"]) == 1
        assert "run cap" in details["budget_skips"][0]

        # Ledger persisted the two successful calls.
        assert ledger.today_calls() == 2
    finally:
        await store.close()


async def test_runner_unlimited_by_default(tmp_path: Path) -> None:
    store, embedder = await _setup_backend(tmp_path)
    try:
        import numpy as np

        clusters = []
        for i in range(2):
            a = _mem(f"m{i}a", f"c-{i}-a", cluster_id=f"c{i}")
            b = _mem(f"m{i}b", f"c-{i}-b", cluster_id=f"c{i}")
            await store.write_memory(
                a, np.array([1] + [0] * 15, dtype=np.float32)
            )
            await store.write_memory(
                b, np.array([1] + [0] * 15, dtype=np.float32)
            )
            clusters.append([a, b])

        llm = MockLLMClient(responses=[_empty_response()] * 10)
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=llm,
            embedder=embedder,
            cluster_min_size=2,
            # No cost_limits / cost_ledger — unlimited.
        )
        details = await _run_consolidate(runner, clusters)
        assert details["budget_skips"] == []
        assert details["llm_calls_made"] == 2
    finally:
        await store.close()


async def test_runner_respects_prior_daily_spend(tmp_path: Path) -> None:
    store, embedder = await _setup_backend(tmp_path)
    try:
        import numpy as np

        clusters = []
        for i in range(3):
            a = _mem(f"m{i}a", f"c-{i}-a", cluster_id=f"c{i}")
            b = _mem(f"m{i}b", f"c-{i}-b", cluster_id=f"c{i}")
            await store.write_memory(
                a, np.array([1] + [0] * 15, dtype=np.float32)
            )
            await store.write_memory(
                b, np.array([1] + [0] * 15, dtype=np.float32)
            )
            clusters.append([a, b])

        ledger = CostLedger(store._require_conn())
        # Pretend a previous run already burned the day's budget.
        for _ in range(5):
            ledger.record_call()

        limits = CostLimits(max_llm_calls_per_day=5)
        runner = DreamRunner(
            store=store,
            params=FormulaParams(),
            llm=MockLLMClient(responses=[_empty_response()] * 10),
            embedder=embedder,
            cluster_min_size=2,
            cost_limits=limits,
            cost_ledger=ledger,
        )
        details = await _run_consolidate(runner, clusters)
        assert details["llm_calls_made"] == 0
        assert any("daily cap" in s for s in details["budget_skips"])
    finally:
        await store.close()
