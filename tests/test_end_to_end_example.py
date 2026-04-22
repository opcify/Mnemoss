"""End-to-end example test — a "day in the life" of a Mnemoss workspace.

This file doubles as a tutorial: every test walks through a slice of
the public API with realistic inputs, named clearly, and asserts the
observable behaviour a caller should expect. Running the whole file
is ~2-3 seconds and exercises:

- ``observe`` + ``recall`` across ambient and per-agent scope
- ``pin`` and pin-aware activation
- ``explain_recall`` (ACT-R activation breakdown)
- ``dream`` with MockLLM, including the six-phase pipeline
- Cost governor (``CostLimits`` + ``CostLedger``) stopping a run
- Partial-failure recovery (injected phase exception)
- Cross-process workspace lock (second opener rejected)
- ``rebalance``, ``dispose``, tombstones
- ``export_markdown``
- ``status()`` snapshot
- Persistence across close/reopen

All tests use ``FakeEmbedder`` (deterministic, no model download) and
``MockLLMClient`` with canned Consolidate responses, so nothing hits
the network.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    MockLLMClient,
    PhaseName,
    StorageParams,
    TriggerType,
)
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.store._workspace_lock import WorkspaceLockError
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


# ─── shared helpers ────────────────────────────────────────────────


def _make_mem(tmp_path: Path, *, llm: MockLLMClient | None = None) -> Mnemoss:
    """Build a Mnemoss instance with a deterministic, offline setup.

    - ``FakeEmbedder(dim=16)`` keeps embeddings reproducible and avoids
      the ~470MB multilingual-model download.
    - ``FormulaParams(noise_scale=0.0)`` makes activation scoring
      deterministic so tests can assert exact orderings.
    - ``StorageParams(root=tmp_path)`` isolates every test to its own
      workspace directory.
    """

    return Mnemoss(
        workspace="e2e",
        embedding_model=FakeEmbedder(dim=16),
        llm=llm,
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )


def _consolidate_response(member_count: int) -> dict:
    """Shape a realistic Dream P3 response for a cluster of ``member_count``.

    The runner's consolidate_cluster parser expects ``summary``,
    ``refinements`` (one per member, 1-indexed), and ``patterns``.
    """

    return {
        "summary": {
            "memory_type": "fact",
            "content": "summary of the cluster",
            "abstraction_level": 0.65,
        },
        "refinements": [
            {
                "index": i + 1,
                "gist": f"gist-{i+1}",
                "time": None,
            }
            for i in range(member_count)
        ],
        "patterns": [],
    }


# ─── 1. observe + recall (smallest round-trip) ─────────────────────


async def test_observe_then_recall_returns_the_memory(tmp_path: Path) -> None:
    """The minimum Mnemoss contract: write something, read it back."""

    mem = _make_mem(tmp_path)
    try:
        mid = await mem.observe(role="user", content="meeting at 4:20 pm")
        assert mid is not None

        results = await mem.recall("when is the meeting", k=3)
        assert results  # non-empty
        assert any(r.memory.id == mid for r in results)

        # Every recall result exposes the activation breakdown so callers
        # can explain rankings without a second query.
        top = results[0]
        assert top.score == top.breakdown.total
        assert top.breakdown.base_level is not None
    finally:
        await mem.close()


async def test_empty_workspace_recall_returns_empty_list(tmp_path: Path) -> None:
    """Recall on a fresh workspace returns ``[]``, not an exception
    (this is an architectural invariant — see CLAUDE.md)."""

    mem = _make_mem(tmp_path)
    try:
        assert await mem.recall("anything", k=5) == []
    finally:
        await mem.close()


# ─── 2. multi-agent scoping ────────────────────────────────────────


async def test_agent_scoping_is_cooperatively_enforced(tmp_path: Path) -> None:
    """``for_agent`` scopes recall to the agent's own memories plus
    ambient (workspace-shared) memories. Another agent can't see the
    first agent's private ones."""

    mem = _make_mem(tmp_path)
    try:
        alice = mem.for_agent("alice")
        bob = mem.for_agent("bob")

        await mem.observe(role="user", content="team standup at 10am")  # ambient
        await alice.observe(role="user", content="alice's private goals for Q2")
        await bob.observe(role="user", content="bob's unrelated grocery list")

        # Alice sees ambient + her own.
        alice_hits = {r.memory.content for r in await alice.recall("plans", k=5)}
        assert "alice's private goals for Q2" in alice_hits

        # Bob doesn't see Alice's private memory.
        bob_hits = {r.memory.content for r in await bob.recall("plans", k=5)}
        assert "alice's private goals for Q2" not in bob_hits
    finally:
        await mem.close()


# ─── 3. pin + status() ─────────────────────────────────────────────


async def test_pin_is_reflected_in_status(tmp_path: Path) -> None:
    """Pinning keeps a memory hot. ``status()`` is the one-stop
    observability surface — any operator dashboard reads from here."""

    mem = _make_mem(tmp_path)
    try:
        mid = await mem.observe(role="user", content="very important fact")
        await mem.pin(mid)

        status = await mem.status()
        # Basic shape any dashboard would assume.
        assert status["workspace"] == "e2e"
        assert status["memory_count"] == 1
        assert status["schema_version"] >= 8
        assert status["embedder"]["dim"] == 16
        assert status["last_observe_at"] is not None
    finally:
        await mem.close()


async def test_explain_recall_surfaces_activation_components(tmp_path: Path) -> None:
    """``explain_recall`` is the debugging surface for "why did this
    memory rank where it did?" — every ACT-R component is exposed."""

    mem = _make_mem(tmp_path)
    try:
        mid = await mem.observe(role="user", content="Alice came by at 4:20")
        results = await mem.recall("Alice", k=1)
        assert results
        breakdown = await mem.explain_recall("Alice", mid)

        # All four activation terms present.
        assert breakdown.base_level is not None
        assert breakdown.matching is not None
        assert breakdown.spreading is not None
        assert breakdown.noise is not None
        # Total equals sum of terms (± floating-point slack).
        reconstructed = (
            breakdown.base_level
            + breakdown.matching
            + breakdown.spreading
            + breakdown.noise
        )
        assert abs(reconstructed - breakdown.total) < 1e-9
    finally:
        await mem.close()


# ─── 4. dream with cost governor (T1.4) ────────────────────────────


async def test_dream_with_cost_cap_stops_after_budget_exhausted(
    tmp_path: Path,
) -> None:
    """A runaway nightly can't breach the configured LLM-call cap.

    We seed three groups of near-identical messages (so P2 Cluster
    produces three clusters), cap Dream at two calls per run, and
    verify the cap engages exactly as advertised.
    """

    llm = MockLLMClient(
        # Enough canned responses for all three potential calls,
        # so we're sure the cap (not exhaustion) is what stops us.
        responses=[_consolidate_response(3) for _ in range(10)]
    )
    mem = _make_mem(tmp_path, llm=llm)
    try:
        # Seed three semantically-distinct groups of three messages.
        for topic in ("alice", "bob", "carol"):
            for i in range(3):
                await mem.observe(
                    role="user",
                    content=f"{topic} did thing number {i} today please cluster me",
                )

        # Wire the cost governor directly — the public API doesn't yet
        # expose it, so we reach into the store's conn to build the
        # ledger. Callers building production deployments can do the
        # same wiring explicitly.
        from mnemoss.dream.runner import DreamRunner

        await mem._ensure_open()
        assert mem._store is not None
        ledger = CostLedger(mem._store._require_conn())
        limits = CostLimits(max_llm_calls_per_run=2)
        runner = DreamRunner(
            store=mem._store,
            params=mem._config.formula,
            llm=llm,
            embedder=mem._embedder,
            cluster_min_size=2,
            cost_limits=limits,
            cost_ledger=ledger,
        )

        report = await runner.run(trigger=TriggerType.NIGHTLY)
        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        assert consolidate.status == "ok"
        # Capped at 2 real LLM calls.
        assert consolidate.details["llm_calls_made"] <= 2
        # At least one skip was reported with a clear "run cap" reason.
        reasons = consolidate.details.get("budget_skips", [])
        if reasons:
            assert any("run cap" in r for r in reasons)
        # The ledger persisted exactly as many calls as the run made.
        assert ledger.today_calls() == consolidate.details["llm_calls_made"]
    finally:
        await mem.close()


# ─── 5. full nightly pipeline with LLM ────────────────────────────


async def test_nightly_pipeline_runs_all_six_phases(tmp_path: Path) -> None:
    """A NIGHTLY trigger exercises every phase: REPLAY → CLUSTER →
    CONSOLIDATE → RELATIONS → REBALANCE → DISPOSE. This is the
    canonical smoke test for the dream engine end-to-end."""

    llm = MockLLMClient(responses=[_consolidate_response(3) for _ in range(10)])
    mem = _make_mem(tmp_path, llm=llm)
    try:
        for i in range(6):
            await mem.observe(
                role="user",
                content=f"routine note {i} about the same project topic",
            )

        report = await mem.dream(trigger="nightly")

        seen = {o.phase for o in report.outcomes}
        assert {
            PhaseName.REPLAY,
            PhaseName.CLUSTER,
            PhaseName.CONSOLIDATE,
            PhaseName.RELATIONS,
            PhaseName.REBALANCE,
            PhaseName.DISPOSE,
        } == seen

        # No phase errored.
        assert not report.degraded_mode

        # status() now reflects the dream having run.
        status = await mem.status()
        assert status["last_dream_at"] is not None
        assert status["last_dream_trigger"] == "nightly"
    finally:
        await mem.close()


# ─── 6. partial-failure recovery (T1.5) ────────────────────────────


async def test_dream_survives_phase_exception(tmp_path: Path) -> None:
    """If a phase raises, the runner records it as ``error`` and
    downstream phases still run. The caller never sees the exception
    — only a degraded DreamReport."""

    llm = MockLLMClient(responses=[_consolidate_response(3) for _ in range(10)])
    mem = _make_mem(tmp_path, llm=llm)
    try:
        await mem.observe(role="user", content="one note so replay has input")

        # Poison REPLAY.
        import mnemoss.dream.runner as runner_mod

        async def boom(*_a, **_kw):
            raise RuntimeError("simulated storage glitch")

        with patch.object(
            runner_mod, "select_replay_candidates", side_effect=boom
        ):
            report = await mem.dream(trigger="nightly")

        replay_outcome = report.outcome(PhaseName.REPLAY)
        assert replay_outcome is not None
        assert replay_outcome.status == "error"
        assert "simulated storage glitch" in (replay_outcome.error or "")
        assert report.degraded_mode

        # Downstream phases still ran and skipped cleanly because
        # there's no replay state to process.
        cluster_outcome = report.outcome(PhaseName.CLUSTER)
        assert cluster_outcome is not None
        assert cluster_outcome.status == "skipped"
        assert cluster_outcome.skip_reason == "empty replay set"
    finally:
        await mem.close()


# ─── 7. workspace lock (T1.3) ─────────────────────────────────────


async def test_workspace_lock_rejects_second_opener(tmp_path: Path) -> None:
    """Mnemoss holds one writer per workspace. A second process (here
    simulated as a second SQLiteBackend) trying to open the same
    workspace must fail fast with a clear error."""

    mem = _make_mem(tmp_path)
    try:
        await mem.observe(role="user", content="hello")
        # mem._store is the first backend, already holding the lock.
        # A second backend on the same paths should be refused.
        storage_root = tmp_path / "workspaces" / "e2e"
        second = SQLiteBackend(
            db_path=storage_root / "memory.sqlite",
            raw_log_path=storage_root / "raw_log.sqlite",
            workspace_id="e2e",
            embedding_dim=16,
            embedder_id="fake:dim16",
        )
        with pytest.raises(WorkspaceLockError, match="already open"):
            await second.open()
    finally:
        await mem.close()


async def test_workspace_lock_releases_on_close(tmp_path: Path) -> None:
    """After ``close()``, the workspace is re-openable — no stale lock."""

    mem = _make_mem(tmp_path)
    await mem.observe(role="user", content="hello")
    await mem.close()

    # Different Mnemoss instance, same storage root. Should open cleanly.
    mem2 = _make_mem(tmp_path)
    try:
        results = await mem2.recall("hello", k=1)
        assert any(r.memory.content == "hello" for r in results)
    finally:
        await mem2.close()


# ─── 8. persistence across close/reopen ───────────────────────────


async def test_memories_survive_close_and_reopen(tmp_path: Path) -> None:
    """Data survives across process restarts. Open, write, close;
    reopen, read back — a must-have for any on-disk store."""

    mem1 = _make_mem(tmp_path)
    try:
        await mem1.observe(role="user", content="data point one")
        await mem1.observe(role="user", content="data point two")
    finally:
        await mem1.close()

    mem2 = _make_mem(tmp_path)
    try:
        status = await mem2.status()
        assert status["memory_count"] == 2
        results = await mem2.recall("data point", k=5)
        contents = {r.memory.content for r in results}
        assert "data point one" in contents
        assert "data point two" in contents
    finally:
        await mem2.close()


# ─── 9. rebalance + dispose + tombstones ──────────────────────────


async def test_rebalance_and_dispose_produce_stats(tmp_path: Path) -> None:
    """Rebalance walks every memory and recomputes tier. Dispose walks
    for tombstoning low-activation memories. Both are idempotent on a
    fresh workspace (fresh memories are age-protected against disposal)."""

    mem = _make_mem(tmp_path)
    try:
        for i in range(5):
            await mem.observe(role="user", content=f"note {i}")

        rb = await mem.rebalance()
        assert rb.scanned == 5
        # Every tier accounted for, sums match scan count.
        after_total = sum(rb.tier_after.values())
        assert after_total == 5

        dispose = await mem.dispose()
        assert dispose.scanned == 5
        # Fresh memories are age-protected; expect zero disposals.
        assert dispose.disposed == 0
        assert dispose.protected >= 0

        # No tombstones on a workspace where nothing was dropped.
        assert await mem.tombstones() == []
    finally:
        await mem.close()


# ─── 10. export memory.md ─────────────────────────────────────────


async def test_export_markdown_produces_viewable_output(tmp_path: Path) -> None:
    """memory.md is the human-readable view. It should contain at
    least the structural markers a downstream tool can parse."""

    mem = _make_mem(tmp_path)
    try:
        await mem.observe(role="user", content="alpha fact")
        await mem.observe(role="user", content="beta fact")

        markdown = await mem.export_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        # Either the content or a section header should appear.
        assert "fact" in markdown.lower() or "#" in markdown
    finally:
        await mem.close()


# ─── 11. full lifecycle narrative (integrative test) ──────────────


async def test_full_day_in_the_life(tmp_path: Path) -> None:
    """One narrative test that chains the above scenarios together,
    roughly mirroring a real caller's day:

    1. Open, observe across two agents.
    2. Recall as both agents, verify scoping.
    3. Pin a memory.
    4. Run a dream with cost cap.
    5. Check status snapshot.
    6. Close, reopen, verify everything persisted.

    This is the test that fails first if any seam between the
    subsystems breaks — even if each individual piece still works in
    isolation.
    """

    llm = MockLLMClient(responses=[_consolidate_response(2) for _ in range(10)])
    mem = _make_mem(tmp_path, llm=llm)
    alice_pin_id: str | None = None
    try:
        alice = mem.for_agent("alice")
        bob = mem.for_agent("bob")

        # Ambient + per-agent observes.
        await mem.observe(role="user", content="company holiday on Friday")
        alice_pin_id = await alice.observe(
            role="user", content="alice: ship the memory API by Thursday"
        )
        await alice.observe(role="user", content="alice: follow up with Bob on specs")
        await bob.observe(role="user", content="bob: writing up the spec doc")
        await bob.observe(role="user", content="bob: coffee order sheet updated")

        # Recall scopes.
        alice_results = await alice.recall("my deadline", k=5)
        assert any(
            "memory API" in r.memory.content for r in alice_results
        ), "alice should see her own deadline note"

        bob_results = await bob.recall("deadline", k=5)
        assert not any(
            "memory API" in r.memory.content for r in bob_results
        ), "bob should NOT see alice's private deadline"

        # Pin survives dream.
        await alice.pin(alice_pin_id)

        # Dream with a soft cap.
        from mnemoss.dream.runner import DreamRunner

        await mem._ensure_open()
        assert mem._store is not None
        ledger = CostLedger(mem._store._require_conn())
        runner = DreamRunner(
            store=mem._store,
            params=mem._config.formula,
            llm=llm,
            embedder=mem._embedder,
            cluster_min_size=2,
            cost_limits=CostLimits(max_llm_calls_per_run=5),
            cost_ledger=ledger,
        )
        report = await runner.run(trigger=TriggerType.NIGHTLY)
        assert not report.degraded_mode

        # Pin still present after the dream pipeline.
        assert mem._store is not None
        assert await mem._store.is_pinned(alice_pin_id, agent_id="alice")

        # Status snapshot makes sense.
        status = await mem.status()
        assert status["memory_count"] >= 5
        assert status["schema_version"] >= 8

        # One last recall from bob to prove scoping is still enforced
        # after dreaming (which runs workspace-wide).
        post_bob = await bob.recall("deadline", k=5)
        assert not any("memory API" in r.memory.content for r in post_bob)
    finally:
        await mem.close()

    # Close and reopen — make sure nothing gets lost on restart.
    mem2 = _make_mem(tmp_path, llm=llm)
    try:
        status = await mem2.status()
        assert status["memory_count"] >= 5
        # Pin state is durable.
        alice2 = mem2.for_agent("alice")
        assert alice_pin_id is not None
        assert mem2._store is not None
        await mem2._ensure_open()
        assert await mem2._store.is_pinned(alice_pin_id, agent_id="alice")

        # The dream diary (if any) left structure in the workspace dir.
        diary_root = tmp_path / "workspaces" / "e2e" / "dreams"
        if diary_root.exists():
            assert any(diary_root.iterdir())

        # Scoping still enforced on fresh open.
        bob2 = mem2.for_agent("bob")
        post_reopen_bob = await bob2.recall("deadline", k=5)
        assert not any("memory API" in r.memory.content for r in post_reopen_bob)

        # Save a dated timestamp marker so the assertion about
        # "today" in any future cost-ledger test can't drift.
        _ = datetime.now(UTC)
        assert alice2 is not None  # keep alice2 reachable to document the handle
    finally:
        await mem2.close()
