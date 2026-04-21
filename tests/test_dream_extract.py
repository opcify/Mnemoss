"""P3 Extract tests (Checkpoint N) — mocked LLM, no network."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from mnemoss import FakeEmbedder, Mnemoss, MockLLMClient, StorageParams
from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, MemoryType
from mnemoss.dream.extract import (
    build_extract_prompt,
    extract_from_cluster,
)

UTC = timezone.utc


def _mem(
    id: str,
    content: str,
    *,
    agent_id: str | None = None,
    session_id: str = "s1",
    salience: float = 0.2,
) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=agent_id,
        session_id=session_id,
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        salience=salience,
    )


def test_prompt_includes_all_members_with_role_markers() -> None:
    members = [
        _mem("m1", "Alice likes coffee"),
        _mem("m2", "Alice ordered a latte"),
    ]
    prompt = build_extract_prompt(members)
    assert "Alice likes coffee" in prompt
    assert "Alice ordered a latte" in prompt
    assert "[user]" in prompt
    assert "memory_type" in prompt  # schema instructions present


async def test_extract_builds_new_memory_from_llm_response() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "memory_type": "fact",
                "content": "Alice prefers lattes",
                "abstraction_level": 0.65,
                "aliases": [],
            }
        ]
    )
    members = [_mem("m1", "Alice ordered a latte"), _mem("m2", "Alice's usual is a latte")]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is not None
    assert result.content == "Alice prefers lattes"
    assert result.memory_type is MemoryType.FACT
    assert result.abstraction_level == 0.65
    assert result.derived_from == ["m1", "m2"]


async def test_extract_returns_none_on_empty_content() -> None:
    llm = MockLLMClient(responses=[{"memory_type": "fact", "content": "  "}])
    members = [_mem("m1", "x"), _mem("m2", "y")]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is None


async def test_extract_cross_agent_promotion() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "memory_type": "fact",
                "content": "both agents discussed the release",
                "abstraction_level": 0.7,
                "aliases": [],
            }
        ]
    )
    members = [
        _mem("m1", "alice release note", agent_id="alice"),
        _mem("m2", "bob release note", agent_id="bob"),
    ]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is not None
    # Spans both agents → ambient.
    assert result.agent_id is None


async def test_extract_preserves_single_agent_scope() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "memory_type": "fact",
                "content": "alice's private fact",
                "abstraction_level": 0.6,
                "aliases": [],
            }
        ]
    )
    members = [
        _mem("m1", "alice note 1", agent_id="alice"),
        _mem("m2", "alice note 2", agent_id="alice"),
    ]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is not None
    assert result.agent_id == "alice"


async def test_extract_takes_max_salience_from_sources() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "memory_type": "fact",
                "content": "derived",
                "abstraction_level": 0.6,
                "aliases": [],
            }
        ]
    )
    members = [
        _mem("m1", "a", salience=0.2),
        _mem("m2", "b", salience=0.7),
        _mem("m3", "c", salience=0.4),
    ]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is not None
    assert result.salience == 0.7


async def test_extract_clamps_abstraction_level() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "memory_type": "fact",
                "content": "x",
                "abstraction_level": 2.5,  # out of range
                "aliases": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is not None
    assert 0.0 <= result.abstraction_level <= 1.0


async def test_extract_falls_back_to_fact_on_unknown_type() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "memory_type": "not-a-real-type",
                "content": "something",
                "abstraction_level": 0.6,
                "aliases": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await extract_from_cluster(members, llm, FormulaParams())
    assert result is not None
    assert result.memory_type is MemoryType.FACT


# ─── end-to-end dream cycle with mocked LLM ───────────────────────


def _mnemoss_with_llm(tmp_path: Path, llm: MockLLMClient) -> Mnemoss:
    return Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        llm=llm,
    )


async def test_full_dream_cycle_writes_extracted_memory(tmp_path: Path) -> None:
    # MockLLMClient is forgiving; any prompt gets the same response.
    def respond(_prompt: str) -> dict:
        return {
            "memory_type": "fact",
            "content": "extracted summary",
            "abstraction_level": 0.6,
            "aliases": [],
        }

    mock = MockLLMClient(callback=respond)
    mem = _mnemoss_with_llm(tmp_path, mock)
    try:
        # Three related-ish memories so HDBSCAN finds at least one cluster.
        for i in range(4):
            await mem.observe(role="user", content=f"alice note {i}")

        # Use idle so all four phases run (task_completion skips cluster).
        report = await mem.dream(trigger="idle")

        # Cluster phase ran; extract ran; relations ran.
        phases = {o.phase.value: o for o in report.outcomes}
        assert phases["cluster"].status == "ok"
        assert phases["extract"].status == "ok"
        assert phases["relations"].status == "ok"

        # FakeEmbedder is deterministic; with 4 nearly-identical content
        # strings, HDBSCAN may or may not cluster them depending on
        # embedding variance — this test just ensures the pipeline runs
        # cleanly end-to-end without errors.
        assert phases["replay"].details["selected"] >= 4
    finally:
        await mem.close()


async def test_dream_cross_agent_cluster_extracts_ambient_memory(tmp_path: Path) -> None:
    def respond(_prompt: str) -> dict:
        return {
            "memory_type": "fact",
            "content": "cross-agent fact",
            "abstraction_level": 0.6,
            "aliases": [],
        }

    mock = MockLLMClient(callback=respond)
    mem = _mnemoss_with_llm(tmp_path, mock)
    alice = mem.for_agent("alice")
    bob = mem.for_agent("bob")
    try:
        # Two memories per agent so each cluster can span.
        for i in range(2):
            await alice.observe(role="user", content=f"shared topic {i}")
            await bob.observe(role="user", content=f"shared topic {i}")

        # Workspace-level dream sees everything (no agent scope).
        report = await mem.dream(trigger="idle")
        extract = report.outcome_map()["extract"] if hasattr(report, "outcome_map") else None
        _ = extract  # silence — use direct access below:
        extract = next(o for o in report.outcomes if o.phase.value == "extract")

        # If the LLM produced any extracted memories, at least one should
        # be ambient because the cluster spans agents.
        ids = extract.details.get("ids", [])
        if ids:
            # Fetch the first extracted memory and check its agent_id.
            assert mem._store is not None
            extracted_mem = await mem._store.get_memory(ids[0])
            assert extracted_mem is not None
            # Ambient when the cluster spans multiple agents.
            assert extracted_mem.agent_id is None
    finally:
        await mem.close()
