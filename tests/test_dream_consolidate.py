"""P3 Consolidate tests — the merged Extract + Refine + Generalize phase.

All exercises use MockLLMClient so no network. Covers the consolidate
function itself plus the runner phase end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mnemoss import FakeEmbedder, Mnemoss, MockLLMClient, StorageParams
from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, MemoryType
from mnemoss.dream.consolidate import (
    build_consolidate_prompt,
    consolidate_cluster,
)

UTC = timezone.utc


class ClusterableFakeEmbedder:
    """Test embedder that places memories sharing a prefix into one tight
    cluster in embedding space. Makes HDBSCAN's output deterministic for
    runner-level tests that care about the consolidation plumbing."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.embedder_id = f"clusterable-fake:{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Anchor on the first token so "alice_1"/"alice_2" cluster,
            # "bob_1"/"bob_2" form their own cluster, etc.
            prefix = text.split("_", 1)[0] if "_" in text else text
            seed = sum(ord(c) for c in prefix) % 97
            v = np.zeros(self.dim, dtype=np.float32)
            v[seed % self.dim] = 1.0
            # Tiny jitter so HDBSCAN doesn't choke on perfectly-identical points.
            v[(seed + 1) % self.dim] = 0.02 * (i % 3)
            norm = float(np.linalg.norm(v))
            out[i] = v / norm if norm > 0 else v
        return out


def _mem(
    id: str,
    content: str,
    *,
    agent_id: str | None = None,
    session_id: str = "s1",
    salience: float = 0.2,
    gist: str | None = None,
    extraction_level: int = 0,
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
        extracted_gist=gist,
        extraction_level=extraction_level,
    )


# ─── prompt shape ──────────────────────────────────────────────────


def test_prompt_includes_members_roles_and_existing_extractions() -> None:
    members = [
        _mem("m1", "Alice likes coffee", gist="existing gist"),
        _mem("m2", "Alice ordered a latte"),
    ]
    prompt = build_consolidate_prompt(members)
    # Content + role markers appear.
    assert "Alice likes coffee" in prompt
    assert "Alice ordered a latte" in prompt
    assert "[user]" in prompt
    # Existing extraction exposed to the LLM for refinement context.
    assert "existing gist" in prompt
    # Schema instructions present.
    assert '"summary"' in prompt
    assert '"refinements"' in prompt
    assert '"patterns"' in prompt


# ─── happy-path consolidation ─────────────────────────────────────


async def test_consolidate_returns_summary_refinements_and_patterns() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "Alice prefers lattes",
                    "abstraction_level": 0.65,
                    "aliases": ["coffee fact"],
                },
                "refinements": [
                    {
                        "index": 1,
                        "gist": "Alice liked coffee",
                        "entities": ["Alice"],
                        "time": None,
                        "location": None,
                        "participants": ["Alice"],
                    },
                    {
                        "index": 2,
                        "gist": "Alice ordered a latte",
                        "entities": ["Alice"],
                        "time": "2026-04-22T09:00:00+00:00",
                        "location": "cafe",
                        "participants": ["Alice"],
                    },
                ],
                "patterns": [
                    {
                        "content": "Alice orders coffee most mornings",
                        "derived_from": [1, 2],
                    }
                ],
            }
        ]
    )
    members = [
        _mem("m1", "Alice likes coffee"),
        _mem("m2", "Alice ordered a latte"),
    ]

    result = await consolidate_cluster(members, llm, FormulaParams())

    assert result.summary is not None
    assert result.summary.content == "Alice prefers lattes"
    assert result.summary.memory_type is MemoryType.FACT
    assert result.summary.abstraction_level == 0.65
    assert result.summary.derived_from == ["m1", "m2"]
    assert result.summary.source_context["extracted_by"] == "dream_consolidate"
    assert result.summary.source_context["aliases"] == ["coffee fact"]

    assert len(result.refinements) == 2
    r0 = result.refinements[0]
    assert r0.member_index == 0
    assert r0.fields.level == 2
    assert r0.fields.gist == "Alice liked coffee"
    r1 = result.refinements[1]
    assert r1.member_index == 1
    assert r1.fields.time is not None
    assert r1.fields.time.year == 2026
    assert r1.fields.location == "cafe"

    assert len(result.patterns) == 1
    p = result.patterns[0]
    assert p.memory_type is MemoryType.PATTERN
    assert p.content == "Alice orders coffee most mornings"
    assert p.derived_from == ["m1", "m2"]
    assert p.source_context["scope"] == "intra_cluster_pattern"


# ─── edge cases ────────────────────────────────────────────────────


async def test_consolidate_singleton_cluster_returns_empty_without_calling_llm() -> None:
    llm = MockLLMClient()
    result = await consolidate_cluster([_mem("m1", "solo")], llm, FormulaParams())
    assert result.is_empty
    assert llm.calls == []


async def test_consolidate_returns_empty_on_llm_failure() -> None:
    class _FailingLLM:
        model = "mock"

        async def complete_text(self, *a, **kw):
            raise RuntimeError("boom")

        async def complete_json(self, *a, **kw):
            raise RuntimeError("boom")

    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, _FailingLLM(), FormulaParams())
    assert result.is_empty


async def test_consolidate_returns_empty_on_empty_content_summary() -> None:
    """Missing summary.content → no summary memory; refinements/patterns
    still parse independently."""

    llm = MockLLMClient(
        responses=[
            {
                "summary": {"memory_type": "fact", "content": "  "},
                "refinements": [],
                "patterns": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is None
    assert result.refinements == []
    assert result.patterns == []


async def test_consolidate_clamps_abstraction_level() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "x",
                    "abstraction_level": 2.5,
                    "aliases": [],
                },
                "refinements": [],
                "patterns": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is not None
    assert 0.0 <= result.summary.abstraction_level <= 1.0


async def test_consolidate_falls_back_to_fact_on_unknown_type() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "not-a-real-type",
                    "content": "x",
                    "abstraction_level": 0.6,
                    "aliases": [],
                },
                "refinements": [],
                "patterns": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is not None
    assert result.summary.memory_type is MemoryType.FACT


async def test_consolidate_cross_agent_summary_is_ambient() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "shared fact",
                    "abstraction_level": 0.7,
                    "aliases": [],
                },
                "refinements": [],
                "patterns": [],
            }
        ]
    )
    members = [
        _mem("m1", "alice thing", agent_id="alice"),
        _mem("m2", "bob thing", agent_id="bob"),
    ]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is not None
    assert result.summary.agent_id is None


async def test_consolidate_preserves_single_agent_scope() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "alice private",
                    "abstraction_level": 0.6,
                    "aliases": [],
                },
                "refinements": [],
                "patterns": [],
            }
        ]
    )
    members = [
        _mem("m1", "alice 1", agent_id="alice"),
        _mem("m2", "alice 2", agent_id="alice"),
    ]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is not None
    assert result.summary.agent_id == "alice"


async def test_consolidate_ignores_out_of_range_refinement_index() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "x",
                    "abstraction_level": 0.6,
                    "aliases": [],
                },
                "refinements": [
                    {
                        "index": 42,
                        "gist": "ghost",
                        "entities": [],
                        "time": None,
                        "location": None,
                        "participants": [],
                    },
                    {
                        "index": 1,
                        "gist": "real",
                        "entities": [],
                        "time": None,
                        "location": None,
                        "participants": [],
                    },
                ],
                "patterns": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, llm, FormulaParams())
    # Only the in-range refinement survives.
    assert len(result.refinements) == 1
    assert result.refinements[0].member_index == 0
    assert result.refinements[0].fields.gist == "real"


async def test_consolidate_drops_patterns_with_fewer_than_two_sources() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "x",
                    "abstraction_level": 0.6,
                    "aliases": [],
                },
                "refinements": [],
                "patterns": [
                    {"content": "single-source", "derived_from": [1]},
                    {"content": "cross-member", "derived_from": [1, 2]},
                    {"content": "empty", "derived_from": []},
                ],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, llm, FormulaParams())
    # Only the cross-member pattern survives.
    assert len(result.patterns) == 1
    assert result.patterns[0].content == "cross-member"


async def test_consolidate_handles_malformed_refinement_entries() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": None,
                "refinements": [
                    "not-a-dict",
                    {"no_index": True},
                    {"index": "not-an-int"},
                    {
                        "index": 1,
                        "gist": "good",
                        "entities": [],
                        "time": None,
                        "location": None,
                        "participants": [],
                    },
                ],
                "patterns": [],
            }
        ]
    )
    members = [_mem("m1", "a"), _mem("m2", "b")]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is None
    # Only the well-formed entry survives.
    assert len(result.refinements) == 1
    assert result.refinements[0].fields.gist == "good"


async def test_consolidate_salience_summary_takes_max_of_members() -> None:
    llm = MockLLMClient(
        responses=[
            {
                "summary": {
                    "memory_type": "fact",
                    "content": "derived",
                    "abstraction_level": 0.6,
                    "aliases": [],
                },
                "refinements": [],
                "patterns": [],
            }
        ]
    )
    members = [
        _mem("m1", "a", salience=0.2),
        _mem("m2", "b", salience=0.7),
        _mem("m3", "c", salience=0.4),
    ]
    result = await consolidate_cluster(members, llm, FormulaParams())
    assert result.summary is not None
    assert result.summary.salience == 0.7


# ─── end-to-end via the runner ─────────────────────────────────────


def _mnemoss(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="consolidate",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


def _mnemoss_clusterable(tmp_path: Path, **kwargs) -> Mnemoss:
    return Mnemoss(
        workspace="consolidate_c",
        embedding_model=ClusterableFakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        **kwargs,
    )


async def test_runner_consolidate_phase_writes_summary_and_refines_members(
    tmp_path: Path,
) -> None:
    """Use DreamRunner directly with cluster_min_size=2 so HDBSCAN
    reliably produces a cluster even from FakeEmbedder + 3 inputs."""

    def canned(_prompt: str) -> dict:
        return {
            "summary": {
                "memory_type": "fact",
                "content": "Alice fact",
                "abstraction_level": 0.65,
                "aliases": [],
            },
            "refinements": [
                {
                    "index": 1,
                    "gist": "refined m1",
                    "entities": ["Alice"],
                    "time": None,
                    "location": None,
                    "participants": ["Alice"],
                },
                {
                    "index": 2,
                    "gist": "refined m2",
                    "entities": ["Alice"],
                    "time": None,
                    "location": None,
                    "participants": ["Alice"],
                },
            ],
            "patterns": [
                {"content": "pattern", "derived_from": [1, 2]},
            ],
        }

    mock = MockLLMClient(callback=canned)
    mem = _mnemoss_clusterable(tmp_path, llm=mock)
    try:
        # Two distinct groups (HDBSCAN needs >1 group to reliably cluster
        # on small inputs — a single near-identical blob can read as noise).
        for i in range(3):
            await mem.observe(role="user", content=f"alice_{i}")
        for i in range(3):
            await mem.observe(role="user", content=f"bob_{i}")

        from mnemoss.dream.runner import DreamRunner
        from mnemoss.dream.types import TriggerType

        assert mem._store is not None
        runner = DreamRunner(
            mem._store,
            mem._config.formula,
            llm=mock,
            embedder=mem._embedder,
            cluster_min_size=2,
        )
        report = await runner.run(TriggerType.IDLE)

        from mnemoss import PhaseName

        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None and consolidate.status == "ok"
        details = consolidate.details
        assert details["clusters_processed"] >= 1
        assert details["summaries"] >= 1
        assert details["refined"] >= 1
        assert mock.calls
    finally:
        await mem.close()


async def test_runner_consolidate_skipped_without_llm(tmp_path: Path) -> None:
    mem = _mnemoss(tmp_path)  # no llm
    try:
        await mem.observe(role="user", content="x")
        report = await mem.dream(trigger="session_end")

        from mnemoss import PhaseName

        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        assert consolidate.status == "skipped"
        assert "llm" in consolidate.details.get("reason", "").lower()
    finally:
        await mem.close()


async def test_runner_consolidate_makes_exactly_one_llm_call_per_cluster(
    tmp_path: Path,
) -> None:
    """The whole point of the merger: one call per cluster, not three."""

    def canned(_prompt: str) -> dict:
        return {
            "summary": {
                "memory_type": "fact",
                "content": "c",
                "abstraction_level": 0.6,
                "aliases": [],
            },
            "refinements": [],
            "patterns": [],
        }

    mock = MockLLMClient(callback=canned)
    mem = _mnemoss_clusterable(tmp_path, llm=mock)
    try:
        # Two separate semantic groups → two clusters (by prefix).
        for i in range(3):
            await mem.observe(role="user", content=f"alice_{i}")
        for i in range(3):
            await mem.observe(role="user", content=f"bob_{i}")

        from mnemoss import PhaseName
        from mnemoss.dream.runner import DreamRunner
        from mnemoss.dream.types import TriggerType

        assert mem._store is not None
        runner = DreamRunner(
            mem._store,
            mem._config.formula,
            llm=mock,
            embedder=mem._embedder,
            cluster_min_size=2,  # guarantee clusters are formed
        )
        report = await runner.run(TriggerType.IDLE)

        consolidate = report.outcome(PhaseName.CONSOLIDATE)
        assert consolidate is not None
        clusters_processed = consolidate.details["clusters_processed"]
        assert clusters_processed >= 1

        json_calls = sum(1 for method, _ in mock.calls if method == "json")
        # Exactly one call per processed cluster — the merger's whole point.
        assert json_calls == clusters_processed
    finally:
        await mem.close()
