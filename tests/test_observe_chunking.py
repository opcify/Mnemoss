"""End-to-end tests for auto-split on ``observe()``.

Verifies that ``EncoderParams.max_memory_chars`` produces multiple
Memory rows for one long observe while keeping the Raw Log a single
row (Principle 3 — Raw Log is the verbatim audit trail).
"""

from __future__ import annotations

from pathlib import Path

from mnemoss import (
    EncoderParams,
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    StorageParams,
)


def _mem(tmp_path: Path, *, max_memory_chars: int | None) -> Mnemoss:
    return Mnemoss(
        workspace="split",
        embedding_model=FakeEmbedder(dim=16),
        formula=FormulaParams(noise_scale=0.0),
        encoder=EncoderParams(max_memory_chars=max_memory_chars),
        storage=StorageParams(root=tmp_path),
    )


# ─── baseline: no split when cap is None ──────────────────────────


async def test_no_split_when_cap_is_none(tmp_path: Path) -> None:
    """Backward-compat: ``max_memory_chars=None`` keeps the old
    1 observe = 1 memory behavior even for very long content."""

    mem = _mem(tmp_path, max_memory_chars=None)
    try:
        big = "abc " * 2000  # 8000 chars
        mid = await mem.observe(role="user", content=big)
        assert mid is not None
        status = await mem.status()
        assert status["memory_count"] == 1
    finally:
        await mem.close()


async def test_no_split_when_content_under_cap(tmp_path: Path) -> None:
    """Even with a cap set, short content stays one memory."""

    mem = _mem(tmp_path, max_memory_chars=10_000)
    try:
        mid = await mem.observe(role="user", content="short message")
        assert mid is not None
        status = await mem.status()
        assert status["memory_count"] == 1
    finally:
        await mem.close()


# ─── the happy path: split produces N memories ────────────────────


async def test_long_observe_produces_multiple_memories(tmp_path: Path) -> None:
    """One long observe above the cap splits into chunks; each chunk
    becomes its own Memory row."""

    mem = _mem(tmp_path, max_memory_chars=50)
    try:
        # Three paragraphs, each ~30 chars → should split into 3 chunks.
        content = (
            "paragraph one about the kickoff\n\n"
            "paragraph two about the agenda\n\n"
            "paragraph three about the team"
        )
        await mem.observe(role="user", content=content)
        status = await mem.status()
        assert status["memory_count"] >= 2  # at least split, exact N depends on regrouping
    finally:
        await mem.close()


async def test_split_chunks_carry_group_marker(tmp_path: Path) -> None:
    """Each chunk Memory's ``source_context.split_part`` tags it with
    the chunk index, total, and shared group id."""

    mem = _mem(tmp_path, max_memory_chars=60)
    try:
        content = (
            "First paragraph with some body text here.\n\n"
            "Second paragraph with different body text.\n\n"
            "Third paragraph to force three chunks."
        )
        await mem.observe(role="user", content=content)

        # Peek at the persisted memories directly.
        assert mem._store is not None
        ids = await mem._store.iter_memory_ids()
        memories = await mem._store.materialize_memories(ids)
        # Expect at least 2 chunks for this content at a 60-char cap.
        assert len(memories) >= 2

        group_ids = set()
        indices = []
        totals = set()
        for m in memories:
            split_part = m.source_context.get("split_part")
            assert split_part is not None, (
                f"memory {m.id} missing split_part: {m.source_context}"
            )
            group_ids.add(split_part["group_id"])
            indices.append(split_part["index"])
            totals.add(split_part["total"])

        # All chunks share one group id, totals agree, indices cover 0..N-1.
        assert len(group_ids) == 1
        assert len(totals) == 1
        assert sorted(indices) == list(range(len(memories)))
    finally:
        await mem.close()


async def test_raw_log_stays_one_row_per_observe(tmp_path: Path) -> None:
    """Principle 3: the Raw Log is verbatim, one row per ``observe()``
    call, regardless of how many Memory rows downstream encoding
    produces."""

    mem = _mem(tmp_path, max_memory_chars=40)
    try:
        content = "\n\n".join(f"paragraph {i} with enough text" for i in range(5))
        await mem.observe(role="user", content=content)

        # Peek at raw_log directly.
        assert mem._store is not None
        raw_conn = mem._store._require_raw_conn()
        row = raw_conn.execute("SELECT COUNT(*) FROM raw_message").fetchone()
        assert row[0] == 1, "Raw Log must stay 1:1 with observe()"

        # But multiple Memory rows exist.
        memory_ids = await mem._store.iter_memory_ids()
        assert len(memory_ids) >= 2
    finally:
        await mem.close()


# ─── recall still finds split content ────────────────────────────


async def test_recall_finds_chunk_by_its_own_tokens(tmp_path: Path) -> None:
    """A query whose terms appear in one specific chunk should surface
    that chunk among the top results.

    This test asserts "findable," not "#1 ranked." With ``FakeEmbedder``
    (hash-based, non-semantic), cosine scores are pseudo-random — a
    distractor chunk can occasionally out-hash the target. The matching
    formula is noisy-OR (either strong signal suffices), so when one
    chunk wins on BM25 and another wins on (random) cosine, they can
    be near-tied. The chunking intent — "each chunk is an independent
    memory row, recall can retrieve it by its unique tokens" — is
    still verified by checking top-k contains the target.
    """

    mem = _mem(tmp_path, max_memory_chars=80)
    try:
        content = (
            "first paragraph about the morning kickoff meeting.\n\n"
            "second paragraph about the midday break and coffee.\n\n"
            "third paragraph about the evening deadline and demo."
        )
        await mem.observe(role="user", content=content)

        # "evening deadline demo" only appears in the third chunk.
        results = await mem.recall("evening deadline demo", k=3)
        assert results
        # Target chunk appears in top-k (by BM25 alone — cosine is
        # hash-noise via FakeEmbedder).
        target_contents = [r.memory.content.lower() for r in results]
        assert any("evening" in c for c in target_contents), (
            f"chunk containing 'evening' should be in top-3; got {target_contents}"
        )
        # Every returned hit is an INDIVIDUAL chunk, not the concatenated
        # whole (no single chunk contains both 'kickoff' and 'deadline').
        for c in target_contents:
            assert not ("kickoff" in c and "deadline" in c)
    finally:
        await mem.close()


# ─── pathological content (hard-split fallback) ──────────────────


async def test_very_long_no_boundary_content_hard_splits(tmp_path: Path) -> None:
    """A single super-long run with no whitespace or sentence
    terminators (e.g. a minified JSON blob) falls through to the
    hard-split fallback and still produces bounded chunks."""

    mem = _mem(tmp_path, max_memory_chars=100)
    try:
        content = "x" * 450  # 450 chars, no boundaries at all
        await mem.observe(role="user", content=content)

        assert mem._store is not None
        ids = await mem._store.iter_memory_ids()
        assert len(ids) == 5  # 100+100+100+100+50 = 450

        # Every chunk is bounded.
        memories = await mem._store.materialize_memories(ids)
        for m in memories:
            assert len(m.content) <= 100
    finally:
        await mem.close()
