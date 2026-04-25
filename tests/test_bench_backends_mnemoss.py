"""Smoke tests for ``bench/backends/mnemoss_backend.py``.

Uses ``FakeEmbedder`` so these run offline and cost $0. The adapter's
embedding-parity contract (pinning ``text-embedding-3-small``) is
tested implicitly by the default-construction path in the module
docstring; verifying it end-to-end requires an ``OPENAI_API_KEY`` and
belongs under the ``integration`` marker if we want it later.
"""

from __future__ import annotations

from bench.backends import MemoryBackend, RecallHit
from bench.backends.mnemoss_backend import MnemossBackend
from mnemoss import FakeEmbedder

# ─── contract ──────────────────────────────────────────────────────


async def test_mnemoss_backend_satisfies_protocol() -> None:
    """Structural Protocol check — caught by mypy too, but a runtime
    smoke assert catches duck-type drift during refactors."""

    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        assert isinstance(be, MemoryBackend)
        assert be.backend_id == "mnemoss"


# ─── roundtrip ─────────────────────────────────────────────────────


async def test_observe_then_recall_returns_the_memory() -> None:
    """The canonical happy path: observe one memory, recall that exact
    text, assert the inserted id comes back at rank 1."""

    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        mem_id = await be.observe("alice joined the quarterly kickoff", ts=0.0)
        assert mem_id, "observe must return a non-empty memory id"

        hits = await be.recall("alice joined the quarterly kickoff", k=5)
        assert hits, "roundtrip recall must return at least one hit"
        assert hits[0].memory_id == mem_id
        assert hits[0].rank == 1
        assert hits[0].score is not None
        assert hits[0].score > 0, "Mnemoss activation must be positive"


async def test_ranks_are_contiguous_and_1_indexed() -> None:
    """Ranks in a recall list must be 1, 2, 3, ... with no gaps."""

    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        for i in range(5):
            await be.observe(f"alice meeting agenda item {i}", ts=float(i))
        hits = await be.recall("alice meeting agenda", k=5)
        assert hits, "recall should return multiple hits for multi-match query"
        for i, hit in enumerate(hits):
            assert hit.rank == i + 1, f"rank at index {i} should be {i + 1}, got {hit.rank}"


# ─── zero-result invariant ─────────────────────────────────────────


async def test_recall_on_empty_workspace_returns_empty_list() -> None:
    """Mirror's Mnemoss's ``recall() -> []`` invariant. The adapter must
    not raise on an empty workspace."""

    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        hits = await be.recall("anything at all", k=10)
        assert hits == [], "empty workspace must return [] not raise"


# ─── cleanup / Gap A (workspace lock) ──────────────────────────────


async def test_tempdir_is_cleaned_up_after_close() -> None:
    """Gap A from eng review: tempdir must be rmtree'd on close so a
    re-run doesn't hit a stale lock file or leftover SQLite."""

    be = MnemossBackend(embedding_model=FakeEmbedder(dim=16))
    tempdir = be._tempdir
    assert tempdir.exists(), "tempdir must exist before close"
    await be.observe("one memory", ts=0.0)
    await be.close()
    assert not tempdir.exists(), "tempdir must be gone after close()"


async def test_close_is_idempotent() -> None:
    """Calling close() twice must not raise. The second call is a no-op."""

    be = MnemossBackend(embedding_model=FakeEmbedder(dim=16))
    await be.observe("one memory", ts=0.0)
    await be.close()
    await be.close()  # must not raise


async def test_two_instances_do_not_collide() -> None:
    """Eng review concern: the workspace advisory lock is per-tempdir.
    Two MnemossBackend instances in the same process must each get
    their own fresh tempdir and not fight over the same lock file."""

    be1 = MnemossBackend(embedding_model=FakeEmbedder(dim=16))
    be2 = MnemossBackend(embedding_model=FakeEmbedder(dim=16))
    try:
        assert be1._tempdir != be2._tempdir
        id1 = await be1.observe("memory from instance 1", ts=0.0)
        id2 = await be2.observe("memory from instance 2", ts=0.0)
        hits1 = await be1.recall("memory from instance 1", k=5)
        hits2 = await be2.recall("memory from instance 2", k=5)
        assert hits1 and hits1[0].memory_id == id1
        assert hits2 and hits2[0].memory_id == id2
        # Critical: instance 1 should NOT see instance 2's memory.
        all_ids_in_be1 = {h.memory_id for h in hits1}
        assert id2 not in all_ids_in_be1, "tempdir isolation broken"
    finally:
        await be1.close()
        await be2.close()


# ─── RecallHit shape ───────────────────────────────────────────────


async def test_recall_hit_fields_are_populated() -> None:
    """RecallHit.memory_id, rank, score are all populated for Mnemoss
    (which has a native activation score)."""

    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        await be.observe("memory alpha", ts=0.0)
        hits = await be.recall("memory alpha", k=1)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, RecallHit)
        assert isinstance(hit.memory_id, str) and hit.memory_id
        assert hit.rank == 1
        assert hit.score is not None
        assert isinstance(hit.score, float)
