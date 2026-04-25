"""Semantic near-duplicate deduplication at ingest time.

The feature is ``EncoderParams.supersede_on_observe`` — when a new
memory has cosine ≥ ``supersede_cosine_threshold`` with an existing
memory in the same agent scope, the old memory is marked
``superseded_by`` the new one and filtered from recall by default.

Named after the config flag, not after "contradiction-aware observe"
(which oversold what the feature does; see the bench report at
``reports/supersession_bench/README.md`` §5.3 for the precision/recall
data that drove the rename). At the shipped threshold 0.85, the
feature catches near-exact duplicates — not semantic contradictions.

Covers the semantics documented in ``EncoderParams.supersede_on_observe``:

1. **Feature off (default)** — two near-duplicate observes both live,
   neither is marked superseded, both come back from recall.
2. **Feature on** — an older memory that's cosine ≥ threshold with a
   newly-observed one gets marked ``superseded_by`` / ``superseded_at``,
   and recall no longer returns it.
3. **Threshold respected** — a moderately-similar but below-threshold
   pair is NOT marked superseded (guards against over-suppression when
   two memories are topic-related but not contradictions).
4. **Agent isolation** — supersession is scoped to the same agent;
   agent A's new memory doesn't silently suppress agent B's memory.
5. **First-writer-wins** — if an already-superseded memory is matched
   by a third observation, the original ``superseded_by`` pointer
   stays intact (the chain doesn't rewrite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss import EncoderParams, FakeEmbedder, Mnemoss, StorageParams


class _DeterministicEmbedder:
    """Tiny hand-rolled embedder that gives us full control over cosines.

    ``FakeEmbedder`` uses hash noise — great for smoke tests, hopeless
    for "cosine X matches cosine Y" assertions. Here we build a tiny
    vocabulary → orthogonal-basis mapping so two strings share vectors
    iff they share all tokens. Close-enough for these tests.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self.embedder_id = f"test_det_embedder:d={dim}"

    def _vec_for(self, text: str) -> list[float]:
        import numpy as np

        v = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            idx = self._vocab.setdefault(tok, len(self._vocab) % self.dim)
            v[idx] += 1.0
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v.tolist()

    def embed(self, texts: list[str]):
        import numpy as np

        return np.array([self._vec_for(t) for t in texts], dtype="float32")


@pytest.mark.asyncio
async def test_feature_off_preserves_both(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="ws_off",
        embedding_model=_DeterministicEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        encoder=EncoderParams(supersede_on_observe=False),
    )
    try:
        old = await mem.observe(role="user", content="coffee every morning")
        new = await mem.observe(role="user", content="coffee every morning")  # exact dup
        hits = await mem.recall("coffee morning", k=5, reconsolidate=False)
        ids = {h.memory.id for h in hits}
        assert old in ids
        assert new in ids, "both memories should still be recallable"
    finally:
        await mem.close()


@pytest.mark.asyncio
async def test_feature_on_supersedes_near_duplicate(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="ws_on",
        embedding_model=_DeterministicEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=0.85,
        ),
    )
    try:
        old = await mem.observe(role="user", content="coffee every morning")
        new = await mem.observe(role="user", content="coffee every morning")  # cosine = 1.0

        hits = await mem.recall("coffee morning", k=5, reconsolidate=False)
        ids = {h.memory.id for h in hits}
        assert new in ids, "new memory should be recallable"
        assert old not in ids, "superseded memory should not appear in recall"

        # The old row is still in storage (for audit / dispose trail),
        # but marked superseded.
        assert mem._store is not None
        raw = await mem._store.get_memory(old)
        assert raw is not None
        assert raw.superseded_by == new
        assert raw.superseded_at is not None
    finally:
        await mem.close()


@pytest.mark.asyncio
async def test_threshold_respected(tmp_path: Path) -> None:
    """Below-threshold similarity must NOT supersede."""

    mem = Mnemoss(
        workspace="ws_thresh",
        embedding_model=_DeterministicEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        encoder=EncoderParams(
            supersede_on_observe=True,
            # Extremely high — only exact duplicates should trigger.
            supersede_cosine_threshold=0.999,
        ),
    )
    try:
        # Disjoint vocabularies → cosine ≈ 0; well under threshold.
        first = await mem.observe(role="user", content="red hat")
        second = await mem.observe(role="user", content="green shoes")

        hits1 = await mem.recall("red hat", k=5, reconsolidate=False)
        hits2 = await mem.recall("green shoes", k=5, reconsolidate=False)
        assert first in {h.memory.id for h in hits1}
        assert second in {h.memory.id for h in hits2}
    finally:
        await mem.close()


@pytest.mark.asyncio
async def test_agent_isolation(tmp_path: Path) -> None:
    """A new memory for agent B must not supersede agent A's memory."""

    mem = Mnemoss(
        workspace="ws_agents",
        embedding_model=_DeterministicEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=0.85,
        ),
    )
    try:
        alice = mem.for_agent("alice")
        bob = mem.for_agent("bob")
        a_old = await alice.observe(role="user", content="coffee every morning")
        b_new = await bob.observe(role="user", content="coffee every morning")

        # Alice's memory should still be hers and recallable for alice.
        a_hits = await alice.recall("coffee morning", k=5, reconsolidate=False)
        assert a_old in {h.memory.id for h in a_hits}

        assert mem._store is not None
        a_raw = await mem._store.get_memory(a_old)
        assert a_raw is not None
        assert a_raw.superseded_by is None, (
            "cross-agent observe must not mark the other agent's memory superseded"
        )
        # Bob's memory is live under his scope too.
        b_hits = await bob.recall("coffee morning", k=5, reconsolidate=False)
        assert b_new in {h.memory.id for h in b_hits}
    finally:
        await mem.close()


@pytest.mark.asyncio
async def test_first_writer_wins(tmp_path: Path) -> None:
    """A third observation doesn't rewrite an existing supersession link."""

    mem = Mnemoss(
        workspace="ws_chain",
        embedding_model=_DeterministicEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=0.85,
        ),
    )
    try:
        v1 = await mem.observe(role="user", content="coffee every morning")
        v2 = await mem.observe(role="user", content="coffee every morning")
        v3 = await mem.observe(role="user", content="coffee every morning")

        assert mem._store is not None
        raw_v1 = await mem._store.get_memory(v1)
        assert raw_v1 is not None
        # v1 was superseded by v2; even though v3 matches v1 too, the
        # link doesn't get rewritten to v3 (first-writer-wins).
        assert raw_v1.superseded_by == v2
        # v2 was then superseded by v3 — chains are allowed, just not
        # overwrites.
        raw_v2 = await mem._store.get_memory(v2)
        assert raw_v2 is not None
        assert raw_v2.superseded_by == v3
        # Only v3 comes back from recall (top of the supersession chain).
        hits = await mem.recall("coffee morning", k=5, reconsolidate=False)
        ids = {h.memory.id for h in hits}
        assert v3 in ids
        assert v1 not in ids
        assert v2 not in ids
    finally:
        await mem.close()


@pytest.mark.asyncio
async def test_fake_embedder_compat_smoke(tmp_path: Path) -> None:
    """Sanity: with FakeEmbedder (hash-noise) the feature doesn't crash.

    Doesn't assert supersession behavior — FakeEmbedder cosines are
    arbitrary — just that the flag being on doesn't explode in a
    workspace that happens to use FakeEmbedder.
    """

    mem = Mnemoss(
        workspace="ws_smoke",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
        encoder=EncoderParams(supersede_on_observe=True),
    )
    try:
        await mem.observe(role="user", content="something")
        await mem.observe(role="user", content="something else")
        hits = await mem.recall("whatever", k=5, reconsolidate=False)
        assert isinstance(hits, list)
    finally:
        await mem.close()
