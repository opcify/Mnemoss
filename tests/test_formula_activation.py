"""End-to-end test: the four formula components sum correctly."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

import numpy as np

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.formula.activation import ActivationBreakdown, compute_activation

UTC = timezone.utc
PARAMS = FormulaParams()


def _make_memory(
    id: str = "m1",
    created_offset_s: float = 0.0,
    salience: float = 0.0,
    emotional_weight: float = 0.0,
) -> Memory:
    now = datetime(2026, 4, 21, 12, 0, 0, tzinfo=UTC)
    created = datetime.fromtimestamp(now.timestamp() - created_offset_s, tz=UTC)
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
        session_id="s1",
        created_at=created,
        content="hello",
        content_embedding=np.zeros(4, dtype=np.float32),
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[created],
        salience=salience,
        emotional_weight=emotional_weight,
        index_tier=IndexTier.HOT,
    )


def test_breakdown_fields_sum_to_total() -> None:
    memory = _make_memory()
    now = memory.created_at
    rng = random.Random(42)

    br: ActivationBreakdown = compute_activation(
        memory=memory,
        query="hello",
        now=now,
        active_set=[],
        relations_from={},
        fan_of={},
        bm25_raw=-5.0,
        cos_sim=0.8,
        pinned=False,
        rng=rng,
        params=PARAMS,
    )

    total = br.base_level + br.spreading + br.matching + br.noise
    assert math.isclose(br.total, total, rel_tol=1e-9)


def test_fresh_unused_memory_scores_positive() -> None:
    """A brand-new memory with a decent FTS hit should easily clear τ=-1."""
    memory = _make_memory()
    now = memory.created_at
    rng = random.Random(0)

    br = compute_activation(
        memory=memory,
        query="hello",
        now=now,
        active_set=[],
        relations_from={},
        fan_of={},
        bm25_raw=-5.0,
        cos_sim=0.8,
        pinned=False,
        rng=rng,
        params=PARAMS,
    )
    # B ≈ 1.0 + matching (≈1.2–1.5) + noise ≈ well above tau.
    assert br.total > PARAMS.tau


def test_spreading_contributes_when_relations_active() -> None:
    memory = _make_memory(id="target")
    now = memory.created_at
    rng = random.Random(0)

    baseline = compute_activation(
        memory=memory,
        query="hello",
        now=now,
        active_set=[],
        relations_from={},
        fan_of={},
        bm25_raw=-5.0,
        cos_sim=0.8,
        pinned=False,
        rng=rng,
        params=PARAMS,
    )
    rng = random.Random(0)  # reset for determinism
    lifted = compute_activation(
        memory=memory,
        query="hello",
        now=now,
        active_set=["friend"],
        relations_from={"friend": {"target"}},
        fan_of={"friend": 1},
        bm25_raw=-5.0,
        cos_sim=0.8,
        pinned=False,
        rng=rng,
        params=PARAMS,
    )
    assert lifted.spreading > 0
    assert baseline.spreading == 0.0
    assert lifted.total > baseline.total


def test_noise_changes_across_seeds() -> None:
    # Explicit noise_scale=0.25 — this test asserts that noise actually
    # varies across RNG seeds, which requires non-zero noise. The shipped
    # default is noise_scale=0.0 (deterministic recall); callers that
    # want stochastic Luce-choice sampling opt in by setting this.
    noisy_params = FormulaParams(noise_scale=0.25)
    memory = _make_memory()
    now = memory.created_at
    args = dict(
        memory=memory,
        query="hello",
        now=now,
        active_set=[],
        relations_from={},
        fan_of={},
        bm25_raw=-5.0,
        cos_sim=0.8,
        pinned=False,
        params=noisy_params,
    )
    samples = {
        compute_activation(rng=random.Random(s), **args).noise  # type: ignore[arg-type]
        for s in range(50)
    }
    assert len(samples) > 10  # noise really is varying
