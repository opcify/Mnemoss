"""Unit tests for derived idx_priority."""

from __future__ import annotations

import math

from mnemoss.core.config import FormulaParams
from mnemoss.formula.idx_priority import compute_idx_priority, sigmoid

PARAMS = FormulaParams()


def test_sigmoid_stability_at_extremes() -> None:
    assert math.isclose(sigmoid(0.0), 0.5, rel_tol=1e-9)
    assert sigmoid(-1000.0) >= 0.0
    assert sigmoid(1000.0) <= 1.0
    assert math.isfinite(sigmoid(-1000.0))
    assert math.isfinite(sigmoid(1000.0))


def test_fresh_memory_is_hot() -> None:
    # Fresh memory: B≈1.0, no salience/emotion/pin → σ(1.0) ≈ 0.73, HOT.
    p = compute_idx_priority(
        base_level=1.0, salience=0.0, emotional_weight=0.0, pinned=False, params=PARAMS
    )
    assert p > 0.7


def test_cold_memory_without_protection() -> None:
    # Long-unused memory with low B.
    p = compute_idx_priority(
        base_level=-3.0, salience=0.0, emotional_weight=0.0, pinned=False, params=PARAMS
    )
    assert p < 0.3


def test_pin_lifts_cold_memory_back_to_hot() -> None:
    # γ = 2.0 should be enough to lift B=-3 to σ(-3+2) ≈ 0.27 — still WARM.
    # But γ=2.0 with B=-1.5 lifts it to σ(0.5) ≈ 0.62, WARM.
    # Pin's real job: lift B≈0 (freshly protected) to σ(2.0) ≈ 0.88, HOT.
    unpinned = compute_idx_priority(0.0, 0.0, 0.0, False, PARAMS)
    pinned = compute_idx_priority(0.0, 0.0, 0.0, True, PARAMS)
    assert pinned > unpinned
    assert pinned > 0.7  # HOT


def test_salience_emotion_additive() -> None:
    base = compute_idx_priority(0.0, 0.0, 0.0, False, PARAMS)
    salient = compute_idx_priority(0.0, 1.0, 0.0, False, PARAMS)
    emotional = compute_idx_priority(0.0, 0.0, 1.0, False, PARAMS)
    both = compute_idx_priority(0.0, 1.0, 1.0, False, PARAMS)
    assert salient > base
    assert emotional > base
    assert both > salient
    assert both > emotional
