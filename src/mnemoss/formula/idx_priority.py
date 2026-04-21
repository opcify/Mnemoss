"""Derived index priority.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.6.

    idx_priority = σ(B + α·salience + β·emotional_weight + γ·1[pinned])

Stage 2 persists this value and recomputes it during P7 Rebalance; the
matching formula still uses the live value during recall so rehearsal
effects are immediate.
"""

from __future__ import annotations

import math

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_idx_priority(
    base_level: float,
    salience: float,
    emotional_weight: float,
    pinned: bool,
    params: FormulaParams,
) -> float:
    x = (
        base_level
        + params.alpha * salience
        + params.beta * emotional_weight
        + (params.gamma if pinned else 0.0)
    )
    return sigmoid(x)


def idx_priority_to_tier(idx_priority: float) -> IndexTier:
    """Map ``idx_priority`` → tier per §2.4.

    Boundaries (from the spec table):
      > 0.7                    → HOT
      0.3 < ip ≤ 0.7           → WARM
      0.1 < ip ≤ 0.3           → COLD
             ip ≤ 0.1          → DEEP
    """

    if idx_priority > 0.7:
        return IndexTier.HOT
    if idx_priority > 0.3:
        return IndexTier.WARM
    if idx_priority > 0.1:
        return IndexTier.COLD
    return IndexTier.DEEP


def initial_idx_priority(params: FormulaParams) -> float:
    """idx_priority for a freshly-encoded memory.

    At t = t_creation: B_i = 0 (history term) + η_0 (grace) = η_0.
    Salience, emotional_weight, and pin are all 0 at encoding time in
    Stage 1, so the Stage-1 initial value reduces to σ(η_0).
    """

    return sigmoid(params.eta_0)
