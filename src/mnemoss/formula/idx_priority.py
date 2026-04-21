"""Derived index priority.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.6.

    idx_priority = σ(B + α·salience + β·emotional_weight + γ·1[pinned])

In Stage 1 we recompute this on the fly at recall time rather than persisting
it. Persistence and P7 migration land in Stage 2.
"""

from __future__ import annotations

import math

from mnemoss.core.config import FormulaParams


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
