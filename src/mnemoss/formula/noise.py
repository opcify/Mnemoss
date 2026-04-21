"""Logistic noise ε.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.5.

Sampled fresh on every retrieval scoring pass — never cached. The caller
passes an explicit ``random.Random`` instance so tests are deterministic.
"""

from __future__ import annotations

import math
import random


def sample_noise(rng: random.Random, scale: float) -> float:
    """Draw one sample from Logistic(0, scale).

    Uses the inverse-CDF method: ``μ + s · ln(u / (1 - u))`` with ``u`` drawn
    uniformly from ``(0, 1)``. Guarding against the open-interval endpoints
    prevents ``log(0)``.
    """

    u = rng.random()
    # Clamp strictly inside (0, 1).
    epsilon = 1e-12
    u = min(max(u, epsilon), 1.0 - epsilon)
    return scale * math.log(u / (1.0 - u))
