"""Base-level activation B_i.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.2.

    B_i = ln(Σ (t - t_k)^-d) + η(t)
    η(t) = η_0 · exp(-(t - t_creation) / τ_η)

The (t - t_k) floor is applied in seconds (not inside the log) so that a
memory queried in the same instant it was created contributes ln(1) = 0 to
the history term and relies on η(t) for the freshness lift.
"""

from __future__ import annotations

import math
from datetime import datetime

from mnemoss.core.config import FormulaParams


def _age_seconds(now: datetime, t_k: datetime, t_floor: float) -> float:
    delta = (now - t_k).total_seconds()
    return max(delta, t_floor)


def compute_base_level(
    access_history: list[datetime],
    now: datetime,
    created_at: datetime,
    params: FormulaParams,
) -> float:
    """Return B_i for a memory at time ``now``.

    ``access_history`` should include ``created_at`` as its first element,
    plus every retrieval timestamp. An empty list is treated as "never
    accessed" and returns only the encoding-grace term (history = -inf
    semantically, but we return 0 for the history term in that edge case).
    """

    if access_history:
        decay_sum = 0.0
        for t_k in access_history:
            age = _age_seconds(now, t_k, params.t_floor_seconds)
            decay_sum += age ** (-params.d)
        history = math.log(decay_sum) if decay_sum > 0.0 else 0.0
    else:
        history = 0.0

    grace_age = max((now - created_at).total_seconds(), 0.0)
    grace = params.eta_0 * math.exp(-grace_age / params.eta_tau_seconds)

    return history + grace
