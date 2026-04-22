"""Unified activation A_i — the whole formula.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.1.

    A_i = B_i + Σ_j W_j·S_ji + MP·[w_F·s̃_F + w_S·s̃_S] + ε

``compute_activation`` returns an ``ActivationBreakdown`` so callers
(``recall.engine``, ``explain_recall``) can inspect each term.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory
from mnemoss.formula.base_level import compute_base_level
from mnemoss.formula.idx_priority import compute_idx_priority
from mnemoss.formula.matching import compute_matching
from mnemoss.formula.noise import sample_noise
from mnemoss.formula.query_bias import compute_query_bias
from mnemoss.formula.spreading import compute_spreading


@dataclass
class ActivationBreakdown:
    """Per-term decomposition of A_i.

    Returned by ``compute_activation`` so retrieval, ``explain_recall``, and
    debugging all see the same view. ``total`` is the final score used
    for ranking.
    """

    base_level: float
    spreading: float
    matching: float
    noise: float
    total: float
    idx_priority: float
    w_f: float
    w_s: float
    query_bias: float


def compute_activation(
    memory: Memory,
    query: str,
    now: datetime,
    active_set: list[str],
    relations_from: Mapping[str, set[str]],
    fan_of: Mapping[str, int],
    bm25_raw: float,
    cos_sim: float,
    pinned: bool,
    rng: random.Random,
    params: FormulaParams,
) -> ActivationBreakdown:
    """Score a single memory against a query.

    ``bm25_raw`` is SQLite FTS5's raw score (negative by convention; the
    matching module takes ``abs``). ``cos_sim`` is raw cosine in ``[-1, 1]``.
    """

    b = compute_base_level(memory.access_history, now, memory.created_at, params)
    idx_priority = compute_idx_priority(b, memory.salience, memory.emotional_weight, pinned, params)

    spread = compute_spreading(memory.id, active_set, relations_from, fan_of, params)

    bias = compute_query_bias(query)
    match = compute_matching(idx_priority, bm25_raw, cos_sim, bias, params)

    noise = sample_noise(rng, params.noise_scale)

    total = b + spread + match + noise

    # Expose the per-term weights used inside matching for explain_recall.
    from mnemoss.formula.matching import matching_weights

    w_f, w_s = matching_weights(idx_priority, bias)

    return ActivationBreakdown(
        base_level=b,
        spreading=spread,
        matching=match,
        noise=noise,
        total=total,
        idx_priority=idx_priority,
        w_f=w_f,
        w_s=w_s,
        query_bias=bias,
    )
