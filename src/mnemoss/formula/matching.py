"""Dynamic hybrid matching — MP · [w_F · s̃_F + w_S · s̃_S].

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.4. The weights are computed with
the symmetric softmax-style formulation:

    w_F_raw = (0.2 + 0.6 · idx_priority) · b_F(q)
    w_S_raw = (0.8 - 0.6 · idx_priority) · b_F(q)^-1
    w_F = w_F_raw / (w_F_raw + w_S_raw)
    w_S = 1 - w_F
"""

from __future__ import annotations

import math

from mnemoss.core.config import FormulaParams


def matching_weights(idx_priority: float, query_bias: float) -> tuple[float, float]:
    """Return ``(w_F, w_S)`` with ``w_F + w_S == 1``.

    ``idx_priority`` is the σ-squashed tier score in ``[0, 1]``.
    ``query_bias`` is ``b_F(q)``.
    """

    w_f_raw = (0.2 + 0.6 * idx_priority) * query_bias
    w_s_raw = (0.8 - 0.6 * idx_priority) / query_bias
    total = w_f_raw + w_s_raw
    if total == 0.0:
        return 0.5, 0.5
    w_f = w_f_raw / total
    return w_f, 1.0 - w_f


def normalize_bm25(bm25_raw: float) -> float:
    """Map SQLite FTS5 BM25 into ``[0, 1]``.

    SQLite FTS5 returns BM25 as a *negative* number (lower magnitude = more
    relevant in SQLite's convention; 0 means no match). We take ``abs`` and
    apply the exponential squash from §1.4:

        s̃_F = 1 - exp(-|bm25| / 5)
    """

    return 1.0 - math.exp(-abs(bm25_raw) / 5.0)


def normalize_cosine(cos_sim: float) -> float:
    """Map cosine similarity from ``[-1, 1]`` into ``[0, 1]``."""

    return (cos_sim + 1.0) / 2.0


def compute_matching(
    idx_priority: float,
    bm25_raw: float,
    cos_sim: float,
    query_bias: float,
    params: FormulaParams,
) -> float:
    """Return ``MP · [w_F · s̃_F + w_S · s̃_S]``."""

    w_f, w_s = matching_weights(idx_priority, query_bias)
    s_f = normalize_bm25(bm25_raw)
    s_s = normalize_cosine(cos_sim)
    return params.mp * (w_f * s_f + w_s * s_s)
