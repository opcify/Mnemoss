"""Spreading activation Σ_j W_j · S_ji.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.3.

    S_ji = S_max - ln(fan_j)

Fan convention for Stage 1: a memory with zero outbound edges is treated
as fan = 1, so S_ji = S_max. This keeps freshly-encoded memories from
contributing -inf before any relations have been written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from mnemoss.core.config import FormulaParams


def compute_spreading(
    target_id: str,
    active_set: list[str],
    relations_from: Mapping[str, set[str]],
    fan_of: Mapping[str, int],
    params: FormulaParams,
    attention_weights: Mapping[str, float] | None = None,
) -> float:
    """Return the spreading-activation contribution for ``target_id``.

    Parameters
    ----------
    target_id : str
        Memory whose activation we're scoring.
    active_set : list[str]
        IDs currently in Working Memory (the set C).
    relations_from : Mapping[str, set[str]]
        Adjacency: ``relations_from[j]`` is the set of memories that ``j``
        points to. ``j`` activates ``i`` if ``i in relations_from[j]``.
    fan_of : Mapping[str, int]
        Out-degree of each memory. ``fan_of.get(j, 0)`` is treated as 1.
    params : FormulaParams
    attention_weights : Mapping[str, float] | None
        Optional per-memory weight W_j. Defaults to uniform 1/|C|.
    """

    if not active_set:
        return 0.0

    default_w = 1.0 / len(active_set)
    total = 0.0

    for j in active_set:
        if target_id not in relations_from.get(j, set()):
            continue
        fan_j = max(fan_of.get(j, 0), 1)  # Stage 1 convention: unseen/empty → 1
        s_ji = params.s_max - math.log(fan_j)
        w_j = attention_weights.get(j, default_w) if attention_weights else default_w
        total += w_j * s_ji

    return total
