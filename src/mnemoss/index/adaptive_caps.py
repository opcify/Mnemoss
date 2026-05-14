"""Adaptive tier-caps controller (Method C).

Runtime self-tuning of ``TierCapacityParams`` from recall telemetry.
Opt-in via ``FormulaParams.adaptive_tier_caps``; default off means a
byte-identical no-op.

This file has three parts:

- ``TierTelemetry`` + ``compute_adjusted_caps`` — the *pure* controller.
  One window of telemetry in, a new ``TierCapacityParams`` out. No I/O.
- ``TierTelemetryLedger`` — ``workspace_meta`` persistence, an exact
  mirror of ``dream/cost.py``'s ``CostLedger``.
- ``maybe_adjust_caps`` — the thin impure orchestrator wired into
  ``rebalance()``.

See docs/superpowers/specs/2026-05-15-adaptive-tier-caps-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import apsw

from mnemoss.core.config import FormulaParams, TierCapacityParams
from mnemoss.core.types import IndexTier

# ─── pure controller ───────────────────────────────────────────────


@dataclass(frozen=True)
class TierTelemetry:
    """One window of recall telemetry, consumed by the controller.

    Counts accumulate since the last controller adjustment; the
    orchestrator resets them after each evaluation that clears the
    min-dwell gate.
    """

    queries: int
    winners_hot: int
    winners_warm: int
    winners_cold: int
    winners_deep: int
    elapsed_ms_sum: float
    reminiscence_events: int


def compute_adjusted_caps(
    current: TierCapacityParams,
    telemetry: TierTelemetry,
    params: FormulaParams,
) -> tuple[TierCapacityParams, float]:
    """Pure controller step. Returns ``(new_caps, delta)``.

    ``delta`` is the raw blended signal before dead-band/clamp. When
    its magnitude is under the dead-band — or there is not enough data
    — ``new_caps is current`` (the caller can use identity to detect
    "no move").

    Signals:

    - ``recall_pressure`` ∈ [0, 1] — fraction of winning results that
      the caps pushed into the slow tiers (COLD/DEEP). High → caps too
      small → grow.
    - ``latency_pressure`` ∈ [0, 1] — mean recall latency against the
      configured budget. High → scans too expensive → shrink.

    Blended: ``delta = (1 - λ)·recall_pressure - λ·latency_pressure``.
    """

    total_winners = (
        telemetry.winners_hot
        + telemetry.winners_warm
        + telemetry.winners_cold
        + telemetry.winners_deep
    )
    if telemetry.queries < params.adaptive_tier_min_queries or total_winners == 0:
        return current, 0.0

    recall_pressure = (
        telemetry.winners_cold + telemetry.winners_deep
    ) / total_winners

    mean_elapsed_ms = telemetry.elapsed_ms_sum / telemetry.queries
    latency_pressure = min(
        mean_elapsed_ms / params.adaptive_tier_latency_budget_ms, 1.0
    )

    lam = params.adaptive_tier_lambda
    delta = (1.0 - lam) * recall_pressure - lam * latency_pressure

    if abs(delta) < params.adaptive_tier_deadband:
        return current, delta

    step = max(
        -params.adaptive_tier_max_step,
        min(params.adaptive_tier_max_step, delta),
    )
    factor = 1.0 + step
    lo, hi = current.min_floor, current.max_cap

    def _adjust(cap: int) -> int:
        return max(lo, min(hi, round(cap * factor)))

    new_hot = _adjust(current.hot_cap)
    new_warm = _adjust(current.warm_cap)
    new_cold = _adjust(current.cold_cap)

    # Preserve hot <= warm <= cold after asymmetric clamping.
    new_warm = max(new_warm, new_hot)
    new_cold = max(new_cold, new_warm)

    new_caps = TierCapacityParams(
        hot_cap=new_hot,
        warm_cap=new_warm,
        cold_cap=new_cold,
        min_floor=current.min_floor,
        max_cap=current.max_cap,
    )
    return new_caps, delta
