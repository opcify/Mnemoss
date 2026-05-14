"""Adaptive tier-caps controller — unit tests.

compute_adjusted_caps is a pure function; these tests pin its
direction, magnitude, dead-band, max-step, clamp, and ordering
behaviour exhaustively.
"""

from __future__ import annotations

from mnemoss import FormulaParams, TierCapacityParams
from mnemoss.index.adaptive_caps import TierTelemetry, compute_adjusted_caps


def _params(**kw: object) -> FormulaParams:
    base = {"adaptive_tier_caps": True, "adaptive_tier_min_queries": 100}
    base.update(kw)
    return FormulaParams(**base)  # type: ignore[arg-type]


def test_insufficient_queries_is_a_no_op() -> None:
    caps = TierCapacityParams()
    tel = TierTelemetry(
        queries=10, winners_hot=0, winners_warm=0, winners_cold=10,
        winners_deep=0, elapsed_ms_sum=200.0, reminiscence_events=0,
    )
    new, delta = compute_adjusted_caps(caps, tel, _params())
    assert new is caps
    assert delta == 0.0


def test_zero_winners_is_a_no_op() -> None:
    caps = TierCapacityParams()
    tel = TierTelemetry(
        queries=300, winners_hot=0, winners_warm=0, winners_cold=0,
        winners_deep=0, elapsed_ms_sum=300.0, reminiscence_events=0,
    )
    new, delta = compute_adjusted_caps(caps, tel, _params())
    assert new is caps
    assert delta == 0.0


def test_recall_leak_grows_caps() -> None:
    # Heavy COLD winner share, fast recall → recall_pressure dominates → grow.
    caps = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    tel = TierTelemetry(
        queries=300, winners_hot=60, winners_warm=0, winners_cold=240,
        winners_deep=0, elapsed_ms_sum=300.0 * 2.0, reminiscence_events=0,
    )
    new, delta = compute_adjusted_caps(caps, tel, _params())
    assert delta > 0.0
    assert new.hot_cap > caps.hot_cap
    assert new.warm_cap > caps.warm_cap
    assert new.cold_cap > caps.cold_cap


def test_no_leak_high_latency_shrinks_caps() -> None:
    # All winners in HOT, slow recall → latency_pressure dominates → shrink.
    caps = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    tel = TierTelemetry(
        queries=300, winners_hot=300, winners_warm=0, winners_cold=0,
        winners_deep=0, elapsed_ms_sum=300.0 * 40.0, reminiscence_events=0,
    )
    new, delta = compute_adjusted_caps(caps, tel, _params())
    assert delta < 0.0
    assert new.hot_cap < caps.hot_cap
    assert new.warm_cap < caps.warm_cap
    assert new.cold_cap < caps.cold_cap


def test_deadband_suppresses_small_signal() -> None:
    # Tiny leak, tiny latency → |delta| under the dead-band → no move.
    caps = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    tel = TierTelemetry(
        queries=300, winners_hot=299, winners_warm=0, winners_cold=1,
        winners_deep=0, elapsed_ms_sum=300.0 * 1.0, reminiscence_events=0,
    )
    p = _params(adaptive_tier_lambda=0.5, adaptive_tier_deadband=0.05)
    new, delta = compute_adjusted_caps(caps, tel, p)
    assert abs(delta) < 0.05
    assert new is caps


def test_max_step_clamps_large_signal() -> None:
    # Extreme leak → raw delta large, but the move is capped at ±max_step.
    caps = TierCapacityParams(hot_cap=1000, warm_cap=2000, cold_cap=20000)
    tel = TierTelemetry(
        queries=300, winners_hot=0, winners_warm=0, winners_cold=300,
        winners_deep=0, elapsed_ms_sum=300.0 * 1.0, reminiscence_events=0,
    )
    p = _params(adaptive_tier_lambda=0.0, adaptive_tier_max_step=0.2)
    new, _ = compute_adjusted_caps(caps, tel, p)
    # λ=0 → delta = recall_pressure = 1.0, clamped to +0.2 → ×1.2.
    assert new.hot_cap == 1200


def test_caps_clamped_to_min_floor() -> None:
    caps = TierCapacityParams(
        hot_cap=22, warm_cap=22, cold_cap=22, min_floor=20, max_cap=100_000
    )
    tel = TierTelemetry(
        queries=300, winners_hot=300, winners_warm=0, winners_cold=0,
        winners_deep=0, elapsed_ms_sum=300.0 * 40.0, reminiscence_events=0,
    )
    new, _ = compute_adjusted_caps(caps, tel, _params(adaptive_tier_lambda=1.0))
    assert new.hot_cap >= 20
    assert new.warm_cap >= 20
    assert new.cold_cap >= 20


def test_caps_clamped_to_max_cap() -> None:
    caps = TierCapacityParams(
        hot_cap=95_000, warm_cap=95_000, cold_cap=95_000,
        min_floor=20, max_cap=100_000,
    )
    tel = TierTelemetry(
        queries=300, winners_hot=0, winners_warm=0, winners_cold=300,
        winners_deep=0, elapsed_ms_sum=300.0 * 1.0, reminiscence_events=0,
    )
    new, _ = compute_adjusted_caps(caps, tel, _params(adaptive_tier_lambda=0.0))
    assert new.hot_cap <= 100_000
    assert new.warm_cap <= 100_000
    assert new.cold_cap <= 100_000


def test_ordering_preserved_after_clamp() -> None:
    # Unordered seed (TierCapacityParams has no ordering constraint).
    # After grow, the max() enforcement must re-impose hot <= warm <= cold.
    caps = TierCapacityParams(
        hot_cap=500, warm_cap=200, cold_cap=100,  # inverted on purpose
        min_floor=20, max_cap=100_000,
    )
    tel = TierTelemetry(
        queries=300, winners_hot=0, winners_warm=0, winners_cold=300,
        winners_deep=0, elapsed_ms_sum=300.0 * 1.0, reminiscence_events=0,
    )
    new, _ = compute_adjusted_caps(caps, tel, _params(adaptive_tier_lambda=0.0))
    assert new.hot_cap <= new.warm_cap <= new.cold_cap


def test_min_floor_max_cap_carried_through() -> None:
    caps = TierCapacityParams(
        hot_cap=200, warm_cap=2000, cold_cap=20000,
        min_floor=30, max_cap=50_000,
    )
    tel = TierTelemetry(
        queries=300, winners_hot=0, winners_warm=0, winners_cold=300,
        winners_deep=0, elapsed_ms_sum=300.0 * 1.0, reminiscence_events=0,
    )
    new, _ = compute_adjusted_caps(caps, tel, _params(adaptive_tier_lambda=0.0))
    assert new.min_floor == 30
    assert new.max_cap == 50_000
