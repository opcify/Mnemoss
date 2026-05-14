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


import apsw  # noqa: E402  (grouped with the ledger tests below)

from mnemoss.core.types import IndexTier  # noqa: E402
from mnemoss.index.adaptive_caps import TierTelemetryLedger  # noqa: E402


def _mem_conn() -> apsw.Connection:
    conn = apsw.Connection(":memory:")
    conn.execute(
        "CREATE TABLE workspace_meta(k TEXT PRIMARY KEY, v TEXT NOT NULL)"
    )
    return conn


def test_ledger_records_and_reads_back() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    ledger.record_recall(
        winner_tiers=[IndexTier.HOT, IndexTier.HOT, IndexTier.COLD],
        elapsed_ms=12.5,
        reminiscence=1,
    )
    ledger.record_recall(
        winner_tiers=[IndexTier.WARM],
        elapsed_ms=7.5,
        reminiscence=0,
    )
    tel = ledger.read()
    assert tel.queries == 2
    assert tel.winners_hot == 2
    assert tel.winners_warm == 1
    assert tel.winners_cold == 1
    assert tel.winners_deep == 0
    assert tel.elapsed_ms_sum == 20.0
    assert tel.reminiscence_events == 1


def test_ledger_read_on_empty_is_zero() -> None:
    ledger = TierTelemetryLedger(_mem_conn())
    tel = ledger.read()
    assert tel.queries == 0
    assert tel.elapsed_ms_sum == 0.0


def test_ledger_read_int_tolerates_garbage() -> None:
    conn = _mem_conn()
    conn.execute(
        "INSERT INTO workspace_meta(k, v) VALUES ('adaptive:queries', 'not-a-number')"
    )
    ledger = TierTelemetryLedger(conn)
    # corrupt value reads as 0, not a crash
    assert ledger.read().queries == 0


def test_ledger_reset_clears_counters_but_keeps_caps() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    ledger.record_recall(
        winner_tiers=[IndexTier.HOT], elapsed_ms=10.0, reminiscence=0
    )
    caps = TierCapacityParams(hot_cap=150, warm_cap=1500, cold_cap=15000)
    ledger.write_effective_caps(caps, delta=-0.12)
    ledger.reset()
    tel = ledger.read()
    assert tel.queries == 0
    # caps survive reset
    back = ledger.read_effective_caps(TierCapacityParams())
    assert back.hot_cap == 150
    assert back.warm_cap == 1500
    assert back.cold_cap == 15000
    # last_delta also survives reset (it is not a counter key)
    assert ledger.snapshot(TierCapacityParams())["last_delta"] == -0.12


def test_ledger_effective_caps_falls_back_to_seed() -> None:
    ledger = TierTelemetryLedger(_mem_conn())
    seed = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    back = ledger.read_effective_caps(seed)
    assert back.hot_cap == 200
    assert back.warm_cap == 2000
    assert back.cold_cap == 20000


def test_ledger_effective_caps_carry_seed_bounds() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    seed = TierCapacityParams(min_floor=33, max_cap=44_444)
    ledger.write_effective_caps(
        TierCapacityParams(hot_cap=100, warm_cap=1000, cold_cap=10000),
        delta=0.0,
    )
    back = ledger.read_effective_caps(seed)
    assert back.hot_cap == 100
    assert back.min_floor == 33
    assert back.max_cap == 44_444


def test_ledger_snapshot_shape() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    ledger.record_recall(
        winner_tiers=[IndexTier.HOT, IndexTier.COLD], elapsed_ms=9.0,
        reminiscence=0,
    )
    snap = ledger.snapshot(TierCapacityParams())
    assert set(snap) == {
        "effective_caps", "last_delta", "queries_since_adjustment", "winners"
    }
    assert snap["effective_caps"] == {
        "hot": 200, "warm": 2000, "cold": 20000
    }
    assert snap["queries_since_adjustment"] == 1
    assert snap["winners"] == {"hot": 1, "warm": 0, "cold": 1, "deep": 0}
    # status() requires json-safe primitives only.
    import json

    json.dumps(snap)
