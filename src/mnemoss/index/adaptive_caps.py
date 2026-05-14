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


# ─── workspace_meta persistence ────────────────────────────────────

# Key namespace in ``workspace_meta`` — groups all adaptive keys under
# one ``adaptive:`` prefix. ``reset`` enumerates the counter subset
# (``_RESETTABLE_KEYS``) explicitly. Mirrors CostLedger's key style.
_QUERIES_KEY = "adaptive:queries"
_ELAPSED_KEY = "adaptive:elapsed_ms_sum"
_REMINISCENCE_KEY = "adaptive:reminiscence_events"
_WINNER_KEYS = {
    IndexTier.HOT: "adaptive:winners_hot",
    IndexTier.WARM: "adaptive:winners_warm",
    IndexTier.COLD: "adaptive:winners_cold",
    IndexTier.DEEP: "adaptive:winners_deep",
}
_CAPS_HOT_KEY = "adaptive:caps_hot"
_CAPS_WARM_KEY = "adaptive:caps_warm"
_CAPS_COLD_KEY = "adaptive:caps_cold"
_LAST_DELTA_KEY = "adaptive:last_delta"

# Counter keys cleared by reset() — caps + last_delta deliberately survive.
_RESETTABLE_KEYS = (
    _QUERIES_KEY,
    _ELAPSED_KEY,
    _REMINISCENCE_KEY,
    *_WINNER_KEYS.values(),
)


class TierTelemetryLedger:
    """Recall-telemetry + effective-caps store in ``workspace_meta``.

    An exact mirror of ``dream/cost.py``'s ``CostLedger``: synchronous
    SQL against the memory DB's KV table. Cheap enough to hit on every
    recall — a handful of tiny UPSERTs on a key-value table.

    Counter keys (cleared by :meth:`reset`):
      ``adaptive:queries``, ``adaptive:winners_{hot,warm,cold,deep}``,
      ``adaptive:elapsed_ms_sum``, ``adaptive:reminiscence_events``.

    Effective-cap keys (survive :meth:`reset`):
      ``adaptive:caps_{hot,warm,cold}``, ``adaptive:last_delta``.
    """

    def __init__(self, conn: apsw.Connection) -> None:
        self._conn = conn

    # ─── writes ────────────────────────────────────────────────────

    def record_recall(
        self,
        *,
        winner_tiers: list[IndexTier],
        elapsed_ms: float,
        reminiscence: int,
    ) -> None:
        """Record one recall call. One tiny transaction, no-op-safe."""

        counts = {tier: 0 for tier in IndexTier}
        for tier in winner_tiers:
            counts[tier] += 1
        with self._conn:
            self._incr_int(_QUERIES_KEY, 1)
            for tier, key in _WINNER_KEYS.items():
                self._incr_int(key, counts[tier])
            self._incr_int(_REMINISCENCE_KEY, int(reminiscence))
            self._incr_float(_ELAPSED_KEY, float(elapsed_ms))

    def write_effective_caps(
        self, caps: TierCapacityParams, *, delta: float = 0.0
    ) -> None:
        """Persist the controller's effective caps + last delta."""

        with self._conn:
            self._set(_CAPS_HOT_KEY, str(caps.hot_cap))
            self._set(_CAPS_WARM_KEY, str(caps.warm_cap))
            self._set(_CAPS_COLD_KEY, str(caps.cold_cap))
            self._set(_LAST_DELTA_KEY, repr(float(delta)))

    def reset(self) -> None:
        """Clear the counter keys. Caps + last_delta are kept."""

        placeholders = ",".join("?" for _ in _RESETTABLE_KEYS)
        with self._conn:
            self._conn.execute(
                f"DELETE FROM workspace_meta WHERE k IN ({placeholders})",
                tuple(_RESETTABLE_KEYS),
            )

    # ─── reads ─────────────────────────────────────────────────────

    def read(self) -> TierTelemetry:
        return TierTelemetry(
            queries=self._read_int(_QUERIES_KEY),
            winners_hot=self._read_int(_WINNER_KEYS[IndexTier.HOT]),
            winners_warm=self._read_int(_WINNER_KEYS[IndexTier.WARM]),
            winners_cold=self._read_int(_WINNER_KEYS[IndexTier.COLD]),
            winners_deep=self._read_int(_WINNER_KEYS[IndexTier.DEEP]),
            elapsed_ms_sum=self._read_float(_ELAPSED_KEY),
            reminiscence_events=self._read_int(_REMINISCENCE_KEY),
        )

    def read_effective_caps(
        self, seed: TierCapacityParams
    ) -> TierCapacityParams:
        """Return the persisted effective caps, or ``seed`` if unset.

        ``min_floor`` / ``max_cap`` always come from ``seed`` — they are
        config, not tuned state.

        A persisted cap of ``0`` is treated as "unset" and triggers the
        ``seed`` fallback. This assumes effective caps are never
        legitimately ``0`` — which holds whenever ``min_floor >= 1``
        (the default ``min_floor`` is 20, so the controller never
        clamps a cap below 1 in practice).
        """

        hot = self._read_int(_CAPS_HOT_KEY)
        warm = self._read_int(_CAPS_WARM_KEY)
        cold = self._read_int(_CAPS_COLD_KEY)
        if hot == 0 or warm == 0 or cold == 0:
            return seed
        return TierCapacityParams(
            hot_cap=hot,
            warm_cap=warm,
            cold_cap=cold,
            min_floor=seed.min_floor,
            max_cap=seed.max_cap,
        )

    def snapshot(self, seed: TierCapacityParams) -> dict[str, Any]:
        """Json-safe view for ``status()``."""

        caps = self.read_effective_caps(seed)
        tel = self.read()
        return {
            "effective_caps": {
                "hot": caps.hot_cap,
                "warm": caps.warm_cap,
                "cold": caps.cold_cap,
            },
            "last_delta": self._read_float(_LAST_DELTA_KEY),
            "queries_since_adjustment": tel.queries,
            "winners": {
                "hot": tel.winners_hot,
                "warm": tel.winners_warm,
                "cold": tel.winners_cold,
                "deep": tel.winners_deep,
            },
        }

    # ─── internals ─────────────────────────────────────────────────

    def _incr_int(self, key: str, amount: int) -> None:
        self._conn.execute(
            "INSERT INTO workspace_meta(k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = CAST(v AS INTEGER) + ?",
            (key, str(int(amount)), int(amount)),
        )

    def _incr_float(self, key: str, amount: float) -> None:
        self._conn.execute(
            "INSERT INTO workspace_meta(k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = CAST(v AS REAL) + ?",
            (key, repr(float(amount)), float(amount)),
        )

    def _set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO workspace_meta(k, v) VALUES (?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (key, value),
        )

    def _read_int(self, key: str) -> int:
        row = self._conn.execute(
            "SELECT v FROM workspace_meta WHERE k = ?", (key,)
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (ValueError, TypeError):
            return 0

    def _read_float(self, key: str) -> float:
        row = self._conn.execute(
            "SELECT v FROM workspace_meta WHERE k = ?", (key,)
        ).fetchone()
        if row is None:
            return 0.0
        try:
            return float(row[0])
        except (ValueError, TypeError):
            return 0.0


# ─── orchestrator ──────────────────────────────────────────────────


def maybe_adjust_caps(
    ledger: TierTelemetryLedger,
    seed: TierCapacityParams,
    params: FormulaParams,
) -> TierCapacityParams:
    """Decide and apply one controller step. Returns the effective caps.

    - Flag off → return ``seed`` unchanged (no reads, no writes).
    - Not enough data yet (queries < min-dwell) → return the current
      effective caps, leave telemetry accumulating.
    - Enough data → compute, persist the new caps + delta, reset the
      telemetry window, return the new caps.

    Wired into ``rebalance()``; runs synchronously against the memory
    DB connection (same pattern as ``CostLedger``).
    """

    if not params.adaptive_tier_caps:
        return seed

    current = ledger.read_effective_caps(seed)
    telemetry = ledger.read()
    if telemetry.queries < params.adaptive_tier_min_queries:
        return current

    new_caps, delta = compute_adjusted_caps(current, telemetry, params)
    ledger.write_effective_caps(new_caps, delta=delta)
    ledger.reset()
    return new_caps
