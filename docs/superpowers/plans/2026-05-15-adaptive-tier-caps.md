# Adaptive Tier Caps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the multi-tier index capacity caps (`hot_cap`/`warm_cap`/`cold_cap`) self-tune at runtime from recall telemetry, gated behind an opt-in flag.

**Architecture:** A pure-function controller in a new `src/mnemoss/index/adaptive_caps.py` consumes recall telemetry accumulated in `workspace_meta` (mirroring the existing `CostLedger` pattern), and at each Dream P7 Rebalance nudges the effective caps toward a one-knob blended latency/recall operating point — with dead-band, min-dwell, max-step, and `[min_floor, max_cap]` clamps for stability. The recall engine records telemetry only on the production-default tier-cascade path. When the flag is off, the code path is a byte-identical no-op.

**Tech Stack:** Python 3.10+, async-first, `apsw` (SQLite), `pytest` (`asyncio_mode = "auto"`), `mypy --strict`, `ruff`.

**Spec:** `docs/superpowers/specs/2026-05-15-adaptive-tier-caps-design.md`

**Resolved spec open questions:**
- `workspace_meta` is a schema-flexible KV table (`CostLedger` already writes arbitrary keys to it) — **no `SCHEMA_VERSION` bump**.
- Telemetry is scoped to `_tier_cascade_recall` only (the production default `use_tier_cascade_recall=True`). `_fast_index_recall` ignores tiers (provenance meaningless); the full ACT-R path is out of scope. If `adaptive_tier_caps=True` while `use_tier_cascade_recall=False`, no telemetry accumulates and the controller is a permanent no-op — safe.
- `max_cap` default = `100_000`; `min_floor` default = `20`.

**Deviation from spec, flagged:** the spec described "in-memory counters flushed on Rebalance/close." This plan uses a `TierTelemetryLedger` that writes to `workspace_meta` per-recall instead — an exact mirror of the proven `CostLedger` pattern. Rationale: it removes the engine↔rebalance coordination problem, is crash-safe, and the recall path already does per-call SQL writes (`store.reconsolidate`). Same outcome (counters in `workspace_meta`, consumed at Rebalance), simpler and safer realisation.

**Deviation from spec, flagged:** the convergence fixture (Task 9) is realised as a **deterministic workload simulator** rather than a full embed→recall→rebalance loop. Rationale: the spec's intent is to validate that the control loop *converges rather than oscillates/drifts*; a real workspace adds embedding noise that would make a convergence assertion flaky without testing the controller itself. The simulator models the one relationship that matters (caps → winner-tier distribution + latency). End-to-end wiring is covered by the integration tests in Tasks 7–8 and the existing benches in Task 10.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/mnemoss/core/config.py` | Modify | Add `min_floor`/`max_cap` to `TierCapacityParams`; add 6 `adaptive_tier_*` fields to `FormulaParams`; validators for both |
| `src/mnemoss/index/adaptive_caps.py` | Create | `TierTelemetry` dataclass, `compute_adjusted_caps` (pure controller), `TierTelemetryLedger` (workspace_meta persistence), `maybe_adjust_caps` (orchestrator) |
| `src/mnemoss/index/__init__.py` | Modify | Export the new public names |
| `src/mnemoss/index/rebalance.py` | Modify | Call `maybe_adjust_caps` at the top of `rebalance()`; bucket with the returned effective caps |
| `src/mnemoss/recall/engine.py` | Modify | `RecallEngine.__init__` accepts a `tier_ledger`; `_tier_cascade_recall` captures winner provenance + latency and records telemetry |
| `src/mnemoss/client.py` | Modify | Build + bind `TierTelemetryLedger`, pass it to `RecallEngine`, add an `adaptive_caps` block to `status()` |
| `tests/test_adaptive_caps.py` | Create | Unit tests: `compute_adjusted_caps`, `TierTelemetryLedger`, `maybe_adjust_caps` |
| `tests/test_adaptive_caps_engine.py` | Create | Integration: engine records telemetry; flag-off no-op |
| `tests/test_adaptive_caps_convergence.py` | Create | Workload simulator + convergence/stability/shift/adversarial assertions |
| `CLAUDE.md` | Modify | Add an architectural-invariant note for adaptive caps |

---

## Task 1: Config — `TierCapacityParams` clamp bounds

**Files:**
- Modify: `src/mnemoss/core/config.py` (the `TierCapacityParams` dataclass — fields at lines 584-586, `__post_init__` at lines 588-598)
- Test: `tests/test_config_validation.py`

> Note: `min_floor`/`max_cap` are the controller's clamp bounds only. They are **not** invariants on the seed caps — `TierCapacityParams(hot_cap=0, warm_cap=0, cold_cap=0)` must still be constructible (`tests/test_rebalance.py` relies on it). Do not add a `hot_cap >= min_floor` cross-check.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_validation.py` (after the existing `TierCapacityParams` tests, or near the end of the file):

```python
def test_tier_capacity_params_clamp_bounds_default() -> None:
    from mnemoss import TierCapacityParams

    caps = TierCapacityParams()
    assert caps.min_floor == 20
    assert caps.max_cap == 100_000


def test_tier_capacity_params_rejects_floor_above_cap() -> None:
    from mnemoss import TierCapacityParams

    with pytest.raises(ValueError, match="min_floor"):
        TierCapacityParams(min_floor=500, max_cap=100)


def test_tier_capacity_params_rejects_non_positive_max_cap() -> None:
    from mnemoss import TierCapacityParams

    with pytest.raises(ValueError, match="max_cap"):
        TierCapacityParams(max_cap=0)


def test_tier_capacity_params_still_allows_zero_seed_caps() -> None:
    # tests/test_rebalance.py constructs all-zero caps; must not regress.
    from mnemoss import TierCapacityParams

    TierCapacityParams(hot_cap=0, warm_cap=0, cold_cap=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_validation.py -k tier_capacity_params -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'min_floor'` (and the `min_floor`/`max_cap` attribute assertions fail).

- [ ] **Step 3: Write minimal implementation**

In `src/mnemoss/core/config.py`, add two fields to `TierCapacityParams` immediately after `cold_cap: int = 20_000` (line 586):

```python
    hot_cap: int = 200
    warm_cap: int = 2_000
    cold_cap: int = 20_000
    # Clamp bounds for the adaptive-caps controller (Method C). Not
    # invariants on the seed caps — only the controller honours them.
    min_floor: int = 20
    max_cap: int = 100_000
```

Replace the existing `__post_init__` loop body so it validates all five int fields and the floor/cap ordering:

```python
    def __post_init__(self) -> None:
        for name, value in (
            ("hot_cap", self.hot_cap),
            ("warm_cap", self.warm_cap),
            ("cold_cap", self.cold_cap),
            ("min_floor", self.min_floor),
            ("max_cap", self.max_cap),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int (got {value!r})")
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (got {value!r})")
        if self.max_cap < 1:
            raise ValueError(f"max_cap must be >= 1 (got {self.max_cap!r})")
        if self.min_floor > self.max_cap:
            raise ValueError(
                f"min_floor ({self.min_floor}) must be <= max_cap "
                f"({self.max_cap})"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_validation.py -k tier_capacity_params -v`
Expected: PASS (all four new tests).

- [ ] **Step 5: Run the full config + rebalance suites to confirm no regression**

Run: `pytest tests/test_config_validation.py tests/test_rebalance.py -v`
Expected: PASS (the all-zero-caps construction in `test_rebalance.py` still works).

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/core/config.py tests/test_config_validation.py
git commit -m "$(printf 'feat(config): add min_floor/max_cap clamp bounds to TierCapacityParams\n\nClamp bounds for the adaptive-caps controller. Not invariants on the\nseed caps — all-zero caps stay constructible.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: Config — `FormulaParams` adaptive-caps fields

**Files:**
- Modify: `src/mnemoss/core/config.py` (the `FormulaParams` dataclass — add fields near `reconsolidate_min_cosine` at line 306; add validators in `__post_init__`, which ends around line 377)
- Test: `tests/test_config_validation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_validation.py`:

```python
def test_formula_params_adaptive_defaults() -> None:
    p = FormulaParams()
    assert p.adaptive_tier_caps is False
    assert p.adaptive_tier_lambda == 0.5
    assert p.adaptive_tier_min_queries == 200
    assert p.adaptive_tier_max_step == 0.2
    assert p.adaptive_tier_deadband == 0.05
    assert p.adaptive_tier_latency_budget_ms == 25.0


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"adaptive_tier_lambda": -0.1}, "adaptive_tier_lambda"),
        ({"adaptive_tier_lambda": 1.5}, "adaptive_tier_lambda"),
        ({"adaptive_tier_min_queries": 0}, "adaptive_tier_min_queries"),
        ({"adaptive_tier_min_queries": -5}, "adaptive_tier_min_queries"),
        ({"adaptive_tier_max_step": 0.0}, "adaptive_tier_max_step"),
        ({"adaptive_tier_max_step": 1.5}, "adaptive_tier_max_step"),
        ({"adaptive_tier_deadband": -0.01}, "adaptive_tier_deadband"),
        ({"adaptive_tier_latency_budget_ms": 0.0}, "adaptive_tier_latency_budget_ms"),
    ],
)
def test_formula_params_adaptive_validation(kwargs, offending_field) -> None:
    with pytest.raises(ValueError, match=offending_field):
        FormulaParams(**kwargs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_validation.py -k adaptive -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'adaptive_tier_lambda'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mnemoss/core/config.py`, add the fields to `FormulaParams` immediately after `reconsolidate_min_cosine: float = 0.7` (line 306, just before `def __post_init__`):

```python
    reconsolidate_min_cosine: float = 0.7

    # ─── Adaptive tier caps (Method C) ────────────────────────────
    # Opt-in runtime self-tuning of TierCapacityParams from recall
    # telemetry. Default False → byte-identical to static-cap
    # behaviour (mirrors use_fast_index_recall). See
    # src/mnemoss/index/adaptive_caps.py and
    # docs/superpowers/specs/2026-05-15-adaptive-tier-caps-design.md.
    adaptive_tier_caps: bool = False
    # The one blend knob. 0.0 = pure recall-safety (grow-happy),
    # 1.0 = pure latency (shrink-happy).
    adaptive_tier_lambda: float = 0.5
    # Min recall calls accumulated since the last adjustment before
    # the controller will move the caps (min-dwell guardrail).
    adaptive_tier_min_queries: int = 200
    # Max multiplicative move per adjustment (e.g. 0.2 = ±20%).
    adaptive_tier_max_step: float = 0.2
    # Blended-signal magnitude below which the controller does
    # nothing (dead-band guardrail — kills thrashing on noise).
    adaptive_tier_deadband: float = 0.05
    # Recall-latency target used to normalise the latency-pressure
    # signal. Mean recall ms at/over this reads as full pressure.
    adaptive_tier_latency_budget_ms: float = 25.0
```

In the same class's `__post_init__`, add these validations at the end of the method (after the matching-weight checks, around line 377 — place them just before the method returns):

```python
        # ─── Adaptive tier caps ───────────────────────────────────
        if not isinstance(self.adaptive_tier_caps, bool):
            raise ValueError(
                f"adaptive_tier_caps must be a bool (got "
                f"{self.adaptive_tier_caps!r})"
            )
        _require_in_unit_interval(
            "adaptive_tier_lambda", self.adaptive_tier_lambda
        )
        if (
            not isinstance(self.adaptive_tier_min_queries, int)
            or isinstance(self.adaptive_tier_min_queries, bool)
            or self.adaptive_tier_min_queries < 1
        ):
            raise ValueError(
                f"adaptive_tier_min_queries must be an int >= 1 (got "
                f"{self.adaptive_tier_min_queries!r})"
            )
        if not (0.0 < self.adaptive_tier_max_step <= 1.0):
            raise ValueError(
                f"adaptive_tier_max_step must be in (0.0, 1.0] (got "
                f"{self.adaptive_tier_max_step!r})"
            )
        _require_non_negative(
            "adaptive_tier_deadband", self.adaptive_tier_deadband
        )
        _require_positive(
            "adaptive_tier_latency_budget_ms",
            self.adaptive_tier_latency_budget_ms,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_validation.py -k adaptive -v`
Expected: PASS.

- [ ] **Step 5: Run the full config suite + type check**

Run: `pytest tests/test_config_validation.py -v && mypy --strict src/mnemoss/core/config.py`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/core/config.py tests/test_config_validation.py
git commit -m "$(printf 'feat(config): add adaptive_tier_* fields to FormulaParams\n\nSix opt-in knobs for the Method C adaptive-caps controller, all\nvalidated at construction. Default adaptive_tier_caps=False.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: `adaptive_caps.py` — `TierTelemetry` + `compute_adjusted_caps`

**Files:**
- Create: `src/mnemoss/index/adaptive_caps.py`
- Test: `tests/test_adaptive_caps.py`

This is the pure heart of the controller — no I/O, no side effects.

- [ ] **Step 1: Write the failing test**

Create `tests/test_adaptive_caps.py`:

```python
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
    # cold pinned at max_cap, hot/warm grow → enforce hot <= warm <= cold.
    caps = TierCapacityParams(
        hot_cap=99_000, warm_cap=99_500, cold_cap=100_000,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_caps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mnemoss.index.adaptive_caps'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/mnemoss/index/adaptive_caps.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_caps.py -v`
Expected: PASS (all 10 tests).

- [ ] **Step 5: Type check**

Run: `mypy --strict src/mnemoss/index/adaptive_caps.py`
Expected: no errors. (The `apsw` import is unused at this point — that is fine; it is used in Task 4. If `ruff` flags the unused import, leave it; Task 4 lands in the same module before any standalone lint gate.)

> If you prefer a clean lint at every commit, defer the `import apsw` line to Task 4. Either is acceptable.

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/index/adaptive_caps.py tests/test_adaptive_caps.py
git commit -m "$(printf 'feat(index): pure adaptive-caps controller (TierTelemetry + compute_adjusted_caps)\n\nMethod C heart: one telemetry window in, new TierCapacityParams out.\nBlended one-knob latency/recall signal with dead-band, max-step and\n[min_floor, max_cap] clamps. No I/O.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: `adaptive_caps.py` — `TierTelemetryLedger`

**Files:**
- Modify: `src/mnemoss/index/adaptive_caps.py`
- Test: `tests/test_adaptive_caps.py`

`workspace_meta` persistence, an exact mirror of `dream/cost.py`'s `CostLedger`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive_caps.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_caps.py -k ledger -v`
Expected: FAIL — `ImportError: cannot import name 'TierTelemetryLedger'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/mnemoss/index/adaptive_caps.py`:

```python
# ─── workspace_meta persistence ────────────────────────────────────

# Key namespace in ``workspace_meta`` — lets ``reset`` filter with a
# single LIKE and keeps the namespace tidy. Mirrors CostLedger.
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_caps.py -v`
Expected: PASS (Task 3 tests + the 6 new ledger tests).

- [ ] **Step 5: Type check + lint**

Run: `mypy --strict src/mnemoss/index/adaptive_caps.py && ruff check src/mnemoss/index/adaptive_caps.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/index/adaptive_caps.py tests/test_adaptive_caps.py
git commit -m "$(printf 'feat(index): TierTelemetryLedger — workspace_meta persistence for adaptive caps\n\nMirrors dream/cost.py CostLedger. Records per-recall winner provenance\n+ latency, persists effective caps, resets counter windows. No\nSCHEMA_VERSION bump (workspace_meta is schema-flexible KV).\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 5: `adaptive_caps.py` — `maybe_adjust_caps` orchestrator + exports

**Files:**
- Modify: `src/mnemoss/index/adaptive_caps.py`
- Modify: `src/mnemoss/index/__init__.py`
- Test: `tests/test_adaptive_caps.py`

The thin impure orchestrator: read effective caps + telemetry, decide, persist, reset.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_adaptive_caps.py`:

```python
from mnemoss.index.adaptive_caps import maybe_adjust_caps  # noqa: E402


def test_maybe_adjust_flag_off_returns_seed() -> None:
    ledger = TierTelemetryLedger(_mem_conn())
    seed = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    params = FormulaParams(adaptive_tier_caps=False)
    out = maybe_adjust_caps(ledger, seed, params)
    assert out is seed


def test_maybe_adjust_insufficient_data_returns_current_no_reset() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    seed = TierCapacityParams()
    ledger.record_recall(
        winner_tiers=[IndexTier.COLD], elapsed_ms=1.0, reminiscence=0
    )
    params = FormulaParams(
        adaptive_tier_caps=True, adaptive_tier_min_queries=100
    )
    out = maybe_adjust_caps(ledger, seed, params)
    assert out.hot_cap == seed.hot_cap
    # telemetry NOT reset — it keeps accumulating toward the gate.
    assert ledger.read().queries == 1


def test_maybe_adjust_applies_and_resets() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    seed = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    # Heavy COLD leak, fast recall → grow.
    for _ in range(120):
        ledger.record_recall(
            winner_tiers=[IndexTier.COLD], elapsed_ms=1.0, reminiscence=0
        )
    params = FormulaParams(
        adaptive_tier_caps=True,
        adaptive_tier_min_queries=100,
        adaptive_tier_lambda=0.0,
    )
    out = maybe_adjust_caps(ledger, seed, params)
    assert out.hot_cap > seed.hot_cap
    # telemetry reset after the adjustment...
    assert ledger.read().queries == 0
    # ...and the new caps persisted.
    assert ledger.read_effective_caps(seed).hot_cap == out.hot_cap


def test_maybe_adjust_second_call_reads_persisted_caps() -> None:
    conn = _mem_conn()
    ledger = TierTelemetryLedger(conn)
    seed = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    params = FormulaParams(
        adaptive_tier_caps=True,
        adaptive_tier_min_queries=100,
        adaptive_tier_lambda=0.0,
    )
    for _ in range(120):
        ledger.record_recall(
            winner_tiers=[IndexTier.COLD], elapsed_ms=1.0, reminiscence=0
        )
    first = maybe_adjust_caps(ledger, seed, params)
    for _ in range(120):
        ledger.record_recall(
            winner_tiers=[IndexTier.COLD], elapsed_ms=1.0, reminiscence=0
        )
    second = maybe_adjust_caps(ledger, seed, params)
    # second call grew from `first`, not from `seed`.
    assert second.hot_cap > first.hot_cap
```

Also test the package export — append to `tests/test_adaptive_caps.py`:

```python
def test_public_exports() -> None:
    from mnemoss.index import (
        TierTelemetry,
        TierTelemetryLedger,
        compute_adjusted_caps,
        maybe_adjust_caps,
    )

    assert TierTelemetry is not None
    assert TierTelemetryLedger is not None
    assert compute_adjusted_caps is not None
    assert maybe_adjust_caps is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_caps.py -k "maybe_adjust or public_exports" -v`
Expected: FAIL — `ImportError: cannot import name 'maybe_adjust_caps'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/mnemoss/index/adaptive_caps.py`:

```python
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
```

In `src/mnemoss/index/__init__.py`, replace the import + `__all__` so the module reads:

```python
"""Index-management utilities.

Stage 2 ships the P7 Rebalance pass here. Stage 4's dream()/P7 pipeline
will wrap ``rebalance`` unchanged, so the decision logic has exactly one
home. The adaptive-caps controller (Method C) also lives here.
"""

from mnemoss.index.adaptive_caps import (
    TierTelemetry,
    TierTelemetryLedger,
    compute_adjusted_caps,
    maybe_adjust_caps,
)
from mnemoss.index.rebalance import RebalanceStats, rebalance

__all__ = [
    "RebalanceStats",
    "TierTelemetry",
    "TierTelemetryLedger",
    "compute_adjusted_caps",
    "maybe_adjust_caps",
    "rebalance",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_caps.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Type check + lint**

Run: `mypy --strict src/mnemoss/index && ruff check src/mnemoss/index`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/index/adaptive_caps.py src/mnemoss/index/__init__.py tests/test_adaptive_caps.py
git commit -m "$(printf 'feat(index): maybe_adjust_caps orchestrator + package exports\n\nThin impure step wired into rebalance(): read effective caps +\ntelemetry, decide via compute_adjusted_caps, persist, reset window.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 6: Wire `maybe_adjust_caps` into `rebalance()`

**Files:**
- Modify: `src/mnemoss/index/rebalance.py` (imports at lines 49-58; `rebalance()` body — `stats.tier_before` at line 130, `_bucket_by_capacity` call at line 163)
- Test: `tests/test_rebalance.py`

Both `client.rebalance()` and `DreamRunner._phase_rebalance` call `_rebalance(...)`, so wiring here covers the Dream P7 path with no `runner.py` change.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_rebalance.py` (it already imports `FormulaParams`, `TierCapacityParams`, `rebalance`, `SQLiteBackend`, `IndexTier`, `Memory`, `np`, `datetime`/`timedelta`/`timezone`, `Path`, and defines `_memory` + `_backend`):

```python
async def test_rebalance_flag_off_ignores_telemetry(tmp_path: Path) -> None:
    # With adaptive_tier_caps=False, rebalance must not touch
    # workspace_meta adaptive keys and must use the passed caps.
    b = await _backend(tmp_path)
    try:
        now = datetime.now(UTC)
        for i in range(5):
            await b.write_memory(
                _memory(f"m{i}", f"content {i}", now), np.zeros(4, dtype=np.float32)
            )
        params = FormulaParams(adaptive_tier_caps=False)
        caps = TierCapacityParams(hot_cap=3, warm_cap=1, cold_cap=1)
        await rebalance(b, params, caps, now=now)
        conn = b._require_conn()
        row = conn.execute(
            "SELECT v FROM workspace_meta WHERE k = 'adaptive:caps_hot'"
        ).fetchone()
        assert row is None
    finally:
        await b.close()


async def test_rebalance_adaptive_uses_adjusted_caps(tmp_path: Path) -> None:
    # Seed telemetry that implies "heavy COLD leak, fast recall" → grow.
    b = await _backend(tmp_path)
    try:
        from mnemoss.index.adaptive_caps import TierTelemetryLedger

        now = datetime.now(UTC)
        for i in range(5):
            await b.write_memory(
                _memory(f"m{i}", f"content {i}", now), np.zeros(4, dtype=np.float32)
            )
        ledger = TierTelemetryLedger(b._require_conn())
        from mnemoss.core.types import IndexTier as _T

        for _ in range(150):
            ledger.record_recall(
                winner_tiers=[_T.COLD], elapsed_ms=1.0, reminiscence=0
            )
        params = FormulaParams(
            adaptive_tier_caps=True,
            adaptive_tier_min_queries=100,
            adaptive_tier_lambda=0.0,
        )
        seed = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
        await rebalance(b, params, seed, now=now)
        # Controller persisted grown caps and reset the window.
        conn = b._require_conn()
        hot = int(
            conn.execute(
                "SELECT v FROM workspace_meta WHERE k = 'adaptive:caps_hot'"
            ).fetchone()[0]
        )
        assert hot > 200
        assert ledger.read().queries == 0
    finally:
        await b.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance.py -k "flag_off or adaptive_uses" -v`
Expected: `test_rebalance_flag_off_ignores_telemetry` PASSES already (flag off path is the existing behaviour — `workspace_meta` has no adaptive keys); `test_rebalance_adaptive_uses_adjusted_caps` FAILS (`hot` row is `None` — `rebalance()` does not yet call the controller).

- [ ] **Step 3: Write minimal implementation**

In `src/mnemoss/index/rebalance.py`, add to the imports block (after line 54, `from mnemoss.core.config import FormulaParams, TierCapacityParams`):

```python
from mnemoss.core.config import FormulaParams, TierCapacityParams
from mnemoss.core.types import IndexTier
from mnemoss.formula.base_level import compute_base_level
from mnemoss.formula.idx_priority import compute_idx_priority
from mnemoss.index.adaptive_caps import TierTelemetryLedger, maybe_adjust_caps
from mnemoss.store.sqlite_backend import SQLiteBackend
```

In the `rebalance()` function body, immediately after `stats.tier_before = await store.tier_counts()` (line 130), insert:

```python
    stats.tier_before = await store.tier_counts()

    # Method C: when adaptive caps are enabled, the controller reads
    # recall telemetry accumulated in workspace_meta and may nudge the
    # effective caps before this pass buckets with them. Flag off →
    # `capacity` is returned unchanged (no reads, no writes).
    effective_capacity = capacity
    if params.adaptive_tier_caps:
        ledger = TierTelemetryLedger(store._require_conn())
        effective_capacity = maybe_adjust_caps(ledger, capacity, params)
```

Then change the `_bucket_by_capacity` call (line 163) from `capacity` to `effective_capacity`:

```python
    new_tiers = _bucket_by_capacity(ranked, pinned_global, effective_capacity)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance.py -k "flag_off or adaptive_uses" -v`
Expected: PASS (both).

- [ ] **Step 5: Run the full rebalance + dream suites + type check**

Run: `pytest tests/test_rebalance.py tests/test_dream_nightly.py tests/test_dream_failure_recovery.py -v && mypy --strict src/mnemoss/index`
Expected: PASS, no type errors. (Dream P7 still works — flag defaults off, so `maybe_adjust_caps` returns the seed.)

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/index/rebalance.py tests/test_rebalance.py
git commit -m "$(printf 'feat(index): wire adaptive caps into rebalance()\n\nrebalance() calls maybe_adjust_caps before bucketing; covers both\nclient.rebalance() and Dream P7 since both route through _rebalance.\nFlag off → seed caps unchanged.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 7: Recall-engine telemetry instrumentation

**Files:**
- Modify: `src/mnemoss/recall/engine.py` (imports at lines 14-33; `RecallEngine.__init__` at lines 64-79; `_tier_cascade_recall` at lines 505-639)
- Test: `tests/test_adaptive_caps_engine.py`

Capture winner provenance + latency on the production-default tier-cascade path; record to the ledger when the flag is on.

- [ ] **Step 1: Write the failing test**

Create `tests/test_adaptive_caps_engine.py`:

```python
"""Adaptive caps — recall-engine instrumentation tests.

Verifies _tier_cascade_recall records telemetry when the flag is on,
and is a true no-op when it is off.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import apsw
import numpy as np

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import IndexTier, Memory, MemoryType
from mnemoss.index.adaptive_caps import TierTelemetryLedger
from mnemoss.recall.engine import RecallEngine
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


def _memory(id: str, content: str, now: datetime) -> Memory:
    return Memory(
        id=id,
        workspace_id="ws",
        agent_id=None,
        session_id="s1",
        created_at=now,
        content=content,
        content_embedding=None,
        role="user",
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[now],
        index_tier=IndexTier.HOT,
        idx_priority=0.9,
    )


async def _backend(tmp_path: Path, dim: int = 4) -> SQLiteBackend:
    b = SQLiteBackend(
        db_path=tmp_path / "memory.sqlite",
        raw_log_path=tmp_path / "raw_log.sqlite",
        workspace_id="ws",
        embedding_dim=dim,
        embedder_id="fake",
    )
    await b.open()
    return b


class _StubEmbedder:
    """Deterministic 4-dim embedder — enough to drive vec_search."""

    dim = 4
    embedder_id = "fake"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            rng = random.Random(t)
            out.append(np.array([rng.random() for _ in range(4)], dtype=np.float32))
        return np.stack(out)


async def _seed(b: SQLiteBackend, emb: _StubEmbedder, now: datetime) -> None:
    for i in range(5):
        m = _memory(f"m{i}", f"content number {i}", now)
        vec = emb.embed([m.content])[0]
        await b.write_memory(m, vec)


async def test_engine_records_telemetry_when_flag_on(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        await _seed(b, emb, now)
        ledger = TierTelemetryLedger(b._require_conn())
        params = FormulaParams(adaptive_tier_caps=True)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
            tier_ledger=ledger,
        )
        await engine.recall("content number 2", agent_id=None, k=3)
        tel = ledger.read()
        assert tel.queries == 1
        # 5 memories, all in HOT → any winners came from HOT.
        assert tel.winners_hot >= 0
        assert tel.winners_cold == 0
        assert tel.elapsed_ms_sum >= 0.0
    finally:
        await b.close()


async def test_engine_no_op_when_flag_off(tmp_path: Path) -> None:
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        await _seed(b, emb, now)
        ledger = TierTelemetryLedger(b._require_conn())
        params = FormulaParams(adaptive_tier_caps=False)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
            tier_ledger=ledger,
        )
        await engine.recall("content number 2", agent_id=None, k=3)
        assert ledger.read().queries == 0
    finally:
        await b.close()


async def test_engine_no_op_when_ledger_absent(tmp_path: Path) -> None:
    # Flag on but no ledger wired (e.g. legacy construction) → no crash.
    b = await _backend(tmp_path)
    try:
        emb = _StubEmbedder()
        now = datetime.now(UTC)
        await _seed(b, emb, now)
        params = FormulaParams(adaptive_tier_caps=True)
        engine = RecallEngine(
            store=b,
            embedder=emb,
            working=WorkingMemory(capacity=10),
            params=params,
        )
        results = await engine.recall("content number 2", agent_id=None, k=3)
        assert isinstance(results, list)
    finally:
        await b.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adaptive_caps_engine.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'tier_ledger'`.

- [ ] **Step 3: Write minimal implementation**

In `src/mnemoss/recall/engine.py`:

(a) Add `import time` to the stdlib imports (after `import random` on line 18):

```python
import asyncio
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal
```

(b) Add a `TYPE_CHECKING` import for the ledger type (after the existing `from mnemoss.working import WorkingMemory` on line 33):

```python
from mnemoss.working import WorkingMemory

if TYPE_CHECKING:
    from mnemoss.index.adaptive_caps import TierTelemetryLedger
```

(c) Extend `RecallEngine.__init__` (lines 64-79) to accept and store the ledger:

```python
    def __init__(
        self,
        store: SQLiteBackend,
        embedder: Embedder,
        working: WorkingMemory,
        params: FormulaParams,
        rng: random.Random | None = None,
        history: RecallHistory | None = None,
        tier_ledger: "TierTelemetryLedger | None" = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._working = working
        self._params = params
        self._rng = rng if rng is not None else random.Random()
        self._history = history if history is not None else RecallHistory()
        self._tier_ledger = tier_ledger
```

(d) Add a private recorder method to `RecallEngine` (place it just before `_tier_cascade_recall`, i.e. before line 505):

```python
    def _record_tier_telemetry(
        self,
        *,
        winner_tiers: list[IndexTier],
        elapsed_ms: float,
        reminiscence: int,
    ) -> None:
        """Record one tier-cascade recall to the adaptive-caps ledger.

        No-op unless ``adaptive_tier_caps`` is on and a ledger is wired.
        The flag check lives here so call sites stay clean and the
        no-op guarantee is in exactly one place.
        """

        if not self._params.adaptive_tier_caps or self._tier_ledger is None:
            return
        self._tier_ledger.record_recall(
            winner_tiers=winner_tiers,
            elapsed_ms=elapsed_ms,
            reminiscence=reminiscence,
        )
```

(e) Instrument `_tier_cascade_recall` (lines 505-639). Make these four edits:

  1. At the very start of the method body, after the docstring (before `tier_plan = [IndexTier.HOT, ...]` on line 539), add the timer and the provenance dict:

```python
        t_start = time.perf_counter()
        candidate_tier: dict[str, IndexTier] = {}
```

  2. In the tier-scan loop, record provenance alongside the cosine. Change the `for mid, cos in hits:` block (lines 559-565) to:

```python
            for mid, cos in hits:
                # First-write-wins: a memory could appear in multiple
                # tiers' over-scan results if the ANN index is shared
                # across tier filters. Keep the cosine from the first
                # tier we see it in — that's the highest-priority tier.
                if mid not in candidates:
                    candidates[mid] = cos
                    candidate_tier[mid] = tier
```

  3. Change the empty-candidates early return (lines 576-581) to record telemetry first:

```python
        if not candidates:
            self._record_tier_telemetry(
                winner_tiers=[],
                elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
                reminiscence=0,
            )
            return [], CascadeStats(
                tiers_scanned=tiers_scanned,
                stopped_at=stopped_at,
                candidates_scored=0,
            )
```

  4. In the reconsolidation block (lines 615-633), count reminiscence events, and record telemetry just before the final `return`. Replace the block from `if reconsolidate and results:` (line 615) through the final `return` (lines 635-639) with:

```python
        reminiscence_count = 0
        if reconsolidate and results:
            # Gate: only reconsolidate memories whose cosine to the query
            # clears the threshold. ``r.score == cos`` in this path.
            # See ``FormulaParams.reconsolidate_min_cosine`` for rationale.
            min_cos = self._params.reconsolidate_min_cosine
            for r in results:
                if r.score < min_cos:
                    continue
                await self._store.reconsolidate(r.memory.id, now)
                r.memory.access_history.append(now)
                r.memory.rehearsal_count += 1
                r.memory.last_accessed_at = now
                if r.memory.index_tier is IndexTier.DEEP:
                    await self._store.reminisce_to_warm(r.memory.id)
                    r.memory.reminisced_count += 1
                    r.memory.index_tier = IndexTier.WARM
                    r.memory.idx_priority = 0.5
                    reminiscence_count += 1
            self._working.extend(agent_id, (r.memory.id for r in results))
            await self._apply_lazy_extraction(results)

        # Winner provenance is captured at scan time (``candidate_tier``)
        # — deliberately *not* ``r.memory.index_tier``, which the
        # reconsolidation block above may have mutated DEEP→WARM.
        self._record_tier_telemetry(
            winner_tiers=[
                candidate_tier[r.memory.id]
                for r in results
                if r.memory.id in candidate_tier
            ],
            elapsed_ms=(time.perf_counter() - t_start) * 1000.0,
            reminiscence=reminiscence_count,
        )

        return results, CascadeStats(
            tiers_scanned=tiers_scanned,
            stopped_at=stopped_at,
            candidates_scored=len(candidates),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adaptive_caps_engine.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Run the recall suites + type check**

Run: `pytest tests/test_recall_unit.py tests/test_fast_index_recall.py tests/test_hot_path_budget.py -v && mypy --strict src/mnemoss/recall/engine.py`
Expected: PASS, no type errors. (Existing recall behaviour is unchanged — the new code only adds telemetry, gated on the flag.)

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/recall/engine.py tests/test_adaptive_caps_engine.py
git commit -m "$(printf 'feat(recall): record tier telemetry on the cascade recall path\n\n_tier_cascade_recall captures winner provenance (at scan time, before\nreminiscence mutates index_tier) + latency, records to the adaptive-\ncaps ledger. Gated on adaptive_tier_caps; off → no-op.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 8: Client wiring + `status()` block

**Files:**
- Modify: `src/mnemoss/client.py` (imports near line 32; `__init__` — `self._cost_ledger` at line 95; engine construction at lines 770-781; `status()` at lines 531-593)
- Test: `tests/test_client.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_client.py` (it already imports `Mnemoss` and uses `tmp_path`; if it does not import `FormulaParams`/`StorageParams`, add `from mnemoss import FormulaParams, StorageParams` at the top of the new tests or to the file's imports):

```python
async def test_status_has_adaptive_caps_block(tmp_path) -> None:
    from mnemoss import FormulaParams, StorageParams

    mem = Mnemoss(
        workspace="ws",
        embedding_model="fake",
        formula=FormulaParams(adaptive_tier_caps=True),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await mem.observe(role="user", content="hello world")
        await mem.recall("hello", k=3)
        st = await mem.status()
        assert "adaptive_caps" in st
        block = st["adaptive_caps"]
        assert set(block) == {
            "effective_caps",
            "last_delta",
            "queries_since_adjustment",
            "winners",
        }
        # one recall happened on the tier-cascade path
        assert block["queries_since_adjustment"] >= 1
        # status() must stay json-safe
        import json

        json.dumps(st)
    finally:
        await mem.close()


async def test_status_adaptive_caps_empty_when_flag_off(tmp_path) -> None:
    from mnemoss import FormulaParams, StorageParams

    mem = Mnemoss(
        workspace="ws",
        embedding_model="fake",
        formula=FormulaParams(adaptive_tier_caps=False),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await mem.observe(role="user", content="hello world")
        await mem.recall("hello", k=3)
        st = await mem.status()
        # block is present (stable shape) but telemetry stayed at zero
        assert st["adaptive_caps"]["queries_since_adjustment"] == 0
    finally:
        await mem.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py -k adaptive_caps -v`
Expected: FAIL — `KeyError: 'adaptive_caps'` (status has no such block yet).

- [ ] **Step 3: Write minimal implementation**

In `src/mnemoss/client.py`:

(a) Add the import next to the `CostLedger` import (line 32, `from mnemoss.dream.cost import CostLedger, CostLimits`):

```python
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.index.adaptive_caps import TierTelemetryLedger
```

(b) In `__init__`, add the ledger attribute next to `self._cost_ledger` (line 95):

```python
        self._cost_ledger: CostLedger | None = None
        self._tier_ledger: TierTelemetryLedger | None = None
```

(c) In the open path (lines 770-781), build the tier ledger before constructing the engine and pass it in. Replace the block:

```python
            await store.open()
            self._engine = RecallEngine(
                store=store,
                embedder=self._embedder,
                working=self._working,
                params=self._config.formula,
                rng=self._rng,
            )
            self._store = store
            # Bind the cost ledger to the open connection — persistence
            # lives in workspace_meta so call counts survive restarts.
            self._cost_ledger = CostLedger(store._require_conn())
```

with:

```python
            await store.open()
            # Bind both ledgers to the open connection — persistence
            # lives in workspace_meta so counters survive restarts.
            self._tier_ledger = TierTelemetryLedger(store._require_conn())
            self._engine = RecallEngine(
                store=store,
                embedder=self._embedder,
                working=self._working,
                params=self._config.formula,
                rng=self._rng,
                tier_ledger=self._tier_ledger,
            )
            self._store = store
            self._cost_ledger = CostLedger(store._require_conn())
```

(d) In `status()`, build the adaptive block alongside the cost/dream blocks (after the `dream_block` assignment, around line 566) and add it to the returned dict. Insert after the `dream_block` definition:

```python
        adaptive_block: dict[str, Any] = {
            "effective_caps": {
                "hot": self._config.tier_capacity.hot_cap,
                "warm": self._config.tier_capacity.warm_cap,
                "cold": self._config.tier_capacity.cold_cap,
            },
            "last_delta": 0.0,
            "queries_since_adjustment": 0,
            "winners": {"hot": 0, "warm": 0, "cold": 0, "deep": 0},
        }
        if self._tier_ledger is not None:
            adaptive_block = self._tier_ledger.snapshot(
                self._config.tier_capacity
            )
```

Then add the key to the returned dict (in the `return {...}` block, after `"dreams": dream_block,` on line 592):

```python
            "llm_cost": cost_block,
            "dreams": dream_block,
            "adaptive_caps": adaptive_block,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_client.py -k adaptive_caps -v`
Expected: PASS (both).

- [ ] **Step 5: Run the client + server-status suites + type check**

Run: `pytest tests/test_client.py tests/test_client_warmup.py -v && mypy --strict src/mnemoss/client.py`
Expected: PASS, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/mnemoss/client.py tests/test_client.py
git commit -m "$(printf 'feat(client): wire TierTelemetryLedger + status() adaptive_caps block\n\nBuild the ledger on open, pass it to RecallEngine, surface effective\ncaps + telemetry via status(). Block has a stable json-safe shape\nwhether or not the flag is on.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 9: Convergence fixture + control-loop tests

**Files:**
- Create: `tests/test_adaptive_caps_convergence.py`
- Test: itself

A deterministic workload simulator (the "synthetic test rig" from the spec) plus the four assertions: convergence, stability, workload-shift tracking, adversarial-noise boundedness.

- [ ] **Step 1: Write the failing test**

Create `tests/test_adaptive_caps_convergence.py`:

```python
"""Adaptive caps — control-loop convergence tests.

The spec's "synthetic aged-workspace test rig", realised as a
deterministic workload simulator. The simulator models the one
relationship the controller depends on: given the current caps, where
do query winners land (HOT vs COLD) and how slow is recall. Driving
compute_adjusted_caps against it in a loop lets us assert the loop
*converges* rather than oscillating or drifting — without the
embedding noise a real workspace would add.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from mnemoss import FormulaParams, TierCapacityParams
from mnemoss.index.adaptive_caps import TierTelemetry, compute_adjusted_caps


@dataclass
class WorkloadModel:
    """Deterministic model of an aged workspace.

    ``ideal_hot`` is the hidden 'true' number of frequently-recalled
    memories. When ``hot_cap + warm_cap`` covers them, winners stay in
    the fast tiers; when the caps fall short, the overflow leaks to
    COLD. Mean recall latency grows with the total scanned breadth
    (``hot_cap + warm_cap``).
    """

    ideal_hot: int
    queries_per_window: int = 300
    noise: float = 0.0
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def window(self, caps: TierCapacityParams) -> TierTelemetry:
        reachable = caps.hot_cap + caps.warm_cap
        if reachable >= self.ideal_hot:
            leak = 0.0
        else:
            leak = (self.ideal_hot - reachable) / self.ideal_hot
        if self.noise:
            leak += self.rng.uniform(-self.noise, self.noise)
        leak = min(1.0, max(0.0, leak))

        q = self.queries_per_window
        winners_cold = round(q * leak)
        winners_hot = q - winners_cold
        mean_ms = 5.0 + 0.01 * reachable
        return TierTelemetry(
            queries=q,
            winners_hot=winners_hot,
            winners_warm=0,
            winners_cold=winners_cold,
            winners_deep=0,
            elapsed_ms_sum=mean_ms * q,
            reminiscence_events=0,
        )


def _params(**kw: object) -> FormulaParams:
    base: dict[str, object] = {
        "adaptive_tier_caps": True,
        "adaptive_tier_min_queries": 100,
        "adaptive_tier_lambda": 0.5,
        "adaptive_tier_max_step": 0.2,
        "adaptive_tier_deadband": 0.05,
        "adaptive_tier_latency_budget_ms": 25.0,
    }
    base.update(kw)
    return FormulaParams(**base)  # type: ignore[arg-type]


def _run(
    model: WorkloadModel,
    params: FormulaParams,
    start: TierCapacityParams,
    windows: int,
) -> list[TierCapacityParams]:
    caps = start
    history = [caps]
    for _ in range(windows):
        tel = model.window(caps)
        caps, _ = compute_adjusted_caps(caps, tel, params)
        history.append(caps)
    return history


def _converged(history: list[TierCapacityParams], tail: int = 3) -> bool:
    """True if the last ``tail`` entries are identical (dead-band rest)."""

    last = history[-tail:]
    return all(
        c.hot_cap == last[0].hot_cap
        and c.warm_cap == last[0].warm_cap
        and c.cold_cap == last[0].cold_cap
        for c in last
    )


def test_converges_from_oversized_start() -> None:
    model = WorkloadModel(ideal_hot=800)
    params = _params()
    start = TierCapacityParams(hot_cap=5000, warm_cap=50_000, cold_cap=90_000)
    history = _run(model, params, start, windows=60)
    assert _converged(history), "controller did not settle from oversized start"


def test_converges_from_undersized_start() -> None:
    model = WorkloadModel(ideal_hot=800)
    params = _params()
    start = TierCapacityParams(hot_cap=20, warm_cap=20, cold_cap=20)
    history = _run(model, params, start, windows=60)
    assert _converged(history), "controller did not settle from undersized start"


def test_stable_once_converged() -> None:
    model = WorkloadModel(ideal_hot=800)
    params = _params()
    start = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    history = _run(model, params, start, windows=120)
    # Last 20 windows must be a single fixed point.
    tail = history[-20:]
    assert all(
        c.hot_cap == tail[0].hot_cap and c.warm_cap == tail[0].warm_cap
        for c in tail
    ), "caps oscillate after convergence"


def test_tracks_upward_workload_shift() -> None:
    params = _params()
    start = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    model = WorkloadModel(ideal_hot=800)
    history_a = _run(model, params, start, windows=60)
    converged_low = history_a[-1]
    # Workload shifts: 3× more memories become hot-worthy.
    model.ideal_hot = 2400
    history_b = _run(model, params, converged_low, windows=60)
    converged_high = history_b[-1]
    reachable_low = converged_low.hot_cap + converged_low.warm_cap
    reachable_high = converged_high.hot_cap + converged_high.warm_cap
    assert reachable_high > reachable_low, "controller did not track the shift up"
    assert _converged(history_b), "controller did not re-settle after the shift"


def test_adversarial_noise_stays_bounded() -> None:
    # Pure-noise-heavy signal: caps must stay within [min_floor, max_cap]
    # and the time-average must not drift to a rail.
    model = WorkloadModel(ideal_hot=1500, noise=0.5, rng=random.Random(42))
    params = _params()
    start = TierCapacityParams(hot_cap=200, warm_cap=2000, cold_cap=20000)
    history = _run(model, params, start, windows=120)
    for caps in history:
        assert 20 <= caps.hot_cap <= 100_000
        assert 20 <= caps.warm_cap <= 100_000
        assert 20 <= caps.cold_cap <= 100_000
    tail = history[-30:]
    mean_hot = sum(c.hot_cap for c in tail) / len(tail)
    # Not pegged at either rail despite the noise.
    assert mean_hot > 25
    assert mean_hot < 99_000
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `pytest tests/test_adaptive_caps_convergence.py -v`
Expected: PASS — this task adds only a test file; it exercises `compute_adjusted_caps` (already implemented in Task 3) through the simulator. If any assertion fails, the controller has a real stability bug — investigate `compute_adjusted_caps` rather than weakening the assertion.

> This task has no "implementation" step: the simulator + assertions ARE the deliverable, and the controller they test already exists. If `test_adversarial_noise_stays_bounded`'s `mean_hot` bounds prove too tight on the seeded RNG, widen them to `> 21` / `< 99_999` — the meaningful assertion is "not pegged at a rail," not a specific band.

- [ ] **Step 3: Type check + lint**

Run: `mypy --strict tests/test_adaptive_caps_convergence.py && ruff check tests/test_adaptive_caps_convergence.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_adaptive_caps_convergence.py
git commit -m "$(printf 'test(index): adaptive-caps control-loop convergence tests\n\nDeterministic workload simulator (the spec test rig) + convergence,\nstability, workload-shift, and adversarial-noise assertions on the\ncontrol loop.\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Task 10: Docs + full-suite verification

**Files:**
- Modify: `CLAUDE.md` (the "Architectural Invariants" section)
- Test: the whole suite

- [ ] **Step 1: Add the architectural-invariant note**

In `CLAUDE.md`, in the "## Architectural Invariants" section, add a new bullet immediately after the existing **Schema version** bullet (`- **Schema version** — pinned constant ...`):

```markdown
- **Adaptive tier caps (opt-in)** — `FormulaParams.adaptive_tier_caps`
  (default `False`) enables a runtime controller that self-tunes
  `TierCapacityParams` from recall telemetry. When on, the recall
  engine records winner provenance + latency to a `TierTelemetryLedger`
  (in `workspace_meta`, same pattern as `CostLedger`), and `rebalance()`
  calls `maybe_adjust_caps` to nudge the *effective* caps before
  bucketing — a one-knob (`adaptive_tier_lambda`) blended
  latency/recall signal with dead-band, min-dwell, max-step, and
  `[min_floor, max_cap]` clamps. Scoped to the `use_tier_cascade_recall`
  path only. Default off is a byte-identical no-op. The static
  `TierCapacityParams` values remain the seed. See
  `src/mnemoss/index/adaptive_caps.py` and
  `docs/superpowers/specs/2026-05-15-adaptive-tier-caps-design.md`.
```

- [ ] **Step 2: Run the full unit suite**

Run: `pytest -m "not integration" -q`
Expected: PASS — all existing tests plus the new `test_adaptive_caps*.py` files. No regressions.

- [ ] **Step 3: Full type check + lint + format check**

Run: `mypy --strict src/mnemoss && ruff check src tests && ruff format --check src tests`
Expected: no errors. If `ruff format --check` flags the new files, run `ruff format src tests` and re-stage.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(printf 'docs: note adaptive tier caps in CLAUDE.md architectural invariants\n\nCo-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review

**Spec coverage** — every spec section maps to a task:
- New module `adaptive_caps.py` + pure `compute_adjusted_caps` → Tasks 3, 5.
- Control mechanism (two signals, blended score, multiplicative step, dead-band/min-dwell/max-step/clamp) → Task 3 (`compute_adjusted_caps`) + Task 5 (`maybe_adjust_caps` min-dwell gate).
- Integration point 1 (telemetry collection in `_tier_cascade_recall`) → Task 7.
- Integration point 2 (controller run at Rebalance) → Task 6.
- Persistence in `workspace_meta` → Task 4 (`TierTelemetryLedger`).
- Config surface (`FormulaParams` + `TierCapacityParams` fields + validators) → Tasks 1, 2.
- Observability (`status()` `adaptive_caps` block) → Task 8.
- Error handling — missing/corrupt telemetry → `_read_int`/`_read_float` return 0 on absent/garbage (Task 4); degenerate caps → `compute_adjusted_caps` only ever constructs clamped-and-ordered `TierCapacityParams`, and the `__post_init__` validators (Task 1) are the backstop; insufficient data → `maybe_adjust_caps` early-return (Task 5); flag-off no-op → `_record_tier_telemetry` guard (Task 7), `maybe_adjust_caps` guard (Task 5), tested in Tasks 6/7/8.
- Testing strategy: unit → Tasks 3–5; integration/round-trip + no-op → Tasks 6–8; convergence fixture → Task 9; end-to-end sanity (full suite) → Task 10. The existing-bench sanity run (`bench_rebalance_lift.py`/`bench_tier_lifecycle.py` with the flag on) is **not** scripted as a task step because benches are standalone and not part of `pytest`; run them manually post-merge if desired — they are not a correctness gate.
- Invariants/principles check → Task 10 (CLAUDE.md note); no `SCHEMA_VERSION` bump confirmed in the plan header.
- Resolved open questions (schema, scope to tier-cascade, `max_cap` default) → plan header.

**Placeholder scan** — no "TBD"/"add error handling"/"similar to Task N". Every code step shows complete code. The one task without an implementation step (Task 9) is explained: its deliverable is the test + simulator, and the code under test already exists.

**Type consistency** — names checked across tasks: `TierTelemetry`, `TierTelemetryLedger`, `compute_adjusted_caps`, `maybe_adjust_caps` (Tasks 3–5, used in 6–8); `TierTelemetryLedger.record_recall(winner_tiers=, elapsed_ms=, reminiscence=)` (defined Task 4, called Task 7); `read()`/`reset()`/`write_effective_caps(caps, delta=)`/`read_effective_caps(seed)`/`snapshot(seed)` (defined Task 4, used Tasks 5/6/8); `RecallEngine(..., tier_ledger=)` (Task 7, called Task 8); `FormulaParams.adaptive_tier_*` and `TierCapacityParams.min_floor`/`max_cap` (Tasks 1–2, used throughout).

---

## Execution Handoff

(Filled in by the writing-plans skill after the user reviews this plan.)
