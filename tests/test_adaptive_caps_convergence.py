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
        leak = 0.0 if reachable >= self.ideal_hot else (self.ideal_hot - reachable) / self.ideal_hot
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
        c.hot_cap == tail[0].hot_cap
        and c.warm_cap == tail[0].warm_cap
        and c.cold_cap == tail[0].cold_cap
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
    spread = max(c.hot_cap for c in tail) - min(c.hot_cap for c in tail)
    assert spread < mean_hot * 3, "caps oscillate wildly under noise"
