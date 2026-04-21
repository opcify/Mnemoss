"""Unit tests for Logistic noise."""

from __future__ import annotations

import math
import random
import statistics

from mnemoss.formula.noise import sample_noise


def test_seeded_rng_is_deterministic() -> None:
    r1 = random.Random(42)
    r2 = random.Random(42)
    assert sample_noise(r1, 0.25) == sample_noise(r2, 0.25)


def test_sample_distribution_properties() -> None:
    rng = random.Random(0)
    samples = [sample_noise(rng, 0.25) for _ in range(20000)]
    # Mean should be close to 0; stddev ≈ s·π/√3 ≈ 0.453 for s=0.25.
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    assert abs(mean) < 0.02
    expected_stdev = 0.25 * math.pi / math.sqrt(3)
    assert abs(stdev - expected_stdev) < 0.02


def test_clamped_endpoints_never_blow_up() -> None:
    # Mock an RNG that returns exactly 0 and 1. Should not raise.
    class Endpoint:
        def __init__(self, value: float) -> None:
            self.value = value

        def random(self) -> float:
            return self.value

    for v in (0.0, 1.0):
        out = sample_noise(Endpoint(v), 0.25)  # type: ignore[arg-type]
        assert math.isfinite(out)
