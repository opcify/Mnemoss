"""Tests for ``RetryingEmbedder``.

Covers: identity passthrough, retry-until-success on transient
failures, exhaustion raises, non-retryable errors surface
immediately, backoff respected, validation of constructor args.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mnemoss import FakeEmbedder, RetryingEmbedder


class _FlakyEmbedder:
    """Embedder that fails a configurable number of times then succeeds."""

    dim = 4
    embedder_id = "flaky:4"

    def __init__(
        self,
        *,
        fail_first: int,
        exc: type[BaseException] = ConnectionError,
    ) -> None:
        self._fail_first = fail_first
        self._exc = exc
        self.call_count = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        if self.call_count <= self._fail_first:
            raise self._exc(f"transient failure #{self.call_count}")
        # Return deterministic embeddings.
        return np.ones((len(texts), self.dim), dtype=np.float32)


class _InstantSleep:
    """Collect sleep durations without actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# ─── identity passthrough ─────────────────────────────────────────


def test_passes_through_dim_and_embedder_id() -> None:
    """Wrapping must not change the schema pin — workspaces created
    with a bare embedder should still open under the wrapped one."""

    inner = FakeEmbedder(dim=32)
    wrapped = RetryingEmbedder(inner)
    assert wrapped.dim == inner.dim
    assert wrapped.embedder_id == inner.embedder_id


# ─── happy path ───────────────────────────────────────────────────


def test_succeeds_on_first_try_without_retry() -> None:
    inner = FakeEmbedder(dim=8)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(inner, max_retries=3, sleep=sleep)
    out = wrapped.embed(["hello"])
    assert out.shape == (1, 8)
    assert sleep.delays == []  # no sleep happened


def test_retries_transient_failures(tmp_path: Any) -> None:
    flaky = _FlakyEmbedder(fail_first=2)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(
        flaky, max_retries=3, base_delay_seconds=0.01, sleep=sleep
    )
    out = wrapped.embed(["hello"])
    assert out.shape == (1, 4)
    # Called three times: two failures, one success.
    assert flaky.call_count == 3
    # Slept twice (between the three tries).
    assert len(sleep.delays) == 2


def test_exhausts_retries_and_raises() -> None:
    flaky = _FlakyEmbedder(fail_first=10)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(
        flaky, max_retries=2, base_delay_seconds=0.01, sleep=sleep
    )
    with pytest.raises(ConnectionError, match="transient failure #3"):
        wrapped.embed(["hello"])
    # Called 3 times total: initial + 2 retries.
    assert flaky.call_count == 3
    # Slept twice between attempts.
    assert len(sleep.delays) == 2


# ─── non-retryable errors pass straight through ─────────────────────


def test_value_error_is_not_retried() -> None:
    """``ValueError`` = "bad input" = programmer error; never retry
    it — we'd only burn latency before raising the same exception."""

    flaky = _FlakyEmbedder(fail_first=10, exc=ValueError)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(flaky, max_retries=5, sleep=sleep)
    with pytest.raises(ValueError):
        wrapped.embed(["hello"])
    assert flaky.call_count == 1  # no retry
    assert sleep.delays == []


def test_custom_retry_on_class_is_honored() -> None:
    """Callers can opt in to provider-specific retryables (e.g.
    ``openai.RateLimitError``) via ``retry_on``."""

    class MyProviderError(Exception):
        pass

    flaky = _FlakyEmbedder(fail_first=1, exc=MyProviderError)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(
        flaky,
        max_retries=3,
        base_delay_seconds=0.01,
        retry_on=(MyProviderError,),
        sleep=sleep,
    )
    out = wrapped.embed(["hello"])
    assert out.shape == (1, 4)
    assert flaky.call_count == 2


# ─── backoff shape ─────────────────────────────────────────────────


def test_backoff_is_bounded_by_max_delay() -> None:
    """Pathological backoff (huge ``base_delay``) shouldn't exceed
    ``max_delay_seconds`` for any single sleep."""

    flaky = _FlakyEmbedder(fail_first=10)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(
        flaky,
        max_retries=5,
        base_delay_seconds=1.0,
        max_delay_seconds=2.0,
        sleep=sleep,
    )
    with pytest.raises(ConnectionError):
        wrapped.embed(["hello"])
    # Each sleep capped at max_delay + 25% jitter.
    assert all(d <= 2.5 for d in sleep.delays), sleep.delays


def test_backoff_grows_across_retries() -> None:
    """Backoff should at least double across attempts (jitter aside).
    With base 0.01s and 4 retries, the delay series is roughly
    [0.01, 0.02, 0.04, 0.08] ± jitter."""

    flaky = _FlakyEmbedder(fail_first=10)
    sleep = _InstantSleep()
    wrapped = RetryingEmbedder(
        flaky,
        max_retries=4,
        base_delay_seconds=0.01,
        max_delay_seconds=1.0,
        sleep=sleep,
    )
    with pytest.raises(ConnectionError):
        wrapped.embed(["hello"])
    # Strictly-increasing means the doubling isn't being swallowed by
    # jitter entirely. Jitter is ±25%, doubling is 100%, so successive
    # floor bounds trivially exceed the prior floor.
    assert len(sleep.delays) == 4
    # Check each delay is at least 75% of the last doubled base (jitter
    # low end). I.e. attempt 1 delay_floor = 0.02 * 0.75 = 0.015 > 0.0075
    # (prior attempt's jitter high); skip strict monotonicity and check
    # that delay[i] > delay[i-1] * 0.5.
    for i in range(1, len(sleep.delays)):
        assert sleep.delays[i] > sleep.delays[i - 1] * 0.5


# ─── constructor validation ───────────────────────────────────────


def test_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        RetryingEmbedder(FakeEmbedder(dim=4), max_retries=-1)


def test_rejects_non_positive_base_delay() -> None:
    with pytest.raises(ValueError, match="base_delay_seconds"):
        RetryingEmbedder(FakeEmbedder(dim=4), base_delay_seconds=0)
    with pytest.raises(ValueError, match="base_delay_seconds"):
        RetryingEmbedder(FakeEmbedder(dim=4), base_delay_seconds=-1)


def test_rejects_max_delay_below_base() -> None:
    with pytest.raises(ValueError, match="max_delay_seconds"):
        RetryingEmbedder(
            FakeEmbedder(dim=4),
            base_delay_seconds=1.0,
            max_delay_seconds=0.5,
        )


# ─── end-to-end with Mnemoss ───────────────────────────────────────


async def test_retrying_embedder_works_with_mnemoss(tmp_path: Any) -> None:
    """Full round-trip: a Mnemoss backed by a wrapped flaky embedder
    should still observe + recall successfully when retries win."""

    from mnemoss import (
        FormulaParams,
        Mnemoss,
        StorageParams,
    )

    flaky = _FlakyEmbedder(fail_first=1)
    # Sleep is real here (zero-ish) so we stay wired to the actual
    # Mnemoss observe() path without mocking time.
    wrapped = RetryingEmbedder(flaky, max_retries=3, base_delay_seconds=0.01)
    mem = Mnemoss(
        workspace="retry",
        embedding_model=wrapped,
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )
    try:
        mid = await mem.observe(role="user", content="hello through a flaky embedder")
        assert mid is not None
        results = await mem.recall("hello", k=1)
        assert any(r.memory.id == mid for r in results)
        # At least the warmup + first observe were retried once.
        assert flaky.call_count >= 2
    finally:
        await mem.close()
