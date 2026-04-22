"""Retry wrapper for flaky embedders.

``LocalEmbedder`` runs in-process and rarely fails transiently.
``OpenAIEmbedder`` / ``GeminiEmbedder`` reach out over the network
and do fail transiently — 429s, 5xx, socket timeouts, DNS blips.

``RetryingEmbedder`` wraps any ``Embedder`` with bounded retries on
I/O-ish exceptions. It is intentionally opt-in: most users don't
need it, and wrapping silently would mask programmer errors.

Usage::

    from mnemoss import OpenAIEmbedder, RetryingEmbedder

    embedder = RetryingEmbedder(
        OpenAIEmbedder(api_key="..."),
        max_retries=3,
        base_delay_seconds=0.2,
    )
    mem = Mnemoss(workspace="prod", embedding_model=embedder)

Design choices:

- Retries only on a curated set of retryable exception types
  (``ConnectionError``, ``TimeoutError``, ``OSError``, and any
  exception the user opts into via ``retry_on``). ``ValueError`` —
  the canonical "bad input" signal — is never retried.
- Exponential backoff with jitter. A retry storm from one flaky
  hour shouldn't DDoS the provider.
- Identity passthrough for ``dim`` / ``embedder_id`` so workspaces
  don't need to distinguish "wrapped" from "plain" embedders in
  their schema pin.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import numpy as np

from mnemoss.encoder.embedder import Embedder

_log = logging.getLogger(__name__)

# Exception types we treat as transient by default. Users can extend
# via ``retry_on=`` for provider-specific errors (e.g. openai.RateLimitError).
_DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class RetryingEmbedder:
    """Wrap an ``Embedder`` with bounded retries on transient failures.

    Parameters
    ----------
    inner
        The embedder to wrap. Its ``dim`` and ``embedder_id`` are
        exposed transparently.
    max_retries
        Number of retry attempts after the initial try. Total calls
        is ``max_retries + 1``. Defaults to 2.
    base_delay_seconds
        Seed delay before the first retry. Doubles each retry.
        Jitter is ±25% uniform. Defaults to 0.2s.
    max_delay_seconds
        Upper bound on any single sleep so pathological backoff
        doesn't block ``observe`` for minutes. Defaults to 5s.
    retry_on
        Additional exception classes (beyond the default network /
        timeout set) that should trigger a retry. Use for
        provider-specific retryables like ``openai.RateLimitError``.
    sleep
        Injection point for tests. Defaults to ``time.sleep``.
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        max_retries: int = 2,
        base_delay_seconds: float = 0.2,
        max_delay_seconds: float = 5.0,
        retry_on: tuple[type[BaseException], ...] = (),
        sleep: Any = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be > 0")
        if max_delay_seconds < base_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be >= base_delay_seconds"
            )
        self._inner = inner
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._retryable: tuple[type[BaseException], ...] = (
            _DEFAULT_RETRYABLE + retry_on
        )
        self._sleep = sleep

    # Passthrough identity so the store's schema pin matches the
    # underlying embedder, not the wrapper — swapping in a wrapper
    # later must not trip SchemaMismatchError.
    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def embedder_id(self) -> str:
        return self._inner.embedder_id

    def embed(self, texts: list[str]) -> np.ndarray:
        attempt = 0
        while True:
            try:
                return self._inner.embed(texts)
            except self._retryable as e:
                if attempt >= self._max_retries:
                    _log.warning(
                        "embed_retry_exhausted",
                        extra={
                            "embedder_id": self._inner.embedder_id,
                            "attempts": attempt + 1,
                            "error": repr(e),
                        },
                    )
                    raise
                delay = self._compute_delay(attempt)
                _log.info(
                    "embed_retry",
                    extra={
                        "embedder_id": self._inner.embedder_id,
                        "attempt": attempt + 1,
                        "next_delay_seconds": delay,
                        "error": repr(e),
                    },
                )
                self._sleep(delay)
                attempt += 1

    def _compute_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, clamped to ``max_delay``."""

        raw: float = self._base_delay * (2**attempt)
        clamped: float = min(raw, self._max_delay)
        # ±25% jitter so a cluster of clients doesn't retry in lockstep.
        jitter: float = random.uniform(0.75, 1.25)
        return clamped * jitter
