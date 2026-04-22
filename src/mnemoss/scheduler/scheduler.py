"""``DreamScheduler`` — background ticker for nightly / idle dreams.

One scheduler per ``Mnemoss`` instance. The loop is an ``asyncio`` task
that wakes every ``check_interval_seconds``, computes whether a trigger
is due, and fires the matching dream. Dreams run inline (``await``);
``check_interval_seconds`` should be comfortably larger than a typical
dream's duration so back-pressure never builds up.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, time, timezone
from typing import Any

UTC = timezone.utc
_log = logging.getLogger("mnemoss.scheduler")


@dataclass
class SchedulerConfig:
    """Tunables for :class:`DreamScheduler`.

    Setting ``nightly_at`` or ``idle_after_seconds`` to ``None`` disables
    that trigger. Setting both to ``None`` means the scheduler still
    runs but fires nothing — useful as a safe no-op when you're toggling
    a global enable flag without tearing down the task.
    """

    nightly_at: time | None = time(3, 0)
    idle_after_seconds: float | None = 600.0
    check_interval_seconds: float = 60.0
    enabled: bool = True


class DreamScheduler:
    """Background task that fires dream triggers on a schedule.

    The scheduler is agnostic to where the backend lives — it calls
    ``backend.dream(trigger=...)`` and reads ``backend.last_observe_at``.
    A ``Mnemoss`` instance conforms; so does anything else that duck-
    types on those two names.
    """

    def __init__(
        self,
        backend: Any,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._backend = backend
        self._config = config or SchedulerConfig()
        self._clock = clock
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._running = False
        # Per-day cooldown: stores the date of the most recent nightly fire.
        self._last_nightly_date: date_cls | None = None
        # Per-idle-window guard: after firing, don't fire again until a
        # new observe resets the idle clock.
        self._last_idle_fire_at: datetime | None = None

    # ─── lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background task. Idempotent."""

        if self._task is not None:
            return
        cfg = self._config
        # Don't fire today's nightly if we started after it already passed.
        # Users who want a one-off can call ``dream(trigger="nightly")``
        # directly — the scheduler is for *recurring* cadence.
        now = self._clock()
        if cfg.nightly_at is not None and self._at_or_past_nightly(now):
            self._last_nightly_date = now.date()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="mnemoss-scheduler")

    async def stop(self) -> None:
        """Stop the background task. Safe to call multiple times."""

        self._running = False
        task = self._task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None

    # ─── ticking ─────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — scheduler must not die
                _log.exception("scheduler tick raised; continuing")
            await self._sleep(self._config.check_interval_seconds)

    async def _tick(self) -> None:
        """One scan of the clock + backend state. Exposed for tests."""

        if not self._config.enabled:
            return
        now = self._clock()

        if self._should_fire_nightly(now):
            _log.info("firing nightly dream")
            await self._fire("nightly")
            self._last_nightly_date = now.date()

        if self._should_fire_idle(now):
            _log.info("firing idle dream")
            await self._fire("idle")
            self._last_idle_fire_at = now

    async def _fire(self, trigger: str) -> None:
        try:
            await self._backend.dream(trigger=trigger)
        except Exception:  # noqa: BLE001 — one failed dream shouldn't kill cadence
            _log.exception("dream(trigger=%s) raised", trigger)

    # ─── predicates ─────────────────────────────────────────────

    def _at_or_past_nightly(self, now: datetime) -> bool:
        nightly_at = self._config.nightly_at
        if nightly_at is None:
            return False
        today_trigger = now.replace(
            hour=nightly_at.hour,
            minute=nightly_at.minute,
            second=nightly_at.second,
            microsecond=nightly_at.microsecond,
        )
        return now >= today_trigger

    def _should_fire_nightly(self, now: datetime) -> bool:
        if self._config.nightly_at is None:
            return False
        if self._last_nightly_date == now.date():
            return False
        return self._at_or_past_nightly(now)

    def _should_fire_idle(self, now: datetime) -> bool:
        threshold = self._config.idle_after_seconds
        if threshold is None:
            return False
        last_obs = getattr(self._backend, "last_observe_at", None)
        if last_obs is None:
            return False
        idle_seconds = (now - last_obs).total_seconds()
        if idle_seconds < threshold:
            return False
        # Fire only if a new observe has happened since the last idle fire.
        return self._last_idle_fire_at is None or last_obs > self._last_idle_fire_at

    # ─── introspection ──────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()
