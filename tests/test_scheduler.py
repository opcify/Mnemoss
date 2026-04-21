"""T1 — DreamScheduler tests.

Unit tests drive ``_tick()`` directly with an injected clock so the
suite is deterministic and fast. One end-to-end test covers the
``start()``/``stop()`` lifecycle with real ``asyncio.sleep`` at short
intervals.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mnemoss import (
    DreamScheduler,
    FakeEmbedder,
    Mnemoss,
    SchedulerConfig,
    StorageParams,
)
from mnemoss.server import ServerConfig, create_app

UTC = timezone.utc


# ─── helpers ──────────────────────────────────────────────────────


class _FakeBackend:
    """Minimal duck-type for DreamScheduler: ``dream()`` + ``last_observe_at``."""

    def __init__(self) -> None:
        self.dreams: list[str] = []
        self.last_observe_at: datetime | None = None

    async def dream(self, *, trigger: str) -> dict:
        self.dreams.append(trigger)
        return {"ok": True}


def _scheduler(backend, *, clock: datetime, **cfg_overrides) -> DreamScheduler:
    """Build a scheduler pinned to a fixed clock."""

    cfg = SchedulerConfig(**cfg_overrides)
    return DreamScheduler(backend, cfg, clock=lambda: clock)


# ─── nightly ─────────────────────────────────────────────────────


async def test_nightly_fires_once_at_or_past_configured_time() -> None:
    backend = _FakeBackend()
    now = datetime(2026, 4, 21, 4, 0, tzinfo=UTC)  # 4 AM
    sched = _scheduler(backend, clock=now, nightly_at=time(3, 0))

    await sched._tick()

    assert backend.dreams == ["nightly"]


async def test_nightly_does_not_fire_before_configured_time() -> None:
    backend = _FakeBackend()
    now = datetime(2026, 4, 21, 2, 59, tzinfo=UTC)  # 1 min before
    sched = _scheduler(backend, clock=now, nightly_at=time(3, 0))

    await sched._tick()

    assert backend.dreams == []


async def test_nightly_fires_at_most_once_per_day() -> None:
    backend = _FakeBackend()
    now = datetime(2026, 4, 21, 4, 0, tzinfo=UTC)
    sched = _scheduler(backend, clock=now, nightly_at=time(3, 0))

    await sched._tick()
    await sched._tick()
    await sched._tick()

    assert backend.dreams == ["nightly"]


async def test_nightly_fires_on_next_day() -> None:
    backend = _FakeBackend()
    sched = SchedulerConfig(nightly_at=time(3, 0), idle_after_seconds=None)
    # Inject mutable clock via closure.
    now_ref = [datetime(2026, 4, 21, 4, 0, tzinfo=UTC)]
    scheduler = DreamScheduler(backend, sched, clock=lambda: now_ref[0])

    await scheduler._tick()
    # Advance 20 hours → 00:00 next day; not yet past 03:00.
    now_ref[0] = datetime(2026, 4, 22, 0, 0, tzinfo=UTC)
    await scheduler._tick()
    assert backend.dreams == ["nightly"]
    # Advance to 03:30 next day → should fire.
    now_ref[0] = datetime(2026, 4, 22, 3, 30, tzinfo=UTC)
    await scheduler._tick()
    assert backend.dreams == ["nightly", "nightly"]


async def test_nightly_disabled_when_nightly_at_is_none() -> None:
    backend = _FakeBackend()
    sched = _scheduler(
        backend,
        clock=datetime(2026, 4, 21, 4, 0, tzinfo=UTC),
        nightly_at=None,
        idle_after_seconds=None,
    )

    await sched._tick()
    assert backend.dreams == []


async def test_start_skips_today_if_nightly_already_passed() -> None:
    """If the scheduler starts at 10 PM with nightly_at=03:00, it must
    not fire tonight at 10 PM — only tomorrow at 03:00."""

    backend = _FakeBackend()
    now_ref = [datetime(2026, 4, 21, 22, 0, tzinfo=UTC)]
    scheduler = DreamScheduler(
        backend,
        SchedulerConfig(
            nightly_at=time(3, 0),
            idle_after_seconds=None,
            check_interval_seconds=0.01,
        ),
        clock=lambda: now_ref[0],
    )
    await scheduler.start()
    try:
        # Let the loop tick a few times under the 10 PM clock.
        await asyncio.sleep(0.05)
        assert backend.dreams == []
    finally:
        await scheduler.stop()


# ─── idle ────────────────────────────────────────────────────────


async def test_idle_fires_after_threshold() -> None:
    backend = _FakeBackend()
    now = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    backend.last_observe_at = now - timedelta(seconds=600)
    sched = _scheduler(
        backend, clock=now, nightly_at=None, idle_after_seconds=300
    )

    await sched._tick()
    assert backend.dreams == ["idle"]


async def test_idle_does_not_fire_before_threshold() -> None:
    backend = _FakeBackend()
    now = datetime(2026, 4, 21, 12, 0, tzinfo=UTC)
    backend.last_observe_at = now - timedelta(seconds=60)
    sched = _scheduler(
        backend, clock=now, nightly_at=None, idle_after_seconds=300
    )

    await sched._tick()
    assert backend.dreams == []


async def test_idle_does_not_refire_without_new_observe() -> None:
    backend = _FakeBackend()
    now_ref = [datetime(2026, 4, 21, 12, 0, tzinfo=UTC)]
    backend.last_observe_at = now_ref[0] - timedelta(seconds=600)
    sched = DreamScheduler(
        backend,
        SchedulerConfig(nightly_at=None, idle_after_seconds=300),
        clock=lambda: now_ref[0],
    )

    await sched._tick()
    assert backend.dreams == ["idle"]

    # Advance 10 minutes with no new observe → should NOT re-fire.
    now_ref[0] += timedelta(minutes=10)
    await sched._tick()
    assert backend.dreams == ["idle"]


async def test_idle_refires_after_new_observe_then_idle_again() -> None:
    backend = _FakeBackend()
    now_ref = [datetime(2026, 4, 21, 12, 0, tzinfo=UTC)]
    backend.last_observe_at = now_ref[0] - timedelta(seconds=600)
    sched = DreamScheduler(
        backend,
        SchedulerConfig(nightly_at=None, idle_after_seconds=300),
        clock=lambda: now_ref[0],
    )

    await sched._tick()
    assert backend.dreams == ["idle"]

    # Advance the clock and simulate a new observe at the new time.
    now_ref[0] += timedelta(seconds=60)
    backend.last_observe_at = now_ref[0]

    # Now wait past threshold again.
    now_ref[0] += timedelta(seconds=400)
    await sched._tick()
    assert backend.dreams == ["idle", "idle"]


async def test_idle_disabled_when_seconds_is_none() -> None:
    backend = _FakeBackend()
    backend.last_observe_at = datetime(2026, 4, 21, 0, 0, tzinfo=UTC)
    now = datetime(2026, 4, 21, 23, 0, tzinfo=UTC)
    sched = _scheduler(
        backend, clock=now, nightly_at=None, idle_after_seconds=None
    )

    await sched._tick()
    assert backend.dreams == []


async def test_idle_requires_last_observe_at_to_be_set() -> None:
    backend = _FakeBackend()
    # backend.last_observe_at stays None — nothing has been observed yet.
    sched = _scheduler(
        backend,
        clock=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
        nightly_at=None,
        idle_after_seconds=1,
    )

    await sched._tick()
    assert backend.dreams == []


# ─── master switch ───────────────────────────────────────────────


async def test_disabled_scheduler_fires_nothing() -> None:
    backend = _FakeBackend()
    backend.last_observe_at = datetime(2026, 4, 21, 0, 0, tzinfo=UTC)
    sched = _scheduler(
        backend,
        clock=datetime(2026, 4, 21, 23, 0, tzinfo=UTC),
        nightly_at=time(3, 0),
        idle_after_seconds=1,
        enabled=False,
    )

    await sched._tick()
    assert backend.dreams == []


# ─── lifecycle ────────────────────────────────────────────────────


async def test_start_and_stop_are_idempotent() -> None:
    backend = _FakeBackend()
    sched = DreamScheduler(
        backend,
        SchedulerConfig(
            nightly_at=None,
            idle_after_seconds=None,
            check_interval_seconds=0.01,
        ),
    )
    await sched.start()
    await sched.start()  # second call is a no-op
    assert sched.is_running

    await sched.stop()
    await sched.stop()  # safe to stop twice
    assert not sched.is_running


async def test_failing_dream_does_not_kill_scheduler() -> None:
    """If one dream raises, the scheduler logs and carries on."""

    backend = _FakeBackend()
    bad_call = {"count": 0}

    async def flaky_dream(*, trigger: str) -> dict:
        bad_call["count"] += 1
        if bad_call["count"] == 1:
            raise RuntimeError("boom")
        backend.dreams.append(trigger)
        return {"ok": True}

    backend.dream = flaky_dream  # type: ignore[method-assign]

    now_ref = [datetime(2026, 4, 21, 4, 0, tzinfo=UTC)]
    sched = DreamScheduler(
        backend,
        SchedulerConfig(nightly_at=time(3, 0), idle_after_seconds=None),
        clock=lambda: now_ref[0],
    )

    # First day's tick raises; we swallow.
    await sched._tick()
    # Next day — fires successfully.
    now_ref[0] = datetime(2026, 4, 22, 4, 0, tzinfo=UTC)
    await sched._tick()
    assert backend.dreams == ["nightly"]


# ─── Mnemoss integration ──────────────────────────────────────────


async def test_mnemoss_sets_last_observe_at(tmp_path: Path) -> None:
    mem = Mnemoss(
        workspace="test",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )
    try:
        assert mem.last_observe_at is None
        before = datetime.now(UTC)
        await mem.observe(role="user", content="hi")
        after = datetime.now(UTC)
        ts = mem.last_observe_at
        assert ts is not None
        assert before <= ts <= after
    finally:
        await mem.close()


async def test_scheduler_end_to_end_over_mnemoss(tmp_path: Path) -> None:
    """Fast end-to-end: a short idle threshold + tight check interval
    exercises the full loop."""

    mem = Mnemoss(
        workspace="sched_e2e",
        embedding_model=FakeEmbedder(dim=16),
        storage=StorageParams(root=tmp_path),
    )
    try:
        await mem.observe(role="user", content="x")
        sched = DreamScheduler(
            mem,
            SchedulerConfig(
                nightly_at=None,
                idle_after_seconds=0.05,
                check_interval_seconds=0.01,
            ),
        )
        await sched.start()
        try:
            # Wait long enough for idle threshold to trip.
            await asyncio.sleep(0.25)
        finally:
            await sched.stop()

        # The idle dream should have landed at least one report — dream
        # runs produce diary files.
        # We can only observe the side effect indirectly; here we
        # verify the scheduler's internal flag advanced.
        assert sched._last_idle_fire_at is not None
    finally:
        await mem.close()


# ─── server pool integration ──────────────────────────────────────


def test_pool_starts_and_stops_scheduler_with_workspace(tmp_path: Path) -> None:
    """When ``ServerConfig.scheduler`` is set, the pool should attach a
    DreamScheduler to each workspace and tear them down on shutdown."""

    config = ServerConfig(
        embedder_override=FakeEmbedder(dim=16),
        storage_root=tmp_path,
        scheduler=SchedulerConfig(
            nightly_at=None,
            idle_after_seconds=None,  # Disabled — we just verify lifecycle.
            check_interval_seconds=0.01,
        ),
    )
    app = create_app(config)
    with TestClient(app) as c:
        c.post(
            "/workspaces/sched_ws/observe",
            json={"role": "user", "content": "hi"},
        )
        assert "sched_ws" in app.state.pool._schedulers
        assert app.state.pool._schedulers["sched_ws"].is_running

    # After TestClient context manager exits, lifespan shutdown ran.
    assert app.state.pool._schedulers == {}


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        ("on", True),
        ("yes", True),
        ("0", False),
        ("", False),
        (None, False),
    ],
)
def test_server_config_scheduler_env_flag(monkeypatch, env_value, expected) -> None:
    if env_value is None:
        monkeypatch.delenv("MNEMOSS_SCHEDULER", raising=False)
    else:
        monkeypatch.setenv("MNEMOSS_SCHEDULER", env_value)
    config = ServerConfig.from_env()
    assert (config.scheduler is not None) == expected
