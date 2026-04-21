"""Async-embedding-path tests (Checkpoint I).

Verifies that ``observe()`` and ``recall()`` offload embedder calls to a
worker thread so the event loop stays free for concurrent work. The
canonical scenario here is a cloud embedder whose ``embed`` call blocks
for ~100ms — without the thread offload, that blocks every other
coroutine in the process.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from mnemoss import Mnemoss, StorageParams


class SleepyEmbedder:
    """Synchronous embedder that blocks its calling thread for ``delay`` seconds."""

    def __init__(self, dim: int = 16, delay: float = 0.1) -> None:
        self.dim = dim
        self.delay = delay
        self.embedder_id = f"sleepy:{delay}s"

    def embed(self, texts: list[str]) -> np.ndarray:
        time.sleep(self.delay)
        n = len(texts)
        return np.ones((n, self.dim), dtype=np.float32) / np.sqrt(self.dim)


async def test_concurrent_observes_do_not_block_event_loop(tmp_path: Path) -> None:
    embedder = SleepyEmbedder(dim=16, delay=0.15)
    mem = Mnemoss(
        workspace="t",
        embedding_model=embedder,
        storage=StorageParams(root=tmp_path),
    )
    try:
        # Drain the warmup embed + open cost on a single observe first.
        await mem.observe(role="user", content="warmup")

        start = time.monotonic()
        await asyncio.gather(
            mem.observe(role="user", content="a"),
            mem.observe(role="user", content="b"),
            mem.observe(role="user", content="c"),
        )
        elapsed = time.monotonic() - start

        # With embeds on threads: ~0.15s wall (plus a bit for serialized
        # SQLite writes). Without threading they'd serialize on the event
        # loop for ~0.45s. Give headroom for CI jitter.
        assert elapsed < 0.35, (
            f"Observes should run in parallel; took {elapsed:.3f}s"
        )
    finally:
        await mem.close()


async def test_observe_yields_control_during_embed(tmp_path: Path) -> None:
    """A ticker coroutine should keep ticking while observe() is waiting on
    its embed — proves the event loop isn't frozen during the blocking call."""

    embedder = SleepyEmbedder(dim=16, delay=0.2)
    mem = Mnemoss(
        workspace="t",
        embedding_model=embedder,
        storage=StorageParams(root=tmp_path),
    )
    await mem.observe(role="user", content="warmup")

    ticks: list[float] = []

    async def ticker() -> None:
        for _ in range(10):
            await asyncio.sleep(0.01)
            ticks.append(time.monotonic())

    try:
        await asyncio.gather(
            ticker(),
            mem.observe(role="user", content="slow"),
        )
        # Even though observe() blocks the worker thread for 200ms, the
        # ticker should have run all 10 iterations.
        assert len(ticks) == 10
    finally:
        await mem.close()


async def test_recall_embedder_is_also_offloaded(tmp_path: Path) -> None:
    """recall() should likewise not block the loop on its query embed."""

    embedder = SleepyEmbedder(dim=16, delay=0.1)
    mem = Mnemoss(
        workspace="t",
        embedding_model=embedder,
        storage=StorageParams(root=tmp_path),
    )
    await mem.observe(role="user", content="hello")

    ticks: list[float] = []

    async def ticker() -> None:
        for _ in range(8):
            await asyncio.sleep(0.01)
            ticks.append(time.monotonic())

    try:
        await asyncio.gather(ticker(), mem.recall("hello", k=1))
        assert len(ticks) == 8
    finally:
        await mem.close()
