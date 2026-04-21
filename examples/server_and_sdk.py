"""Mnemoss REST server + Python SDK, end-to-end in one process.

Runs the full stack without opening a socket: a FastAPI app lives in
this process, ``httpx.ASGITransport`` routes SDK calls straight into
it. The same SDK code works against a remote server — swap the
transport for a real URL.

    python examples/server_and_sdk.py

To run against a real ``mnemoss-server`` instead:

    # terminal 1:
    MNEMOSS_API_KEY=s3cret mnemoss-server
    # terminal 2 (or equivalent):
    async with MnemossClient("http://127.0.0.1:8000", api_key="s3cret") as c:
        ...
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import httpx

from mnemoss import FakeEmbedder
from mnemoss.sdk import MnemossClient
from mnemoss.server import ServerConfig, create_app


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # FakeEmbedder so the script runs offline. In production, the
        # server uses LocalEmbedder (multilingual, ~470MB first run) or
        # OpenAIEmbedder if MNEMOSS_EMBEDDING_MODEL=openai.
        config = ServerConfig(
            embedder_override=FakeEmbedder(dim=16),
            storage_root=Path(tmp),
        )
        app = create_app(config)
        transport = httpx.ASGITransport(app=app)

        async with MnemossClient(
            "http://testserver", transport=transport
        ) as client:
            ws = client.workspace("example")

            print("─── observe ───")
            mid = await ws.observe(
                role="user", content="I meet Alice tomorrow at 4:20 PM"
            )
            print(f"memory_id: {mid}")
            await ws.observe(
                role="user", content="The meeting is at the Sydney Opera House"
            )

            print()
            print("─── recall ───")
            for i, r in enumerate(await ws.recall("When do I meet Alice?", k=3), 1):
                print(f"  {i}. [{r.score:.3f}] {r.memory.content}")

            print()
            print("─── per-agent ───")
            alice = ws.for_agent("alice")
            await alice.observe(role="user", content="my secret project")
            for r in await alice.recall("secret", k=3):
                print(f"  [{r.score:.3f}] {r.memory.content}")

            print()
            print("─── dream (no LLM configured) ───")
            report = await ws.dream(trigger="idle")
            print(f"trigger: {report.trigger.value}")
            for o in report.outcomes:
                print(f"  {o.phase.value:<10} {o.status}")

            print()
            print("─── rebalance + tiers ───")
            stats = await ws.rebalance()
            print(f"scanned={stats.scanned}, migrated={stats.migrated}")
            print(f"tier counts: {await ws.tier_counts()}")

            print()
            print("─── explain_recall ───")
            breakdown = await ws.explain_recall("Alice", mid)
            print(f"base_level={breakdown.base_level:.3f}")
            print(f"matching=  {breakdown.matching:.3f}")
            print(f"idx_prio=  {breakdown.idx_priority:.3f}")
            print(f"total=     {breakdown.total:.3f}")

        # Close the pooled Mnemoss instances so SQLite WAL checkpoints
        # before the tempdir goes away.
        await app.state.pool.close_all()


if __name__ == "__main__":
    asyncio.run(main())
