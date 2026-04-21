"""Mnemoss Quick Start.

Runs the Stage 1 success criterion, the per-agent usage example, and
shows the Stage 2 additions — tier counts, rebalancing, and DEEP-cue
cascade recall. First run downloads ~470MB (sentence-transformers
multilingual model).

    python examples/basic.py
"""

from __future__ import annotations

import asyncio

from mnemoss import FormulaParams, Mnemoss


async def main() -> None:
    # Uses a Stage-2 workspace name; Stage-1 workspaces are schema-v1 and
    # can't be opened by Stage-2 code (by design — see CLAUDE.md D2).
    mem = Mnemoss(
        workspace="quickstart_stage2",
        # Noise off so example output is reproducible.
        formula=FormulaParams(noise_scale=0.0),
    )
    try:
        print("─── Workspace-level (ambient) ───")
        await mem.observe(role="user", content="我明天下午 4:20 和 Alice 见面")
        await mem.observe(role="user", content="见面地点在悉尼歌剧院旁边")

        results = await mem.recall("什么时候见 Alice?", k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.memory.content}")

        print()
        print("─── Per-agent (private) ───")
        alice = mem.for_agent("alice")
        await alice.observe(role="user", content="my secret side project idea")
        await alice.observe(role="user", content="alice-only plan for Thursday")

        bob = mem.for_agent("bob")
        await bob.observe(role="user", content="bob's unrelated grocery list")

        print("Alice asks 'what's my secret plan?' — she should see only her own:")
        for r in await alice.recall("secret plan", k=3):
            print(f"  [{r.score:.3f}] {r.memory.content}")

        print()
        print("Bob can't see Alice's private memories:")
        for r in await bob.recall("secret plan", k=3):
            print(f"  [{r.score:.3f}] {r.memory.content}")

        print()
        print("─── Stage 2: tiers + rebalance ───")
        counts = await mem.tier_counts()
        print(f"Tier counts before rebalance: {counts}")

        stats = await mem.rebalance()
        print(
            f"Rebalance: scanned {stats.scanned}, migrated {stats.migrated} "
            f"(fresh memories stay HOT)."
        )
        after = {t.value: c for t, c in stats.tier_after.items()}
        print(f"Tier counts after rebalance:  {after}")

        print()
        print("DEEP auto-include on temporal cue ('long ago'):")
        # In a real session, rebalance over time would land ancient memories in
        # DEEP. Here we just demonstrate that the cascade scans DEEP for queries
        # like "what was the original plan" without needing include_deep=True.
        for r in await mem.recall("what did we decide long ago about Alice", k=3):
            print(f"  [{r.score:.3f}] {r.memory.content}")
    finally:
        await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
