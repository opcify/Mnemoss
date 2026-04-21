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
    # Uses a Stage-5 workspace name; older-schema workspaces can't be
    # opened by Stage-5 code (by design — see CLAUDE.md D2/D7/D12/D17).
    mem = Mnemoss(
        workspace="quickstart_stage5",
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

        print()
        print("─── Stage 3: lazy extraction on top-k ───")
        # Extraction ran automatically during the recall calls above.
        # Peek at one of the memories to show populated extracted_* fields.
        peek = await mem.recall("Alice", k=1)
        if peek:
            m = peek[0].memory
            print(f"gist:     {m.extracted_gist!r}")
            print(f"entities: {m.extracted_entities}")
            print(f"time:     {m.extracted_time}")
            print(f"level:    {m.extraction_level}")

        print()
        print("─── Dreaming (no LLM configured) ───")
        # dream() runs P1 Replay and P2 Cluster; P3 Consolidate is
        # skipped without an LLM; P4 Relations still writes similar_to
        # edges. Pass a MockLLMClient / OpenAI / Anthropic / Gemini
        # client to exercise P3.
        report = await mem.dream(trigger="idle")
        print(f"Dream trigger: {report.trigger.value}")
        print(f"Duration:      {report.duration_seconds():.3f}s")
        for outcome in report.outcomes:
            # Summarise by dropping the bulky `memories` list.
            summary = {k: v for k, v in outcome.details.items() if k != "memories"}
            print(f"  {outcome.phase.value:<10} {outcome.status:<8} {summary}")
        print(f"Diary:         {report.diary_path}")

        print()
        print("─── Stage 4: memory.md (ambient view) ───")
        print(await mem.export_markdown())

        print()
        print("─── Stage 5: standalone dispose pass ───")
        dispose_stats = await mem.dispose()
        print(
            f"Scanned: {dispose_stats.scanned}, "
            f"disposed: {dispose_stats.disposed}, "
            f"protected: {dispose_stats.protected}"
        )
        print("(Fresh memories are age-protected; nothing disposed.)")

        print()
        print("─── Stage 5: tombstones (disposal audit trail) ───")
        tombs = await mem.tombstones()
        if tombs:
            for t in tombs:
                print(f"  · {t.original_id} dropped ({t.reason}): {t.gist_snapshot!r}")
        else:
            print("(none yet)")
    finally:
        await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
