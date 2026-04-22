"""Mnemoss Quick Start.

A quickstart that actually proves recall works. Every section seeds
a mix of topic-relevant and deliberately irrelevant messages, then
runs two queries — one on each topic — and prints the top hits so
you can see the ACT-R activation scoring pick the right memories
out of the noise, not just "return whatever was most recent."

Covers: ambient + per-agent observes, recall (with selectivity
against distractors), tiering / rebalance, DEEP-cue cascade, lazy
extraction, a dream cycle (no LLM → P3 skips), memory.md export,
disposal pass, tombstones. First run downloads ~470MB for the
multilingual sentence-transformers model.

    python examples/basic.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from mnemoss import FormulaParams, Mnemoss, StorageParams


def _print_hits(
    title: str, results: list, *, mark_relevant: str | None = None
) -> None:
    """Render recall results with scores, truncated for readability.

    ``mark_relevant`` is a case-insensitive substring that marks a
    result as "expected" with a ✓ glyph; everything else gets a
    space. Makes the selectivity visible at a glance even when the
    absolute scores cluster tightly (which they do on fresh
    workspaces, because base-level activation gives every
    just-observed memory an encoding-grace bonus ≈ 1.0).
    """

    print(f"  {title}")
    if not results:
        print("    (no hits)")
        return
    needle = mark_relevant.lower() if mark_relevant else None
    for i, r in enumerate(results, 1):
        content = r.memory.content
        flag = " "
        if needle is not None and needle in content.lower():
            flag = "✓"
        if len(content) > 100:
            content = content[:100] + "..."
        print(f"    {flag} {i}. [{r.score:+.3f}] {content}")


async def main() -> None:
    # The example is self-contained: each run creates a fresh
    # workspace inside a tempdir so repeated runs don't accumulate
    # memories (which would muddle the recall output). A production
    # deployment would of course keep state — default storage root
    # is ``~/.mnemoss``.
    tmp = tempfile.TemporaryDirectory(prefix="mnemoss_quickstart_")
    mem = Mnemoss(
        workspace="quickstart",
        # Noise off so this example's rankings are reproducible.
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=Path(tmp.name)),
    )
    try:
        # ─── Workspace-level (ambient) observes ─────────────────────
        # "Ambient" memories belong to the workspace, not to any
        # specific agent — every agent that ``for_agent(...)``'s the
        # workspace can see them alongside their own private ones.
        #
        # We seed three Q3-kickoff-related messages interleaved with
        # three deliberately unrelated messages (office logistics,
        # on-call rotation, gym perk) so you can see recall pick out
        # the kickoff ones from noise instead of just returning
        # "whatever was most recent."
        print("─── Workspace-level (ambient) ───")
        print("seeding 6 ambient messages (3 about kickoff, 3 distractors)...")
        await mem.observe(
            role="user",
            content=(
                "Alice and I scheduled the Q3 planning kickoff for "
                "next Tuesday at 4:20 PM in the main conference room "
                "on the fifteenth floor, and we agreed the meeting "
                "should end no later than 5:30 so people can make "
                "the 6 PM commuter train."
            ),
        )
        await mem.observe(
            role="user",
            content=(
                "The kickoff agenda, which Alice circulated over "
                "email this morning, covers three topics in order: "
                "the hiring budget for the rest of the year, the "
                "product roadmap revisions, and a go/no-go decision "
                "on adopting the new payment provider."
            ),
        )
        await mem.observe(
            role="user",
            content=(
                "Bob mentioned that the platform migration is blocked "
                "on the legal review of the data residency "
                "requirements, so we should expect the rollout "
                "timeline to slip by at least two weeks — worth "
                "flagging in the Q3 kickoff."
            ),
        )
        # ─── distractors — intentionally unrelated to the kickoff ──
        await mem.observe(
            role="user",
            content=(
                "The espresso machine in the third-floor kitchen is "
                "broken again; facilities says a replacement grinder "
                "will arrive Friday, so please use the second-floor "
                "kitchen until then or bring your own from home."
            ),
        )
        await mem.observe(
            role="user",
            content=(
                "Reminder to the platform team: this weekend's on-call "
                "shift is covered by Dana, with Ramon as backup; file "
                "incident tickets in the usual channel and page "
                "either of them directly if the pagerduty rotation "
                "falls through."
            ),
        )
        await mem.observe(
            role="user",
            content=(
                "HR announced a new corporate gym partnership that "
                "gives employees 50% off monthly memberships at any "
                "of the downtown branches; sign-up details went out "
                "in an email from benefits@company this morning."
            ),
        )

        # A ✓ next to a row means the result contains the topic-
        # specific token we're looking for. On a fresh workspace
        # absolute scores cluster tightly (encoding-grace dominates);
        # the ✓ markers let you see selectivity is working even when
        # rank-1 isn't always the "most relevant" hit.
        print("\nquery 1: 'Q3 planning kickoff agenda hiring roadmap'")
        _print_hits(
            "top-3 (kickoff memories should show ✓):",
            await mem.recall(
                "Q3 planning kickoff agenda hiring roadmap", k=3
            ),
            mark_relevant="kickoff",
        )

        print("\nquery 2: 'on-call rotation Dana Ramon this weekend'")
        _print_hits(
            "top-3 (on-call reminder should show ✓):",
            await mem.recall(
                "on-call rotation Dana Ramon this weekend", k=3
            ),
            mark_relevant="on-call",
        )

        print("\nquery 3: 'espresso machine broken grinder kitchen'")
        _print_hits(
            "top-3 (espresso distractor should show ✓):",
            await mem.recall(
                "espresso machine broken grinder kitchen", k=3
            ),
            mark_relevant="espresso",
        )

        # ─── Per-agent (private) observes ───────────────────────────
        # ``for_agent`` binds an agent_id. An agent sees its own
        # private memories + workspace ambient memories, but NEVER
        # another agent's privates. We give both Alice and Bob a mix
        # of topic-relevant + irrelevant messages so the scoping is
        # easy to see in the recall output.
        print()
        print("─── Per-agent (private) ───")
        alice = mem.for_agent("alice")
        bob = mem.for_agent("bob")

        print("seeding Alice: 2 side-project notes + 2 distractors...")
        await alice.observe(
            role="user",
            content=(
                "Private note for Alice: drafting a side-project "
                "proposal about predictive caching for recall-heavy "
                "agent systems; plan to share with the research "
                "leads only after the Q3 kickoff so it doesn't get "
                "rolled into the main roadmap discussion."
            ),
        )
        await alice.observe(
            role="user",
            content=(
                "Alice's follow-up to the caching idea: sketched a "
                "two-page design doc covering the cold-path cost, "
                "the expected hit rate under realistic workloads, "
                "and the open question of whether dream-phase "
                "pre-warming would interact badly with disposal."
            ),
        )
        # ─── Alice's distractors ─────────────────────────────────
        await alice.observe(
            role="user",
            content=(
                "Alice: passport renewal appointment is on May 14th "
                "at 10 AM at the federal building downtown; bring "
                "two passport photos, the current passport, and the "
                "DS-82 form already filled out."
            ),
        )
        await alice.observe(
            role="user",
            content=(
                "Alice: reminder to book the Portland flight for the "
                "conference in early June; prefer the 7 AM departure "
                "so there's enough buffer before the opening keynote "
                "on the 8th."
            ),
        )

        print("seeding Bob:  1 grocery list + 2 distractors...")
        await bob.observe(
            role="user",
            content=(
                "Bob's grocery list for tomorrow's team breakfast: "
                "sourdough from the bakery on Fifth, half a pound of "
                "smoked salmon, one flat of eggs, and a dozen "
                "oranges — nothing unusual, just the standard "
                "Friday spread."
            ),
        )
        await bob.observe(
            role="user",
            content=(
                "Bob's dentist appointment got rescheduled from "
                "Thursday afternoon to the following Monday morning "
                "at 9:30 AM, so Thursday's team sync doesn't need "
                "to be shifted after all."
            ),
        )
        await bob.observe(
            role="user",
            content=(
                "Bob's manager recommended reading 'Working in "
                "Public' by Nadia Eghbal before the next OSS "
                "strategy review; public library has three copies "
                "and one is on hold for pickup under Bob's name."
            ),
        )

        print("\nalice asks 'predictive caching side project design doc':")
        _print_hits(
            "(alice's side-project notes should show ✓):",
            await alice.recall(
                "predictive caching side project design doc", k=5
            ),
            mark_relevant="caching",
        )

        print("\nbob asks 'sourdough smoked salmon eggs team breakfast':")
        _print_hits(
            "(bob's grocery list should show ✓ — alice's private notes NEVER appear):",
            await bob.recall(
                "sourdough smoked salmon eggs team breakfast", k=5
            ),
            mark_relevant="grocery list",
        )

        print("\nbob asks 'Alice caching side project design doc':")
        print("  (scoping invariant: alice's private notes are NOT in bob's scope)")
        bob_on_alice = await bob.recall(
            "Alice caching side project design doc", k=5
        )
        _print_hits("top-5 for bob:", bob_on_alice)
        leaked = [
            r for r in bob_on_alice if "caching" in r.memory.content.lower()
        ]
        if leaked:
            print(
                f"  ⚠ SCOPING LEAK: {len(leaked)} of alice's caching "
                "note(s) appeared in bob's result — this would be a bug!"
            )
        else:
            print(
                "  ✓ scoping enforced: none of alice's caching notes "
                "surfaced in bob's recall."
            )

        # ─── Index tiers + rebalance ────────────────────────────────
        # Every memory is assigned to one of four tiers
        # (HOT/WARM/COLD/DEEP) by its ``idx_priority``. Rebalance
        # walks the whole workspace and migrates memories whose
        # priority has drifted past a tier boundary.
        print()
        print("─── Index tiers + rebalance ───")
        counts = await mem.tier_counts()
        print(f"tier counts before rebalance: {counts}")

        stats = await mem.rebalance()
        print(
            f"rebalance: scanned {stats.scanned}, migrated {stats.migrated} "
            f"(fresh memories stay HOT thanks to the encoding-grace bonus)."
        )
        after = {t.value: c for t, c in stats.tier_after.items()}
        print(f"tier counts after rebalance:  {after}")

        print()
        # The cascade normally scans HOT → WARM → COLD and stops once
        # it has enough high-confidence hits. DEEP is only scanned
        # when a caller flags ``include_deep=True`` or when the
        # query itself contains a temporal-distance cue (like "long
        # ago") that strongly implies the answer is in deep memory.
        #
        # With a fresh workspace there's nothing *in* DEEP yet, so
        # this demo shows the cue is still recognized — you can
        # verify by grepping the query for "long ago" against the
        # cue list in ``has_deep_cue``.
        print("DEEP auto-include on a temporal cue ('long ago'):")
        _print_hits(
            "query: 'what did we decide long ago about the payment provider'",
            await mem.recall(
                "what did we decide long ago about the payment provider",
                k=3,
            ),
        )

        print()
        # ─── Lazy extraction on recalled top-k ──────────────────────
        # Recall populates ``extracted_gist`` + ``extracted_time`` on
        # returned top-k memories the first time they surface.
        # Entities / location / participants intentionally stay None
        # — NER is out of scope (see §9.7 of PROJECT_KNOWLEDGE).
        print("─── Lazy extraction on top-k ───")
        peek = await mem.recall("Q3 kickoff schedule", k=1)
        if peek:
            m = peek[0].memory
            print(f"  gist:             {m.extracted_gist!r}")
            print(f"  time:             {m.extracted_time}")
            print(
                f"  extraction_level: {m.extraction_level}  "
                "(1 = heuristic, 2 = Dream P3)"
            )

        print()
        # ─── Dreaming (no LLM configured → P3 skips) ───────────────
        # ``dream()`` without an LLM still runs REPLAY, CLUSTER, and
        # RELATIONS — only CONSOLIDATE (the LLM phase) skips. Pass a
        # MockLLMClient / OpenAIClient / AnthropicClient / GeminiClient
        # to exercise the full six-phase pipeline.
        print("─── Dreaming (no LLM configured) ───")
        report = await mem.dream(trigger="idle")
        print(f"  trigger:  {report.trigger.value}")
        print(f"  duration: {report.duration_seconds():.3f}s")
        if report.degraded_mode:
            print("  (degraded: some phases errored)")
        for outcome in report.outcomes:
            # Drop the bulky ``memories`` blob from the summary.
            summary = {
                k: v for k, v in outcome.details.items() if k != "memories"
            }
            status_line = f"{outcome.phase.value:<12} {outcome.status:<8}"
            if outcome.skip_reason:
                status_line += f" ({outcome.skip_reason})"
            print(f"  {status_line} {summary}")
        print(f"  diary:    {report.diary_path}")

        print()
        # ─── memory.md — human-readable ambient view ───────────────
        # Generates a Markdown doc of the workspace's high-priority
        # memories, grouped by type. Tuned to be readable by humans
        # AND feedable as LLM context in downstream systems.
        print("─── memory.md (ambient view) ───")
        print(await mem.export_markdown())

        print()
        # ─── Standalone dispose pass ───────────────────────────────
        # Dispose walks every memory, computes max activation across
        # a sweep of plausible queries, and tombstones ones whose
        # peak activation falls below ``tau - delta``. Fresh
        # memories are age-protected, so a just-built workspace
        # never disposes anything.
        print("─── Standalone dispose pass ───")
        dispose_stats = await mem.dispose()
        print(
            f"  scanned: {dispose_stats.scanned}, "
            f"disposed: {dispose_stats.disposed}, "
            f"protected: {dispose_stats.protected}"
        )
        print("  (Fresh memories are age-protected; nothing disposed on first run.)")

        print()
        # ─── Tombstones — audit trail for disposed memories ────────
        # Each disposal leaves a tombstone with the memory's gist
        # snapshot and the reason for dropping, so you can debug
        # "why did Mnemoss forget X?" after the fact.
        print("─── Tombstones (disposal audit trail) ───")
        tombs = await mem.tombstones()
        if tombs:
            for t in tombs:
                print(
                    f"  · {t.original_id} dropped "
                    f"({t.reason}): {t.gist_snapshot!r}"
                )
        else:
            print("  (none yet)")
    finally:
        await mem.close()
        tmp.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
