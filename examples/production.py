"""Mnemoss — production setup example.

Shows the hardened patterns that matter once you move past a single
developer machine:

- ``RetryingEmbedder`` around a cloud provider so a transient 429
  doesn't kill ``observe``.
- ``CostLimits`` capping nightly LLM spend.
- Config-file driven setup so credentials don't live in code.
- ``status()`` interpretation for an ops dashboard.

``examples/basic.py`` is the 30-second tour; this file is closer to
what you'd paste into a service's startup code.

Run with::

    python examples/production.py

By default it uses ``FakeEmbedder`` + ``MockLLMClient`` so it works
offline without any API keys. Set ``MNEMOSS_PRODUCTION_USE_REAL=1``
and provide ``OPENAI_API_KEY`` to swap in real providers.
"""

from __future__ import annotations

import asyncio
import json
import os

from mnemoss import (
    CostLimits,
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    MockLLMClient,
    RetryingEmbedder,
    StorageParams,
)


def _build_embedder() -> object:
    """Return a wrapped embedder appropriate for the environment.

    The shape is the same either way: a primary embedder with a
    retry wrapper around it. Swapping FakeEmbedder for OpenAI/Gemini
    needs only this one change — every other part of the pipeline
    sees the same ``Embedder`` interface.
    """

    if os.environ.get("MNEMOSS_PRODUCTION_USE_REAL") == "1":
        from mnemoss import OpenAIEmbedder

        primary = OpenAIEmbedder()  # reads OPENAI_API_KEY
    else:
        primary = FakeEmbedder(dim=384)

    # Retry transient network errors up to 3 times with exponential
    # backoff. ValueError (programmer errors) is never retried.
    return RetryingEmbedder(
        primary,
        max_retries=3,
        base_delay_seconds=0.2,
        max_delay_seconds=5.0,
    )


def _build_llm() -> object:
    """Return an LLM client for Dream P3 Consolidate.

    In the default (fake) path we return a MockLLMClient with canned
    responses so the dream pipeline runs end-to-end without billable
    API calls. In real mode swap in your provider of choice.
    """

    if os.environ.get("MNEMOSS_PRODUCTION_USE_REAL") == "1":
        from mnemoss import AnthropicClient

        return AnthropicClient()  # reads ANTHROPIC_API_KEY

    canned = {
        "summary": {
            "memory_type": "fact",
            "content": "team has discussed the Q3 roadmap",
            "abstraction_level": 0.65,
        },
        "refinements": [],
        "patterns": [],
    }
    return MockLLMClient(responses=[canned for _ in range(100)])


async def main() -> None:
    # ─── 1. Cost ceiling ────────────────────────────────────────────
    # Production workspaces should cap LLM spend. Per-run cap is the
    # cheap insurance; per-day is the real spend ceiling. Per-month
    # is the accountant's safety net.
    cost_limits = CostLimits(
        max_llm_calls_per_run=50,
        max_llm_calls_per_day=500,
        max_llm_calls_per_month=10_000,
    )

    # ─── 2. Build Mnemoss ───────────────────────────────────────────
    mem = Mnemoss(
        workspace="production_demo",
        embedding_model=_build_embedder(),
        llm=_build_llm(),
        cost_limits=cost_limits,
        # Turn noise off only if you need deterministic rankings.
        # Production usually leaves the default (0.25) for human-like
        # retrieval variance.
        formula=FormulaParams(noise_scale=0.25),
        storage=StorageParams(root=None),  # defaults to ~/.mnemoss
    )

    try:
        # ─── 3. Observe some data ───────────────────────────────────
        print("observing 20 team messages ...")
        messages = [
            ("alice", "kickoff for Q3 roadmap happens Thursday"),
            ("alice", "bob is leading the migration work"),
            ("bob",   "tracked the new deps upgrade in #platform"),
            ("bob",   "need a decision on the payment provider by Friday"),
            ("carol", "designs for the dashboard are done, review tomorrow"),
        ] * 4  # 20 messages total
        for who, content in messages:
            await mem.for_agent(who).observe(role="user", content=content)

        # ─── 4. Recall ─────────────────────────────────────────────
        # FakeEmbedder vectors are near-orthogonal hashes so semantic
        # recall here leans on FTS + activation, not cosine. With a
        # real embedder this query returns the kickoff note first.
        print("\nrecall('migration'):")
        for r in await mem.recall("migration", k=3):
            print(f"  [{r.score:+.3f}] {r.memory.content}")

        # ─── 5. Dream (bounded by cost limits) ─────────────────────
        print("\nrunning a nightly dream ...")
        report = await mem.dream(trigger="nightly")
        if report.degraded_mode:
            print(f"  degraded — {len(report.errors())} phase(s) errored:")
            for e in report.errors():
                print(f"    · {e.phase.value}: {e.error}")
        else:
            print("  completed cleanly")
        for outcome in report.outcomes:
            print(f"  {outcome.phase.value:<12} {outcome.status}")

        # ─── 6. Operational status ─────────────────────────────────
        # This is what a production dashboard would poll. Every field
        # is JSON-safe so you can ship it over REST / Prometheus /
        # whatever observability stack you run.
        status = await mem.status()
        print("\nstatus() snapshot:")
        print(json.dumps(
            {
                "memory_count": status["memory_count"],
                "tier_counts": status["tier_counts"],
                "tombstones": status["tombstone_count"],
                "llm_cost": status["llm_cost"],
                "dreams_recent": len(status["dreams"]["recent"]),
                "dreams_degraded": status["dreams"]["recent_degraded_count"],
            },
            indent=2,
        ))

        # ─── 7. Explainability ─────────────────────────────────────
        # When a ranking looks wrong, explain_recall() gives you the
        # per-component ACT-R breakdown — JSON-exportable.
        top = await mem.recall("migration", k=1)
        if top:
            breakdown = await mem.explain_recall(
                "migration", top[0].memory.id
            )
            if breakdown is not None:
                print("\nexplain_recall for top result:")
                print(json.dumps(breakdown.to_dict(), indent=2))
    finally:
        await mem.close()


if __name__ == "__main__":
    asyncio.run(main())
