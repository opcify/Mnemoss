"""Mnemoss adapter for the launch-comparison benchmark.

Default config for published-chart runs
---------------------------------------

- Embedder: ``text-embedding-3-small`` via ``OpenAIEmbedder`` wrapped
  in ``RetryingEmbedder`` (Issue 1.1A + 2.b from eng review). Pinning
  this way gives fair numbers against Chroma, which uses the same
  model, and keeps transient network hiccups from tanking a multi-
  hour Chart 1 run.
- Recall: ``auto_expand=False, include_deep=True``. Benchmark wants
  the primary recall path (Mem0 and Chroma have no auto-expand
  analogue) at Mnemoss's full tier cascade (HOT → WARM → COLD → DEEP).
  The simulation in ``demo/simulate.py`` flips these — see Issue 2.e.
- Noise: ``FormulaParams.noise_scale=0.0`` by default. Deterministic
  ranking across benchmark runs so Chart 4 Kendall tau is 1.0 by
  construction on the Mnemoss side. Override to a non-zero value if
  you're deliberately measuring noise-sensitivity.

Gap A cleanup (eng review)
--------------------------

Each instance gets a fresh ``tempfile.mkdtemp`` workspace root. A
leftover workspace from a crashed previous run would re-trigger
``WorkspaceLockError`` on re-run; fresh tempdirs per instance and
unconditional cleanup on ``close()`` close that gap.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from bench.backends.base import RecallHit
from mnemoss import (
    Embedder,
    FormulaParams,
    Mnemoss,
    OpenAIEmbedder,
    RetryingEmbedder,
    StorageParams,
    TierCapacityParams,
)


class MnemossBackend:
    """``MemoryBackend`` adapter around ``Mnemoss`` for cross-system benchmarks.

    Parameters
    ----------
    formula:
        Override the full ``FormulaParams``. If supplied, ``noise_scale``
        is ignored (use the field on the supplied dataclass). Use this
        to benchmark non-default activation configs — e.g. a
        "semantic-only" preset that kills recency bias for bulk-ingest
        workloads like LoCoMo where all memories were just created at
        once and the recency term otherwise drowns out matching.

        Example::

            FormulaParams(noise_scale=0.0, eta_0=0.0, d=0.01)

        ``eta_0=0`` disables the grace term; ``d=0.01`` (can't be zero
        per the validator) nearly flattens the history term. What's
        left is dominated by ``MP * cosine_sim``, which is essentially
        raw-stack cosine scoring with Mnemoss's candidate-selection +
        query-bias layer on top.
    """

    backend_id = "mnemoss"

    def __init__(
        self,
        *,
        embedding_model: Embedder | None = None,
        workspace: str = "bench",
        noise_scale: float = 0.0,
        include_deep: bool = True,
        formula: FormulaParams | None = None,
        encoder: Any | None = None,
        tier_capacity: TierCapacityParams | None = None,
        llm_client: Any | None = None,
        cost_limits: Any | None = None,
        expand_via_streak: bool = False,
        dreamer: Any | None = None,
    ) -> None:
        # Tempdir first, before Mnemoss() — stored on self so close()
        # can rmtree it even if the constructor below raises. This is
        # what closes Gap A from the eng review.
        self._tempdir = Path(tempfile.mkdtemp(prefix="mnemoss_bench_"))
        self._closed = False

        if embedding_model is None:
            embedding_model = RetryingEmbedder(
                OpenAIEmbedder(model="text-embedding-3-small"),
                max_retries=3,
            )

        if formula is None:
            formula = FormulaParams(noise_scale=noise_scale)

        kwargs: dict[str, Any] = {
            "workspace": workspace,
            "embedding_model": embedding_model,
            "formula": formula,
            "storage": StorageParams(root=self._tempdir),
        }
        if encoder is not None:
            kwargs["encoder"] = encoder
        if tier_capacity is not None:
            kwargs["tier_capacity"] = tier_capacity
        # ``llm_client`` is needed for ``mem.dream()`` — the consolidate
        # phase makes one LLM call per cluster. ``cost_limits`` caps
        # that spend so a runaway dream on a long haystack doesn't
        # turn into a surprise bill. The Mnemoss kwarg is ``llm=`` (not
        # ``llm_client=``); we keep the explicit name on the bench
        # adapter so callers don't confuse it with the chat-LLM client
        # the LongMemEval-S harness uses for its own generator + judge.
        if llm_client is not None:
            kwargs["llm"] = llm_client
        if cost_limits is not None:
            kwargs["cost_limits"] = cost_limits
        if dreamer is not None:
            kwargs["dreamer"] = dreamer

        self._mem = Mnemoss(**kwargs)
        self._include_deep = include_deep
        self._expand_via_streak = expand_via_streak

    async def observe(self, text: str, ts: float) -> str:
        """Observe ``text`` as a user message. Returns Mnemoss's memory id.

        ``ts`` is accepted for protocol uniformity but not passed to
        Mnemoss — Mnemoss uses its own ``datetime.now(UTC)`` inside
        ``observe()`` (see ``mnemoss/client.py:171``). Benchmarks that
        need replay-able timestamps should run against a corpus whose
        natural ordering matches ingestion order.

        ``turn_id`` is intentionally NOT supplied. Per ``client.py:173``,
        a missing ``turn_id`` sets ``auto_close=True`` so each observe
        produces exactly one Memory row that flushes before the call
        returns. Matches ``bench/bench_recall.py`` and gives us the
        predictable "one observe = one memory" semantics the benchmark
        needs.
        """

        # ``role="user"`` passes ``EncoderParams.encoded_roles`` by default.
        memory_id = await self._mem.observe(
            role="user",
            content=text,
            session_id="bench",
        )
        if memory_id is None:
            # Role filter would have to be actively reconfigured to drop
            # ``user`` for this to trigger — treat as a loud misconfig,
            # not a silent skip. If it fires, the caller's Mnemoss
            # construction has an encoder override we didn't expect.
            raise RuntimeError(
                "MnemossBackend.observe: Mnemoss returned None for "
                "role='user'. Check EncoderParams.encoded_roles — "
                "benchmark runs expect all roles enabled."
            )
        return memory_id

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Recall top-k with Mnemoss's full cascade, no auto-expand.

        ``reconsolidate=False`` is critical for benchmark fairness:
        Mnemoss's default reconsolidation appends ``now`` to each
        returned memory's ``access_history``, growing ``B_i`` for the
        next recall. Across hundreds of benchmark queries, the
        earliest-recalled memories get massive B_i boosts regardless
        of actual relevance, biasing the remaining queries' rankings.
        See ``docs/ROOT_CAUSE.md``.
        """

        results = await self._mem.recall(
            query,
            k=k,
            include_deep=self._include_deep,
            auto_expand=False,
            reconsolidate=False,
        )
        return [
            RecallHit(
                memory_id=r.memory.id,
                rank=i + 1,
                score=float(r.score),
            )
            for i, r in enumerate(results)
        ]

    async def ingest_session(
        self,
        *,
        session_id: str,
        ts: float,
        turns: list[dict[str, str]],
    ) -> None:
        """Ingest one multi-turn session, one ``observe()`` per turn.

        LongMemEval-S parity: each turn's native ``role`` is passed
        through (Mnemoss filters by ``EncoderParams.encoded_roles``;
        defaults accept all four roles). ``session_id`` matches the
        haystack's session id so Mnemoss's per-session structure
        survives ingestion. ``ts`` is accepted for protocol uniformity
        but Mnemoss uses its own clock inside ``observe()`` — see the
        ``MnemossBackend.observe`` docstring.
        """

        for turn in turns:
            await self._mem.observe(
                role=turn["role"],
                content=turn["content"],
                session_id=session_id,
            )

    async def recall_text(self, query: str, k: int = 10) -> list[str]:
        """Return the top-K hits' ``content`` strings in rank order.

        LongMemEval-S composes the QA-generator prompt from memory
        text. Routing through ``recall_text`` keeps the harness
        backend-agnostic; the Mnemoss-specific scoring breakdown
        stays accessible via :meth:`explain`.

        Each snippet is prefixed with ``[YYYY-MM-DD type]`` derived
        from ``Memory.created_at`` and ``Memory.memory_type``. The
        text content of recalled memories carries no temporal
        signal on its own (an ``observe()`` from yesterday and one
        from last year look identical when read back), so the QA
        generator has no way to disambiguate competing values for
        the same fact (LongMemEval-S knowledge-update slice) or
        order events in time (temporal-reasoning slice). The
        ``type`` token (``episode`` / ``summary``) tells the LLM
        when a snippet is a Dream-consolidated abstraction over
        many sources, which it should weight differently from a
        single raw turn. The format is intentionally compact so
        ``snippet_max_chars`` budget isn't dominated by metadata.

        When ``expand_via_streak=True`` we issue a primer recall
        first. ``recall(auto_expand=True)`` only fires the relation-
        graph BFS when the query history shows a same-topic streak
        (cosine ≥ ``same_topic_cosine``, gap ≤ ``streak_reset_seconds``).
        Bench questions arrive in fresh workspaces with empty history,
        so a single call would never engage spreading. Issuing the
        same query twice produces a streak of 1 → 2 with cosine 1.0
        and gap 0 s, and the second call returns top-K plus expanded
        candidates pulled from the relations graph (``co_occurs_in_session``
        plus the ``derived_from``/``derived_to`` edges Dream Consolidate
        writes when ``--dream-after-ingest`` is on). This is the
        bench-side surface for measuring whether cross-session
        synthesis actually helps the multi-session / temporal-
        reasoning / knowledge-update slices.
        """

        if self._expand_via_streak:
            await self._mem.recall(
                query,
                k=k,
                include_deep=self._include_deep,
                auto_expand=True,
                reconsolidate=False,
            )
            results = await self._mem.recall(
                query,
                k=k,
                include_deep=self._include_deep,
                auto_expand=True,
                reconsolidate=False,
            )
        else:
            results = await self._mem.recall(
                query,
                k=k,
                include_deep=self._include_deep,
                auto_expand=False,
                reconsolidate=False,
            )
        snippets: list[str] = []
        for r in results:
            m = r.memory
            stamp = m.created_at.strftime("%Y-%m-%d") if m.created_at else "????-??-??"
            mtype = getattr(m.memory_type, "value", str(m.memory_type))
            snippets.append(f"[{stamp} {mtype}] {m.content}")
        return snippets

    async def dream(self, *, trigger: str = "session_end") -> Any:
        """Run Mnemoss's dream pipeline. Returns the ``DreamReport``.

        LongMemEval-S parity: multi-session and temporal-reasoning
        questions need the relations graph that Dream populates
        (P4 Relations) for spreading activation to engage. Without
        it, recall is limited to direct embedding match — fine for
        single-session questions, useless when the answer requires
        connecting evidence across sessions.

        Trigger ``"session_end"`` is the natural choice for "ingestion
        is done, please consolidate." ``"nightly"`` is heavier and
        would also rebalance + dispose; benches typically don't want
        disposal in the middle of a 24-question run.
        """

        return await self._mem.dream(trigger=trigger)

    async def explain(self, query: str, memory_id: str) -> dict | None:
        """Return Mnemoss's ``ActivationBreakdown`` for one memory as a
        JSON-safe dict, or ``None`` if the memory isn't in the workspace.

        This is the hook :mod:`demo.simulate` uses to capture the per-
        component scores (``B_i``, spreading, matching, noise) into
        trace events so the player can render the stacked-bar
        breakdown on recall events. Not part of the generic
        ``MemoryBackend`` protocol — Mnemoss-specific.
        """

        breakdown = await self._mem.explain_recall(query, memory_id)
        return breakdown.to_dict() if breakdown is not None else None

    async def close(self) -> None:
        """Close the Mnemoss handle and rmtree the tempdir. Idempotent."""

        if self._closed:
            return
        self._closed = True
        try:
            await self._mem.close()
        finally:
            # Unconditional cleanup — Gap A. ignore_errors so a
            # stuck lock file doesn't mask a real close() failure.
            shutil.rmtree(self._tempdir, ignore_errors=True)

    # Context-manager sugar so tests can ``async with MnemossBackend(...)``.
    async def __aenter__(self) -> MnemossBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
