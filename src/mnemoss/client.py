"""Public Mnemoss client.

Three core methods (``observe``, ``recall``, ``pin``) plus a thin
per-agent handle from ``for_agent(id)``. Everything is async.

The constructor does no I/O — it just resolves paths and picks the
embedder. First ``observe()`` auto-creates the workspace DB (Stage 1
settled decision).
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any

import ulid

from mnemoss.core.config import (
    SCHEMA_VERSION,
    DreamerParams,
    EncoderParams,
    FormulaParams,
    MnemossConfig,
    SegmentationParams,
    StorageParams,
    TierCapacityParams,
)
from mnemoss.core.types import RawMessage, Tombstone
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.dream.diary import append_entry, dream_diary_path
from mnemoss.dream.dispose import DisposalStats, dispose_pass
from mnemoss.dream.runner import DreamRunner
from mnemoss.dream.types import DreamReport, TriggerType
from mnemoss.encoder import Embedder, make_embedder
from mnemoss.encoder.chunking import split_content
from mnemoss.encoder.event_encoder import encode_event, should_encode
from mnemoss.encoder.event_segmentation import ClosedEvent, EventSegmenter
from mnemoss.export import render_memory_md
from mnemoss.formula.activation import ActivationBreakdown
from mnemoss.index import RebalanceStats
from mnemoss.index import rebalance as _rebalance
from mnemoss.llm.client import LLMClient
from mnemoss.recall import RecallEngine, RecallResult
from mnemoss.relations import write_cooccurrence_edges
from mnemoss.store.paths import raw_log_db_path, workspace_db_path
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc
_log = logging.getLogger("mnemoss.client")


class Mnemoss:
    """The user-facing memory object.

    >>> mem = Mnemoss(workspace="my_agent")
    >>> await mem.observe(role="user", content="hello")
    >>> results = await mem.recall("hello", k=3)
    """

    def __init__(
        self,
        workspace: str,
        *,
        embedding_model: str | Embedder = "local",
        formula: FormulaParams | None = None,
        encoder: EncoderParams | None = None,
        storage: StorageParams | None = None,
        segmentation: SegmentationParams | None = None,
        dreamer: DreamerParams | None = None,
        tier_capacity: TierCapacityParams | None = None,
        llm: LLMClient | None = None,
        cost_limits: CostLimits | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = MnemossConfig(
            workspace=workspace,
            formula=formula or FormulaParams(),
            encoder=encoder or EncoderParams(),
            storage=storage or StorageParams(),
            segmentation=segmentation or SegmentationParams(),
            dreamer=dreamer or DreamerParams(),
            tier_capacity=tier_capacity or TierCapacityParams(),
        )
        self._embedder = make_embedder(embedding_model)
        self._llm = llm
        # Cost governance — ``cost_limits=None`` means unlimited
        # (historical behavior). The ledger is lazily bound to the
        # store on ``_ensure_open`` so its persistence lives in the
        # same workspace DB as the memories it's counting calls for.
        self._cost_limits = cost_limits or CostLimits()
        self._cost_ledger: CostLedger | None = None
        self._store: SQLiteBackend | None = None
        self._working = WorkingMemory(capacity=self._config.encoder.working_memory_capacity)
        self._engine: RecallEngine | None = None
        self._segmenter = EventSegmenter()
        self._open_lock = asyncio.Lock()
        self._segment_lock = asyncio.Lock()
        self._rng = rng if rng is not None else random.Random()
        # Operational timestamps, exposed via ``status()``. ``None``
        # means the corresponding operation has not run in this
        # instance's lifetime.
        self._last_observe_at: datetime | None = None
        self._last_dream_at: datetime | None = None
        self._last_dream_trigger: str | None = None
        self._last_rebalance_at: datetime | None = None
        self._last_dispose_at: datetime | None = None
        # Bounded in-memory history of recent dream runs surfaced via
        # ``status()``. Holds lightweight summaries, not full
        # ``DreamReport``s, to keep the dashboard payload small.
        self._dream_history: list[dict[str, Any]] = []
        self._dream_history_cap = 10

    @classmethod
    def from_config_file(
        cls,
        workspace: str,
        *,
        path: str | None = None,
        **overrides: Any,
    ) -> Mnemoss:
        """Construct a ``Mnemoss`` from a ``mnemoss.toml``.

        The file supplies ``embedding_model`` and ``llm``; other keyword
        arguments can be passed through ``**overrides`` to override or
        complement it. Raises if no config file is found.
        """

        from mnemoss.core.config_file import load_config_file

        cfg = load_config_file(path)
        if cfg is None:
            raise FileNotFoundError(
                "No mnemoss config file found. Set MNEMOSS_CONFIG, create "
                "./mnemoss.toml, or ~/.mnemoss/config.toml, or pass path=."
            )
        kwargs: dict[str, Any] = {
            "embedding_model": cfg.build_embedder(),
            "llm": cfg.build_llm(),
        }
        kwargs.update(overrides)
        return cls(workspace=workspace, **kwargs)

    # ─── public API ───────────────────────────────────────────────────

    async def observe(
        self,
        role: str,
        content: str,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Observe a message.

        Always writes to the Raw Log (Principle 3). Only stages a Memory
        row if ``role`` is in ``encoder.encoded_roles``. Returns the new
        memory's id (pre-assigned by the segmenter), or ``None`` if the
        role was filtered out.

        ``turn_id`` controls event batching: messages sharing an explicit
        ``turn_id`` accumulate into one Memory that flushes when a
        closing rule fires (turn shift, time gap, size cap). Callers
        that omit ``turn_id`` get Stage-1/2 semantics — one observe =
        one Memory, persisted before observe() returns.
        """

        await self._ensure_open()
        assert self._store is not None

        now = datetime.now(UTC)
        effective_session = session_id or "default"
        auto_close = turn_id is None
        effective_turn = turn_id or str(ulid.new())
        msg = RawMessage(
            id=str(ulid.new()),
            workspace_id=self._config.workspace,
            agent_id=agent_id,
            session_id=effective_session,
            turn_id=effective_turn,
            parent_id=parent_id,
            timestamp=now,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        await self._store.write_raw_message(msg)
        self._last_observe_at = now

        if not should_encode(msg, self._config.encoder):
            return None

        async with self._segment_lock:
            step = self._segmenter.on_observe(
                msg, now, self._config.segmentation, auto_close=auto_close
            )

        for event in step.closed_events:
            await self._persist_event(event)

        _log.info(
            "observed",
            extra={
                "workspace": self._config.workspace,
                "agent_id": agent_id,
                "role": role,
                "encoded": step.pending_memory_id is not None,
                "events_closed": len(step.closed_events),
            },
        )
        return step.pending_memory_id

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        agent_id: str | None = None,
        include_deep: bool = False,
        auto_expand: bool = True,
        reconsolidate: bool = True,
    ) -> list[RecallResult]:
        """Retrieve the top-k memories for ``query``.

        ``reconsolidate`` (default ``True``) follows the ACT-R story:
        every returned memory gets its access history extended, rehearsal
        count bumped, last-accessed timestamp refreshed, and (if in DEEP)
        promoted to WARM. This is the whole point of ACT-R's "recall
        strengthens memory" design — the more often you fetch a memory,
        the more accessible it becomes for future recalls.

        Set ``reconsolidate=False`` for read-only callers that must not
        mutate memory state: benchmark harnesses (every query otherwise
        boosts early-recalled memories' ``B_i`` and biases later
        queries), audit / explain paths, external evaluation runs, or
        any scenario where "does Mnemoss return the right top-k for
        THIS query in isolation" is the only question being asked.
        """

        await self._ensure_open()
        assert self._engine is not None
        results = await self._engine.recall(
            query,
            agent_id=agent_id,
            k=k,
            include_deep=include_deep,
            auto_expand=auto_expand,
            reconsolidate=reconsolidate,
        )
        _log.info(
            "recalled",
            extra={
                "workspace": self._config.workspace,
                "agent_id": agent_id,
                "k": k,
                "include_deep": include_deep,
                "auto_expand": auto_expand,
                "results": len(results),
                "expanded": sum(1 for r in results if r.source == "expanded"),
            },
        )
        return results

    async def pin(self, memory_id: str, *, agent_id: str | None = None) -> None:
        await self._ensure_open()
        assert self._store is not None
        await self._store.pin(memory_id, agent_id)

    async def explain_recall(
        self,
        query: str,
        memory_id: str,
        *,
        agent_id: str | None = None,
    ) -> ActivationBreakdown | None:
        """Return the ``ActivationBreakdown`` for a specific memory (no reconsolidation)."""

        await self._ensure_open()
        assert self._engine is not None
        return await self._engine.explain(query, memory_id, agent_id=agent_id)

    async def expand(
        self,
        memory_id: str,
        *,
        agent_id: str | None = None,
        query: str | None = None,
        hops: int = 1,
        k: int = 5,
    ) -> list[RecallResult]:
        """Explicitly expand from one memory via the relation graph.

        Companion to ``recall()``'s auto-expand: auto-expand fires on
        heuristic same-topic detection; this method fires whenever the
        caller wants it. Use for "show related" buttons, agent-framework
        follow-up detectors that have their own signals, or "drill into
        this" planner steps. Returns memories reachable via the relation
        graph, ranked by spreading activation from the seed. All results
        are tagged ``source="expanded"``.
        """

        await self._ensure_open()
        assert self._engine is not None
        results = await self._engine.expand_recall(
            memory_id,
            agent_id=agent_id,
            query=query,
            hops=hops,
            k=k,
        )
        _log.info(
            "expanded",
            extra={
                "workspace": self._config.workspace,
                "agent_id": agent_id,
                "memory_id": memory_id,
                "hops": hops,
                "results": len(results),
            },
        )
        return results

    def for_agent(self, agent_id: str) -> AgentHandle:
        """Return a thin per-agent handle that binds ``agent_id`` on every call."""

        return AgentHandle(self, agent_id)

    @property
    def last_observe_at(self) -> datetime | None:
        """Timestamp of the most recent ``observe()`` call, or ``None``
        if this instance has never observed. Used by the DreamScheduler's
        idle trigger."""

        return self._last_observe_at

    async def flush_session(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Force-close in-flight event buffers and persist them.

        Returns the number of events that were flushed. Scope defaults to
        everything; narrowing is available via ``agent_id`` and/or
        ``session_id``.
        """

        await self._ensure_open()
        async with self._segment_lock:
            closed = self._segmenter.flush(agent_id=agent_id, session_id=session_id)
        for event in closed:
            await self._persist_event(event)
        return len(closed)

    async def close(self) -> None:
        if self._store is not None:
            # Drain any pending event buffers before the store goes away.
            async with self._segment_lock:
                closed = self._segmenter.flush_all()
            for event in closed:
                await self._persist_event(event)
            await self._store.close()
            self._store = None
            self._engine = None

    async def dispose(self) -> DisposalStats:
        """Run P8 Dispose standalone (without the rest of the dream pipeline).

        Scans every memory, enforces hard protections (pin / high
        salience / high emotional_weight / age < 30 days), and removes
        any that satisfy ``activation_dead`` or ``redundant`` criteria.
        Writes a Tombstone per disposal; content never silently
        disappears without an audit entry.
        """

        await self._ensure_open()
        assert self._store is not None
        stats = await dispose_pass(self._store, self._config.formula)
        self._last_dispose_at = datetime.now(UTC)
        _log.info(
            "dispose",
            extra={
                "workspace": self._config.workspace,
                "scanned": stats.scanned,
                "disposed": stats.disposed,
                "protected": stats.protected,
            },
        )
        return stats

    async def tombstones(self, *, agent_id: str | None = None, limit: int = 100) -> list[Tombstone]:
        """Return recent tombstones, newest first, scoped the same way as
        recall (agent + ambient, or ambient only)."""

        await self._ensure_open()
        assert self._store is not None
        return await self._store.list_tombstones(agent_id=agent_id, limit=limit)

    async def rebalance(self) -> RebalanceStats:
        """Recompute ``idx_priority`` and tier for every memory.

        Stage 2 surfaces this as a direct call; Stage 4 will call the same
        function from the Dreaming P7 phase. Runs entirely on metadata —
        content, embeddings, and relations are untouched.
        """

        await self._ensure_open()
        assert self._store is not None
        stats = await _rebalance(
            self._store, self._config.formula, self._config.tier_capacity
        )
        self._last_rebalance_at = datetime.now(UTC)
        _log.info(
            "rebalance",
            extra={
                "workspace": self._config.workspace,
                "scanned": stats.scanned,
                "migrated": stats.migrated,
            },
        )
        return stats

    async def tier_counts(self) -> dict[str, int]:
        """Convenience view for observability — mirrors ``store.tier_counts``
        but returns string keys so callers don't need to import IndexTier."""

        await self._ensure_open()
        assert self._store is not None
        counts = await self._store.tier_counts()
        return {tier.value: count for tier, count in counts.items()}

    async def dream(
        self,
        trigger: str = "session_end",
        *,
        agent_id: str | None = None,
        phases: set[str] | None = None,
    ) -> DreamReport:
        """Run a dreaming cycle for ``trigger``.

        Valid triggers: ``"idle"``, ``"session_end"``, ``"surprise"``,
        ``"cognitive_load"``, ``"nightly"``. Each one maps to a specific
        phase subset (see ``dream.runner.TRIGGER_PHASES``). If no LLM is
        configured, LLM-dependent phases record ``status="skipped"``
        with an explanatory reason and the rest still run.

        ``phases`` (optional) is an ablation mask used by
        ``bench/ablate_dreaming.py``. Pass a set of phase ``value`` strings
        (e.g. ``{"replay", "cluster"}``) to run only those phases for
        this trigger; excluded phases record
        ``status="excluded_by_mask"``. Default ``None`` runs the full
        trigger pipeline.
        """

        await self._ensure_open()
        assert self._store is not None
        # Flush pending event buffers so dreaming sees a consistent snapshot.
        async with self._segment_lock:
            closed = self._segmenter.flush(agent_id=agent_id)
        for event in closed:
            await self._persist_event(event)

        runner = DreamRunner(
            self._store,
            self._config.formula,
            tier_capacity=self._config.tier_capacity,
            llm=self._llm,
            embedder=self._embedder,
            replay_limit=self._config.dreamer.replay_limit,
            replay_min_base_level=self._config.dreamer.replay_min_base_level,
            cluster_min_size=self._config.dreamer.cluster_min_size,
            cost_limits=self._cost_limits,
            cost_ledger=self._cost_ledger,
        )
        report = await runner.run(TriggerType(trigger), agent_id=agent_id, phases=phases)

        # Dream Diary (§2.5) — append a Markdown audit entry for this run.
        diary_path = dream_diary_path(self._config.storage.resolve_root(), self._config.workspace)
        append_entry(diary_path, report)
        report.diary_path = diary_path
        self._last_dream_at = report.finished_at
        self._last_dream_trigger = report.trigger.value
        # Bounded history surfaced via status() so operators see recent
        # pipeline health without cracking open the dream diary.
        self._dream_history.append(_summarize_dream(report))
        if len(self._dream_history) > self._dream_history_cap:
            self._dream_history = self._dream_history[-self._dream_history_cap :]
        _log.info(
            "dream",
            extra={
                "workspace": self._config.workspace,
                "trigger": report.trigger.value,
                "agent_id": agent_id,
                "phases": [o.phase.value for o in report.outcomes],
                "duration_seconds": report.duration_seconds(),
            },
        )
        return report

    async def export_markdown(
        self, *, agent_id: str | None = None, min_idx_priority: float = 0.5
    ) -> str:
        """Render a deterministic Markdown view of high-priority memories.

        Scope follows the recall rule: the root ``Mnemoss`` returns a
        workspace-ambient view; a ``for_agent(id)`` handle returns that
        agent's private + ambient memories. No LLM — only memories that
        already cross the threshold are rendered.
        """

        await self._ensure_open()
        assert self._store is not None
        memories = await self._store.list_memories_for_export(agent_id, min_idx_priority=0.0)
        pinned = await self._store.pinned_ids_in_scope(agent_id)
        return render_memory_md(
            memories,
            pinned_ids=pinned,
            agent_id=agent_id,
            min_idx_priority=min_idx_priority,
        )

    async def status(self) -> dict[str, Any]:
        """Return a snapshot of the workspace's operational state.

        Provides everything an observability dashboard needs without
        a full scan: counts, timestamps, embedder info, schema version,
        recent dream activity, and the LLM cost ledger.
        """

        await self._ensure_open()
        assert self._store is not None
        tier_counts_raw = await self._store.tier_counts()
        tier_counts = {tier.value: count for tier, count in tier_counts_raw.items()}
        memory_count = sum(tier_counts.values())
        tombstone_count = await self._store.count_tombstones()

        # Cost + dream-health summary. Both are cheap reads:
        # - Cost ledger is three workspace_meta rows.
        # - Dream history is in-memory, bounded at 10 entries.
        cost_block: dict[str, Any] = {
            "today_calls": 0,
            "month_calls": 0,
            "total_calls": 0,
            "limits": _cost_limits_to_dict(self._cost_limits),
        }
        if self._cost_ledger is not None:
            snap = self._cost_ledger.snapshot()
            cost_block["today_calls"] = snap.today_calls
            cost_block["month_calls"] = snap.month_calls
            cost_block["total_calls"] = snap.total_calls

        recent_dreams = list(self._dream_history)
        dream_block: dict[str, Any] = {
            "recent": recent_dreams,
            "recent_count": len(recent_dreams),
            "recent_degraded_count": sum(1 for d in recent_dreams if d["degraded"]),
        }

        return {
            "workspace": self._config.workspace,
            "schema_version": SCHEMA_VERSION,
            "embedder": {
                "id": self._embedder.embedder_id,
                "dim": self._embedder.dim,
            },
            "memory_count": memory_count,
            "tier_counts": tier_counts,
            "tombstone_count": tombstone_count,
            "last_observe_at": (
                self._last_observe_at.isoformat() if self._last_observe_at is not None else None
            ),
            "last_dream_at": (
                self._last_dream_at.isoformat() if self._last_dream_at is not None else None
            ),
            "last_dream_trigger": self._last_dream_trigger,
            "last_rebalance_at": (
                self._last_rebalance_at.isoformat() if self._last_rebalance_at is not None else None
            ),
            "last_dispose_at": (
                self._last_dispose_at.isoformat() if self._last_dispose_at is not None else None
            ),
            "llm_cost": cost_block,
            "dreams": dream_block,
        }

    # ─── internal ─────────────────────────────────────────────────────

    async def _persist_event(self, event: ClosedEvent) -> None:
        """Write one closed event as Memory row(s) + embeddings + edges.

        Normal path: one Memory row per event. Long-content path:
        when ``encoder.max_memory_chars`` is set and the event's
        encoded content exceeds it, the content is split at the
        nearest natural boundary and each chunk becomes its own
        Memory row. Raw Log stays 1-to-1 with ``observe()``; chunks
        share ``source_message_ids`` and carry a
        ``source_context.split_part = {"index": i, "total": n,
        "group_id": <first_chunk_id>}`` marker so callers can
        de-duplicate in recall if they want.
        """

        assert self._store is not None
        memory = encode_event(
            event.messages,
            memory_id=event.memory_id,
            now=event.closed_at,
            formula=self._config.formula,
        )

        cap = self._config.encoder.max_memory_chars
        if cap is None or len(memory.content) <= cap:
            # Happy path — one event, one Memory.
            embedding = (await asyncio.to_thread(self._embedder.embed, [memory.content]))[0]
            await self._store.write_memory(memory, embedding)
            await self._mark_semantic_supersession(memory, embedding)
            await write_cooccurrence_edges(
                self._store,
                memory.id,
                memory.session_id or "default",
                self._config.encoder,
            )
            self._working.append(memory.agent_id, memory.id)
            return

        # Long-content path: split + emit N memories.
        chunks = split_content(memory.content, cap)
        if len(chunks) == 1:
            # ``split_content`` returned the content unchanged (fits
            # the cap after all). Fall back to the single-row path.
            embedding = (await asyncio.to_thread(self._embedder.embed, [memory.content]))[0]
            await self._store.write_memory(memory, embedding)
            await write_cooccurrence_edges(
                self._store,
                memory.id,
                memory.session_id or "default",
                self._config.encoder,
            )
            self._working.append(memory.agent_id, memory.id)
            return

        # Batch-embed all chunks so the embedder can amortize setup
        # over the whole list (materially faster for LocalEmbedder).
        embeddings = await asyncio.to_thread(self._embedder.embed, chunks)
        total = len(chunks)
        group_id = memory.id  # first chunk keeps the original event id
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_memory = self._chunk_memory(memory, chunk, i, total, group_id)
            await self._store.write_memory(chunk_memory, emb)
            await write_cooccurrence_edges(
                self._store,
                chunk_memory.id,
                chunk_memory.session_id or "default",
                self._config.encoder,
            )
            self._working.append(chunk_memory.agent_id, chunk_memory.id)

    async def _mark_semantic_supersession(
        self, new_memory: Any, embedding: Any
    ) -> None:
        """Mark any existing near-duplicate memory as superseded by ``new_memory``.

        No-op when ``encoder.supersede_on_observe`` is False. Otherwise
        runs one ANN query (same scope as the new memory's agent_id),
        picks matches whose cosine ≥ ``encoder.supersede_cosine_threshold``,
        and marks each older memory's ``superseded_by`` = ``new_memory.id``.

        The feature is intentionally conservative:
        - Matches are scoped to the same agent (cross-agent supersession
          would leak through agent isolation).
        - The new memory itself is excluded (vec_search will return it
          as self-match with cosine ≈ 1.0).
        - Already-superseded rows are skipped (they live under
          ``superseded_by IS NOT NULL`` in the filter helper).
        """

        if not self._config.encoder.supersede_on_observe:
            return
        assert self._store is not None

        threshold = self._config.encoder.supersede_cosine_threshold
        # Small candidate pool — we don't need the full corpus, just
        # the handful of memories semantically nearest to the new one.
        candidates = await self._store.vec_search(
            embedding, 10, new_memory.agent_id, tier_filter=None
        )
        now = datetime.now(UTC)
        for mid, cos in candidates:
            if mid == new_memory.id:
                continue
            if cos < threshold:
                # vec_search returns in descending cosine order — once
                # we cross the threshold, remaining candidates are all
                # below it and we can stop.
                break
            await self._store.mark_superseded(mid, new_memory.id, now)

    def _chunk_memory(
        self,
        template: Any,
        chunk_content: str,
        index: int,
        total: int,
        group_id: str,
    ) -> Any:
        """Derive a chunk Memory from a template.

        The first chunk reuses the template's id so any Raw Log
        ``source_message_ids`` cross-references that the segmenter
        emitted stay valid. Subsequent chunks get fresh ULIDs but
        share the group id in ``source_context`` so callers can
        recognize the relationship on recall.
        """

        from dataclasses import replace

        chunk_id = template.id if index == 0 else str(ulid.new())
        return replace(
            template,
            id=chunk_id,
            content=chunk_content,
            source_context={
                **template.source_context,
                "split_part": {
                    "index": index,
                    "total": total,
                    "group_id": group_id,
                },
            },
        )

    async def _ensure_open(self) -> None:
        if self._store is not None:
            return
        async with self._open_lock:
            if self._store is not None:
                return
            # Warm the embedder BEFORE constructing the SQLite backend.
            # The backend pins ``embedding_dim`` at open time; for
            # embedders whose dim is only known after the first
            # ``embed()`` call (e.g. custom SentenceTransformer models,
            # OpenAI with a custom dim, Gemini MRL), calling warmup
            # first resolves ``embedder.dim`` to the real value.
            #
            # ``"warmup"`` is deliberately boring — the returned vector
            # is discarded so its semantic content doesn't matter. Must
            # be a non-empty, non-whitespace token: OpenAI's embeddings
            # API rejects empty strings with a 400, and future
            # tokenizers may normalize pure whitespace to empty.
            await asyncio.to_thread(self._embedder.embed, ["warmup"])

            db_path = workspace_db_path(self._config.storage.root, self._config.workspace)
            raw_path = raw_log_db_path(self._config.storage.root, self._config.workspace)
            store = SQLiteBackend(
                db_path=db_path,
                raw_log_path=raw_path,
                workspace_id=self._config.workspace,
                embedding_dim=self._embedder.dim,
                embedder_id=self._embedder.embedder_id,
                use_ann_index=self._config.storage.use_ann_index,
            )
            await store.open()
            self._engine = RecallEngine(
                store=store,
                embedder=self._embedder,
                working=self._working,
                params=self._config.formula,
                rng=self._rng,
            )
            self._store = store
            # Bind the cost ledger to the open connection — persistence
            # lives in workspace_meta so call counts survive restarts.
            self._cost_ledger = CostLedger(store._require_conn())


class AgentHandle:
    """Sugar over a workspace-level ``Mnemoss`` that binds ``agent_id``."""

    def __init__(self, mem: Mnemoss, agent_id: str) -> None:
        self._mem = mem
        self._agent_id = agent_id

    async def observe(
        self,
        role: str,
        content: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        return await self._mem.observe(
            role=role,
            content=content,
            agent_id=self._agent_id,
            session_id=session_id,
            turn_id=turn_id,
            parent_id=parent_id,
            metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        include_deep: bool = False,
        auto_expand: bool = True,
        reconsolidate: bool = True,
    ) -> list[RecallResult]:
        return await self._mem.recall(
            query,
            k=k,
            agent_id=self._agent_id,
            include_deep=include_deep,
            auto_expand=auto_expand,
            reconsolidate=reconsolidate,
        )

    async def pin(self, memory_id: str) -> None:
        await self._mem.pin(memory_id, agent_id=self._agent_id)

    async def explain_recall(self, query: str, memory_id: str) -> ActivationBreakdown | None:
        return await self._mem.explain_recall(query, memory_id, agent_id=self._agent_id)

    async def expand(
        self,
        memory_id: str,
        *,
        query: str | None = None,
        hops: int = 1,
        k: int = 5,
    ) -> list[RecallResult]:
        return await self._mem.expand(
            memory_id,
            agent_id=self._agent_id,
            query=query,
            hops=hops,
            k=k,
        )

    async def export_markdown(self) -> str:
        return await self._mem.export_markdown(agent_id=self._agent_id)


# ─── status() helpers ────────────────────────────────────────────


def _summarize_dream(report: DreamReport) -> dict[str, Any]:
    """Lightweight summary of a ``DreamReport`` for ``status()``.

    Keeps only the fields an operator dashboard actually renders —
    the full report lives in the dream diary for deep inspection.
    """

    return {
        "trigger": report.trigger.value,
        "started_at": report.started_at.isoformat(),
        "finished_at": report.finished_at.isoformat(),
        "duration_seconds": report.duration_seconds(),
        "degraded": report.degraded_mode,
        "phase_statuses": {o.phase.value: o.status for o in report.outcomes},
        "errors": [{"phase": o.phase.value, "error": o.error} for o in report.errors()],
    }


def _cost_limits_to_dict(limits: CostLimits) -> dict[str, int | None]:
    """Serialize ``CostLimits`` for ``status()``.

    ``None`` fields are kept as-is to signal "no cap for this
    dimension" — clearer than omitting the key entirely.
    """

    return {
        "max_llm_calls_per_run": limits.max_llm_calls_per_run,
        "max_llm_calls_per_day": limits.max_llm_calls_per_day,
        "max_llm_calls_per_month": limits.max_llm_calls_per_month,
    }
