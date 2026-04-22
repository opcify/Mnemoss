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
    EncoderParams,
    FormulaParams,
    MnemossConfig,
    SegmentationParams,
    StorageParams,
)
from mnemoss.core.types import RawMessage, Tombstone
from mnemoss.dream.diary import append_entry, dream_diary_path
from mnemoss.dream.dispose import DisposalStats, dispose_pass
from mnemoss.dream.runner import DreamRunner
from mnemoss.dream.types import DreamReport, TriggerType
from mnemoss.encoder import Embedder, make_embedder
from mnemoss.encoder.event_encoder import encode_event, should_encode
from mnemoss.encoder.event_segmentation import ClosedEvent, EventSegmenter
from mnemoss.export import render_memory_md
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
        llm: LLMClient | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = MnemossConfig(
            workspace=workspace,
            formula=formula or FormulaParams(),
            encoder=encoder or EncoderParams(),
            storage=storage or StorageParams(),
            segmentation=segmentation or SegmentationParams(),
        )
        self._embedder = make_embedder(embedding_model)
        self._llm = llm
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
    ) -> list[RecallResult]:
        await self._ensure_open()
        assert self._engine is not None
        results = await self._engine.recall(
            query,
            agent_id=agent_id,
            k=k,
            include_deep=include_deep,
            auto_expand=auto_expand,
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
    ):
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
        stats = await _rebalance(self._store, self._config.formula)
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
    ) -> DreamReport:
        """Run a dreaming cycle for ``trigger``.

        Valid triggers: ``"idle"``, ``"session_end"``, ``"surprise"``,
        ``"cognitive_load"``, ``"nightly"``. Each one maps to a specific
        phase subset (see ``dream.runner.TRIGGER_PHASES``). If no LLM is
        configured, LLM-dependent phases record ``status="skipped"``
        with an explanatory reason and the rest still run.
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
            llm=self._llm,
            embedder=self._embedder,
        )
        report = await runner.run(TriggerType(trigger), agent_id=agent_id)

        # Dream Diary (§2.5) — append a Markdown audit entry for this run.
        diary_path = dream_diary_path(self._config.storage.resolve_root(), self._config.workspace)
        append_entry(diary_path, report)
        report.diary_path = diary_path
        self._last_dream_at = report.finished_at
        self._last_dream_trigger = report.trigger.value
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
        a full scan: counts, timestamps, embedder info, schema version.
        """

        await self._ensure_open()
        assert self._store is not None
        tier_counts_raw = await self._store.tier_counts()
        tier_counts = {tier.value: count for tier, count in tier_counts_raw.items()}
        memory_count = sum(tier_counts.values())
        tombstone_count = await self._store.count_tombstones()
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
        }

    # ─── internal ─────────────────────────────────────────────────────

    async def _persist_event(self, event: ClosedEvent) -> None:
        """Write one closed event as a Memory + embedding + edges."""

        assert self._store is not None
        memory = encode_event(
            event.messages,
            memory_id=event.memory_id,
            now=event.closed_at,
            formula=self._config.formula,
        )
        embedding = (await asyncio.to_thread(self._embedder.embed, [memory.content]))[0]
        await self._store.write_memory(memory, embedding)
        await write_cooccurrence_edges(
            self._store, memory.id, memory.session_id or "default", self._config.encoder
        )
        self._working.append(memory.agent_id, memory.id)

    async def _ensure_open(self) -> None:
        if self._store is not None:
            return
        async with self._open_lock:
            if self._store is not None:
                return
            db_path = workspace_db_path(self._config.storage.root, self._config.workspace)
            raw_path = raw_log_db_path(self._config.storage.root, self._config.workspace)
            store = SQLiteBackend(
                db_path=db_path,
                raw_log_path=raw_path,
                workspace_id=self._config.workspace,
                embedding_dim=self._embedder.dim,
                embedder_id=self._embedder.embedder_id,
            )
            await store.open()
            # Warm the embedder before we start writing memories.
            # Otherwise the first observe() pays the full model-load cost
            # (seconds) and ends up "older" than subsequent memories at
            # recall time — which skews B_i. A single empty encode is
            # enough to trigger lazy model loads.
            await asyncio.to_thread(self._embedder.embed, [""])
            self._engine = RecallEngine(
                store=store,
                embedder=self._embedder,
                working=self._working,
                params=self._config.formula,
                rng=self._rng,
            )
            self._store = store


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
    ) -> list[RecallResult]:
        return await self._mem.recall(
            query,
            k=k,
            agent_id=self._agent_id,
            include_deep=include_deep,
            auto_expand=auto_expand,
        )

    async def pin(self, memory_id: str) -> None:
        await self._mem.pin(memory_id, agent_id=self._agent_id)

    async def explain_recall(self, query: str, memory_id: str):
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
