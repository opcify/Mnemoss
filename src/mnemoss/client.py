"""Public Mnemoss client.

Three core methods (``observe``, ``recall``, ``pin``) plus a thin
per-agent handle from ``for_agent(id)``. Everything is async.

The constructor does no I/O — it just resolves paths and picks the
embedder. First ``observe()`` auto-creates the workspace DB (Stage 1
settled decision).
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import Any

import ulid

from mnemoss.core.config import (
    EncoderParams,
    FormulaParams,
    MnemossConfig,
    SegmentationParams,
    StorageParams,
)
from mnemoss.core.types import RawMessage
from mnemoss.dream.runner import DreamRunner
from mnemoss.dream.types import DreamReport, TriggerType
from mnemoss.encoder import Embedder, make_embedder
from mnemoss.encoder.event_encoder import encode_event, should_encode
from mnemoss.encoder.event_segmentation import ClosedEvent, EventSegmenter
from mnemoss.index import RebalanceStats
from mnemoss.index import rebalance as _rebalance
from mnemoss.llm.client import LLMClient
from mnemoss.recall import RecallEngine, RecallResult
from mnemoss.relations import write_cooccurrence_edges
from mnemoss.store.paths import workspace_db_path
from mnemoss.store.sqlite_backend import SQLiteBackend
from mnemoss.working import WorkingMemory

UTC = timezone.utc


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

        if not should_encode(msg, self._config.encoder):
            return None

        async with self._segment_lock:
            step = self._segmenter.on_observe(
                msg, now, self._config.segmentation, auto_close=auto_close
            )

        for event in step.closed_events:
            await self._persist_event(event)

        return step.pending_memory_id

    async def recall(
        self,
        query: str,
        *,
        k: int = 5,
        agent_id: str | None = None,
        include_deep: bool = False,
    ) -> list[RecallResult]:
        await self._ensure_open()
        assert self._engine is not None
        return await self._engine.recall(
            query, agent_id=agent_id, k=k, include_deep=include_deep
        )

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

    def for_agent(self, agent_id: str) -> AgentHandle:
        """Return a thin per-agent handle that binds ``agent_id`` on every call."""

        return AgentHandle(self, agent_id)

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

    async def rebalance(self) -> RebalanceStats:
        """Recompute ``idx_priority`` and tier for every memory.

        Stage 2 surfaces this as a direct call; Stage 4 will call the same
        function from the Dreaming P7 phase. Runs entirely on metadata —
        content, embeddings, and relations are untouched.
        """

        await self._ensure_open()
        assert self._store is not None
        return await _rebalance(self._store, self._config.formula)

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

        Valid triggers in Stage 4: ``"idle"``, ``"session_end"``,
        ``"task_completion"``. The remaining three (surprise,
        cognitive_load, nightly) plus the deep phases P6–P8 arrive in
        Stage 5. If no LLM is configured, LLM-dependent phases record
        ``status="skipped"`` with an explanatory reason and the rest
        still run.
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
        return await runner.run(TriggerType(trigger), agent_id=agent_id)

    # ─── stubs for deferred stages ────────────────────────────────────

    async def status(self) -> dict[str, Any]:
        raise NotImplementedError("status() lands in Stage 4")

    async def export_markdown(self, *, agent_id: str | None = None) -> str:
        raise NotImplementedError("memory.md generation lands in Stage 4")

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
        embedding = (
            await asyncio.to_thread(self._embedder.embed, [memory.content])
        )[0]
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
            db_path = workspace_db_path(
                self._config.storage.root, self._config.workspace
            )
            store = SQLiteBackend(
                db_path=db_path,
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
        self, query: str, *, k: int = 5, include_deep: bool = False
    ) -> list[RecallResult]:
        return await self._mem.recall(
            query, k=k, agent_id=self._agent_id, include_deep=include_deep
        )

    async def pin(self, memory_id: str) -> None:
        await self._mem.pin(memory_id, agent_id=self._agent_id)

    async def explain_recall(self, query: str, memory_id: str):
        return await self._mem.explain_recall(
            query, memory_id, agent_id=self._agent_id
        )

    async def export_markdown(self) -> str:
        return await self._mem.export_markdown(agent_id=self._agent_id)
