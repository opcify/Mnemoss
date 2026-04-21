"""FastAPI application factory.

``create_app()`` returns a fully-wired ``FastAPI`` instance. The CLI uses
this via uvicorn; tests pass a custom ``ServerConfig`` (typically with a
``FakeEmbedder``) so suites run without downloading models.

Stage-6 endpoints:

- ``GET  /health``                              — liveness
- ``POST /workspaces/{id}/observe``             — record a message
- ``POST /workspaces/{id}/recall``              — cascade retrieval
- ``POST /workspaces/{id}/pin``                 — pin a memory
- ``POST /workspaces/{id}/explain``             — ActivationBreakdown for one memory
- ``POST /workspaces/{id}/dream``               — run a dream cycle
- ``POST /workspaces/{id}/rebalance``           — P7 rebalance (standalone)
- ``POST /workspaces/{id}/dispose``             — P8 dispose (standalone)
- ``GET  /workspaces/{id}/tombstones``          — list disposal audit rows
- ``GET  /workspaces/{id}/tiers``               — tier counts
- ``POST /workspaces/{id}/export``              — render memory.md
- ``POST /workspaces/{id}/flush``               — force-close event buffers
- ``GET  /workspaces/{id}/status``              — operational snapshot
- ``GET  /metrics``                             — Prometheus (when [observability] installed)

Per-agent scoping is a query parameter (``?agent_id=alice``). Omitting
it means "workspace-ambient", matching ``Mnemoss.for_agent`` semantics
on the library side.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from mnemoss import __version__
from mnemoss.server import metrics
from mnemoss.server.auth import verify_api_key
from mnemoss.server.config import ServerConfig
from mnemoss.server.pool import WorkspaceNotAllowedError, WorkspacePool
from mnemoss.server.schemas import (
    DisposeResponse,
    DreamRequest,
    DreamResponse,
    ExplainRequest,
    ExplainResponse,
    ExportMarkdownRequest,
    ExportMarkdownResponse,
    FlushSessionRequest,
    FlushSessionResponse,
    ObserveRequest,
    ObserveResponse,
    OkResponse,
    PinRequest,
    RebalanceResponse,
    RecallRequest,
    RecallResponse,
    StatusResponse,
    TierCountsResponse,
    TombstonesResponse,
    breakdown_to_dto,
    disposal_stats_to_dto,
    dream_report_to_dto,
    rebalance_stats_to_dto,
    recall_result_to_dto,
    tombstone_to_dto,
)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI app. ``config`` defaults to ``ServerConfig.from_env()``."""

    effective_config = config if config is not None else ServerConfig.from_env()
    pool = WorkspacePool(effective_config)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await pool.close_all()

    app = FastAPI(
        title="Mnemoss",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.config = effective_config
    app.state.pool = pool

    # ─── liveness ────────────────────────────────────────────────

    @app.get("/health", response_model=OkResponse)
    async def health() -> OkResponse:
        return OkResponse(ok=True)

    # ─── observe ─────────────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/observe",
        response_model=ObserveResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def observe(
        workspace_id: str,
        body: ObserveRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> ObserveResponse:
        mem = await _resolve(request, workspace_id)
        start = time.perf_counter()
        memory_id = await mem.observe(
            role=body.role,
            content=body.content,
            agent_id=agent_id,
            session_id=body.session_id,
            turn_id=body.turn_id,
            parent_id=body.parent_id,
            metadata=body.metadata,
        )
        metrics.record_observe(
            workspace_id,
            encoded=memory_id is not None,
            duration=time.perf_counter() - start,
        )
        return ObserveResponse(memory_id=memory_id)

    # ─── recall ──────────────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/recall",
        response_model=RecallResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def recall(
        workspace_id: str,
        body: RecallRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> RecallResponse:
        mem = await _resolve(request, workspace_id)
        start = time.perf_counter()
        results = await mem.recall(
            body.query,
            k=body.k,
            agent_id=agent_id,
            include_deep=body.include_deep,
        )
        metrics.record_recall(
            workspace_id, duration=time.perf_counter() - start
        )
        return RecallResponse(
            results=[recall_result_to_dto(r) for r in results],
        )

    # ─── pin ─────────────────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/pin",
        response_model=OkResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def pin(
        workspace_id: str,
        body: PinRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> OkResponse:
        mem = await _resolve(request, workspace_id)
        await mem.pin(body.memory_id, agent_id=agent_id)
        return OkResponse()

    # ─── explain_recall ─────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/explain",
        response_model=ExplainResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def explain(
        workspace_id: str,
        body: ExplainRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> ExplainResponse:
        mem = await _resolve(request, workspace_id)
        breakdown = await mem.explain_recall(
            body.query, body.memory_id, agent_id=agent_id
        )
        return ExplainResponse(breakdown=breakdown_to_dto(breakdown))

    # ─── dream ──────────────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/dream",
        response_model=DreamResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def dream(
        workspace_id: str,
        body: DreamRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> DreamResponse:
        mem = await _resolve(request, workspace_id)
        start = time.perf_counter()
        try:
            report = await mem.dream(trigger=body.trigger, agent_id=agent_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metrics.record_dream(
            workspace_id,
            trigger=report.trigger.value,
            duration=time.perf_counter() - start,
        )
        return dream_report_to_dto(report)

    # ─── rebalance ──────────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/rebalance",
        response_model=RebalanceResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def rebalance(
        workspace_id: str,
        request: Request,
    ) -> RebalanceResponse:
        mem = await _resolve(request, workspace_id)
        stats = await mem.rebalance()
        return rebalance_stats_to_dto(stats)

    # ─── dispose ────────────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/dispose",
        response_model=DisposeResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def dispose(
        workspace_id: str,
        request: Request,
    ) -> DisposeResponse:
        mem = await _resolve(request, workspace_id)
        stats = await mem.dispose()
        metrics.record_disposal(
            workspace_id,
            activation_dead=stats.activation_dead,
            redundant=stats.redundant,
        )
        return disposal_stats_to_dto(stats)

    # ─── tombstones ─────────────────────────────────────────────

    @app.get(
        "/workspaces/{workspace_id}/tombstones",
        response_model=TombstonesResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def tombstones(
        workspace_id: str,
        request: Request,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> TombstonesResponse:
        mem = await _resolve(request, workspace_id)
        rows = await mem.tombstones(agent_id=agent_id, limit=limit)
        return TombstonesResponse(
            tombstones=[tombstone_to_dto(t) for t in rows],
        )

    # ─── tier_counts ────────────────────────────────────────────

    @app.get(
        "/workspaces/{workspace_id}/tiers",
        response_model=TierCountsResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def tiers(
        workspace_id: str,
        request: Request,
    ) -> TierCountsResponse:
        mem = await _resolve(request, workspace_id)
        return TierCountsResponse(tiers=await mem.tier_counts())

    # ─── export_markdown ────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/export",
        response_model=ExportMarkdownResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def export_markdown(
        workspace_id: str,
        body: ExportMarkdownRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> ExportMarkdownResponse:
        mem = await _resolve(request, workspace_id)
        md = await mem.export_markdown(
            agent_id=agent_id, min_idx_priority=body.min_idx_priority
        )
        return ExportMarkdownResponse(markdown=md)

    # ─── flush_session ──────────────────────────────────────────

    @app.post(
        "/workspaces/{workspace_id}/flush",
        response_model=FlushSessionResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def flush_session(
        workspace_id: str,
        body: FlushSessionRequest,
        request: Request,
        agent_id: str | None = None,
    ) -> FlushSessionResponse:
        mem = await _resolve(request, workspace_id)
        n = await mem.flush_session(agent_id=agent_id, session_id=body.session_id)
        return FlushSessionResponse(flushed=n)

    # ─── status ─────────────────────────────────────────────────

    @app.get(
        "/workspaces/{workspace_id}/status",
        response_model=StatusResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def status(
        workspace_id: str,
        request: Request,
    ) -> StatusResponse:
        mem = await _resolve(request, workspace_id)
        snapshot = await mem.status()
        return StatusResponse.model_validate(snapshot)

    # ─── /metrics (Prometheus, optional) ────────────────────────

    if metrics.HAS_PROMETHEUS:

        @app.get("/metrics", include_in_schema=False)
        async def metrics_endpoint(request: Request) -> Response:
            # Refresh gauges at scrape time so values are current.
            await metrics.refresh_memory_gauges(request.app.state.pool)
            body, content_type = metrics.latest_metrics()
            return Response(content=body, media_type=content_type)

    return app


# ─── helpers ─────────────────────────────────────────────────────


async def _resolve(request: Request, workspace_id: str):
    """Pull the workspace instance out of the pool, mapping allow-list
    violations to a 403."""

    pool: WorkspacePool = request.app.state.pool
    try:
        return await pool.get(workspace_id)
    except WorkspaceNotAllowedError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"Workspace not allowed: {exc}",
        ) from exc
