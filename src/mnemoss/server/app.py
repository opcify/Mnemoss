"""FastAPI application factory.

``create_app()`` returns a fully-wired ``FastAPI`` instance. The CLI uses
this via uvicorn; tests pass a custom ``ServerConfig`` (typically with a
``FakeEmbedder``) so suites run without downloading models.

Stage-6 S1 surface:

- ``GET  /health``                              — liveness
- ``POST /workspaces/{id}/observe``             — record a message
- ``POST /workspaces/{id}/recall``              — cascade retrieval
- ``POST /workspaces/{id}/pin``                 — pin a memory

Per-agent scoping is a query parameter (``?agent_id=alice``). Omitting
it means "workspace-ambient", matching ``Mnemoss.for_agent`` semantics
on the library side.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from mnemoss import __version__
from mnemoss.server.auth import verify_api_key
from mnemoss.server.config import ServerConfig
from mnemoss.server.pool import WorkspaceNotAllowedError, WorkspacePool
from mnemoss.server.schemas import (
    ObserveRequest,
    ObserveResponse,
    OkResponse,
    PinRequest,
    RecallRequest,
    RecallResponse,
    recall_result_to_dto,
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
        memory_id = await mem.observe(
            role=body.role,
            content=body.content,
            agent_id=agent_id,
            session_id=body.session_id,
            turn_id=body.turn_id,
            parent_id=body.parent_id,
            metadata=body.metadata,
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
        results = await mem.recall(
            body.query,
            k=body.k,
            agent_id=agent_id,
            include_deep=body.include_deep,
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
