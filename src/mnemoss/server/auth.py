"""Bearer-token auth dependency.

If ``ServerConfig.api_key`` is ``None`` (default for local dev), every
request passes. Otherwise, clients must send
``Authorization: Bearer <key>`` with a matching token.
"""

from __future__ import annotations

from fastapi import HTTPException, Request


async def verify_api_key(request: Request) -> None:
    config = request.app.state.config
    if config.api_key is None:
        return
    header = request.headers.get("authorization")
    if header is None or not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )
    token = header.split(" ", 1)[1].strip()
    if token != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
