"""Mem0 adapter for the LongMemEval-S benchmark.

`Mem0 <https://github.com/mem0ai/mem0>`_ ships its memory layer behind a
``Memory`` client with three relevant operations:

- ``add(messages, user_id, metadata)`` — ingest a single message or a
  list of role/content turns. Mem0 internally extracts "memories" via
  an LLM pass; the call returns the surviving memory ids.
- ``search(query, user_id, limit)`` — hybrid retrieval over the
  extracted memories. Returns each hit's ``id``, ``memory`` text, and
  a ``score``.
- ``delete_all(user_id)`` — wipe a user's namespace.

The adapter sits behind the same boundary the other LongMemEval
backends use (``bench/longmemeval.py``): one ``LongMemEvalBackend``
per question, ingest sessions in chronological order, recall top-K
for the question, close.

Embedder parity with Mnemoss
----------------------------

Mem0's defaults pick OpenAI ``text-embedding-3-small`` for the vector
side and ``gpt-4o-mini`` for its extraction LLM. We don't override
those — Mnemoss's published-chart config also pins
``text-embedding-3-small`` (Issue 1.1A in the launch eng review), so
the *vector* side is matched. The extraction LLM is intrinsic to
Mem0's architecture; we measure Mem0 as it ships, not a stripped-
down configuration that wouldn't be a fair representation.

The OpenAI fallbacks need ``OPENAI_API_KEY`` in env. Pass an explicit
``config`` dict to override (e.g. for an Anthropic LLM or a local
embedder via Ollama).

Optional dependency
-------------------

``mem0ai`` is not in Mnemoss's normal install surface — it pulls in
its own LLM/vector-DB stack. Install it just for benchmarking::

    pip install mem0ai

The import is lazy so missing-extra failures fire only when a caller
actually constructs ``Mem0Backend``.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

from bench.backends.base import RecallHit


class Mem0Backend:
    """LongMemEval-S backend wrapping ``mem0.Memory``.

    Parameters
    ----------
    user_id:
        Mem0 namespaces all memories by ``user_id``. We default to a
        per-instance random id so two backends in the same Python
        process can't accidentally see each other's memories — Mem0
        with the default in-memory vector store still shares state
        across instances of the same user_id.
    config:
        Optional Mem0 ``Memory.from_config`` dict. ``None`` uses the
        library defaults (OpenAI on both LLM and embedder side).
    score_floor:
        Drop hits with ``score`` below this threshold. Mem0 sometimes
        returns long tails of weak matches; benches care about the
        top-K it would actually surface to a downstream agent.
    """

    backend_id = "mem0"

    def __init__(
        self,
        *,
        user_id: str | None = None,
        config: dict[str, Any] | None = None,
        score_floor: float = 0.0,
        cleanup_dir: Path | None = None,
    ) -> None:
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover — exercised at install time
            raise ImportError(
                "Mem0Backend requires the optional `mem0ai` package. "
                "Install with `pip install mem0ai`."
            ) from exc

        # Mem0's history layer is a SQLite connection opened with the
        # default ``check_same_thread=True``. The harness runs the
        # blocking mem0 calls via ``loop.run_in_executor``; with the
        # default executor the thread changes between calls and SQLite
        # raises ``Programming Error: ... created in thread X used in
        # thread Y``. A single-thread executor pins every mem0 call
        # to the same thread so the SQLite invariant holds. We
        # construct the ``Memory`` client INSIDE the executor for the
        # same reason: ``Memory()`` opens the SQLite history during
        # __init__, so it must happen on the same thread that will
        # later use it.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0")
        self._user_id = user_id or f"mnemoss-bench-{uuid4().hex}"
        self._client: Any = self._executor.submit(
            (lambda: Memory.from_config(config)) if config is not None else Memory
        ).result()
        self._score_floor = score_floor
        self._closed = False
        self._cleanup_dir = cleanup_dir

    async def _run(self, fn: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn)

    async def ingest_session(
        self,
        *,
        session_id: str,
        ts: float,
        turns: list[dict[str, str]],
    ) -> None:
        """Ingest one session as a single Mem0 ``add`` call.

        Mem0's docs recommend batching turns into one ``add`` so the
        extraction LLM sees the full conversational context — that's
        how it produces decent memories. Per-turn calls produce
        choppier "the user said X" memories that hurt downstream
        recall on multi-session questions.

        ``ts`` is recorded in Mem0's ``metadata`` so any future
        recency-aware retrieval mode in Mem0 can use it. Mem0's stable
        retrieval today is hybrid semantic + keyword and ignores ``ts``,
        but recording it costs us nothing and futureproofs the bench.
        """

        if not turns:
            return

        def _add() -> None:
            self._client.add(
                turns,
                user_id=self._user_id,
                metadata={"session_id": session_id, "ts": ts},
            )

        await self._run(_add)

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Hybrid search over extracted memories. Empty list if nothing matches."""

        def _search() -> list[dict[str, Any]]:
            raw = self._client.search(
                query, top_k=k, filters={"user_id": self._user_id}
            )
            # Mem0 v1 returned a bare list; v2 wraps it as
            # ``{"results": [...]}``. Handle both.
            if isinstance(raw, dict) and "results" in raw:
                return list(raw["results"])
            return list(raw or [])

        results = await self._run(_search)

        hits: list[RecallHit] = []
        for i, r in enumerate(results):
            score = r.get("score")
            if score is not None and float(score) < self._score_floor:
                continue
            hits.append(
                RecallHit(
                    memory_id=str(r.get("id", f"mem0-{i}")),
                    rank=i + 1,
                    score=float(score) if score is not None else None,
                )
            )
        return hits

    async def recall_text(self, query: str, k: int = 10) -> list[str]:
        """Convenience: return just the recalled memory text strings.

        The LongMemEval harness composes the QA generator prompt from
        memory *text*, not ids — Mem0's ids are opaque internal handles.
        Wrapping this on the backend keeps the harness backend-agnostic.
        """

        def _search() -> list[str]:
            raw = self._client.search(
                query, top_k=k, filters={"user_id": self._user_id}
            )
            if isinstance(raw, dict) and "results" in raw:
                results = list(raw["results"])
            else:
                results = list(raw or [])
            out: list[str] = []
            for r in results:
                text = r.get("memory") or r.get("text") or ""
                if text:
                    out.append(text)
            return out

        return await self._run(_search)

    async def close(self) -> None:
        """Wipe this user's namespace and drop the client. Idempotent."""

        if self._closed:
            return
        self._closed = True

        def _wipe() -> None:
            # delete_all is a courtesy — if Mem0's vector store has
            # gone away (e.g. ephemeral instance), don't tank the
            # bench teardown.
            with contextlib.suppress(Exception):
                self._client.delete_all(user_id=self._user_id)

        await self._run(_wipe)
        # Drop the in-memory client reference before tearing down the
        # executor — Qdrant's local-mode finalizer wants the same
        # thread that opened the SQLite store, and pinning everything
        # to the executor makes sure that holds.
        def _drop_client() -> None:
            self._client = None

        with contextlib.suppress(Exception):
            await self._run(_drop_client)
        self._executor.shutdown(wait=True)
        if self._cleanup_dir is not None and self._cleanup_dir.exists():
            shutil.rmtree(self._cleanup_dir, ignore_errors=True)

    # Generic single-text observe so Mem0Backend can also stand in for
    # the older ``MemoryBackend`` protocol used by ``launch_comparison``.
    # The LongMemEval harness uses ``ingest_session`` instead.
    async def observe(self, text: str, ts: float) -> str:
        def _add() -> str:
            res = self._client.add(
                [{"role": "user", "content": text}],
                user_id=self._user_id,
                metadata={"ts": ts},
            )
            # Mem0's ``add`` returns a structured result — pull the
            # first surviving memory id, or synthesize one if Mem0
            # decided not to extract anything.
            if isinstance(res, dict) and res.get("results"):
                first = res["results"][0]
                if isinstance(first, dict) and "id" in first:
                    return str(first["id"])
            if isinstance(res, list) and res:
                first = res[0]
                if isinstance(first, dict) and "id" in first:
                    return str(first["id"])
            return f"mem0-noop-{int(ts * 1000)}-{uuid4().hex[:8]}"

        return await self._run(_add)

    async def __aenter__(self) -> Mem0Backend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def _now_unix() -> float:
    """Tiny utility used by tests that fake ``Mem0Backend`` without
    a real Mem0 install. Kept here so the test stays in-tree."""

    return time.time()
