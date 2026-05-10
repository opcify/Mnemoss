"""MemOS 2.0 adapter for the LongMemEval-S benchmark.

`MemOS <https://github.com/MemTensor/MemOS>`_ ("Memory Operating
System") wraps multi-cube memory under a ``MOS`` facade with three
relevant operations:

- ``MOS.add(messages=..., user_id=..., mem_cube_id=...)`` — ingest a
  list of role/content turns into a cube. MemOS internally extracts
  textual memories (``TextualMemory``) and optional activation /
  parametric memories.
- ``MOS.search(query, user_id, top_k, install_cube_ids=None)`` —
  hybrid retrieval across a user's cubes. Returns a structured dict
  whose ``text_mem`` slice carries the surviving memory items.
- ``MOS.delete_cube(cube_id, user_id)`` and friends — namespace
  cleanup.

The 2.0 line stabilised the ``MOSConfig`` / ``GeneralMemCubeConfig``
schemas and the ``MOS`` constructor signature. We pin against that
public surface, lazy-importing so the bench module loads cleanly
without ``MemoryOS`` installed.

Embedder parity with Mnemoss
----------------------------

MemOS's defaults pick OpenAI ``text-embedding-3-small`` for the vector
side (matches Mnemoss + Mem0) and ``gpt-4o-mini`` for its extraction
LLM. We don't override unless the caller passes ``mos_config`` —
benches measure each system "as it ships," not in a custom-tuned
config that hides architectural cost.

OpenAI fallbacks need ``OPENAI_API_KEY`` in env. To run on Anthropic
or Ollama, build a ``MOSConfig`` and pass it via ``mos_config=``.

Optional dependency
-------------------

``MemoryOS`` is not in Mnemoss's normal install surface (it carries
its own LLM + embedder + Neo4j stack). Install it just for
benchmarking::

    pip install MemoryOS

The import is lazy so missing-extra failures fire only when a caller
actually constructs ``MemOSBackend``.

Naming note
-----------

Two distinct projects circulate as "MemOS"-flavored memory layers:
**MemOS** (the MemTensor project, PyPI ``MemoryOS``) and **MemoryOS**
(BAI-LAB). This adapter targets MemTensor's MemOS 2.x, which is what
the user-facing "MemOS 2.0" version label refers to. If you're on
BAI-LAB's MemoryOS, the API is different — write a parallel backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from bench.backends.base import RecallHit


def _build_default_mos_config(
    *,
    user_id: str,
    cube_id: str,
    workspace: Path,
) -> Any:
    """Construct a minimal MOSConfig with one TreeTextMemory cube.

    A bare ``MOS()`` call requires no arguments in MemOS's "quick start"
    docs, but the actual constructor signature wants a ``MOSConfig``.
    We build the smallest one that exercises the textual-memory path —
    that's the slice LongMemEval-S evaluates.
    """

    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.configs.mem_os import MOSConfig

    cube_dir = workspace / cube_id
    cube_dir.mkdir(parents=True, exist_ok=True)
    # GeneralMemCubeConfig is imported for the symbol-existence side
    # effect: a wrong MemoryOS install will fail-fast here rather than
    # mid-add. We don't actually thread the per-cube config through —
    # MOS 2.x builds cubes from the workspace dir registered later via
    # ``register_mem_cube``.
    _ = GeneralMemCubeConfig

    return MOSConfig(
        user_id=user_id,
        chat_model={"backend": "openai", "config": {"model_name_or_path": "gpt-4o-mini"}},
        mem_reader={
            "backend": "simple_struct",
            "config": {
                "llm": {"backend": "openai", "config": {"model_name_or_path": "gpt-4o-mini"}},
                "embedder": {
                    "backend": "openai",
                    "config": {"model_name_or_path": "text-embedding-3-small"},
                },
            },
        },
        enable_textual_memory=True,
        enable_activation_memory=False,
        enable_parametric_memory=False,
        top_k=10,
    )


class MemOSBackend:
    """LongMemEval-S backend wrapping MemOS 2.x ``MOS``.

    Parameters
    ----------
    user_id:
        MemOS namespaces memories by ``user_id``. Defaults to a fresh
        random id per instance so two backends in the same process
        can't see each other's state.
    cube_id:
        The MemOS "cube" the memories live in. One cube per backend
        instance keeps cleanup straightforward — we drop the whole
        cube on close.
    mos_config:
        Optional pre-built ``MOSConfig``. ``None`` builds the minimal
        default above (one textual cube, OpenAI LLM + embedder).
    workspace:
        Where MemOS persists cube state. Defaults to a fresh tempdir;
        unconditionally rmtree'd on close so a crashed run doesn't
        leak gigabytes of vector index files.
    """

    backend_id = "memos"

    def __init__(
        self,
        *,
        user_id: str | None = None,
        cube_id: str | None = None,
        mos_config: Any | None = None,
        workspace: Path | None = None,
    ) -> None:
        try:
            from memos.mem_os.main import MOS  # noqa: F401  — imported below
        except ImportError as exc:  # pragma: no cover — exercised at install time
            raise ImportError(
                "MemOSBackend requires the optional `MemoryOS` package "
                "(MemTensor's MemOS 2.x). Install with `pip install MemoryOS`."
            ) from exc

        self._user_id = user_id or f"mnemoss-bench-{uuid4().hex}"
        self._cube_id = cube_id or f"bench-cube-{uuid4().hex[:8]}"
        self._owned_workspace = workspace is None
        self._workspace = workspace or Path(tempfile.mkdtemp(prefix="memos_bench_"))
        self._closed = False
        self._mos: Any | None = None
        self._mos_config = mos_config
        # Lazy MOS construction — defer until first ingest so a
        # MOSConfig validation error surfaces in the harness loop with
        # the right per-question context, not during __init__.

    def _ensure_mos(self) -> Any:
        if self._mos is not None:
            return self._mos
        from memos.mem_os.main import MOS

        config = self._mos_config or _build_default_mos_config(
            user_id=self._user_id,
            cube_id=self._cube_id,
            workspace=self._workspace,
        )
        self._mos = MOS(config)
        # MemOS requires the user + cube to be registered before
        # add/search. The methods are best-effort — older 2.0 builds
        # auto-create on first ``add``; newer ones require explicit
        # registration. Try both, ignore "already exists".
        for method_name, args in (
            ("create_user", {"user_id": self._user_id}),
            (
                "register_mem_cube",
                {"mem_cube_name_or_path": self._cube_id, "user_id": self._user_id},
            ),
        ):
            method = getattr(self._mos, method_name, None)
            if method is None:
                continue
            with contextlib.suppress(Exception):
                method(**args)
        return self._mos

    async def ingest_session(
        self,
        *,
        session_id: str,
        ts: float,
        turns: list[dict[str, str]],
    ) -> None:
        """Ingest one session as a single MemOS ``add`` call.

        MemOS's textual-memory extractor benefits from seeing the full
        session at once — same rationale as Mem0's batched ``add``.
        Per-turn calls produce shallower memories.
        """

        if not turns:
            return

        def _add() -> None:
            mos = self._ensure_mos()
            # MemOS 2.x ``add`` accepts ``messages=`` for chat-style
            # ingestion. ``mem_cube_id`` is required when multiple
            # cubes exist; we always pass ours explicitly.
            mos.add(
                messages=list(turns),
                user_id=self._user_id,
                mem_cube_id=self._cube_id,
            )

        await asyncio.get_running_loop().run_in_executor(None, _add)

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Search the textual memories. Empty list if nothing matches."""

        def _search() -> list[dict[str, Any]]:
            mos = self._ensure_mos()
            res = mos.search(
                query=query,
                user_id=self._user_id,
                top_k=k,
                install_cube_ids=[self._cube_id],
            )
            # MemOS returns ``{"text_mem": [{"cube_id": ..., "memories": [...]}, ...], ...}``.
            # Flatten across cubes; 0 cubes → 0 memories → [].
            text_mem = (res or {}).get("text_mem", []) if isinstance(res, dict) else []
            flat: list[dict[str, Any]] = []
            for entry in text_mem:
                for mem in entry.get("memories", []):
                    flat.append(mem)
            return flat[:k]

        results = await asyncio.get_running_loop().run_in_executor(None, _search)
        hits: list[RecallHit] = []
        for i, m in enumerate(results):
            mid = m.get("id") or m.get("memory_id") or f"memos-{i}"
            score = m.get("score") or m.get("relevance")
            hits.append(
                RecallHit(
                    memory_id=str(mid),
                    rank=i + 1,
                    score=float(score) if score is not None else None,
                )
            )
        return hits

    async def recall_text(self, query: str, k: int = 10) -> list[str]:
        """Return the recalled memory ``memory`` text strings, in rank order."""

        def _search() -> list[str]:
            mos = self._ensure_mos()
            res = mos.search(
                query=query,
                user_id=self._user_id,
                top_k=k,
                install_cube_ids=[self._cube_id],
            )
            text_mem = (res or {}).get("text_mem", []) if isinstance(res, dict) else []
            out: list[str] = []
            for entry in text_mem:
                for mem in entry.get("memories", []):
                    text = mem.get("memory") or mem.get("content") or mem.get("text") or ""
                    if text:
                        out.append(text)
                    if len(out) >= k:
                        return out
            return out

        return await asyncio.get_running_loop().run_in_executor(None, _search)

    async def close(self) -> None:
        """Drop this user's cube and rmtree the workspace. Idempotent."""

        if self._closed:
            return
        self._closed = True

        def _cleanup() -> None:
            if self._mos is not None:
                # delete_cube is the canonical 2.x cleanup; older
                # builds may have ``unregister_mem_cube``. Try both.
                for method_name in ("delete_cube", "unregister_mem_cube"):
                    method = getattr(self._mos, method_name, None)
                    if method is None:
                        continue
                    try:
                        method(self._cube_id, user_id=self._user_id)
                    except TypeError:
                        # Some signatures take positional only.
                        with contextlib.suppress(Exception):
                            method(self._cube_id)
                    except Exception:  # noqa: BLE001
                        pass
                # Best-effort full close.
                close_method = getattr(self._mos, "close", None)
                if close_method is not None:
                    with contextlib.suppress(Exception):
                        close_method()

        try:
            await asyncio.get_running_loop().run_in_executor(None, _cleanup)
        finally:
            if self._owned_workspace:
                shutil.rmtree(self._workspace, ignore_errors=True)

    # Generic single-text observe so MemOSBackend can also stand in for
    # the older ``MemoryBackend`` protocol used by ``launch_comparison``.
    async def observe(self, text: str, ts: float) -> str:
        def _add() -> str:
            mos = self._ensure_mos()
            mos.add(
                messages=[{"role": "user", "content": text}],
                user_id=self._user_id,
                mem_cube_id=self._cube_id,
            )
            # MemOS's add doesn't synchronously return memory ids the
            # way Mem0 does — it queues extraction. Synthesize a
            # deterministic id from the input so the benchmark's
            # mapping bookkeeping doesn't choke. The id is opaque
            # to LongMemEval-S (we only score the final answer).
            return f"memos-{int(ts * 1000)}-{uuid4().hex[:8]}"

        return await asyncio.get_running_loop().run_in_executor(None, _add)

    async def __aenter__(self) -> MemOSBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
