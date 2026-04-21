"""Mnemoss MemoryProvider for Hermes Agent.

Implements the ``agent.memory_provider.MemoryProvider`` contract so
Hermes treats Mnemoss as a first-class memory backend alongside the
built-in MEMORY.md/USER.md store. The Honcho plugin is the reference
for Hermes's provider lifecycle; see ``plugins/memory/honcho/`` in
hermes-agent.

Lifecycle summary (Hermes invokes in this order):

    initialize(session_id, **kwargs)
        → construct / resolve the Mnemoss backend
    system_prompt_block()
        → returned once per turn; static tool hint injected into prompt
    queue_prefetch(query)                       [end of turn N]
        → prepare recall for turn N+1 (fire-and-forget; Hermes prefers
          background work so the hot path stays under its latency budget)
    prefetch(query)                             [start of turn N+1]
        → returns the per-turn memory block (pulled from the queued work)
    sync_turn(user_content, assistant_content)  [after each turn]
        → persist the turn into Mnemoss
    on_session_end(messages)                    [session boundary]
        → trigger a session_end dream for consolidation
    on_pre_compress(messages)                   [before compression]
        → hand extra text to the compressor so insights survive
    on_memory_write(action, target, content)    [builtin writes]
        → mirror builtin MEMORY.md writes into Mnemoss as observations
    shutdown()
        → flush + close

Backend options:

- **Embedded** (default): constructs a ``mnemoss.Mnemoss`` in-process.
  Zero network hop; the database lives at ``{hermes_home}/mnemoss/``.
- **Remote**: points at a shared Mnemoss REST server via the SDK. Set
  ``baseUrl`` (and optionally ``apiKey``) in the provider config.

Config resolution order (highest first):

1. Explicit ``init(config=...)`` dict (mostly for tests).
2. ``$HERMES_HOME/mnemoss.json`` — user-editable, persists across runs.
3. Environment variables ``MNEMOSS_BASE_URL``, ``MNEMOSS_API_KEY``,
   ``MNEMOSS_WORKSPACE``.
4. Defaults (embedded mode, workspace = agent_identity or "hermes").
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    # In a real Hermes deployment this resolves through hermes-agent's
    # own package root (the plugin lives at
    # ``hermes-agent/plugins/memory/mnemoss/``).
    from agent.memory_provider import MemoryProvider  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — covered by stub in tests/conftest
    # Standalone development / CI outside hermes-agent. ``tests/conftest.py``
    # drops a stub ``agent.memory_provider`` module on ``sys.path`` so this
    # import succeeds; shipped into hermes-agent, this branch never runs.
    from mnemoss_hermes._stub_memory_provider import MemoryProvider  # type: ignore[no-redef]

logger = logging.getLogger("mnemoss_hermes")


# ─── tool schemas (exposed to the Hermes-hosted model) ────────────


_RECALL_SCHEMA = {
    "name": "mnemoss_recall",
    "description": (
        "Search Mnemoss memory for context relevant to a query. Returns "
        "memories ranked by ACT-R activation — a blend of recency, "
        "frequency, spreading activation from recent context, and "
        "literal/semantic match. Use when you need to pull in prior "
        "context the built-in prefetch didn't surface."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up.",
            },
            "k": {
                "type": "integer",
                "description": "Number of memories to return (default 5).",
            },
            "include_deep": {
                "type": "boolean",
                "description": (
                    "Force a scan of the DEEP tier (cold, rarely "
                    "accessed memories). Default false."
                ),
            },
        },
        "required": ["query"],
    },
}

_EXPAND_SCHEMA = {
    "name": "mnemoss_expand",
    "description": (
        "Given a specific memory id, return memories reachable through "
        "the relation graph ranked by spreading activation. Use after a "
        "recall when you want to dig deeper into one surfaced memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Memory id from a prior recall result.",
            },
            "hops": {
                "type": "integer",
                "description": "Graph radius (1–3). Default 1.",
            },
            "k": {
                "type": "integer",
                "description": "Max memories to return (default 5).",
            },
        },
        "required": ["memory_id"],
    },
}

_PIN_SCHEMA = {
    "name": "mnemoss_pin",
    "description": (
        "Mark a memory as pinned. Pinned memories are protected from "
        "disposal and stay in the HOT tier regardless of recency. Use "
        "sparingly — for facts the user has explicitly said matter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Memory id to pin.",
            },
        },
        "required": ["memory_id"],
    },
}


_ALL_TOOL_SCHEMAS = [_RECALL_SCHEMA, _EXPAND_SCHEMA, _PIN_SCHEMA]


# ─── provider ─────────────────────────────────────────────────────


class MnemossMemoryProvider(MemoryProvider):  # type: ignore[misc]
    """Hermes memory provider backed by Mnemoss.

    Thread-safe for concurrent turn handling as long as the underlying
    Mnemoss backend is — the embedded library and the SDK handle both
    serialize through their own internal locks.
    """

    # Sentinel marking "no prefetch queued yet" so we can distinguish
    # "not asked" from "asked and got empty."
    _PREFETCH_UNSET = "__mnemoss_prefetch_unset__"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._explicit_config: dict[str, Any] | None = config
        self._config: dict[str, Any] = {}
        self._backend: Any = None  # Mnemoss or WorkspaceHandle
        self._client: Any = None  # SDK client, kept alive for remote mode
        self._session_id: str = ""
        self._agent_id: str | None = None
        self._prefetch_cache: str = self._PREFETCH_UNSET
        self._skipped_reason: str | None = None
        # Hermes's prefetch() is synchronous but Mnemoss is all async.
        # Cache a loop we own so sync_* methods can await into it cheaply.
        self._loop: asyncio.AbstractEventLoop | None = None

    # ─── identity ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "mnemoss"

    # ─── availability + setup ────────────────────────────────────

    def is_available(self) -> bool:
        """Return True when Mnemoss is importable. We don't need creds
        for the embedded mode, so a bare install is enough; remote mode
        reports unavailable if neither the SDK nor a base URL is
        configured, but that's resolved inside ``initialize``."""

        try:
            import mnemoss  # noqa: F401
        except ImportError:
            return False
        return True

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "baseUrl",
                "description": (
                    "Mnemoss REST server URL. Leave blank for embedded "
                    "mode (Mnemoss runs in-process)."
                ),
            },
            {
                "key": "api_key",
                "description": "Mnemoss API key (only for remote mode).",
                "secret": True,
                "env_var": "MNEMOSS_API_KEY",
            },
            {
                "key": "workspace",
                "description": (
                    "Mnemoss workspace id. Defaults to the agent identity "
                    "when omitted."
                ),
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Persist non-secret config to ``$HERMES_HOME/mnemoss.json``."""

        path = Path(hermes_home) / "mnemoss.json"
        existing: dict[str, Any] = {}
        if path.exists():
            with contextlib.suppress(Exception):
                existing = json.loads(path.read_text())
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2))

    # ─── lifecycle: initialize ───────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:  # noqa: ANN003
        """Wire up the Mnemoss backend for this session.

        Skips the whole plugin under non-primary agent contexts (cron,
        flush) because those are read-only passes that shouldn't mutate
        memory — matching Honcho's behaviour.
        """

        context = kwargs.get("agent_context", "primary")
        platform = kwargs.get("platform", "cli")
        if context in ("cron", "flush") or platform == "cron":
            self._skipped_reason = f"context={context}, platform={platform}"
            logger.debug("Mnemoss skipped: %s", self._skipped_reason)
            return

        self._session_id = session_id or "hermes-default"
        self._agent_id = kwargs.get("user_id") or None

        cfg = self._resolve_config(
            hermes_home=kwargs.get("hermes_home"),
            agent_identity=kwargs.get("agent_identity"),
        )
        self._config = cfg

        try:
            self._backend = self._build_backend(cfg)
        except Exception as exc:  # noqa: BLE001
            # Never hard-fail a Hermes startup because memory plumbing
            # had a bad day. Log and let the rest of the provider go no-op.
            logger.warning(
                "Mnemoss init failed (%s); plugin will be inactive for this session",
                exc,
            )
            self._backend = None

    def _resolve_config(
        self,
        *,
        hermes_home: str | None,
        agent_identity: str | None,
    ) -> dict[str, Any]:
        """Merge config from explicit > file > env > defaults."""

        cfg: dict[str, Any] = {}

        # Lowest precedence first — each layer updates the running dict.
        if agent_identity:
            cfg["workspace"] = agent_identity

        for env_key, cfg_key in [
            ("MNEMOSS_BASE_URL", "baseUrl"),
            ("MNEMOSS_API_KEY", "api_key"),
            ("MNEMOSS_WORKSPACE", "workspace"),
        ]:
            value = os.environ.get(env_key)
            if value:
                cfg[cfg_key] = value

        if hermes_home:
            path = Path(hermes_home) / "mnemoss.json"
            if path.exists():
                try:
                    cfg.update(json.loads(path.read_text()))
                except Exception:
                    logger.debug("mnemoss.json unreadable; ignoring file config")

        if self._explicit_config:
            cfg.update(self._explicit_config)

        cfg.setdefault("workspace", "hermes")
        return cfg

    def _build_backend(self, cfg: dict[str, Any]) -> Any:
        """Construct either an embedded Mnemoss or a REST-backed
        WorkspaceHandle based on whether ``baseUrl`` is set."""

        workspace = cfg.get("workspace") or "hermes"
        base_url = cfg.get("baseUrl")

        if base_url:
            # Remote: HTTP SDK. Requires the [sdk] extra.
            from mnemoss.sdk import MnemossClient  # type: ignore[import-not-found]

            self._client = MnemossClient(
                base_url=base_url, api_key=cfg.get("api_key")
            )
            handle = self._client.workspace(workspace)
            return handle

        # Embedded: construct Mnemoss directly. Storage root lives under
        # the same HERMES_HOME the rest of the agent uses so it migrates
        # with profile-level backups.
        from mnemoss import Mnemoss, StorageParams  # type: ignore[import-not-found]

        storage_root = None
        hermes_home = os.environ.get("HERMES_HOME")
        if hermes_home:
            storage_root = Path(hermes_home) / "mnemoss"

        return Mnemoss(
            workspace=workspace,
            storage=StorageParams(root=storage_root),
        )

    # ─── lifecycle: per-turn ─────────────────────────────────────

    def system_prompt_block(self) -> str:
        if self._skipped_reason or self._backend is None:
            return ""
        return (
            "# Mnemoss Memory\n"
            "Active. Past conversation context is auto-recalled before each "
            "turn (ACT-R ranked). When you need to dig deeper, use the "
            "`mnemoss_recall` tool for ad-hoc lookups, `mnemoss_expand` to "
            "follow relation links from a surfaced memory, and "
            "`mnemoss_pin` to protect a memory from disposal."
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Run the recall synchronously at queue time and stash the
        result for ``prefetch`` to return on the next turn.

        Hermes expects ``queue_prefetch`` to be cheap and non-blocking.
        We satisfy that by running the async recall to completion on a
        dedicated event loop — a local Mnemoss round-trip is a few
        milliseconds; the shared-server path is dominated by HTTP RTT
        and still safely sub-100ms. Keeping it synchronous sidesteps
        the cross-loop handoff pitfalls Honcho solves with threads.
        """

        _ = session_id  # reserved — Hermes includes it for future per-session fan-out.
        if self._skipped_reason or self._backend is None or not query:
            return

        try:
            results = self._run_coro(
                self._backend.recall(query, k=5, agent_id=self._agent_id)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mnemoss queue_prefetch recall failed: %s", exc)
            self._prefetch_cache = ""
            return

        self._prefetch_cache = self._format_recall(results)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return the memory block for the upcoming turn.

        Returns what was queued by the previous turn's ``queue_prefetch``.
        On the first turn (no prior queue) we issue a synchronous recall
        in-place — otherwise the first turn would be empty, which is
        exactly when context is most useful.
        """

        _ = session_id
        if self._skipped_reason or self._backend is None:
            return ""

        if self._prefetch_cache is self._PREFETCH_UNSET:
            # First turn: fill the cache synchronously. The embedded
            # backend resolves this in ms; the SDK path is an HTTP RTT.
            self.queue_prefetch(query)

        cached = self._prefetch_cache
        self._prefetch_cache = self._PREFETCH_UNSET
        if cached is self._PREFETCH_UNSET:  # queue_prefetch errored
            return ""
        return cached or ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Observe both sides of the turn."""

        _ = session_id
        if self._skipped_reason or self._backend is None:
            return

        async def _write() -> None:
            if user_content:
                await self._backend.observe(
                    role="user",
                    content=user_content,
                    agent_id=self._agent_id,
                    session_id=self._session_id,
                )
            if assistant_content:
                await self._backend.observe(
                    role="assistant",
                    content=assistant_content,
                    agent_id=self._agent_id,
                    session_id=self._session_id,
                )

        try:
            self._run_coro(_write())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mnemoss sync_turn failed: %s", exc)

    # ─── lifecycle: tools ────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if self._skipped_reason or self._backend is None:
            return []
        return list(_ALL_TOOL_SCHEMAS)

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs  # noqa: ANN003
    ) -> str:
        """Dispatch to the Mnemoss backend. Returns JSON (the tool result
        contract Hermes expects)."""

        if self._skipped_reason or self._backend is None:
            return json.dumps(
                {"error": "Mnemoss is not active for this session."}
            )

        try:
            if tool_name == "mnemoss_recall":
                query = args.get("query", "").strip()
                if not query:
                    return json.dumps(
                        {"error": "Missing required parameter: query"}
                    )
                k = int(args.get("k") or 5)
                include_deep = bool(args.get("include_deep", False))
                results = self._run_coro(
                    self._backend.recall(
                        query,
                        k=k,
                        agent_id=self._agent_id,
                        include_deep=include_deep,
                    )
                )
                return json.dumps(
                    {
                        "results": [
                            {
                                "id": r.memory.id,
                                "content": r.memory.content,
                                "score": float(r.score),
                                "source": getattr(r, "source", "direct"),
                            }
                            for r in results
                        ]
                    }
                )

            if tool_name == "mnemoss_expand":
                memory_id = (args.get("memory_id") or "").strip()
                if not memory_id:
                    return json.dumps(
                        {"error": "Missing required parameter: memory_id"}
                    )
                hops = int(args.get("hops") or 1)
                k = int(args.get("k") or 5)
                results = self._run_coro(
                    self._backend.expand(
                        memory_id,
                        hops=hops,
                        k=k,
                        agent_id=self._agent_id,
                    )
                )
                return json.dumps(
                    {
                        "results": [
                            {
                                "id": r.memory.id,
                                "content": r.memory.content,
                                "score": float(r.score),
                                "source": getattr(r, "source", "expanded"),
                            }
                            for r in results
                        ]
                    }
                )

            if tool_name == "mnemoss_pin":
                memory_id = (args.get("memory_id") or "").strip()
                if not memory_id:
                    return json.dumps(
                        {"error": "Missing required parameter: memory_id"}
                    )
                self._run_coro(
                    self._backend.pin(memory_id, agent_id=self._agent_id)
                )
                return json.dumps({"result": f"Pinned {memory_id}."})

            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as exc:  # noqa: BLE001
            logger.warning("Mnemoss tool %s failed: %s", tool_name, exc)
            return json.dumps({"error": f"Mnemoss {tool_name} failed: {exc}"})

    # ─── lifecycle: session boundaries ───────────────────────────

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Consolidate via a session_end dream."""

        _ = messages  # Mnemoss reads its own store; no need to replay.
        if self._skipped_reason or self._backend is None:
            return
        try:
            self._run_coro(
                self._backend.dream(
                    trigger="session_end", agent_id=self._agent_id
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mnemoss session_end dream failed: %s", exc)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Hand Hermes's compressor a short summary of what Mnemoss has
        so it doesn't compress away context the user cares about."""

        if self._skipped_reason or self._backend is None:
            return ""
        # Lean implementation: export the standing-memory block at the
        # HOT threshold. The compressor interpolates this alongside its
        # own summary so cross-session knowledge survives compression.
        try:
            md = self._run_coro(
                self._backend.export_markdown(
                    agent_id=self._agent_id, min_idx_priority=0.7
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mnemoss export_markdown failed in pre_compress: %s", exc)
            return ""
        return md or ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror Hermes's builtin MEMORY.md/USER.md writes into Mnemoss
        so the two stores don't drift apart."""

        if self._skipped_reason or self._backend is None or not content:
            return
        if action != "add":
            return  # removals/replaces are out of scope for mirroring
        role = "user" if target == "user" else "assistant"
        try:
            self._run_coro(
                self._backend.observe(
                    role=role,
                    content=content,
                    agent_id=self._agent_id,
                    session_id=self._session_id,
                    metadata={"mirror_of": f"builtin.{target}"},
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Mnemoss memory mirror failed: %s", exc)

    # ─── shutdown ────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Close the Mnemoss backend + SDK client."""

        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if callable(close):
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        self._run_coro(result)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Mnemoss backend close failed: %s", exc)

        if self._client is not None:
            aclose = getattr(self._client, "aclose", None)
            if callable(aclose):
                try:
                    self._run_coro(aclose())
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Mnemoss client aclose failed: %s", exc)

        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()

    # ─── helpers ─────────────────────────────────────────────────

    def _run_coro(self, coro: Any) -> Any:
        """Run an async coroutine synchronously.

        Provider methods are sync; Mnemoss is async. We own a private
        event loop so repeated calls don't pay per-call loop-setup cost
        (the embedded backend's own locks serialize correctness on top).

        If the current thread is already running a loop (e.g. the caller
        is itself async — rare for Hermes's sync provider contract but
        possible in tests), we fall back to a fresh loop to avoid the
        "cannot be called from a running event loop" error.
        """

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            # A loop is running on this thread. We can't nest; spin a
            # temporary one on a dedicated executor.
            import concurrent.futures

            def _target() -> Any:
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_target).result()

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def _format_recall(self, results: list[Any]) -> str:
        """Render a list of ``RecallResult``s into a Markdown block."""

        if not results:
            return ""
        lines = ["## Relevant memories"]
        for r in results:
            tag = ""
            if getattr(r, "source", "direct") == "expanded":
                tag = " *(associated)*"
            lines.append(f"- {r.memory.content}{tag}")
        return "\n".join(lines)


# ─── Hermes plugin entry ─────────────────────────────────────────


def register(ctx: Any) -> None:
    """Hermes plugin entry point.

    Hermes invokes this during plugin discovery. We hand it our
    :class:`MnemossMemoryProvider` instance; the agent's
    ``MemoryManager`` activates it if ``is_available`` returns ``True``
    and the user has configured Mnemoss as the active memory provider.
    """

    ctx.register_memory_provider(MnemossMemoryProvider())
