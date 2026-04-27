"""Async SQLite backend for Mnemoss — thin orchestrator over sync op modules.

The actual SQL lives in ``_memory_ops``, ``_graph_ops``, and
``_raw_log_ops``. ``SQLiteBackend`` owns the apsw connections, the
write lock, and the dedicated worker thread that every DB call is
funneled through (apsw connections are pinned to their creating
thread). Each public method is a thin wrapper that takes the write
lock when needed and dispatches the corresponding sync function on
the worker thread.

This structure keeps:
- SQL centralized in focused modules, unit-testable against a plain
  ``apsw.Connection`` without the async machinery.
- The threading + lock policy in one place, easy to audit.
- The façade small enough that adding or renaming a read/write is
  one wrapper line, not a mix of SQL + scheduling.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import struct
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import apsw
import numpy as np
import sqlite_vec

from mnemoss.core.config import SCHEMA_VERSION
from mnemoss.core.types import IndexTier, Memory, RawMessage, Tombstone
from mnemoss.store import _graph_ops, _memory_ops, _raw_log_ops
from mnemoss.store._sql_helpers import build_trigram_query, filter_by_agent_and_tier
from mnemoss.store._workspace_lock import WorkspaceLock, WorkspaceLockError
from mnemoss.store.ann_index import HNSWLIB_AVAILABLE, ANNIndex
from mnemoss.store.migrations import (
    MigrationError,
    apply_migrations,
    apply_raw_log_migrations,
)
from mnemoss.store.schema import (
    FTS_DDL,
    MEMORY_DDL_STATEMENTS,
    MIN_SQLITE_VERSION,
    RAW_LOG_DDL_STATEMENTS,
    vec_ddl,
)

_log = logging.getLogger(__name__)

__all__ = [
    "SQLiteBackend",
    "SchemaMismatchError",
    "WorkspaceLockError",
    "build_trigram_query",
]

# Milliseconds SQLite will retry a SQLITE_BUSY write before raising.
# Belt-and-braces next to the workspace lock: the lock prevents most
# conflicts, but intra-process WAL contention (e.g. Dream + recall
# overlapping) can still transiently BUSY on heavy writes.
_BUSY_TIMEOUT_MS = 5000


class SchemaMismatchError(RuntimeError):
    """Raised when a workspace DB's schema / embedding dim doesn't match
    the embedder the caller is trying to open it with."""


class SQLiteBackend:
    """Low-level persistence.

    The constructor does no I/O — it just stashes paths and dimensions.
    Call :meth:`open` before use. First call to :meth:`open` for a
    fresh path creates the DB file; subsequent calls validate the
    pinned schema version and embedding dim.
    """

    def __init__(
        self,
        db_path: Path,
        raw_log_path: Path,
        workspace_id: str,
        embedding_dim: int,
        embedder_id: str,
        *,
        use_ann_index: bool = True,
    ) -> None:
        self._db_path = db_path
        self._raw_log_path = raw_log_path
        self._workspace_id = workspace_id
        self._embedding_dim = embedding_dim
        self._embedder_id = embedder_id
        self._conn: apsw.Connection | None = None
        self._raw_conn: apsw.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._memory_columns: list[str] = []
        self._use_ann_index = use_ann_index
        # ANN index: built on open, populated from memory_vec, updated
        # on write_memory / delete_memory_completely. See
        # mnemoss/store/ann_index.py for the HNSW design rationale.
        # None means "fall back to sqlite-vec linear scan" — either
        # because use_ann_index=False or hnswlib isn't installed.
        self._ann: ANNIndex | None = None
        # Cross-process advisory lock on the workspace directory.
        # Acquired in _open_sync and released in close(), so a second
        # process trying to open the same workspace fails fast.
        self._workspace_lock = WorkspaceLock(db_path.parent / ".mnemoss.lock")
        # apsw connections are pinned to the thread that created them;
        # funnel every DB call through one worker so concurrent callers
        # are safe without per-thread connections.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mnemoss-db"
        )

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
        )

    # ─── open / close ─────────────────────────────────────────────────

    async def open(self) -> None:
        await self._run(self._open_sync)

    def _open_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Acquire the cross-process lock *before* we touch any DB files.
        # If a second process tries to open the same workspace we raise
        # WorkspaceLockError here and never create partial state.
        self._workspace_lock.acquire()
        try:
            self._open_with_lock_held()
        except BaseException:
            # Schema mismatch / corrupt DB / platform error — release the
            # lock so a corrected reopen (or a different process) can
            # proceed. Use BaseException so KeyboardInterrupt also cleans up.
            self._workspace_lock.release()
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._raw_conn is not None:
                self._raw_conn.close()
                self._raw_conn = None
            raise

    def _open_with_lock_held(self) -> None:
        is_new = not self._db_path.exists()

        conn = apsw.Connection(str(self._db_path))
        sqlite_version = tuple(int(p) for p in apsw.sqlite_lib_version().split("."))
        if sqlite_version < MIN_SQLITE_VERSION:
            raise RuntimeError(
                f"SQLite >= {MIN_SQLITE_VERSION} required for FTS5 trigram "
                f"tokenizer; got {sqlite_version}"
            )

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        except Exception as e:  # pragma: no cover - platform-specific
            raise RuntimeError(
                "Failed to load sqlite-vec extension. On macOS with pyenv "
                "Python, ensure apsw is installed (it ships its own SQLite)."
            ) from e
        conn.enable_load_extension(False)

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")

        if is_new:
            self._create_schema(conn)
        else:
            self._validate_meta(conn)

        self._conn = conn
        self._memory_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(memory)")
        ]

        raw_is_new = not self._raw_log_path.exists()
        raw_conn = apsw.Connection(str(self._raw_log_path))
        raw_conn.execute("PRAGMA journal_mode=WAL")
        raw_conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        if raw_is_new:
            self._create_raw_schema(raw_conn)
        else:
            self._validate_raw_meta(raw_conn)
        self._raw_conn = raw_conn

        # Build the ANN index last — after both DBs are open so
        # rehydrate sees a valid memory_vec table. A no-op on fresh
        # workspaces (memory_vec is empty).
        self._maybe_build_ann_index(conn)

    def _maybe_build_ann_index(self, conn: apsw.Connection) -> None:
        """Initialize the HNSW index and rehydrate from memory_vec.

        Honours the ``use_ann_index`` flag and falls back silently (but
        with a one-line log notice) if ``hnswlib`` isn't installed.
        Safe on an empty workspace — just creates an empty index ready
        to receive the first write.
        """

        if not self._use_ann_index:
            return
        if not HNSWLIB_AVAILABLE:
            _log.info(
                "mnemoss: hnswlib not installed; vec_search will use "
                "sqlite-vec linear scan. `pip install mnemoss[ann]` for "
                "O(log N) vector recall."
            )
            return

        # Start the index sized to current workspace; grows on demand.
        n_rows = int(
            conn.execute("SELECT COUNT(*) FROM memory_vec").fetchone()[0]
        )
        initial_capacity = max(1024, n_rows * 2)
        self._ann = ANNIndex(
            dim=self._embedding_dim,
            initial_capacity=initial_capacity,
        )

        if n_rows == 0:
            return

        # Rehydrate: pull all (memory_id, embedding) rows and batch-add.
        # vec0 stores embeddings as little-endian float32 blobs — the
        # same pack_vec format we wrote on insert, so decoding is a
        # single struct.unpack per row.
        batch_ids: list[str] = []
        batch_emb: list[np.ndarray] = []
        for mid, blob in conn.execute(
            "SELECT memory_id, embedding FROM memory_vec"
        ):
            vec = np.array(
                struct.unpack(f"{self._embedding_dim}f", bytes(blob)),
                dtype=np.float32,
            )
            batch_ids.append(str(mid))
            batch_emb.append(vec)

        if batch_ids:
            self._ann.add_batch(batch_ids, np.vstack(batch_emb))

    def _create_schema(self, conn: apsw.Connection) -> None:
        with conn:
            for ddl in MEMORY_DDL_STATEMENTS:
                conn.execute(ddl)
            conn.execute(vec_ddl(self._embedding_dim))
            conn.execute(FTS_DDL)
            conn.execute(
                "INSERT INTO workspace_meta(k, v) VALUES (?, ?), (?, ?), (?, ?)",
                (
                    "schema_version",
                    str(SCHEMA_VERSION),
                    "embedding_dim",
                    str(self._embedding_dim),
                    "embedder_id",
                    self._embedder_id,
                ),
            )

    def _create_raw_schema(self, conn: apsw.Connection) -> None:
        with conn:
            for ddl in RAW_LOG_DDL_STATEMENTS:
                conn.execute(ddl)
            conn.execute(
                "INSERT INTO raw_log_meta(k, v) VALUES (?, ?), (?, ?)",
                (
                    "schema_version",
                    str(SCHEMA_VERSION),
                    "workspace_id",
                    self._workspace_id,
                ),
            )

    def _validate_meta(self, conn: apsw.Connection) -> None:
        rows = dict(conn.execute("SELECT k, v FROM workspace_meta"))
        stored_version = int(rows.get("schema_version", "-1"))
        stored_dim = int(rows.get("embedding_dim", "-1"))
        stored_embedder = rows.get("embedder_id", "")

        # Embedding dim and embedder id must match exactly — migrations
        # can handle schema shape, but vector store shape and embedder
        # output must line up or recall silently misbehaves.
        if stored_dim != self._embedding_dim:
            raise SchemaMismatchError(
                f"Embedding dim mismatch on workspace {self._workspace_id!r}: "
                f"DB was created at dim={stored_dim}, caller is opening "
                f"with dim={self._embedding_dim}. The workspace is pinned "
                "to its creation-time embedder; open with the matching "
                "embedder or create a new workspace."
            )
        if stored_embedder != self._embedder_id:
            raise SchemaMismatchError(
                f"Embedder id mismatch on workspace {self._workspace_id!r}: "
                f"DB was created with embedder_id={stored_embedder!r}, "
                f"caller is opening with embedder_id={self._embedder_id!r}. "
                "Switching embedders on a live workspace would invalidate "
                "every stored vector; use a new workspace or restore "
                "from a backup made under the original embedder."
            )

        # Schema version drift is recoverable via migrations.
        if stored_version != SCHEMA_VERSION:
            try:
                apply_migrations(conn, stored_version, SCHEMA_VERSION)
            except MigrationError as e:
                raise SchemaMismatchError(str(e)) from e

    def _validate_raw_meta(self, conn: apsw.Connection) -> None:
        rows = dict(conn.execute("SELECT k, v FROM raw_log_meta"))
        stored_version = int(rows.get("schema_version", "-1"))
        if stored_version != SCHEMA_VERSION:
            try:
                apply_raw_log_migrations(conn, stored_version, SCHEMA_VERSION)
            except MigrationError as e:
                raise SchemaMismatchError(str(e)) from e

    async def close(self) -> None:
        if self._conn is not None:
            await self._run(self._conn.close)
            self._conn = None
        if self._raw_conn is not None:
            await self._run(self._raw_conn.close)
            self._raw_conn = None
        self._executor.shutdown(wait=True)
        # Release the workspace lock last — after both DBs are closed
        # there's no more writer activity to protect.
        self._workspace_lock.release()

    # ─── memory table — writes ───────────────────────────────────────

    async def write_memory(self, memory: Memory, embedding: np.ndarray) -> None:
        async with self._write_lock:
            await self._run(
                _memory_ops.write_memory,
                self._require_conn(),
                memory,
                embedding,
                self._embedding_dim,
            )
            # Mirror into the ANN index so vec_search returns this
            # memory on the very next recall. Done on the DB worker
            # thread to keep hnswlib confined to a single thread.
            if self._ann is not None:
                await self._run(self._ann.add, memory.id, embedding)

    async def delete_memory_completely(self, memory_id: str) -> None:
        """Remove a memory from ``memory``, ``memory_vec``, ``memory_fts``,
        and every edge in ``relation`` / ``pin``. Raw Log is untouched."""

        async with self._write_lock:
            await self._run(
                _memory_ops.delete_memory_completely, self._require_conn(), memory_id
            )
            if self._ann is not None:
                await self._run(self._ann.remove, memory_id)

    async def update_idx_priority(
        self, memory_id: str, idx_priority: float, tier: IndexTier
    ) -> None:
        """P7 Rebalance: persist a fresh priority + tier for one memory."""

        async with self._write_lock:
            await self._run(
                _memory_ops.update_idx_priority,
                self._require_conn(),
                memory_id,
                idx_priority,
                tier,
            )

    async def update_extraction(
        self,
        memory_id: str,
        *,
        gist: str | None,
        entities: list[str] | None,
        time: datetime | None,
        location: str | None,
        participants: list[str] | None,
        level: int,
    ) -> None:
        """Persist a lazy-extraction pass's output.

        Called by ``RecallEngine`` after it fills heuristic extraction
        fields on returned top-k memories, and by Dream P3 Consolidate
        after its LLM-level refinement (level=2).
        """

        async with self._write_lock:
            await self._run(
                _memory_ops.update_extraction,
                self._require_conn(),
                memory_id,
                gist=gist,
                entities=entities,
                time=time,
                location=location,
                participants=participants,
                level=level,
            )

    async def update_cluster_assignment(
        self,
        memory_id: str,
        cluster_id: str | None,
        similarity: float | None,
        is_representative: bool,
    ) -> None:
        """P2 Cluster: write back cluster_id / similarity / representative."""

        async with self._write_lock:
            await self._run(
                _memory_ops.update_cluster_assignment,
                self._require_conn(),
                memory_id,
                cluster_id,
                similarity,
                is_representative,
            )

    async def reconsolidate(self, memory_id: str, now: datetime) -> None:
        async with self._write_lock:
            await self._run(
                _memory_ops.reconsolidate, self._require_conn(), memory_id, now
            )

    async def mark_superseded(
        self, old_id: str, new_id: str, at: datetime
    ) -> None:
        """Mark an existing memory as superseded by a newer one.

        Used by the contradiction-aware observe path. ``old_id`` stays
        in storage (for audit / dispose trail / future un-supersede)
        but is filtered out of recall by default.
        """

        async with self._write_lock:
            await self._run(
                _memory_ops.mark_superseded,
                self._require_conn(),
                old_id,
                new_id,
                at,
            )

    async def reminisce_to_warm(self, memory_id: str) -> None:
        """Reactivate a DEEP memory: bump reminisced_count + set tier=WARM.

        Called when a DEEP memory is surfaced in cascade recall (§1.9
        "If reactivated from DEEP, ``reminisced_count += 1``, jump to
        WARM"). The next rebalance recomputes ``idx_priority``
        precisely; we set it to mid-WARM here so state stays consistent
        in the interim.
        """

        async with self._write_lock:
            await self._run(
                _memory_ops.reminisce_to_warm, self._require_conn(), memory_id
            )

    async def link_derived(self, parent_ids: Iterable[str], child_id: str) -> int:
        """Append ``child_id`` to each parent's ``derived_to`` list.

        Returns the number of parents that were actually updated. Silently
        skips missing parents and no-ops when the edge already exists.
        """

        return await self._run(
            _memory_ops.link_derived,
            self._require_conn(),
            list(parent_ids),
            child_id,
        )

    # ─── memory table — reads ────────────────────────────────────────

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self._run(
            _memory_ops.get_memory,
            self._require_conn(),
            memory_id,
            self._memory_columns,
        )

    async def materialize_memories(self, ids: Iterable[str]) -> list[Memory]:
        return await self._run(
            _memory_ops.materialize_memories,
            self._require_conn(),
            list(ids),
            self._memory_columns,
        )

    async def list_memories_for_export(
        self,
        agent_id: str | None,
        *,
        min_idx_priority: float = 0.0,
        limit: int = 500,
    ) -> list[Memory]:
        """In-scope memories ordered by idx_priority desc.

        Used by ``Mnemoss.export_markdown``. Scope follows recall: a
        non-None ``agent_id`` returns that agent's + ambient; ``None``
        returns ambient-only.
        """

        return await self._run(
            _memory_ops.list_memories_for_export,
            self._require_conn(),
            self._memory_columns,
            agent_id,
            min_idx_priority,
            limit,
        )

    async def iter_memory_ids(self, batch_size: int = 500) -> list[str]:
        """Every memory id in the workspace, for batch rebalance passes."""

        return await self._run(_memory_ops.iter_memory_ids, self._require_conn())

    async def list_recent_in_session(
        self, session_id: str, limit: int
    ) -> list[str]:
        """Recent memory IDs in a session, newest first."""

        return await self._run(
            _memory_ops.list_recent_in_session,
            self._require_conn(),
            session_id,
            limit,
        )

    async def get_embeddings(
        self, memory_ids: Iterable[str]
    ) -> dict[str, np.ndarray]:
        """``{memory_id: embedding}`` for every requested id present.

        Missing ids are omitted from the result. Used by P2 Cluster.
        """

        return await self._run(
            _memory_ops.get_embeddings, self._require_conn(), list(memory_ids)
        )

    async def tier_counts(self) -> dict[IndexTier, int]:
        """``{tier: count}`` across every tier, including empty ones."""

        return await self._run(_memory_ops.tier_counts, self._require_conn())

    async def get_idx_priorities(
        self,
        memory_ids: Iterable[str],
        agent_id: str | None,
    ) -> dict[str, float]:
        """``{memory_id: idx_priority}`` for ids passing agent scope.

        Backs the fast-index recall path: after ANN returns the top-K
        candidates, we batch-fetch their cached priority in a single
        indexed SQL round-trip. O(K) — not O(N) — so it stays fast as
        the workspace grows.
        """

        return await self._run(
            _memory_ops.get_idx_priorities,
            self._require_conn(),
            list(memory_ids),
            agent_id,
        )

    async def cluster_size(self, cluster_id: str) -> int:
        """Number of memories currently registered in ``cluster_id``."""

        return await self._run(
            _memory_ops.cluster_size, self._require_conn(), cluster_id
        )

    async def vec_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None = None,
    ) -> list[tuple[str, float]]:
        """``[(memory_id, cosine_similarity)]`` sorted by similarity desc.

        Filters by agent scope: ``agent_id`` means ``WHERE memory.agent_id
        = id OR memory.agent_id IS NULL``; ``None`` means ambient-only.
        ``tier_filter`` lets cascade retrieval scan one tier at a time.

        Uses the HNSW ANN index when available (O(log N)); falls back to
        ``sqlite-vec``'s linear scan otherwise. Behavior and return
        contract are identical either way — ANN is approximate but with
        default config the recall@10 vs exact is >0.99.
        """

        if self._ann is None:
            return await self._run(
                _memory_ops.vec_search,
                self._require_conn(),
                query_embedding,
                k,
                agent_id,
                tier_filter,
                self._embedding_dim,
            )
        return await self._run(
            self._ann_vec_search_sync,
            query_embedding,
            k,
            agent_id,
            tier_filter,
        )

    def _ann_vec_search_sync(
        self,
        query_embedding: np.ndarray,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None,
    ) -> list[tuple[str, float]]:
        """HNSW top-K + agent/tier filter in one SQL round-trip.

        Over-scans by the same factor as ``_memory_ops.vec_search`` so
        the filter stage still has enough candidates to find ``k``
        survivors when scope filtering removes some.
        """

        assert self._ann is not None
        over_scan = max(k * 8, 64) if tier_filter is not None else max(k * 4, 32)
        hits = self._ann.query(query_embedding, over_scan)
        if not hits:
            return []
        ids = [mid for mid, _ in hits]
        conn = self._require_conn()
        allowed = filter_by_agent_and_tier(conn, ids, agent_id, tier_filter)
        out: list[tuple[str, float]] = []
        for mid, sim in hits:
            if mid in allowed:
                out.append((mid, sim))
                if len(out) >= k:
                    break
        return out

    async def fts_search(
        self,
        query: str,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None = None,
    ) -> list[tuple[str, float]]:
        """``[(memory_id, bm25_raw)]`` — SQLite BM25 (negative)."""

        return await self._run(
            _memory_ops.fts_search,
            self._require_conn(),
            query,
            k,
            agent_id,
            tier_filter,
        )

    # ─── relations ───────────────────────────────────────────────────

    async def write_relation(
        self,
        src_id: str,
        dst_id: str,
        predicate: str,
        confidence: float = 1.0,
    ) -> None:
        async with self._write_lock:
            await self._run(
                _graph_ops.write_relation,
                self._require_conn(),
                src_id,
                dst_id,
                predicate,
                confidence,
            )

    async def fan_out(self, memory_ids: Iterable[str]) -> dict[str, int]:
        return await self._run(
            _graph_ops.fan_out, self._require_conn(), list(memory_ids)
        )

    async def relations_from(
        self, memory_ids: Iterable[str]
    ) -> dict[str, set[str]]:
        return await self._run(
            _graph_ops.relations_from, self._require_conn(), list(memory_ids)
        )

    async def expand_via_relations(
        self,
        seed_ids: Iterable[str],
        *,
        hops: int,
        predicates: Iterable[str] | None = None,
        max_candidates: int | None = None,
    ) -> set[str]:
        """BFS over the relation graph from ``seed_ids``, ``hops`` deep.

        Returns every memory id reachable in ``[1, hops]`` steps via the
        directed ``relation`` table, restricted to ``predicates`` when
        provided. Seeds themselves are excluded from the result. Used
        by the auto-expand path on repeated same-topic recalls.

        ``max_candidates`` caps the reachable set size. BFS stops at
        the first hop boundary where the cap would be exceeded; any ids
        already collected are returned.
        """

        return await self._run(
            _graph_ops.expand_via_relations,
            self._require_conn(),
            list(seed_ids),
            hops,
            list(predicates) if predicates is not None else None,
            max_candidates,
        )

    # ─── pins ────────────────────────────────────────────────────────

    async def pin(self, memory_id: str, agent_id: str | None) -> None:
        async with self._write_lock:
            await self._run(
                _graph_ops.pin, self._require_conn(), memory_id, agent_id
            )

    async def is_pinned(self, memory_id: str, agent_id: str | None) -> bool:
        return await self._run(
            _graph_ops.is_pinned, self._require_conn(), memory_id, agent_id
        )

    async def pinned_any(self, memory_ids: Iterable[str]) -> set[str]:
        """Subset of ``memory_ids`` pinned by *any* agent.

        Used by P7 Rebalance: ``idx_priority`` is a memory-wide property,
        so a per-agent pin counts for the memory's tier decision.
        """

        return await self._run(
            _graph_ops.pinned_any, self._require_conn(), list(memory_ids)
        )

    async def pinned_by_agent(
        self, memory_ids: Iterable[str], agent_id: str | None
    ) -> set[str]:
        """Subset of ``memory_ids`` pinned *by this agent*.

        Per-agent pin semantics — matches ``is_pinned(id, agent_id)``
        but batched. Used by expansion scoring where the candidate list
        may be in the hundreds.
        """

        return await self._run(
            _graph_ops.pinned_by_agent,
            self._require_conn(),
            list(memory_ids),
            agent_id,
        )

    async def pinned_ids_in_scope(self, agent_id: str | None) -> set[str]:
        """Memory ids pinned by the caller's scope.

        ``agent_id=None`` matches ambient pins only; a non-None id
        matches that agent's pins *plus* ambient pins.
        """

        return await self._run(
            _graph_ops.pinned_ids_in_scope, self._require_conn(), agent_id
        )

    # ─── tombstones ──────────────────────────────────────────────────

    async def write_tombstone(self, tombstone: Tombstone) -> None:
        """Persist a disposal record."""

        async with self._write_lock:
            await self._run(
                _graph_ops.write_tombstone, self._require_conn(), tombstone
            )

    async def list_tombstones(
        self, *, agent_id: str | None = None, limit: int = 100
    ) -> list[Tombstone]:
        """Recent tombstones, newest first.

        Scope follows recall: a non-None ``agent_id`` returns that
        agent's + ambient; ``None`` returns ambient-only.
        """

        return await self._run(
            _graph_ops.list_tombstones, self._require_conn(), agent_id, limit
        )

    async def count_tombstones(self) -> int:
        """Count of tombstone rows across the whole workspace."""

        return await self._run(_graph_ops.count_tombstones, self._require_conn())

    # ─── raw log ─────────────────────────────────────────────────────

    async def write_raw_message(self, msg: RawMessage) -> None:
        async with self._write_lock:
            await self._run(
                _raw_log_ops.write_raw_message, self._require_raw_conn(), msg
            )

    # ─── helpers ─────────────────────────────────────────────────────

    def _require_conn(self) -> apsw.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteBackend.open() must be called first")
        return self._conn

    def _require_raw_conn(self) -> apsw.Connection:
        if self._raw_conn is None:
            raise RuntimeError("SQLiteBackend.open() must be called first")
        return self._raw_conn
