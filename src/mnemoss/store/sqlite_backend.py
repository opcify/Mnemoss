"""SQLite backend for Mnemoss.

Uses apsw so sqlite-vec's loadable extension works on stock macOS/pyenv
interpreters. All public methods are ``async`` and wrap blocking work via
``asyncio.to_thread``, so the rest of the codebase can ``await`` uniformly
without an extra driver dependency.

Stage 1 scope: single-process workspace. Cross-process coordination is
Stage 2+.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import struct
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw
import numpy as np
import sqlite_vec

from mnemoss.core.config import SCHEMA_VERSION
from mnemoss.core.types import IndexTier, Memory, MemoryType, RawMessage
from mnemoss.store.schema import DDL_STATEMENTS, FTS_DDL, MIN_SQLITE_VERSION, vec_ddl

UTC = timezone.utc

_FTS_STRIP_RE = re.compile(r"[\"()*:!?\\\n\r\t]")


class SchemaMismatchError(RuntimeError):
    """Raised when a workspace DB's schema / embedding dim doesn't match
    the embedder the caller is trying to open it with."""


class SQLiteBackend:
    """Low-level persistence.

    The constructor does no I/O — it just stashes paths and dimensions. Call
    :meth:`open` before use. First call to :meth:`open` for a fresh path
    creates the DB file; subsequent calls validate the pinned schema version
    and embedding dim.
    """

    def __init__(
        self,
        db_path: Path,
        workspace_id: str,
        embedding_dim: int,
        embedder_id: str,
    ) -> None:
        self._db_path = db_path
        self._workspace_id = workspace_id
        self._embedding_dim = embedding_dim
        self._embedder_id = embedder_id
        self._conn: apsw.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._memory_columns: list[str] = []

    # ─── open / close ─────────────────────────────────────────────────

    async def open(self) -> None:
        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
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

        if is_new:
            self._create_schema(conn)
        else:
            self._validate_meta(conn)

        self._conn = conn
        self._memory_columns = [row[1] for row in conn.execute("PRAGMA table_info(memory)")]

    def _create_schema(self, conn: apsw.Connection) -> None:
        with conn:
            for ddl in DDL_STATEMENTS:
                conn.execute(ddl)
            conn.execute(vec_ddl(self._embedding_dim))
            conn.execute(FTS_DDL)
            conn.execute(
                "INSERT INTO workspace_meta(k, v) VALUES (?, ?), (?, ?), (?, ?)",
                (
                    "schema_version", str(SCHEMA_VERSION),
                    "embedding_dim", str(self._embedding_dim),
                    "embedder_id", self._embedder_id,
                ),
            )

    def _validate_meta(self, conn: apsw.Connection) -> None:
        # Ensure schema tables exist (legacy guard) but mainly validate pins.
        rows = dict(conn.execute("SELECT k, v FROM workspace_meta"))
        stored_version = int(rows.get("schema_version", "-1"))
        stored_dim = int(rows.get("embedding_dim", "-1"))
        stored_embedder = rows.get("embedder_id", "")
        if stored_version != SCHEMA_VERSION:
            raise SchemaMismatchError(
                f"Schema version mismatch: DB={stored_version}, code={SCHEMA_VERSION}"
            )
        if stored_dim != self._embedding_dim:
            raise SchemaMismatchError(
                f"Embedding dim mismatch: DB={stored_dim}, embedder={self._embedding_dim}"
            )
        if stored_embedder != self._embedder_id:
            raise SchemaMismatchError(
                f"Embedder id mismatch: DB={stored_embedder!r}, "
                f"code={self._embedder_id!r}"
            )

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    # ─── writes ───────────────────────────────────────────────────────

    async def write_memory(self, memory: Memory, embedding: np.ndarray) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._write_memory_sync, memory, embedding)

    def _write_memory_sync(self, memory: Memory, embedding: np.ndarray) -> None:
        conn = self._require_conn()
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.shape != (self._embedding_dim,):
            raise ValueError(
                f"Embedding shape {emb.shape} != ({self._embedding_dim},)"
            )
        with conn:
            conn.execute(
                """
                INSERT INTO memory (
                    id, workspace_id, agent_id, session_id, created_at, content,
                    role, memory_type, abstraction_level, access_history,
                    last_accessed_at, rehearsal_count, salience, emotional_weight,
                    reminisced_count, index_tier, idx_priority,
                    source_message_ids, source_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.workspace_id,
                    memory.agent_id,
                    memory.session_id,
                    memory.created_at.timestamp(),
                    memory.content,
                    memory.role,
                    memory.memory_type.value,
                    memory.abstraction_level,
                    json.dumps([dt.timestamp() for dt in memory.access_history]),
                    memory.last_accessed_at.timestamp() if memory.last_accessed_at else None,
                    memory.rehearsal_count,
                    memory.salience,
                    memory.emotional_weight,
                    memory.reminisced_count,
                    memory.index_tier.value,
                    memory.idx_priority,
                    json.dumps(memory.source_message_ids),
                    json.dumps(_json_safe(memory.source_context)),
                ),
            )
            conn.execute(
                "INSERT INTO memory_vec(memory_id, embedding) VALUES (?, ?)",
                (memory.id, _pack_vec(emb)),
            )
            conn.execute(
                "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
                (memory.id, memory.content),
            )

    async def write_raw_message(self, msg: RawMessage) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._write_raw_sync, msg)

    def _write_raw_sync(self, msg: RawMessage) -> None:
        conn = self._require_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO raw_message (
                    id, workspace_id, agent_id, session_id, turn_id, parent_id,
                    timestamp, role, content, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.id,
                    msg.workspace_id,
                    msg.agent_id,
                    msg.session_id,
                    msg.turn_id,
                    msg.parent_id,
                    msg.timestamp.timestamp(),
                    msg.role,
                    msg.content,
                    json.dumps(_json_safe(msg.metadata)),
                ),
            )

    async def write_relation(
        self, src_id: str, dst_id: str, predicate: str, confidence: float = 1.0
    ) -> None:
        async with self._write_lock:
            await asyncio.to_thread(
                self._write_relation_sync, src_id, dst_id, predicate, confidence
            )

    def _write_relation_sync(
        self, src_id: str, dst_id: str, predicate: str, confidence: float
    ) -> None:
        conn = self._require_conn()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO relation VALUES (?, ?, ?, ?, ?)",
                (src_id, dst_id, predicate, confidence, datetime.now(UTC).timestamp()),
            )

    async def pin(self, memory_id: str, agent_id: str | None) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._pin_sync, memory_id, agent_id)

    def _pin_sync(self, memory_id: str, agent_id: str | None) -> None:
        conn = self._require_conn()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO pin VALUES (?, ?, ?)",
                (memory_id, agent_id, datetime.now(UTC).timestamp()),
            )

    async def reconsolidate(self, memory_id: str, now: datetime) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._reconsolidate_sync, memory_id, now)

    def _reconsolidate_sync(self, memory_id: str, now: datetime) -> None:
        conn = self._require_conn()
        with conn:
            row = conn.execute(
                "SELECT access_history, rehearsal_count FROM memory WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return
            history = json.loads(row[0])
            history.append(now.timestamp())
            conn.execute(
                "UPDATE memory SET access_history = ?, rehearsal_count = ?, "
                "last_accessed_at = ? WHERE id = ?",
                (json.dumps(history), row[1] + 1, now.timestamp(), memory_id),
            )

    async def update_idx_priority(
        self, memory_id: str, idx_priority: float, tier: IndexTier
    ) -> None:
        """P7 Rebalance entry point: persist a fresh priority + tier for one memory."""

        async with self._write_lock:
            await asyncio.to_thread(
                self._update_idx_priority_sync, memory_id, idx_priority, tier
            )

    def _update_idx_priority_sync(
        self, memory_id: str, idx_priority: float, tier: IndexTier
    ) -> None:
        conn = self._require_conn()
        with conn:
            conn.execute(
                "UPDATE memory SET idx_priority = ?, index_tier = ? WHERE id = ?",
                (idx_priority, tier.value, memory_id),
            )

    async def tier_counts(self) -> dict[IndexTier, int]:
        """Return ``{tier: count}`` across every tier, including empty ones."""

        return await asyncio.to_thread(self._tier_counts_sync)

    def _tier_counts_sync(self) -> dict[IndexTier, int]:
        conn = self._require_conn()
        counts = {tier: 0 for tier in IndexTier}
        rows = conn.execute(
            "SELECT index_tier, COUNT(*) FROM memory GROUP BY index_tier"
        ).fetchall()
        for tier_name, count in rows:
            with contextlib.suppress(ValueError):
                counts[IndexTier(tier_name)] = count
        return counts

    async def iter_memory_ids(self, batch_size: int = 500) -> list[str]:
        """Return every memory id in the workspace, for batch rebalance passes."""

        return await asyncio.to_thread(self._iter_memory_ids_sync)

    def _iter_memory_ids_sync(self) -> list[str]:
        conn = self._require_conn()
        rows = conn.execute("SELECT id FROM memory").fetchall()
        return [r[0] for r in rows]

    # ─── reads ────────────────────────────────────────────────────────

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await asyncio.to_thread(self._get_memory_sync, memory_id)

    def _get_memory_sync(self, memory_id: str) -> Memory | None:
        conn = self._require_conn()
        row = conn.execute(
            "SELECT * FROM memory WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_memory(dict(zip(self._memory_columns, row, strict=True)))

    async def is_pinned(self, memory_id: str, agent_id: str | None) -> bool:
        return await asyncio.to_thread(self._is_pinned_sync, memory_id, agent_id)

    async def pinned_any(self, memory_ids: Iterable[str]) -> set[str]:
        """Return the subset of ``memory_ids`` pinned by *any* agent.

        Used by P7 Rebalance: ``idx_priority`` is a memory-wide property,
        so a per-agent pin counts for the memory's tier decision.
        """

        return await asyncio.to_thread(self._pinned_any_sync, list(memory_ids))

    def _pinned_any_sync(self, memory_ids: list[str]) -> set[str]:
        if not memory_ids:
            return set()
        conn = self._require_conn()
        placeholders = ",".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"SELECT DISTINCT memory_id FROM pin WHERE memory_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        return {r[0] for r in rows}

    def _is_pinned_sync(self, memory_id: str, agent_id: str | None) -> bool:
        conn = self._require_conn()
        if agent_id is None:
            row = conn.execute(
                "SELECT 1 FROM pin WHERE memory_id = ? AND agent_id IS NULL",
                (memory_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM pin WHERE memory_id = ? AND agent_id = ?",
                (memory_id, agent_id),
            ).fetchone()
        return row is not None

    async def fan_out(self, memory_ids: Iterable[str]) -> dict[str, int]:
        return await asyncio.to_thread(self._fan_out_sync, list(memory_ids))

    def _fan_out_sync(self, memory_ids: list[str]) -> dict[str, int]:
        conn = self._require_conn()
        out = {mid: 0 for mid in memory_ids}
        if not memory_ids:
            return out
        placeholders = ",".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"SELECT src_id, COUNT(*) FROM relation WHERE src_id IN ({placeholders}) "
            "GROUP BY src_id",
            tuple(memory_ids),
        ).fetchall()
        for src_id, count in rows:
            out[src_id] = count
        return out

    async def relations_from(self, memory_ids: Iterable[str]) -> dict[str, set[str]]:
        return await asyncio.to_thread(self._relations_from_sync, list(memory_ids))

    def _relations_from_sync(self, memory_ids: list[str]) -> dict[str, set[str]]:
        conn = self._require_conn()
        out: dict[str, set[str]] = {mid: set() for mid in memory_ids}
        if not memory_ids:
            return out
        placeholders = ",".join("?" for _ in memory_ids)
        rows = conn.execute(
            f"SELECT src_id, dst_id FROM relation WHERE src_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        for src_id, dst_id in rows:
            out[src_id].add(dst_id)
        return out

    async def vec_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(memory_id, cosine_similarity)]`` sorted by similarity desc.

        Filters by agent scope: ``agent_id`` means ``WHERE memory.agent_id = id
        OR memory.agent_id IS NULL``; ``None`` means ambient-only. If
        ``tier_filter`` is provided, candidates whose ``index_tier`` is not in
        the set are dropped (cascade retrieval uses this to scan one tier at
        a time).
        """

        return await asyncio.to_thread(
            self._vec_search_sync, query_embedding, k, agent_id, tier_filter
        )

    def _vec_search_sync(
        self,
        query_embedding: np.ndarray,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None,
    ) -> list[tuple[str, float]]:
        conn = self._require_conn()
        emb = np.asarray(query_embedding, dtype=np.float32)
        if emb.shape != (self._embedding_dim,):
            raise ValueError(
                f"Query embedding shape {emb.shape} != ({self._embedding_dim},)"
            )
        # Pull more than k from vec0, then filter by agent / tier in Python.
        # sqlite-vec MATCH doesn't compose cleanly with SQL filters in one
        # statement. We over-scan so the tier filter still yields ~k hits.
        over_scan = max(k * 8, 64) if tier_filter is not None else max(k * 4, 32)
        rows = conn.execute(
            "SELECT memory_id, distance FROM memory_vec "
            "WHERE embedding MATCH ? AND k = ?",
            (_pack_vec(emb), over_scan),
        ).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return []
        allowed = self._filter_by_agent_and_tier(conn, ids, agent_id, tier_filter)
        results: list[tuple[str, float]] = []
        for mid, dist in rows:
            if mid in allowed:
                similarity = 1.0 - float(dist)
                results.append((mid, similarity))
            if len(results) >= k:
                break
        return results

    async def fts_search(
        self,
        query: str,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``[(memory_id, bm25_raw)]`` — SQLite BM25 (negative)."""

        return await asyncio.to_thread(
            self._fts_search_sync, query, k, agent_id, tier_filter
        )

    def _fts_search_sync(
        self,
        query: str,
        k: int,
        agent_id: str | None,
        tier_filter: set[IndexTier] | None,
    ) -> list[tuple[str, float]]:
        conn = self._require_conn()
        fts_query = build_trigram_query(query)
        if fts_query is None:
            return []
        over_scan = max(k * 8, 64) if tier_filter is not None else max(k * 4, 32)
        rows = conn.execute(
            "SELECT memory_id, bm25(memory_fts) FROM memory_fts "
            "WHERE memory_fts MATCH ? "
            "ORDER BY bm25(memory_fts) "
            "LIMIT ?",
            (fts_query, over_scan),
        ).fetchall()
        ids = [r[0] for r in rows]
        if not ids:
            return []
        allowed = self._filter_by_agent_and_tier(conn, ids, agent_id, tier_filter)
        return [(mid, float(score)) for mid, score in rows if mid in allowed][:k]

    async def materialize_memories(self, ids: Iterable[str]) -> list[Memory]:
        return await asyncio.to_thread(self._materialize_sync, list(ids))

    def _materialize_sync(self, ids: list[str]) -> list[Memory]:
        conn = self._require_conn()
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM memory WHERE id IN ({placeholders})", tuple(ids)
        ).fetchall()
        if not rows:
            return []
        by_id = {
            row[0]: _row_to_memory(dict(zip(self._memory_columns, row, strict=True)))
            for row in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    async def list_recent_in_session(
        self, session_id: str, limit: int
    ) -> list[str]:
        """Return recent memory IDs in a session, newest first."""

        return await asyncio.to_thread(self._recent_session_sync, session_id, limit)

    def _recent_session_sync(self, session_id: str, limit: int) -> list[str]:
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT id FROM memory WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [r[0] for r in rows]

    # ─── helpers ──────────────────────────────────────────────────────

    def _require_conn(self) -> apsw.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteBackend.open() must be called first")
        return self._conn

    def _filter_by_agent_and_tier(
        self,
        conn: apsw.Connection,
        ids: list[str],
        agent_id: str | None,
        tier_filter: set[IndexTier] | None,
    ) -> set[str]:
        placeholders = ",".join("?" for _ in ids)
        clauses = [f"id IN ({placeholders})"]
        params: list[Any] = list(ids)

        if agent_id is None:
            clauses.append("agent_id IS NULL")
        else:
            clauses.append("(agent_id = ? OR agent_id IS NULL)")
            params.append(agent_id)

        if tier_filter is not None:
            if not tier_filter:
                return set()
            tier_placeholders = ",".join("?" for _ in tier_filter)
            clauses.append(f"index_tier IN ({tier_placeholders})")
            params.extend(t.value for t in tier_filter)

        sql = f"SELECT id FROM memory WHERE {' AND '.join(clauses)}"
        rows = conn.execute(sql, tuple(params)).fetchall()
        return {r[0] for r in rows}


# ─── module-level helpers ─────────────────────────────────────────────


def _pack_vec(emb: np.ndarray) -> bytes:
    return struct.pack(f"{len(emb)}f", *emb.tolist())


def build_trigram_query(query: str) -> str | None:
    """Tokenize ``query`` into overlapping trigrams and OR-join as phrases.

    FTS5 with the trigram tokenizer silently returns nothing for queries
    shorter than 3 characters, and its default query parser treats multi-
    token input as an implicit AND — which fails when some trigrams don't
    exist in any document. We tokenize at the Python layer and OR the
    trigrams so partial matches still score.

    Returns ``None`` if the query has no usable trigrams.
    """

    stripped = _FTS_STRIP_RE.sub(" ", query).strip()
    if len(stripped) < 3:
        return None
    grams: list[str] = []
    seen: set[str] = set()
    for i in range(len(stripped) - 2):
        g = stripped[i : i + 3]
        # Skip grams that are all whitespace (trigram tokenizer wouldn't match).
        if g.strip() == "" or g in seen:
            continue
        seen.add(g)
        grams.append(g)
    if not grams:
        return None
    # Escape embedded quotes just in case (already stripped above).
    return " OR ".join(f'"{g}"' for g in grams)


def _row_to_memory(row: dict[str, Any]) -> Memory:
    access_history = [datetime.fromtimestamp(t, tz=UTC) for t in json.loads(row["access_history"])]
    last_accessed_raw = row.get("last_accessed_at")
    last_accessed = (
        datetime.fromtimestamp(last_accessed_raw, tz=UTC)
        if last_accessed_raw is not None
        else None
    )
    return Memory(
        id=row["id"],
        workspace_id=row["workspace_id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        created_at=datetime.fromtimestamp(row["created_at"], tz=UTC),
        content=row["content"],
        content_embedding=None,  # Not hydrated by default; callers query memory_vec directly
        role=row["role"],
        memory_type=MemoryType(row["memory_type"]),
        abstraction_level=row["abstraction_level"],
        access_history=access_history,
        last_accessed_at=last_accessed,
        rehearsal_count=row["rehearsal_count"],
        salience=row["salience"],
        emotional_weight=row["emotional_weight"],
        reminisced_count=row["reminisced_count"],
        index_tier=IndexTier(row["index_tier"]),
        idx_priority=row.get("idx_priority", 0.5),
        source_message_ids=json.loads(row["source_message_ids"]),
        source_context=json.loads(row["source_context"]),
    )


def _json_safe(obj: Any) -> Any:
    """Best-effort conversion of common non-JSON types."""

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return _json_safe(asdict(obj))
    return obj
