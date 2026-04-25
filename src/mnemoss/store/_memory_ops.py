"""Sync SQL operations against the ``memory`` table and its indexes.

Covers: Memory CRUD, vec/FTS search, metadata updates (extraction,
cluster assignment, idx_priority, reconsolidation, reminisce), embeddings
retrieval, export listing, session recency.

All functions take an already-open ``apsw.Connection``. Transactions
are the caller's responsibility — these helpers execute individual
statements and let ``SQLiteBackend`` wrap groups of them in
``with conn:`` where needed.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from typing import Any, cast

import apsw
import numpy as np

from mnemoss.core.types import IndexTier, Memory
from mnemoss.store._sql_helpers import (
    UTC,
    build_trigram_query,
    dump_json_or_none,
    filter_by_agent_and_tier,
    json_safe,
    pack_vec,
    row_to_memory,
)

# ─── writes ───────────────────────────────────────────────────────


def write_memory(
    conn: apsw.Connection,
    memory: Memory,
    embedding: np.ndarray,
    embedding_dim: int,
) -> None:
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape != (embedding_dim,):
        raise ValueError(f"Embedding shape {emb.shape} != ({embedding_dim},)")
    with conn:
        conn.execute(
            """
            INSERT INTO memory (
                id, workspace_id, agent_id, session_id, created_at, content,
                role, memory_type, abstraction_level, access_history,
                last_accessed_at, rehearsal_count, salience, emotional_weight,
                reminisced_count, index_tier, idx_priority,
                extracted_gist, extracted_entities, extracted_time,
                extracted_location, extracted_participants, extraction_level,
                cluster_id, cluster_similarity, is_cluster_representative,
                derived_from, derived_to,
                source_message_ids, source_context,
                superseded_by, superseded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?,
                      ?, ?,
                      ?, ?)
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
                memory.extracted_gist,
                dump_json_or_none(memory.extracted_entities),
                memory.extracted_time.timestamp() if memory.extracted_time else None,
                memory.extracted_location,
                dump_json_or_none(memory.extracted_participants),
                memory.extraction_level,
                memory.cluster_id,
                memory.cluster_similarity,
                1 if memory.is_cluster_representative else 0,
                json.dumps(memory.derived_from),
                json.dumps(memory.derived_to),
                json.dumps(memory.source_message_ids),
                json.dumps(json_safe(memory.source_context)),
                memory.superseded_by,
                memory.superseded_at.timestamp() if memory.superseded_at else None,
            ),
        )
        conn.execute(
            "INSERT INTO memory_vec(memory_id, embedding) VALUES (?, ?)",
            (memory.id, pack_vec(emb)),
        )
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
            (memory.id, memory.content),
        )


def delete_memory_completely(conn: apsw.Connection, memory_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM memory_vec WHERE memory_id = ?", (memory_id,))
        conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
        conn.execute(
            "DELETE FROM relation WHERE src_id = ? OR dst_id = ?",
            (memory_id, memory_id),
        )
        conn.execute("DELETE FROM pin WHERE memory_id = ?", (memory_id,))


def update_idx_priority(
    conn: apsw.Connection,
    memory_id: str,
    idx_priority: float,
    tier: IndexTier,
) -> None:
    with conn:
        conn.execute(
            "UPDATE memory SET idx_priority = ?, index_tier = ? WHERE id = ?",
            (idx_priority, tier.value, memory_id),
        )


def mark_superseded(
    conn: apsw.Connection,
    old_id: str,
    new_id: str,
    at: datetime,
) -> None:
    """Mark ``old_id`` as superseded by ``new_id`` at time ``at``.

    No-op if ``old_id`` is already superseded (first writer wins — we
    don't overwrite an earlier supersession with a later one, so the
    tombstone-like chain of supersessions stays stable).
    """

    with conn:
        conn.execute(
            "UPDATE memory SET superseded_by = ?, superseded_at = ? "
            "WHERE id = ? AND superseded_by IS NULL",
            (new_id, at.timestamp(), old_id),
        )


def update_extraction(
    conn: apsw.Connection,
    memory_id: str,
    *,
    gist: str | None,
    entities: list[str] | None,
    time: datetime | None,
    location: str | None,
    participants: list[str] | None,
    level: int,
) -> None:
    with conn:
        conn.execute(
            """
            UPDATE memory SET
                extracted_gist = ?,
                extracted_entities = ?,
                extracted_time = ?,
                extracted_location = ?,
                extracted_participants = ?,
                extraction_level = ?
            WHERE id = ?
            """,
            (
                gist,
                json.dumps(entities) if entities is not None else None,
                time.timestamp() if time else None,
                location,
                json.dumps(participants) if participants is not None else None,
                level,
                memory_id,
            ),
        )


def update_cluster_assignment(
    conn: apsw.Connection,
    memory_id: str,
    cluster_id: str | None,
    similarity: float | None,
    is_representative: bool,
) -> None:
    with conn:
        conn.execute(
            "UPDATE memory SET cluster_id = ?, cluster_similarity = ?, "
            "is_cluster_representative = ? WHERE id = ?",
            (
                cluster_id,
                similarity,
                1 if is_representative else 0,
                memory_id,
            ),
        )


def reconsolidate(conn: apsw.Connection, memory_id: str, now: datetime) -> None:
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


def reminisce_to_warm(conn: apsw.Connection, memory_id: str) -> None:
    with conn:
        conn.execute(
            "UPDATE memory SET reminisced_count = reminisced_count + 1, "
            "index_tier = ?, idx_priority = ? WHERE id = ?",
            (IndexTier.WARM.value, 0.5, memory_id),
        )


def link_derived(
    conn: apsw.Connection, parent_ids: list[str], child_id: str
) -> int:
    if not parent_ids:
        return 0
    updated = 0
    with conn:
        for parent_id in parent_ids:
            row = conn.execute(
                "SELECT derived_to FROM memory WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if row is None:
                continue
            current = json.loads(row[0] or "[]")
            if child_id in current:
                continue
            current.append(child_id)
            conn.execute(
                "UPDATE memory SET derived_to = ? WHERE id = ?",
                (json.dumps(current), parent_id),
            )
            updated += 1
    return updated


# ─── reads ────────────────────────────────────────────────────────


def get_memory(
    conn: apsw.Connection, memory_id: str, memory_columns: list[str]
) -> Memory | None:
    row = conn.execute("SELECT * FROM memory WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        return None
    return row_to_memory(dict(zip(memory_columns, row, strict=True)))


def materialize_memories(
    conn: apsw.Connection, ids: list[str], memory_columns: list[str]
) -> list[Memory]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT * FROM memory WHERE id IN ({placeholders})", tuple(ids)
    ).fetchall()
    if not rows:
        return []
    by_id = {
        row[0]: row_to_memory(dict(zip(memory_columns, row, strict=True)))
        for row in rows
    }
    return [by_id[i] for i in ids if i in by_id]


def list_memories_for_export(
    conn: apsw.Connection,
    memory_columns: list[str],
    agent_id: str | None,
    min_idx_priority: float,
    limit: int,
) -> list[Memory]:
    if agent_id is None:
        sql = (
            "SELECT * FROM memory "
            "WHERE agent_id IS NULL AND idx_priority >= ? "
            "ORDER BY idx_priority DESC LIMIT ?"
        )
        rows = conn.execute(sql, (min_idx_priority, limit)).fetchall()
    else:
        sql = (
            "SELECT * FROM memory "
            "WHERE (agent_id = ? OR agent_id IS NULL) "
            "AND idx_priority >= ? "
            "ORDER BY idx_priority DESC LIMIT ?"
        )
        rows = conn.execute(sql, (agent_id, min_idx_priority, limit)).fetchall()
    if not rows:
        return []
    return [row_to_memory(dict(zip(memory_columns, row, strict=True))) for row in rows]


def iter_memory_ids(conn: apsw.Connection) -> list[str]:
    rows = conn.execute("SELECT id FROM memory").fetchall()
    return [cast(str, r[0]) for r in rows]


def list_recent_in_session(
    conn: apsw.Connection, session_id: str, limit: int
) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM memory WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [cast(str, r[0]) for r in rows]


def get_embeddings(
    conn: apsw.Connection, memory_ids: list[str]
) -> dict[str, np.ndarray]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT memory_id, embedding FROM memory_vec WHERE memory_id IN ({placeholders})",
        tuple(memory_ids),
    ).fetchall()
    out: dict[str, np.ndarray] = {}
    for mid_raw, blob in rows:
        mid = cast(str, mid_raw)
        out[mid] = np.frombuffer(cast(bytes, blob), dtype=np.float32).copy()
    return out


def tier_counts(conn: apsw.Connection) -> dict[IndexTier, int]:
    counts = {tier: 0 for tier in IndexTier}
    rows = conn.execute(
        "SELECT index_tier, COUNT(*) FROM memory GROUP BY index_tier"
    ).fetchall()
    for tier_name, count in rows:
        with contextlib.suppress(ValueError):
            counts[IndexTier(cast(str, tier_name))] = cast(int, count)
    return counts


def get_idx_priorities(
    conn: apsw.Connection,
    memory_ids: list[str],
    agent_id: str | None,
) -> dict[str, float]:
    """Return ``{memory_id: idx_priority}`` for ids passing scope filter.

    Used by the fast-index recall path — it wants just the cached
    ``idx_priority`` scalar per candidate, filtered by agent scope, in
    one round-trip. Ids that fail scope (or don't exist) are omitted
    from the result rather than raising.
    """

    if not memory_ids:
        return {}
    placeholders = ",".join("?" for _ in memory_ids)
    clauses = [f"id IN ({placeholders})"]
    params: list[Any] = list(memory_ids)
    if agent_id is None:
        clauses.append("agent_id IS NULL")
    else:
        clauses.append("(agent_id = ? OR agent_id IS NULL)")
        params.append(agent_id)
    # Exclude superseded memories from fast-index recall — same
    # contract as the full ACT-R path.
    clauses.append("superseded_by IS NULL")

    rows = conn.execute(
        f"SELECT id, idx_priority FROM memory WHERE {' AND '.join(clauses)}",
        tuple(params),
    ).fetchall()
    return {cast(str, mid): float(cast(float, p)) for mid, p in rows}


def cluster_size(conn: apsw.Connection, cluster_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM memory WHERE cluster_id = ?",
        (cluster_id,),
    ).fetchone()
    return int(row[0]) if row else 0


# ─── search ───────────────────────────────────────────────────────


def vec_search(
    conn: apsw.Connection,
    query_embedding: np.ndarray,
    k: int,
    agent_id: str | None,
    tier_filter: set[IndexTier] | None,
    embedding_dim: int,
) -> list[tuple[str, float]]:
    emb = np.asarray(query_embedding, dtype=np.float32)
    if emb.shape != (embedding_dim,):
        raise ValueError(f"Query embedding shape {emb.shape} != ({embedding_dim},)")
    over_scan = max(k * 8, 64) if tier_filter is not None else max(k * 4, 32)
    rows = conn.execute(
        "SELECT memory_id, distance FROM memory_vec WHERE embedding MATCH ? AND k = ?",
        (pack_vec(emb), over_scan),
    ).fetchall()
    ids = [cast(str, r[0]) for r in rows]
    if not ids:
        return []
    allowed = filter_by_agent_and_tier(conn, ids, agent_id, tier_filter)
    results: list[tuple[str, float]] = []
    for mid_raw, dist in rows:
        mid = cast(str, mid_raw)
        if mid in allowed:
            similarity = 1.0 - float(cast(float, dist))
            results.append((mid, similarity))
        if len(results) >= k:
            break
    return results


def fts_search(
    conn: apsw.Connection,
    query: str,
    k: int,
    agent_id: str | None,
    tier_filter: set[IndexTier] | None,
) -> list[tuple[str, float]]:
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
    ids = [cast(str, r[0]) for r in rows]
    if not ids:
        return []
    allowed = filter_by_agent_and_tier(conn, ids, agent_id, tier_filter)
    return [
        (cast(str, mid), float(cast(float, score)))
        for mid, score in rows
        if cast(str, mid) in allowed
    ][:k]


# Convenience for callers that only care about iteration — kept to keep
# the ``SQLiteBackend`` façade purely delegating.
__all__ = [
    "cluster_size",
    "delete_memory_completely",
    "fts_search",
    "get_embeddings",
    "get_memory",
    "iter_memory_ids",
    "link_derived",
    "list_memories_for_export",
    "list_recent_in_session",
    "materialize_memories",
    "reconsolidate",
    "reminisce_to_warm",
    "tier_counts",
    "update_cluster_assignment",
    "update_extraction",
    "update_idx_priority",
    "vec_search",
    "write_memory",
    "UTC",  # re-exported for backend orchestration
]
