"""Sync SQL operations against the secondary state tables.

Covers three table families that sit alongside ``memory`` but aren't
part of a Memory row:

- ``relation`` — the directed edge graph (co_occurs, similar_to,
  derived_from). Read during spreading activation and expansion.
- ``pin`` — per-agent (and ambient) pinning so a memory stays hot.
- ``tombstone`` — disposal receipts with the pre-drop gist snapshot.

All functions take an open ``apsw.Connection``; transactions are the
caller's concern. Grouped in one module because the three tables
share the "secondary state, tiny SQL surface, read-heavy" shape.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, cast

import apsw

from mnemoss.core.types import Tombstone
from mnemoss.store._sql_helpers import UTC

# ─── relations ───────────────────────────────────────────────────


def write_relation(
    conn: apsw.Connection,
    src_id: str,
    dst_id: str,
    predicate: str,
    confidence: float,
) -> None:
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO relation VALUES (?, ?, ?, ?, ?)",
            (src_id, dst_id, predicate, confidence, datetime.now(UTC).timestamp()),
        )


def fan_out(conn: apsw.Connection, memory_ids: list[str]) -> dict[str, int]:
    out = {mid: 0 for mid in memory_ids}
    if not memory_ids:
        return out
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT src_id, COUNT(*) FROM relation WHERE src_id IN ({placeholders}) "
        "GROUP BY src_id",
        tuple(memory_ids),
    ).fetchall()
    for src_id_raw, count in rows:
        out[cast(str, src_id_raw)] = cast(int, count)
    return out


def relations_from(
    conn: apsw.Connection, memory_ids: list[str]
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {mid: set() for mid in memory_ids}
    if not memory_ids:
        return out
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT src_id, dst_id FROM relation WHERE src_id IN ({placeholders})",
        tuple(memory_ids),
    ).fetchall()
    for src_id_raw, dst_id_raw in rows:
        out[cast(str, src_id_raw)].add(cast(str, dst_id_raw))
    return out


def expand_via_relations(
    conn: apsw.Connection,
    seed_ids: list[str],
    hops: int,
    predicates: list[str] | None,
    max_candidates: int | None,
) -> set[str]:
    if not seed_ids or hops <= 0:
        return set()
    seed_set = set(seed_ids)
    seen: set[str] = set(seed_set)
    frontier: set[str] = set(seed_set)
    for _ in range(hops):
        if not frontier:
            break
        ph = ",".join("?" for _ in frontier)
        params: tuple[str, ...] = tuple(frontier)
        sql = f"SELECT dst_id FROM relation WHERE src_id IN ({ph})"
        if predicates:
            pp = ",".join("?" for _ in predicates)
            sql += f" AND predicate IN ({pp})"
            params = params + tuple(predicates)
        rows = conn.execute(sql, params).fetchall()
        next_frontier: set[str] = {cast(str, r[0]) for r in rows} - seen
        seen.update(next_frontier)
        if max_candidates is not None and len(seen - seed_set) >= max_candidates:
            break
        frontier = next_frontier
    return seen - seed_set


# ─── pins ────────────────────────────────────────────────────────


def pin(conn: apsw.Connection, memory_id: str, agent_id: str | None) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO pin VALUES (?, ?, ?)",
            (memory_id, agent_id, datetime.now(UTC).timestamp()),
        )


def is_pinned(conn: apsw.Connection, memory_id: str, agent_id: str | None) -> bool:
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


def pinned_any(conn: apsw.Connection, memory_ids: list[str]) -> set[str]:
    """Subset of ``memory_ids`` pinned by *any* agent (or ambient)."""

    if not memory_ids:
        return set()
    placeholders = ",".join("?" for _ in memory_ids)
    rows = conn.execute(
        f"SELECT DISTINCT memory_id FROM pin WHERE memory_id IN ({placeholders})",
        tuple(memory_ids),
    ).fetchall()
    return {cast(str, r[0]) for r in rows}


def pinned_by_agent(
    conn: apsw.Connection, memory_ids: list[str], agent_id: str | None
) -> set[str]:
    if not memory_ids:
        return set()
    placeholders = ",".join("?" for _ in memory_ids)
    if agent_id is None:
        sql = (
            f"SELECT DISTINCT memory_id FROM pin "
            f"WHERE memory_id IN ({placeholders}) AND agent_id IS NULL"
        )
        params: tuple[Any, ...] = tuple(memory_ids)
    else:
        sql = (
            f"SELECT DISTINCT memory_id FROM pin "
            f"WHERE memory_id IN ({placeholders}) AND agent_id = ?"
        )
        params = tuple(memory_ids) + (agent_id,)
    rows = conn.execute(sql, params).fetchall()
    return {cast(str, r[0]) for r in rows}


def pinned_ids_in_scope(
    conn: apsw.Connection, agent_id: str | None
) -> set[str]:
    if agent_id is None:
        rows = conn.execute(
            "SELECT memory_id FROM pin WHERE agent_id IS NULL"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT memory_id FROM pin WHERE agent_id = ? OR agent_id IS NULL",
            (agent_id,),
        ).fetchall()
    return {cast(str, r[0]) for r in rows}


# ─── tombstones ──────────────────────────────────────────────────


def write_tombstone(conn: apsw.Connection, t: Tombstone) -> None:
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tombstone
            (original_id, workspace_id, agent_id, dropped_at, reason,
             gist_snapshot, b_at_drop, source_message_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.original_id,
                t.workspace_id,
                t.agent_id,
                t.dropped_at.timestamp(),
                t.reason,
                t.gist_snapshot,
                t.b_at_drop,
                json.dumps(t.source_message_ids),
            ),
        )


def list_tombstones(
    conn: apsw.Connection, agent_id: str | None, limit: int
) -> list[Tombstone]:
    if agent_id is None:
        sql = (
            "SELECT * FROM tombstone WHERE agent_id IS NULL "
            "ORDER BY dropped_at DESC LIMIT ?"
        )
        rows = conn.execute(sql, (limit,)).fetchall()
    else:
        sql = (
            "SELECT * FROM tombstone "
            "WHERE agent_id = ? OR agent_id IS NULL "
            "ORDER BY dropped_at DESC LIMIT ?"
        )
        rows = conn.execute(sql, (agent_id, limit)).fetchall()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tombstone)")]
    out = []
    for row in rows:
        d = dict(zip(cols, row, strict=True))
        out.append(
            Tombstone(
                original_id=d["original_id"],
                workspace_id=d["workspace_id"],
                agent_id=d["agent_id"],
                dropped_at=datetime.fromtimestamp(d["dropped_at"], tz=UTC),
                reason=d["reason"],
                gist_snapshot=d["gist_snapshot"],
                b_at_drop=d["b_at_drop"],
                source_message_ids=json.loads(d["source_message_ids"]),
            )
        )
    return out


def count_tombstones(conn: apsw.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM tombstone").fetchone()
    return int(row[0]) if row is not None else 0
