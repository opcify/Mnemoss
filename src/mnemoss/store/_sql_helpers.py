"""Shared SQL helpers used across the ``*_ops`` modules.

Pure, stateless functions. All three store-op modules (``_memory_ops``,
``_graph_ops``, ``_raw_log_ops``) import from here to avoid circular
imports between them and ``sqlite_backend``.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, cast

import apsw
import numpy as np

from mnemoss.core.types import IndexTier, Memory, MemoryType

UTC = timezone.utc

# FTS query sanitization — strip FTS5 special syntax so user queries can't
# accidentally build malformed MATCH expressions.
_FTS_STRIP_RE = re.compile(r"[\"()*:!?\\\n\r\t]")


def pack_vec(emb: np.ndarray) -> bytes:
    """Pack a float32 embedding as the little-endian blob sqlite-vec expects."""

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
        if g.strip() == "" or g in seen:
            continue
        seen.add(g)
        grams.append(g)
    if not grams:
        return None
    return " OR ".join(f'"{g}"' for g in grams)


def row_to_memory(row: dict[str, Any]) -> Memory:
    """Reconstruct a Memory from a ``SELECT * FROM memory`` row dict."""

    access_history = [
        datetime.fromtimestamp(t, tz=UTC) for t in json.loads(row["access_history"])
    ]
    last_accessed_raw = row.get("last_accessed_at")
    last_accessed = (
        datetime.fromtimestamp(last_accessed_raw, tz=UTC)
        if last_accessed_raw is not None
        else None
    )
    entities_raw = row.get("extracted_entities")
    participants_raw = row.get("extracted_participants")
    time_raw = row.get("extracted_time")
    derived_from_raw = row.get("derived_from") or "[]"
    derived_to_raw = row.get("derived_to") or "[]"
    return Memory(
        id=row["id"],
        workspace_id=row["workspace_id"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        created_at=datetime.fromtimestamp(row["created_at"], tz=UTC),
        content=row["content"],
        content_embedding=None,
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
        extracted_gist=row.get("extracted_gist"),
        extracted_entities=json.loads(entities_raw) if entities_raw else None,
        extracted_time=(
            datetime.fromtimestamp(time_raw, tz=UTC) if time_raw is not None else None
        ),
        extracted_location=row.get("extracted_location"),
        extracted_participants=(json.loads(participants_raw) if participants_raw else None),
        extraction_level=row.get("extraction_level", 0),
        cluster_id=row.get("cluster_id"),
        cluster_similarity=row.get("cluster_similarity"),
        is_cluster_representative=bool(row.get("is_cluster_representative") or 0),
        derived_from=json.loads(derived_from_raw),
        derived_to=json.loads(derived_to_raw),
        source_message_ids=json.loads(row["source_message_ids"]),
        source_context=json.loads(row["source_context"]),
        superseded_by=row.get("superseded_by"),
        superseded_at=(
            datetime.fromtimestamp(row["superseded_at"], tz=UTC)
            if row.get("superseded_at") is not None
            else None
        ),
    )


def dump_json_or_none(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


def json_safe(obj: Any) -> Any:
    """Best-effort conversion of common non-JSON types before ``json.dumps``."""

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(asdict(obj))
    return obj


def filter_by_agent_and_tier(
    conn: apsw.Connection,
    ids: list[str],
    agent_id: str | None,
    tier_filter: set[IndexTier] | None,
    *,
    include_superseded: bool = False,
) -> set[str]:
    """Return the subset of ``ids`` that pass the agent + tier scope filter.

    Used by ``vec_search`` and ``fts_search``: both pull more than ``k``
    candidates from their respective indexes, then refine against the
    ``memory`` table's ``agent_id`` / ``index_tier`` columns in one SQL
    round-trip.

    By default, memories marked ``superseded_by`` (contradiction-aware
    observe has seen a newer version) are excluded from recall. Pass
    ``include_superseded=True`` for paths that need the full ledger
    (inspection CLI, Dream replay / disposal audit). The partial
    index ``idx_memory_superseded`` keeps the filter cheap when the
    feature is inactive.
    """

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

    if not include_superseded:
        clauses.append("superseded_by IS NULL")

    sql = f"SELECT id FROM memory WHERE {' AND '.join(clauses)}"
    rows = conn.execute(sql, tuple(params)).fetchall()
    return {cast(str, r[0]) for r in rows}
