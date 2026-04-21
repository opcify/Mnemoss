"""Relation-graph maintenance.

Stage 1 ships only ``co_occurs_in_session`` edges, written at encode time.
Richer relation types (``supersedes``, ``part_of``, ``derived_from``) are
populated by Dreaming P5 in Stage 4+.

Fan values are not cached in Stage 1 — ``SQLiteBackend.fan_out`` computes
them on demand with a single aggregate query per recall.
"""

from __future__ import annotations

from mnemoss.core.config import EncoderParams
from mnemoss.store.sqlite_backend import SQLiteBackend


async def write_cooccurrence_edges(
    store: SQLiteBackend,
    new_memory_id: str,
    session_id: str | None,
    params: EncoderParams,
) -> None:
    """Link the new memory to the most recent N memories in the same session.

    Bidirectional edges (both directions get the ``co_occurs_in_session``
    predicate) so fan-based spreading works regardless of which direction
    a recall happens to traverse.
    """

    if session_id is None:
        return
    recent = await store.list_recent_in_session(
        session_id, params.session_cooccurrence_window + 1
    )
    # The first result is the memory we just wrote; skip it.
    for mid in recent:
        if mid == new_memory_id:
            continue
        await store.write_relation(
            new_memory_id, mid, "co_occurs_in_session", confidence=0.5
        )
        await store.write_relation(
            mid, new_memory_id, "co_occurs_in_session", confidence=0.5
        )
