"""Event encoder.

Stage 1: one RawMessage = one Event = one Memory row. Stage 3 adds
``encode_event`` which collapses multiple messages in the same
(agent, session, turn) buffer into one Memory.
"""

from __future__ import annotations

from datetime import datetime, timezone

import ulid

from mnemoss.core.config import EncoderParams, FormulaParams
from mnemoss.core.types import Memory, MemoryType, RawMessage
from mnemoss.formula.idx_priority import idx_priority_to_tier, initial_idx_priority

UTC = timezone.utc


def should_encode(msg: RawMessage, params: EncoderParams) -> bool:
    """Return True if ``msg.role`` is in the configured encoded roles."""

    return msg.role in params.encoded_roles


def encode_message(
    msg: RawMessage,
    now: datetime | None = None,
    formula: FormulaParams | None = None,
) -> Memory:
    """Create a Memory from a single RawMessage.

    Content is kept verbatim. The caller attaches the embedding separately
    (embedding comes from the embedder, not from a pure encode step).

    The initial ``idx_priority`` and ``index_tier`` are derived from
    ``formula`` (default ``FormulaParams()``) so fresh memories land in
    HOT without waiting for the first rebalance pass.
    """

    params = formula if formula is not None else FormulaParams()
    t = now if now is not None else datetime.now(UTC)
    ip = initial_idx_priority(params)
    return Memory(
        id=str(ulid.new()),
        workspace_id=msg.workspace_id,
        agent_id=msg.agent_id,
        session_id=msg.session_id,
        created_at=t,
        content=msg.content,
        content_embedding=None,
        role=msg.role,
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[t],
        last_accessed_at=None,
        idx_priority=ip,
        index_tier=idx_priority_to_tier(ip),
        source_message_ids=[msg.id],
    )


def encode_event(
    messages: list[RawMessage],
    memory_id: str,
    now: datetime | None = None,
    formula: FormulaParams | None = None,
) -> Memory:
    """Create a Memory from one or more RawMessages that share a turn.

    For 1-message events this matches ``encode_message`` verbatim so the
    Stage 1/2 content contract (and all the integration tests that pin
    specific content strings) keeps working. Multi-message events are
    synthesized by newline-joining the raw contents; role markers are
    intentionally omitted so they don't become FTS / embedding noise —
    the role metadata is still recoverable via ``source_message_ids``
    → Raw Log.

    ``created_at`` anchors to the first message's timestamp (when the
    event *started*), not the close time, so Base-Level decay reflects
    the event's real age.
    """

    if not messages:
        raise ValueError("encode_event needs at least one message")

    params = formula if formula is not None else FormulaParams()
    t = now if now is not None else datetime.now(UTC)
    first = messages[0]

    content = (
        first.content
        if len(messages) == 1
        else "\n".join(m.content for m in messages)
    )

    ip = initial_idx_priority(params)
    return Memory(
        id=memory_id,
        workspace_id=first.workspace_id,
        agent_id=first.agent_id,
        session_id=first.session_id,
        created_at=first.timestamp,
        content=content,
        content_embedding=None,
        role=first.role,
        memory_type=MemoryType.EPISODE,
        abstraction_level=0.0,
        access_history=[t],
        last_accessed_at=None,
        idx_priority=ip,
        index_tier=idx_priority_to_tier(ip),
        source_message_ids=[m.id for m in messages],
    )
