"""Simplified event encoder.

Stage 1: one RawMessage = one Event = one Memory row. Proper event
segmentation (topic shift, time gap, turn completion, task done) lands
in Stage 3.
"""

from __future__ import annotations

from datetime import datetime, timezone

import ulid

from mnemoss.core.config import EncoderParams
from mnemoss.core.types import Memory, MemoryType, RawMessage

UTC = timezone.utc


def should_encode(msg: RawMessage, params: EncoderParams) -> bool:
    """Return True if ``msg.role`` is in the configured encoded roles."""

    return msg.role in params.encoded_roles


def encode_message(msg: RawMessage, now: datetime | None = None) -> Memory:
    """Create a Memory from a single RawMessage.

    Content is kept verbatim. The caller attaches the embedding separately
    (embedding comes from the embedder, not from a pure encode step).
    """

    t = now if now is not None else datetime.now(UTC)
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
        source_message_ids=[msg.id],
    )
