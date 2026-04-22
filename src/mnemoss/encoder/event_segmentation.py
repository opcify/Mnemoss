"""Rule-based event segmentation (Stage 3).

Sits between the Raw Log append and the Memory write: accumulates
messages into a per-(agent, session, turn) buffer, and closes the
buffer when a rule fires. Each closed buffer becomes exactly one
Memory row.

Rules (evaluated on each new observation, in order):

1. **turn_shift** — a same-(agent, session) buffer with a different
   ``turn_id`` is closed before the new message is appended.
2. **time_gap** — any buffer whose last append is older than
   ``time_gap_seconds`` is closed.
3. **size_limit** — after appending, if the buffer has ≥
   ``max_event_messages`` messages or its content length has reached
   ``max_event_characters``, it is closed.
4. **auto_1to1** — when the caller omits ``turn_id`` (auto-generated
   unique turn), the buffer is closed inside the same observation so
   the Stage-1/2 "one observe = one memory" contract still holds.

Ancillary operations ``flush(agent_id, session_id)`` and
``flush_all()`` drain buffers on demand; these are called by
``Mnemoss.flush_session()`` and ``close()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import ulid

from mnemoss.core.config import SegmentationParams
from mnemoss.core.types import RawMessage

UTC = timezone.utc


@dataclass
class ClosedEvent:
    """Contiguous messages that are now ready to become a single Memory."""

    messages: list[RawMessage]
    memory_id: str
    closed_at: datetime
    closed_by: str  # "turn_shift" | "time_gap" | "size_limit" | "auto_1to1" | "flush"


@dataclass
class SegmentStep:
    """Result of ``EventSegmenter.on_observe``.

    ``pending_memory_id`` is the id of the Memory this message will
    ultimately belong to — whether the event is still open or was just
    closed by the same call. Callers can return this from ``observe()``
    without caring which side of the close the message landed on.

    ``closed_events`` is the list of events that became Memory-ready as
    a side effect of this observation (including this message's own
    event if a rule fired). Callers iterate ``closed_events`` to persist
    Memory rows.
    """

    pending_memory_id: str
    closed_events: list[ClosedEvent] = field(default_factory=list)


@dataclass
class _Buffer:
    memory_id: str
    messages: list[RawMessage]
    opened_at: datetime
    last_updated_at: datetime
    total_chars: int


_BufferKey = tuple[str | None, str, str]  # (agent_id, session_id, turn_id)


class EventSegmenter:
    """Owns all in-flight event buffers for a workspace."""

    def __init__(self) -> None:
        self._buffers: dict[_BufferKey, _Buffer] = {}

    # ─── public API ───────────────────────────────────────────────────

    def on_observe(
        self,
        msg: RawMessage,
        now: datetime,
        params: SegmentationParams,
        *,
        auto_close: bool = False,
    ) -> SegmentStep:
        """Route ``msg`` through the segmenter, returning its pending
        memory id and any events that closed as a side effect."""

        closed: list[ClosedEvent] = []

        # Rules 1 & 2: close stale or turn-shifted buffers before this
        # message joins the current buffer.
        for key in list(self._buffers.keys()):
            buf_agent, buf_session, buf_turn = key
            msg_key = (msg.agent_id, msg.session_id, msg.turn_id)
            if key == msg_key:
                continue  # This buffer will receive the new message.
            elapsed = (now - self._buffers[key].last_updated_at).total_seconds()
            if elapsed >= params.time_gap_seconds:
                closed.append(self._close(key, now, "time_gap"))
            elif buf_agent == msg.agent_id and buf_session == msg.session_id:
                # Same-(agent, session), different turn → shift.
                closed.append(self._close(key, now, "turn_shift"))

        # Open or reuse this message's buffer.
        msg_key = (msg.agent_id, msg.session_id, msg.turn_id)
        buf = self._buffers.get(msg_key)
        if buf is None:
            buf = _Buffer(
                memory_id=str(ulid.new()),
                messages=[],
                opened_at=now,
                last_updated_at=now,
                total_chars=0,
            )
            self._buffers[msg_key] = buf

        buf.messages.append(msg)
        buf.last_updated_at = now
        buf.total_chars += len(msg.content)
        pending_id = buf.memory_id

        # Rule 3: size cap.
        if (
            len(buf.messages) >= params.max_event_messages
            or buf.total_chars >= params.max_event_characters
        ):
            closed.append(self._close(msg_key, now, "size_limit"))
        elif auto_close:
            # Rule 4: legacy 1:1 observe (caller omitted turn_id).
            closed.append(self._close(msg_key, now, "auto_1to1"))

        return SegmentStep(pending_memory_id=pending_id, closed_events=closed)

    def flush(
        self,
        agent_id: str | None = None,
        session_id: str | None = None,
        *,
        now: datetime | None = None,
    ) -> list[ClosedEvent]:
        """Force-close buffers in the given scope.

        ``agent_id=None`` + ``session_id=None`` matches every buffer.
        A non-None filter on either dimension narrows the scope.
        Passing ``agent_id=None`` explicitly only matches ambient buffers
        (those created without an ``agent_id``) when combined with a
        non-None session; otherwise all agents are matched.
        """

        t = now if now is not None else datetime.now(UTC)
        closed: list[ClosedEvent] = []
        # Use a sentinel to distinguish "unfiltered" from "filter by None".
        for key in list(self._buffers.keys()):
            buf_agent, buf_session, _ = key
            if agent_id is not None and buf_agent != agent_id:
                continue
            if session_id is not None and buf_session != session_id:
                continue
            closed.append(self._close(key, t, "flush"))
        return closed

    def flush_all(self, *, now: datetime | None = None) -> list[ClosedEvent]:
        return self.flush(now=now)

    def pending_buffer_count(self) -> int:
        return len(self._buffers)

    # ─── internals ────────────────────────────────────────────────────

    def _close(self, key: _BufferKey, now: datetime, closed_by: str) -> ClosedEvent:
        buf = self._buffers.pop(key)
        return ClosedEvent(
            messages=buf.messages,
            memory_id=buf.memory_id,
            closed_at=now,
            closed_by=closed_by,
        )
