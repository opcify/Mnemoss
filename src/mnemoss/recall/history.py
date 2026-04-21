"""Per-agent recall history for the same-topic auto-expand trigger.

Each ``Mnemoss`` instance carries one ``RecallHistory`` shared across
every agent handle. The history is **in-memory only** — on process
restart the buffer clears and the first recall cannot be a follow-up.
This matches the cognitive intuition: same-topic detection is about
"is this the next recall in an ongoing thread?", which is short-lived
state, not persistent memory.

Detection is purely semantic (see ``is_same_topic``): result overlap
or query-embedding cosine. Time is only used by the engine to decide
whether the hop-count streak should keep escalating or reset to 1; it
does not gate detection itself.

Agent scoping: entries are keyed by ``agent_id`` (``None`` for ambient),
so Alice's queries never influence whether Bob's next recall counts as
a follow-up.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class PastQuery:
    """One entry in the per-agent recall buffer.

    ``result_ids`` stores the memory ids returned — a zero-cost way to
    detect follow-ups that happen to share results even when the query
    text rewording drops the cosine below the threshold.
    """

    query: str
    query_vec: np.ndarray
    timestamp: datetime
    result_ids: set[str]
    streak: int = 1  # Consecutive same-topic queries; seeds expand-hop count.


class RecallHistory:
    """Bounded per-agent ring buffer of recent recalls.

    Capacity of 5 per agent is plenty — same-topic detection only ever
    consults the *most recent* entry, so older ones exist only for
    debugging or future heuristics.
    """

    def __init__(self, capacity: int = 5) -> None:
        self._capacity = capacity
        self._buffers: dict[str | None, deque[PastQuery]] = {}

    def latest(self, agent_id: str | None) -> PastQuery | None:
        buf = self._buffers.get(agent_id)
        if not buf:
            return None
        return buf[-1]

    def record(self, agent_id: str | None, entry: PastQuery) -> None:
        buf = self._buffers.get(agent_id)
        if buf is None:
            buf = deque(maxlen=self._capacity)
            self._buffers[agent_id] = buf
        buf.append(entry)

    def clear(self, agent_id: str | None = None) -> None:
        """Reset the buffer for one agent, or all agents if ``None``."""

        if agent_id is None:
            self._buffers.clear()
        else:
            self._buffers.pop(agent_id, None)


def is_same_topic(
    prev: PastQuery,
    *,
    current_query_vec: np.ndarray,
    current_result_ids: set[str],
    cosine_threshold: float,
) -> bool:
    """Decide whether ``current`` continues the topic of ``prev``.

    Detection is purely semantic — no time gate. A user returning to
    the same thread hours later is still asking about the same topic,
    and expansion should still fire. Hop-count escalation, which *does*
    depend on recency, is handled separately in the engine via
    ``streak_reset_seconds``.

    Rules (any satisfied → same topic):

    - **Result overlap.** If this recall returned at least one memory
      in common with the previous, the follow-up is clearly about the
      same cluster regardless of wording.
    - **Query cosine** above ``cosine_threshold``. Different wording
      but semantically close.
    """

    if current_result_ids & prev.result_ids:
        return True
    return _cosine(current_query_vec, prev.query_vec) >= cosine_threshold


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
