"""Per-agent working-memory buffers.

Populated by ``observe()`` (writer side) and by ``recall()`` on returned
top-k (reconsolidation side). The active set ``C`` here feeds the
spreading-activation term.

Keyed by ``agent_id`` with ``None`` mapped to ``"__ambient__"`` so that
ambient memories participate in ambient-scope spreading without leaking
into agent-private spreading.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable


AMBIENT_KEY = "__ambient__"


class WorkingMemory:
    def __init__(self, capacity: int = 10) -> None:
        self._capacity = capacity
        self._buffers: dict[str, deque[str]] = {}

    def _key(self, agent_id: str | None) -> str:
        return agent_id if agent_id is not None else AMBIENT_KEY

    def append(self, agent_id: str | None, memory_id: str) -> None:
        key = self._key(agent_id)
        buf = self._buffers.get(key)
        if buf is None:
            buf = deque(maxlen=self._capacity)
            self._buffers[key] = buf
        if memory_id in buf:
            buf.remove(memory_id)  # move to most-recent
        buf.append(memory_id)

    def extend(self, agent_id: str | None, memory_ids: Iterable[str]) -> None:
        for mid in memory_ids:
            self.append(agent_id, mid)

    def active_set(self, agent_id: str | None) -> list[str]:
        """Return the active set for recall scoping.

        An agent's active set is its own buffer plus the ambient buffer;
        ambient recall sees the ambient buffer only.
        """

        ambient = list(self._buffers.get(AMBIENT_KEY, ()))
        if agent_id is None:
            return ambient
        own = list(self._buffers.get(agent_id, ()))
        # De-dupe while preserving order (own first — "you just wrote it").
        seen: set[str] = set()
        merged: list[str] = []
        for mid in own + ambient:
            if mid in seen:
                continue
            seen.add(mid)
            merged.append(mid)
        return merged

    def clear(self, agent_id: str | None = None) -> None:
        if agent_id is None and AMBIENT_KEY in self._buffers:
            self._buffers[AMBIENT_KEY].clear()
        elif agent_id is not None and agent_id in self._buffers:
            self._buffers[agent_id].clear()
