"""Core data types for Mnemoss.

All domain objects live here. Schema rows use these as their Python-side
representation; the store layer handles (de)serialization. Every memory-bearing
type carries ``agent_id: str | None`` — non-null is private to that agent,
null is workspace-shared (ambient). See MNEMOSS_PROJECT_KNOWLEDGE.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class MemoryType(str, Enum):
    """Discriminator for Memory rows. Stage 1 only writes ``EPISODE``."""

    EPISODE = "episode"
    FACT = "fact"
    ENTITY = "entity"
    PATTERN = "pattern"


class IndexTier(str, Enum):
    """Physical index tier. Stage 1 only uses ``HOT``."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DEEP = "deep"


@dataclass
class Relation:
    """An edge in the memory graph, used for spreading activation."""

    predicate: str
    target_id: str
    confidence: float = 1.0
    created_at: datetime | None = None


@dataclass
class Memory:
    """A single unit of the Memory Store.

    Pin state is NOT carried on Memory rows — it lives in a separate ``pin``
    table keyed by ``(memory_id, agent_id)`` because the same ambient memory
    can be pinned by multiple agents independently.
    """

    id: str
    workspace_id: str
    agent_id: str | None
    session_id: str | None
    created_at: datetime
    content: str
    content_embedding: np.ndarray | None
    role: str | None
    memory_type: MemoryType
    abstraction_level: float
    access_history: list[datetime] = field(default_factory=list)
    last_accessed_at: datetime | None = None
    rehearsal_count: int = 0
    salience: float = 0.0
    emotional_weight: float = 0.0
    reminisced_count: int = 0
    index_tier: IndexTier = IndexTier.HOT
    idx_priority: float = 0.5
    source_message_ids: list[str] = field(default_factory=list)
    source_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawMessage:
    """A single message in the append-only Raw Log.

    Principle 3: every observed message lands here unconditionally,
    regardless of whether it produces a Memory row.
    """

    id: str
    workspace_id: str
    agent_id: str | None
    session_id: str
    turn_id: str
    parent_id: str | None
    timestamp: datetime
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """A segmented event — a group of RawMessages forming one episode.

    Stage 1 simplification: one RawMessage = one Event = one Memory.
    Full segmentation (topic shift, time gap, task completion) lands in Stage 3.
    """

    id: str
    agent_id: str | None
    session_id: str
    messages: list[RawMessage]
    started_at: datetime
    ended_at: datetime
    closed_by: str
