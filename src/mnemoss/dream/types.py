"""Dreaming types (Stage 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TriggerType(str, Enum):
    """Light triggers ship in Stage 4; surprise / cognitive_load / nightly
    arrive with Stage 5's P6–P8 phases."""

    IDLE = "idle"
    SESSION_END = "session_end"
    TASK_COMPLETION = "task_completion"


class PhaseName(str, Enum):
    """Stage 4 only wires P1/P2/P3/P5.

    P4 Refine is in the spec but deferred — see CLAUDE.md D10. P6 / P7 /
    P8 arrive with Stage 5 (P7 is already exposed on Mnemoss.rebalance).
    """

    REPLAY = "replay"
    CLUSTER = "cluster"
    EXTRACT = "extract"
    RELATIONS = "relations"


@dataclass
class PhaseOutcome:
    """What one phase produced in a single dream run."""

    phase: PhaseName
    status: str  # "ok" | "skipped" | "error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamReport:
    """The full transcript of a ``Mnemoss.dream(...)`` call."""

    trigger: TriggerType
    started_at: datetime
    finished_at: datetime
    agent_id: str | None = None
    outcomes: list[PhaseOutcome] = field(default_factory=list)

    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def outcome(self, phase: PhaseName) -> PhaseOutcome | None:
        for outcome in self.outcomes:
            if outcome.phase is phase:
                return outcome
        return None
