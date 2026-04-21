"""Dreaming types (Stage 4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TriggerType(str, Enum):
    """The five triggers from §2.5.

    The two "light" triggers (idle / session_end) fire often and cheap;
    the deeper ones (surprise / cognitive_load / nightly) run the
    LLM-heavy phases or, for nightly, everything.
    """

    IDLE = "idle"
    SESSION_END = "session_end"
    SURPRISE = "surprise"
    COGNITIVE_LOAD = "cognitive_load"
    NIGHTLY = "nightly"


class PhaseName(str, Enum):
    """Names of dreaming phases per §6.3.

    Stage 4 wired P1/P2/P3/P5. Stage 5 adds P4 Refine, P6 Generalize, P8
    Dispose. P7 Rebalance is already exposed standalone on
    Mnemoss.rebalance() but is also dispatched from the nightly
    trigger.
    """

    REPLAY = "replay"
    CLUSTER = "cluster"
    EXTRACT = "extract"
    REFINE = "refine"
    RELATIONS = "relations"
    GENERALIZE = "generalize"
    REBALANCE = "rebalance"
    DISPOSE = "dispose"


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
    diary_path: Path | None = None

    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def outcome(self, phase: PhaseName) -> PhaseOutcome | None:
        for outcome in self.outcomes:
            if outcome.phase is phase:
                return outcome
        return None
