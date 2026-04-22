"""Configuration dataclasses for Mnemoss.

Pydantic is listed as a dependency for future validation work; Stage 1 uses
plain dataclasses because config is programmatic only (no YAML yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 8


@dataclass
class FormulaParams:
    """Parameters of the ACT-R activation formula.

    Defaults come from MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.10. Real
    deployments should recalibrate ``d``, ``tau``, and ``mp`` against
    benchmarks.
    """

    d: float = 0.5
    tau: float = -1.0
    mp: float = 1.5
    noise_scale: float = 0.25
    s_max: float = 2.0
    alpha: float = 0.5
    beta: float = 0.4
    gamma: float = 2.0
    delta: float = 1.0
    epsilon_max: float = 0.75
    t_floor_seconds: float = 1.0
    eta_0: float = 1.0
    eta_tau_seconds: float = 3600.0
    confidence_hot_offset: float = 2.0
    confidence_warm_offset: float = 1.0
    confidence_cold_offset: float = 0.0
    # Same-topic auto-expand (§recall/expand.py). Detection is purely
    # semantic: a follow-up recall is "same topic" when it either shares
    # at least one returned memory with the previous recall, or its query
    # embedding has cosine >= ``same_topic_cosine`` with the previous
    # query. Time does not gate detection — the user coming back to a
    # thread hours later still benefits from expansion.
    #
    # ``streak_reset_seconds`` only controls hop-count escalation: while
    # a same-topic streak continues within this window, the hop count
    # grows (capped at ``expand_hops_max``). After a gap longer than
    # this, the streak resets to 1 — the user is restarting the thread,
    # so expansion starts shallow again.
    same_topic_cosine: float = 0.7
    streak_reset_seconds: float = 600.0
    expand_hops_max: int = 3
    # Hard cap on how many memories the relation-graph BFS will pull in.
    # A densely co-occurring workspace can reach thousands of candidates
    # at 3 hops; each additional candidate is one activation-formula
    # evaluation downstream. The cap short-circuits BFS once we've
    # collected this many reachable ids, keeping worst-case expansion
    # latency bounded.
    expand_candidates_max: int = 200


@dataclass
class EncoderParams:
    """Encoder configuration.

    ``encoded_roles`` controls which Raw Log roles produce Memory rows. The
    Raw Log itself is unfiltered — see Principle 3.
    """

    encoded_roles: set[str] = field(
        default_factory=lambda: {"user", "assistant", "tool_call", "tool_result"}
    )
    session_cooccurrence_window: int = 5
    working_memory_capacity: int = 10


@dataclass
class SegmentationParams:
    """Rule-based event segmentation thresholds (Stage 3).

    Messages that share an explicit ``turn_id`` accumulate into one event
    until a closing rule fires: (a) a new message in the same
    (agent, session) arrives with a different ``turn_id``; (b) the buffer
    has been idle longer than ``time_gap_seconds``; (c) the buffer has
    hit ``max_event_messages`` or ``max_event_characters``.

    When a caller omits ``turn_id``, observe() auto-generates a unique
    id *and* closes the resulting 1-message event immediately so the
    Stage-1/2 "one message = one memory" contract is preserved.
    """

    time_gap_seconds: float = 60.0
    max_event_messages: int = 20
    max_event_characters: int = 8000


@dataclass
class StorageParams:
    """Storage-layer configuration."""

    root: Path | None = None

    def resolve_root(self) -> Path:
        return self.root if self.root is not None else Path.home() / ".mnemoss"


@dataclass
class MnemossConfig:
    """Top-level config bundle passed to ``Mnemoss(...)``."""

    workspace: str
    formula: FormulaParams = field(default_factory=FormulaParams)
    encoder: EncoderParams = field(default_factory=EncoderParams)
    storage: StorageParams = field(default_factory=StorageParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
