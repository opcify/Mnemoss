"""Configuration dataclasses for Mnemoss.

Each dataclass validates its fields in ``__post_init__`` so invalid
values raise ``ValueError`` at construction time with a clear
message. Silent acceptance of e.g. ``FormulaParams(d=-1)`` or
``EncoderParams(working_memory_capacity=0)`` would cause subtle
wrong-answer bugs much later, when a caller tries to recall and
gets nonsense.

Validators stay intentionally lenient: we check for obvious
impossibility (negatives where they can't be negative, zero where
divide-by-zero would follow, upper bounds where ``[0,1]`` is the
only sensible domain). Fine tuning is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 8


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value!r})")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0 (got {value!r})")


def _require_in_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0] (got {value!r})")


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

    def __post_init__(self) -> None:
        # Decay / scaling parameters must be strictly positive — zero
        # would zero out B_i or matching entirely and produce
        # degenerate rankings.
        _require_positive("d", self.d)
        _require_positive("mp", self.mp)
        _require_positive("s_max", self.s_max)
        _require_positive("t_floor_seconds", self.t_floor_seconds)
        _require_positive("eta_tau_seconds", self.eta_tau_seconds)

        # Non-negative scalars — zero means "feature off" which is
        # legal, negative means misconfiguration.
        _require_non_negative("noise_scale", self.noise_scale)
        _require_non_negative("alpha", self.alpha)
        _require_non_negative("beta", self.beta)
        _require_non_negative("gamma", self.gamma)
        _require_non_negative("delta", self.delta)
        _require_non_negative("eta_0", self.eta_0)
        _require_non_negative("epsilon_max", self.epsilon_max)
        _require_non_negative("streak_reset_seconds", self.streak_reset_seconds)

        # Tier confidence offsets must be ordered HOT >= WARM >= COLD
        # so cascade cutoffs widen as we scan deeper tiers.
        if not (
            self.confidence_hot_offset >= self.confidence_warm_offset >= self.confidence_cold_offset
        ):
            raise ValueError(
                "tier confidence offsets must satisfy "
                "hot >= warm >= cold (got "
                f"hot={self.confidence_hot_offset}, "
                f"warm={self.confidence_warm_offset}, "
                f"cold={self.confidence_cold_offset})"
            )

        # Cosine similarity threshold is a [0,1] quantity after
        # renormalization — passing 2.0 would mean "never match."
        _require_in_unit_interval("same_topic_cosine", self.same_topic_cosine)

        # BFS hop + candidate caps must be positive integers.
        if self.expand_hops_max <= 0:
            raise ValueError(f"expand_hops_max must be > 0 (got {self.expand_hops_max!r})")
        if self.expand_candidates_max <= 0:
            raise ValueError(
                f"expand_candidates_max must be > 0 (got {self.expand_candidates_max!r})"
            )


@dataclass
class EncoderParams:
    """Encoder configuration.

    ``encoded_roles`` controls which Raw Log roles produce Memory rows. The
    Raw Log itself is unfiltered — see Principle 3.

    ``max_memory_chars`` is an optional soft cap on Memory ``content``
    length. When a single observe exceeds the cap the encoder splits
    the content at the nearest paragraph / line / sentence boundary and
    emits multiple Memory rows; the Raw Log still sees one row. The
    split avoids silent embedder truncation (MiniLM drops tokens past
    ~512) and keeps Dream P3 cluster prompts bounded. ``None`` = no
    split (backward-compatible default). A sensible explicit value for
    LocalEmbedder deployments is ``2000``; for OpenAI's
    ``text-embedding-3-small`` it's ``30000``.
    """

    encoded_roles: set[str] = field(
        default_factory=lambda: {"user", "assistant", "tool_call", "tool_result"}
    )
    session_cooccurrence_window: int = 5
    working_memory_capacity: int = 10
    max_memory_chars: int | None = None

    def __post_init__(self) -> None:
        if not self.encoded_roles:
            raise ValueError(
                "encoded_roles must be non-empty — an encoder that "
                "rejects every role would produce zero memories."
            )
        if self.session_cooccurrence_window < 0:
            raise ValueError(
                "session_cooccurrence_window must be >= 0 "
                f"(got {self.session_cooccurrence_window!r})"
            )
        if self.working_memory_capacity <= 0:
            raise ValueError(
                f"working_memory_capacity must be > 0 (got {self.working_memory_capacity!r})"
            )
        if self.max_memory_chars is not None and self.max_memory_chars <= 0:
            raise ValueError(
                f"max_memory_chars must be > 0 or None for no split (got {self.max_memory_chars!r})"
            )


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

    def __post_init__(self) -> None:
        _require_positive("time_gap_seconds", self.time_gap_seconds)
        if self.max_event_messages <= 0:
            raise ValueError(f"max_event_messages must be > 0 (got {self.max_event_messages!r})")
        if self.max_event_characters <= 0:
            raise ValueError(
                f"max_event_characters must be > 0 (got {self.max_event_characters!r})"
            )


@dataclass
class StorageParams:
    """Storage-layer configuration."""

    root: Path | None = None

    def resolve_root(self) -> Path:
        return self.root if self.root is not None else Path.home() / ".mnemoss"


@dataclass
class DreamerParams:
    """Knobs for the dream pipeline that the harness needs to pin.

    Mirrors the ``FormulaParams`` / ``EncoderParams`` / ``SegmentationParams``
    pattern. Plumbed through ``Mnemoss(dreamer=...)`` →
    ``Mnemoss.dream()`` → ``DreamRunner(replay_limit=...,
    cluster_min_size=..., replay_min_base_level=...)``. Defaults match
    the historical hardcoded values in ``DreamRunner.__init__``.

    Note: there is no ``cluster_cosine_threshold`` knob — Mnemoss uses
    HDBSCAN (``dream/cluster.py``) which only exposes ``min_cluster_size``.
    """

    cluster_min_size: int = 3
    replay_limit: int = 100
    replay_min_base_level: float | None = None

    def __post_init__(self) -> None:
        if self.cluster_min_size <= 0:
            raise ValueError(f"cluster_min_size must be > 0 (got {self.cluster_min_size!r})")
        if self.replay_limit <= 0:
            raise ValueError(f"replay_limit must be > 0 (got {self.replay_limit!r})")
        # replay_min_base_level is None (no floor) or any float — negative
        # floors are legitimate ("only memories above near-dead activation").


@dataclass
class MnemossConfig:
    """Top-level config bundle passed to ``Mnemoss(...)``."""

    workspace: str
    formula: FormulaParams = field(default_factory=FormulaParams)
    encoder: EncoderParams = field(default_factory=EncoderParams)
    storage: StorageParams = field(default_factory=StorageParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    dreamer: DreamerParams = field(default_factory=DreamerParams)

    def __post_init__(self) -> None:
        if not self.workspace or not self.workspace.strip():
            raise ValueError("workspace must be a non-empty string")
