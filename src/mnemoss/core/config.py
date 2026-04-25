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

SCHEMA_VERSION = 9


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0 (got {value!r})")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0 (got {value!r})")


def _require_in_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"{name} must be in [0.0, 1.0] (got {value!r})"
        )


def _require_in_cosine_range(name: str, value: float) -> None:
    """Cosine similarity is bounded in [-1, 1]. Used for thresholds
    that gate behavior on raw cosine values (e.g.
    ``reconsolidate_min_cosine``), where -1 means "always pass" and
    1 means "always fail." Real embedders typically produce cosines
    in [0, 1] post-normalization, but ``FakeEmbedder`` emits the full
    [-1, 1] range — accepting that range avoids needing a separate
    "gate disabled" flag."""

    if not (-1.0 <= value <= 1.0):
        raise ValueError(
            f"{name} must be in [-1.0, 1.0] (got {value!r})"
        )


@dataclass
class FormulaParams:
    """Parameters of the ACT-R activation formula.

    Defaults tuned against LoCoMo bulk-ingest recall AND the
    supersession benchmark (``bench/bench_stale_fact.py``), which
    exposed a fundamental tension one ``d`` can't resolve:

    - **Recall path** wants *gentle* decay. With ``d=0.5`` (textbook
      ACT-R), base_level becomes a steep function of ingest order at
      bulk ingest, so last-ingested memories dominate regardless of
      relevance — cost ~45pp recall vs cosine baseline at N=500.
    - **Storage path** (disposal, tier migration) wants *aggressive*
      decay. With ``d=0.01``, a 2-year-old unaccessed memory's B_i
      barely drops — disposal-by-decay becomes effectively dead, and
      old contradicting facts rank equally with new ones on
      supersession queries.

    Resolution (April 2026): split ``d`` into two knobs. The
    retrieval ranking uses ``d_recall`` (default 0.2 — the Pareto
    knee from ``bench/bench_d_recall_sweep.py``: +40pp supersession
    new@1 over d_recall=0.1 at only 2.1pp LoCoMo recall cost. At
    0.2 the B_i lead over a 60s-old stale fact is ~0.82, large
    enough to beat query-token-match cosine gaps in 92% of the
    benchmark's contradiction pairs, small enough that last-
    ingested distractors don't crowd gold at bulk ingest).
    Disposal and tier migration use ``d_storage`` (default 0.5 —
    classic ACT-R; old unaccessed memories genuinely decay).

    The popularity signal that *really* differentiates memories
    comes from ``access_history``: reconsolidate bumps
    ``access_history`` on every recall hit, and ``B_i = ln(Σ t^-d)``
    aggregates all hits so memories queried frequently rise above
    memories queried once. Time decay is a tiebreaker layered on
    top of usage, not the primary signal.

    The legacy ``d`` field remains as a backwards-compat fallback:
    explicit callers that constructed ``FormulaParams(d=0.5)`` still
    get d=0.5 applied to both paths. New code should set
    ``d_recall`` and ``d_storage`` directly.

    Also retuned in April 2026: ``noise_scale: 0.25 → 0.0``. Noise
    scrambled rankings when per-candidate activation differentials
    were smaller than the noise SD, which was the common case at
    bulk ingest — cost ~40pp recall. Real deployments that want
    Luce-choice-style stochastic recall opt in explicitly.
    """

    # Backwards-compat legacy field. If set to a value other than the
    # sentinel 0.01, it overrides any d_recall / d_storage that were
    # left at their default (the old behavior: one d for everything).
    # Explicit d_recall / d_storage values win over legacy d.
    # Prefer setting d_recall and d_storage directly in new code.
    d: float = 0.01
    # Retrieval-time decay — applied when computing B_i for candidate
    # ranking in recall/activation paths. ``None`` triggers the
    # default-resolution logic in ``__post_init__``: gentle 0.1 unless
    # a legacy ``d=X`` was supplied, in which case it inherits X.
    d_recall: float | None = None
    # Storage-time decay — applied when computing B_i for disposal and
    # tier-migration decisions. Same default-resolution rule as
    # ``d_recall``, but the unset default is 0.5 (classic ACT-R).
    d_storage: float | None = None
    tau: float = -1.0
    mp: float = 1.5
    noise_scale: float = 0.0
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
    # Matching-term weighted-sum constants. The matching term is
    # ``MP · [w_F · s̃_F + w_S · s̃_S]`` where
    #
    #     w_F_raw = (match_w_f_base + match_w_f_slope · idx_priority) · b_F(q)
    #     w_S_raw = (match_w_s_base − match_w_f_slope · idx_priority) / b_F(q)
    #
    # Defaults (0.02, 0.05, 0.98) are tuned for modern dense embedders
    # on conversational corpora: fresh memory + plain query gives
    # w_F ≈ 0.07, w_S ≈ 0.93. Cosine carries the discriminative load;
    # BM25 is a small tiebreaker that grows on literal queries
    # (quoted strings, IDs, version numbers push w_F toward ~0.14) and
    # shrinks on old memories (w_F → ~0.03, gist-dominant recall). On
    # LoCoMo 2024 these defaults close the gap to raw-stack-cosine to
    # within ~1pp; BM25-heavy defaults cost ~22pp on conversational data
    # where shared vocabulary is noisy.
    #
    # Raise ``match_w_f_base`` and ``match_w_f_slope`` while lowering
    # ``match_w_s_base`` for workloads where BM25 shines (literal IDs,
    # code identifiers, quoted version strings). See
    # ``docs/ROOT_CAUSE.md`` for the full calibration trail.
    match_w_f_base: float = 0.02
    match_w_f_slope: float = 0.05
    match_w_s_base: float = 0.98
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
    # Skip the FTS5 trigram scan on queries with no literal markers
    # (quotes, URLs, IDs, numbers, CamelCase, ALL-CAPS, etc.) — i.e.
    # ``compute_query_bias(query) == 1.0``. With cosine-dominant
    # matching weights (defaults), BM25 contributes ≤7% of the
    # matching score on plain queries, so skipping FTS is almost free
    # on recall and removes a linear-in-N scan that ``sqlite-vec``'s
    # ANN index can't speed up. Saves ~20-40ms per recall at N=10K.
    #
    # Default ``False`` preserves the existing hybrid behavior (BM25
    # as a tiebreaker on every query). Turn on for production
    # workloads that are overwhelmingly plain English + want the
    # latency win; keep off if the workload has lots of IDs / code
    # / quoted strings where BM25 carries real weight.
    skip_fts_when_no_literal_markers: bool = False
    # Skip cascade round-trips into tiers that are empty at recall
    # time. The tier plan is ``HOT → WARM → COLD [→ DEEP]``; on
    # bulk-ingest workloads (benchmarks, batch imports) every memory
    # lands in HOT because ``initial_idx_priority = σ(η_0) ≈ 0.73``
    # barely clears the 0.7 HOT boundary. Without rebalance the
    # other tiers stay empty — each cascade scan into them is a
    # wasted SQL round-trip that adds ~2-5ms. When True, the engine
    # queries ``tier_counts`` once per recall and drops empty tiers
    # from the plan. Safe because an empty tier cannot produce new
    # candidates, only waste a query. Default False preserves
    # behaviour for callers that rely on cascade fall-through
    # semantics even on empty tiers.
    skip_empty_tiers: bool = False
    # ─── Fast-index recall mode ───────────────────────────────────
    # Mnemoss's defining architectural bet: expensive cognition runs
    # **async** (observe, reconsolidate, dream); recall is a pure
    # index lookup. With this flag on, ``RecallEngine.recall`` skips
    # the FTS scan, tier cascade, active-set spreading, base-level
    # recomputation, and noise sampling entirely. It does:
    #
    #   1. ANN top-K on ``vec_search`` (O(log N) with hnswlib).
    #   2. Fetch the K candidates with their precomputed
    #      ``idx_priority`` (a single indexed SQL query, O(K)).
    #   3. Rank by ``fast_index_semantic_weight · cos_sim +
    #      fast_index_priority_weight · idx_priority``.
    #   4. Return top-k.
    #
    # Correctness depends on ``idx_priority`` being kept up-to-date
    # via the async paths — ``store.reconsolidate`` recomputes it on
    # every recall-hit, Dream P7 Rebalance recomputes it for the
    # whole workspace, idle triggers can recompute opportunistically.
    # At launch Mnemoss's store.reconsolidate already does this.
    #
    # Tradeoff: you give up query-dependent matching weights (BM25
    # tiebreak on literal queries) and spreading activation on
    # return. For LoCoMo-style conversational QA these contribute
    # <1pp on recall; the latency payoff is large and grows with N
    # (at N=100K the cosine-baseline scan is ~10× slower than the
    # Mnemoss ANN+priority lookup).
    #
    # Default ``False`` preserves the full ACT-R recall path. Turn
    # on when you want the scale story at large N. (Historical note:
    # this flag used to also carry implicit tuning because the full
    # ACT-R path was mis-configured by default. As of April 2026,
    # ``d`` and ``noise_scale`` are tuned to not collapse at bulk
    # ingest regardless of this flag — see the ``FormulaParams``
    # docstring.)
    use_fast_index_recall: bool = False
    fast_index_semantic_weight: float = 1.0
    fast_index_priority_weight: float = 0.0

    # Tier cascade with pure cosine — the new default recall path.
    #
    # Mnemoss's async-cognition split: Dream/Rebalance does the
    # expensive ACT-R math (B_i, spreading, salience, pin) and writes
    # ``idx_priority`` + tier; recall reads tiers and uses pure cosine.
    # Cascade scans HOT → WARM → COLD (and DEEP if asked); within each
    # tier returns top-K by cosine. No per-candidate activation
    # evaluation, no τ filter, no FTS scoring layer at recall.
    #
    # Mutually exclusive with ``use_fast_index_recall`` (which ignores
    # tiers entirely and ranks by ``cos·sem_w + pri·pri_w``). When both
    # are False the original ACT-R recall path runs (cascade + per-
    # candidate ``compute_activation``); kept as opt-in for research /
    # debugging.
    use_tier_cascade_recall: bool = True
    # Cascade early-stop threshold. If the cascade collects at least k
    # candidates whose lowest cosine clears this value, lower tiers are
    # skipped. Effectively disabled at 0.99 — real-world cosines rarely
    # reach that, so the cascade exhausts all populated tiers.
    #
    # **Default 0.99 (disabled).** Empirical finding from
    # ``bench_rebalance_lift`` on N=20K MiniLM: short-circuit at 0.5
    # caused a 4.7pp recall regression vs raw_stack because realistic
    # Rebalance can't reliably put gold in HOT. With short-circuit
    # disabled, recall matches raw_stack exactly while latency stays
    # 2× faster (per-tier ANN with tier_filter is still cheaper than
    # flat scan, even when scanning every populated tier).
    #
    # When to revisit: if the Rebalance signal improves (selective
    # reconsolidation API, query-aware classification, etc.), HOT
    # becomes a high-precision pre-filter and short-circuit at e.g.
    # 0.6-0.8 could buy back the latency without recall cost. The
    # tier-oracle bench is the canonical measurement: if its gap
    # between realistic Rebalance and oracle ceiling closes
    # significantly, lower this default.
    cascade_min_cosine: float = 0.99

    # Cosine threshold for reconsolidation. When ``reconsolidate=True``
    # is passed to ``recall()``, only memories whose query-time cosine
    # similarity clears this threshold get their ``access_history``
    # bumped. The motivation: indiscriminate reconsolidation
    # strengthens "popular distractors" (memories that happen to land
    # in top-K across many queries) just as much as actual gold
    # answers, which pollutes ``idx_priority`` and weakens Rebalance's
    # ability to put gold in HOT. With the gate, only memories that
    # look genuinely relevant to the query reinforce.
    #
    # Default ``0.7`` was selected from a sweep on N=20K LoCoMo +
    # MiniLM (the rebalance-lift bench) — values 0.5 / 0.7 / 0.8
    # produced test-phase recall@10 of 0.3737 / 0.3882 / 0.3975
    # respectively. 0.7 is the practical knee: meaningful lift over
    # ungated (+1.45pp) at modest latency cost (+7ms p50) without
    # being so aggressive that legitimate-but-loose-cosine matches
    # are excluded. See §17 of MNEMOSS_PROJECT_KNOWLEDGE.md.
    #
    # Range is [-1.0, 1.0] — cosine similarity range. Set to -1.0
    # to disable the gate entirely (every returned top-K memory
    # gets reconsolidated).
    reconsolidate_min_cosine: float = 0.7

    def __post_init__(self) -> None:
        # Resolve d_recall / d_storage defaults with backwards compat.
        # If the caller passed a legacy d=X (non-default), inherit X
        # for any d_recall / d_storage they left unset. Otherwise fall
        # back to the split defaults (0.1 / 0.5). Explicit d_recall /
        # d_storage always win.
        _SENTINEL_D = 0.01
        if self.d_recall is None:
            self.d_recall = self.d if self.d != _SENTINEL_D else 0.2
        if self.d_storage is None:
            self.d_storage = self.d if self.d != _SENTINEL_D else 0.5
        # Decay / scaling parameters must be strictly positive — zero
        # would zero out B_i or matching entirely and produce
        # degenerate rankings.
        _require_positive("d", self.d)
        _require_positive("d_recall", self.d_recall)
        _require_positive("d_storage", self.d_storage)
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
        _require_non_negative(
            "streak_reset_seconds", self.streak_reset_seconds
        )

        # Tier confidence offsets must be ordered HOT >= WARM >= COLD
        # so cascade cutoffs widen as we scan deeper tiers.
        if not (
            self.confidence_hot_offset
            >= self.confidence_warm_offset
            >= self.confidence_cold_offset
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

        # Matching-weight constants must be non-negative; at least one
        # of base/slope must be > 0 so w_F isn't identically zero (pure-
        # cosine is still legal at idx_priority=0 if slope > 0, but we
        # need SOME variation across the memory lifetime).
        _require_non_negative("match_w_f_base", self.match_w_f_base)
        _require_non_negative("match_w_f_slope", self.match_w_f_slope)
        _require_positive("match_w_s_base", self.match_w_s_base)
        _require_non_negative(
            "fast_index_semantic_weight", self.fast_index_semantic_weight
        )
        _require_non_negative(
            "fast_index_priority_weight", self.fast_index_priority_weight
        )
        if (
            self.use_fast_index_recall
            and self.fast_index_semantic_weight <= 0.0
            and self.fast_index_priority_weight <= 0.0
        ):
            raise ValueError(
                "use_fast_index_recall requires at least one of "
                "fast_index_semantic_weight or fast_index_priority_weight "
                "to be strictly positive (got "
                f"semantic={self.fast_index_semantic_weight}, "
                f"priority={self.fast_index_priority_weight})"
            )

        # Tier-cascade and fast-index are alternative read paths;
        # if the caller explicitly opts into fast-index that wins —
        # auto-disable tier cascade so the new default doesn't shadow
        # explicit opt-ins (e.g. tests/benches that predate
        # ``use_tier_cascade_recall``).
        if self.use_fast_index_recall and self.use_tier_cascade_recall:
            self.use_tier_cascade_recall = False
        _require_in_unit_interval("cascade_min_cosine", self.cascade_min_cosine)
        _require_in_cosine_range(
            "reconsolidate_min_cosine", self.reconsolidate_min_cosine
        )

        # BFS hop + candidate caps must be positive integers.
        if self.expand_hops_max <= 0:
            raise ValueError(
                f"expand_hops_max must be > 0 (got {self.expand_hops_max!r})"
            )
        if self.expand_candidates_max <= 0:
            raise ValueError(
                f"expand_candidates_max must be > 0 "
                f"(got {self.expand_candidates_max!r})"
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
    # Semantic near-duplicate deduplication at ingest time.
    #
    # When True, each new memory triggers one ANN query against
    # existing memories in the same agent scope. Any existing memory
    # with cosine ≥ ``supersede_cosine_threshold`` gets marked
    # ``superseded_by`` the new one and filtered from recall by
    # default. It's a dedup mechanism — not a contradiction detector.
    #
    # What it actually catches (at the shipped 0.85 threshold):
    # - Re-ingests: a pipeline that accidentally observes the same
    #   message twice produces a single live memory, not two.
    # - Multi-writer races: two agents observing the same user turn
    #   milliseconds apart result in one live row.
    # - Verbatim repeats: user types "I'm hungry." three times in
    #   a row; the third marks the first two superseded.
    #
    # What it does NOT catch (measured on 25 handcrafted
    # contradiction pairs; see ``bench/bench_false_positive.py``):
    # - State changes: "move to Seattle" vs "stay in Boston"
    #   (cosine 0.50) — does not trigger.
    # - Preference shifts: "love coffee" vs "quit coffee"
    #   (cosine 0.32) — does not trigger.
    # - Fact corrections: flight time change, office address move
    #   (cosines 0.52–0.64) — do not trigger at 0.85.
    #
    # These are what ``d_recall > 0`` (time-decay) handles on the
    # retrieval path — see FormulaParams docstring. The two
    # mechanisms cover different cases:
    # - ``supersede_on_observe``: zero-gap exact duplicates.
    # - ``d_recall``: wall-clock-aware supersession of older facts
    #   regardless of whether their content is cosine-close.
    #
    # **Lowering the threshold is tempting and costly.** Measured at
    # 0.50: 80% contradiction catch rate, but 24% of topic-similar
    # valid memories get incorrectly superseded. There is no
    # threshold that catches most contradictions without destroying
    # many valid memories — the failure mode is that cosine itself
    # can't distinguish "contradicting" from "sibling fact about the
    # same topic." See ``docs/ROOT_CAUSE.md`` and
    # ``reports/supersession_bench/README.md`` §5.3 for the full
    # precision/recall curve and the reasoning behind keeping 0.85.
    #
    # **On by default as of April 2026.** Empirical finding from the
    # ``bench_multi_step`` LoCoMo+supersession blended bench at
    # N=20K MiniLM: enabling supersede_on_observe at the conservative
    # 0.85 threshold lifts mnemoss recall@10 from 0.4026 (off) to
    # 0.4622 (on), a +4.17pp gain over raw_stack's 0.4205 baseline.
    # This is the first config under which mnemoss cleanly beats
    # raw_stack on recall on a realistic aged corpus.
    #
    # The mechanism does mutate state — memories with cosine ≥ 0.85
    # to a newer observation get ``superseded_by`` set and are
    # filtered from default recall. At 0.85 this only catches
    # genuine near-duplicates (verbatim repeats, multi-writer races,
    # accidental re-ingests) and the false-positive rate is < 1%
    # on the 50-pair non-contradiction bench.
    #
    # Disable per workspace by passing
    # ``EncoderParams(supersede_on_observe=False)`` to ``Mnemoss(...)``.
    supersede_on_observe: bool = True
    supersede_cosine_threshold: float = 0.85

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
                "working_memory_capacity must be > 0 "
                f"(got {self.working_memory_capacity!r})"
            )
        if self.max_memory_chars is not None and self.max_memory_chars <= 0:
            raise ValueError(
                "max_memory_chars must be > 0 or None for no split "
                f"(got {self.max_memory_chars!r})"
            )
        if not 0.0 < self.supersede_cosine_threshold <= 1.0:
            raise ValueError(
                "supersede_cosine_threshold must be in (0, 1] "
                f"(got {self.supersede_cosine_threshold!r})"
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
            raise ValueError(
                f"max_event_messages must be > 0 (got {self.max_event_messages!r})"
            )
        if self.max_event_characters <= 0:
            raise ValueError(
                f"max_event_characters must be > 0 "
                f"(got {self.max_event_characters!r})"
            )


@dataclass
class TierCapacityParams:
    """Capacity caps for the multi-tier index (HOT/WARM/COLD/DEEP).

    Replaces the older threshold-based bucketing (HOT iff
    ``idx_priority > 0.7`` etc). Threshold-based buckets degenerate at
    scale: under any realistic time-skewed workload, ``B_i`` collapses
    for older memories, ``idx_priority`` falls below 0.1 for ~99% of
    them, and HOT empties out — defeating the architecture.

    Capacity-based bucketing instead: at each Rebalance, sort memories
    by ``idx_priority`` descending and fill tiers top-down by absolute
    cap. The result is structurally bounded HOT/WARM/COLD regardless
    of corpus size or formula tuning. DEEP receives whatever doesn't
    fit. Maps directly onto cognitive structure: working memory has a
    hard cap (Miller 1956, Cowan 2001), not a threshold.

    Defaults are cognitive-realistic order-of-magnitude estimates:

    - ``hot_cap = 200`` — working memory + recently-primed long-term
    - ``warm_cap = 2000`` — easily accessible long-term
    - ``cold_cap = 20000`` — recallable with effort
    - everything else → DEEP

    At any N ≥ 22,200 the three tier caps stay constant; only DEEP
    grows. At small N the tiers fill top-down (a 500-memory workspace
    has HOT=200, WARM=300, COLD=0, DEEP=0).

    Pinned memories bypass the cap (they're forced HOT regardless of
    their idx_priority rank). Pinning thus *displaces* the lowest-
    priority non-pinned HOT entries — exactly the cognitive analogue
    of consciously holding something in working memory.
    """

    hot_cap: int = 200
    warm_cap: int = 2_000
    cold_cap: int = 20_000

    def __post_init__(self) -> None:
        for name, value in (
            ("hot_cap", self.hot_cap),
            ("warm_cap", self.warm_cap),
            ("cold_cap", self.cold_cap),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int (got {value!r})")
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (got {value!r})")


@dataclass
class StorageParams:
    """Storage-layer configuration.

    ``use_ann_index`` controls the HNSW approximate-nearest-neighbor
    index that backs ``vec_search``. When True (the default),
    Mnemoss builds an in-memory HNSW index on workspace open and uses
    it for all vector queries — O(log N) per recall vs the O(N) linear
    scan ``sqlite-vec`` does internally. If ``hnswlib`` isn't installed
    (``pip install mnemoss[ann]``) Mnemoss falls back to the linear
    scan and logs a one-line notice on open. Set to False to force the
    linear scan regardless of hnswlib availability (useful for tests
    that want exact-NN ordering or for debugging).
    """

    root: Path | None = None
    use_ann_index: bool = True

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
    tier_capacity: TierCapacityParams = field(default_factory=TierCapacityParams)

    def __post_init__(self) -> None:
        if not self.workspace or not self.workspace.strip():
            raise ValueError("workspace must be a non-empty string")
