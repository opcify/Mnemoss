"""P3 Consolidate — one LLM call per cluster does extract + refine + pattern.

Replaces the former P3/P4/P6 trio. Inside a single cluster the LLM sees:

- Every member's content + role + its current heuristic extraction
  (``extracted_gist``, ``entities``, ``time``, ``location``,
  ``participants``).

And returns, in one structured JSON response:

- ``summary``       — a new higher-abstraction Memory to persist
                      (formerly P3 Extract).
- ``refinements``   — upgraded ``extracted_*`` fields for each cluster
                      member, bumping ``extraction_level`` to 2
                      (formerly P4 Refine).
- ``patterns``      — zero or more cross-episode patterns *within* this
                      cluster (formerly P6 Generalize, minus cross-
                      cluster discovery).

Why one call beats three:

- Sleep replay IS one integrated process in the cognitive-science story;
  three separate "look at the same cluster for different reasons" calls
  were an implementation artifact, not a theoretical commitment.
- Cost: from (N_clusters + M_refine_batch + 1) calls per nightly down
  to ~N_clusters.
- Coherence: the LLM sees everything about the cluster at once, so its
  summary, refinements, and patterns tell a single consistent story.

Trade-offs this bakes in:

- Cross-cluster pattern discovery is gone. The former P6 scanned across
  every extracted fact at the end of a run; Consolidate operates one
  cluster at a time. If cross-cluster signal matters it can be re-added
  as a small run-level synthesis call, but in practice clusters are
  already the semantic boundary — patterns that truly span them are rare.
- Non-clustered high-B_i memories no longer get refined. They'll cluster
  in a future run when neighbours arrive; isolating a singleton and
  asking the LLM to refine it has no cluster signal to draw on anyway.
- One call, one failure. A malformed response loses summary +
  refinements + patterns for that cluster. LLM failures still never
  kill the dream run — the phase records the skip and moves on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import ulid
from dateutil import parser as dateutil_parser

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory, MemoryType
from mnemoss.encoder.extraction import ExtractionFields
from mnemoss.formula.idx_priority import idx_priority_to_tier, initial_idx_priority
from mnemoss.llm.client import LLMClient

UTC = timezone.utc
log = logging.getLogger(__name__)

PATTERN_ABSTRACTION_LEVEL = 0.85

# Bonus added to the cluster's max idx_priority when promoting a summary
# or pattern memory above its source members. Small positive offset so
# the derived memory leads recall on topical queries; capped to 1.0 in
# the call sites so summaries-of-summaries don't drift to absolute top.
# 2026-04-27 — see docs/dreaming-decision.md.
SUMMARY_PRIORITY_BONUS = 0.05


@dataclass
class Refinement:
    """An upgraded extraction for one cluster member."""

    member_index: int  # 0-indexed into the cluster_members list passed in
    fields: ExtractionFields


@dataclass
class ConsolidationResult:
    """What one Consolidate call produced for one cluster.

    Any of the four outputs can be empty/None independently — the LLM
    may emit a summary with no patterns, atomic-facts only when nothing
    rises to a cluster-level summary, refine-only when there's nothing
    worth distilling, etc.

    ``atomic_facts`` was added 2026-05-09 after the LongMemEval-S pilot
    showed mem0 winning four questions Mnemoss missed (single-session
    Target lookup, Sony preference, multi-session model-kit list,
    knowledge-update 5K time) — all because mem0 surfaces atomic
    propositional facts as discrete memories. Mnemoss's narrative
    summary collapses too much into one row to surface those. Atomic
    facts coexist with the summary; the summary is "what was the
    cluster about" and atomic facts are "what discrete things were
    asserted."
    """

    summary: Memory | None = None
    refinements: list[Refinement] = field(default_factory=list)
    patterns: list[Memory] = field(default_factory=list)
    atomic_facts: list[Memory] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            self.summary is None
            and not self.refinements
            and not self.patterns
            and not self.atomic_facts
        )


def build_consolidate_prompt(cluster_members: list[Memory]) -> str:
    """Render the one-call consolidation prompt.

    Refinements cover ``gist`` and ``time`` only — NER (entities,
    location, participants) is intentionally not requested. See
    MNEMOSS_PROJECT_KNOWLEDGE.md §9.7 for the rationale.
    """

    lines = [
        "The following memories have been clustered together as part of "
        "a consolidation pass (they are semantically related):",
        "",
    ]
    for i, m in enumerate(cluster_members, 1):
        role = m.role or "note"
        stamp = m.created_at.strftime("%Y-%m-%d") if m.created_at else "????-??-??"
        lines.append(f"{i}. [{stamp} {role}] {m.content}")
        existing = {
            "gist": m.extracted_gist,
            "time": (m.extracted_time.isoformat() if m.extracted_time else None),
        }
        lines.append(f"   current extraction: {json.dumps(existing, default=str)}")

    lines.extend(
        [
            "",
            "Do four things in one pass:",
            "",
            "(A) SUMMARY — distil the cluster into ONE higher-abstraction "
            "memory (a fact or pattern). Keep the content in the same "
            "language as the input.",
            "",
            "(B) REFINEMENTS — for EACH memory above, improve or correct "
            "its gist (concise one-sentence summary) and time (ISO-8601 "
            "timestamp if the content implies one). Do not invent facts "
            "not implied by content.",
            "",
            "(C) PATTERNS — identify any recurring patterns WITHIN this "
            "cluster (behaviours, preferences, regularities spanning ≥2 "
            "members). Zero patterns is a valid answer — don't invent.",
            "",
            "(D) ATOMIC FACTS — extract zero or more standalone, "
            "self-contained propositional facts asserted in the members. "
            "Each fact must be a single complete sentence that names its "
            "subject and object explicitly so it makes sense without the "
            "surrounding conversation. Atomic facts are different from "
            "the SUMMARY: the summary is ONE narrative line about the "
            "cluster; atomic facts are INDIVIDUAL distillable "
            "propositions, possibly many per cluster.",
            "",
            "Three patterns you MUST cover when present in the cluster:",
            "",
            "  - NAMED ENTITIES: every fact mentioning a brand, store, "
            "model name/number, person, organization, or product MUST "
            "include the entity name verbatim. Wrong: 'User redeemed a "
            "coupon.' Right: 'User redeemed a $5 coffee creamer coupon "
            "at Target.' Wrong: 'User uses a Sony camera.' Right: 'User "
            "owns a Sony A7R IV camera and uses Sony-compatible "
            "accessories.'",
            "",
            "  - USER PREFERENCES: when content reveals a user "
            "preference, extract it as an explicit preference fact "
            "phrased 'User prefers X (over Y)' or 'User prefers X "
            "because Z'. Recurring usage, repeated approvals, repeated "
            "complaints, and 'I always X' all count as preferences. "
            "Wrong: 'User asked about cameras.' Right: 'User prefers "
            "Sony-compatible photography accessories over other brands.'",
            "",
            "  - KNOWLEDGE UPDATES: when the cluster contains multiple "
            "values for the same underlying fact (e.g. earlier $350K "
            "pre-approval, later $400K pre-approval; earlier 27:12 5K "
            "time, later 25:50 5K time), emit ONE atomic fact that "
            "states the CURRENT (most recent) value and explicitly "
            "marks it as such. Right: 'User Anna's current Wells Fargo "
            "mortgage pre-approval is $400,000 (raised from an earlier "
            "$350,000).' Right: 'User's current 5K personal best is "
            "25:50 (improved from 27:12).' Do NOT emit separate facts "
            "for each value — collapse them into the current-value "
            "fact.",
            "",
            "  - EVENT DATES: when a fact describes an event with a "
            "concrete date (visit, attendance, meeting, milestone, "
            "purchase), include the YYYY-MM-DD date in the fact text. "
            "Right: 'User attended the Walk for Hunger charity event "
            "on 2025-03-15.' Right: 'User visited the Metropolitan "
            "Museum of Art Ancient Civilizations exhibit on "
            "2025-04-22.' If the original member's content states "
            'the date relative to a known anchor ("last Sunday", '
            '"three weeks ago"), and you can resolve it from the '
            "[YYYY-MM-DD] prefix on the member, write the absolute "
            "date in the fact.",
            "",
            "Prefer atomic facts when the cluster contains multiple "
            "discrete claims (entities, dates, amounts, preferences). "
            "Zero is a valid answer.",
            "",
            "Respond with a single JSON object of this exact shape:",
            "{",
            '  "summary": {',
            '    "memory_type": "fact" | "pattern",',
            '    "content": "one short sentence",',
            '    "abstraction_level": 0.6',
            "  },",
            '  "refinements": [',
            "    {",
            '      "index": 1,',
            '      "gist": "concise one-sentence summary",',
            '      "time": "ISO-8601 timestamp or null"',
            "    }",
            "  ],",
            '  "patterns": [',
            "    {",
            '      "content": "pattern description, one sentence",',
            '      "derived_from": [1, 2]',
            "    }",
            "  ],",
            '  "atomic_facts": [',
            "    {",
            '      "content": "self-contained propositional fact, one sentence",',
            '      "derived_from": [1]',
            "    }",
            "  ]",
            "}",
            "",
            "Rules:",
            "- 'fact' is propositional ('user prefers dark mode').",
            "- 'pattern' is a recurring behaviour ('user commits on Fridays').",
            "- summary.abstraction_level ∈ [0.5, 0.9]: 0.5 concrete, 0.9 abstract.",
            "- refinements: preserve null on fields not implied by the content.",
            "- patterns: each must reference ≥2 members via 'derived_from'.",
            "- atomic_facts: each must reference ≥1 member via 'derived_from' "
            "and stand alone semantically (no pronouns referring to context).",
            "- All indices are 1-indexed references to the numbered list above.",
        ]
    )
    return "\n".join(lines)


def _cluster_agent_id(members: list[Memory]) -> str | None:
    """Cross-agent promotion: mixed scopes collapse to ambient (None)."""

    agents = {m.agent_id for m in members}
    return next(iter(agents)) if len(agents) == 1 else None


async def consolidate_cluster(
    cluster_members: list[Memory],
    llm: LLMClient,
    params: FormulaParams,
    *,
    now: datetime | None = None,
) -> ConsolidationResult:
    """Call the LLM once, parse the three-part response.

    Returns an empty ``ConsolidationResult`` on LLM failure or empty
    cluster. Partial output is preserved: if refinements parse but
    the summary is malformed, refinements still land.
    """

    if len(cluster_members) < 2:
        return ConsolidationResult()

    prompt = build_consolidate_prompt(cluster_members)
    try:
        response = await llm.complete_json(prompt, max_tokens=2048)
    except Exception as e:  # pragma: no cover — provider errors vary
        log.warning("consolidate_cluster: LLM failure: %s", e)
        return ConsolidationResult()

    t = now if now is not None else datetime.now(UTC)
    return ConsolidationResult(
        summary=_parse_summary(response, cluster_members, params, t),
        refinements=_parse_refinements(response, cluster_members),
        patterns=_parse_patterns(response, cluster_members, params, t),
        atomic_facts=_parse_atomic_facts(response, cluster_members, params, t),
    )


# ─── summary parse (P3 Extract equivalent) ─────────────────────────


def _parse_summary(
    response: dict[str, Any],
    members: list[Memory],
    params: FormulaParams,
    now: datetime,
) -> Memory | None:
    raw = response.get("summary")
    if not isinstance(raw, dict):
        return None

    content = (raw.get("content") or "").strip()
    if not content:
        return None

    try:
        memory_type = MemoryType(raw.get("memory_type", "fact"))
    except ValueError:
        memory_type = MemoryType.FACT

    try:
        abstraction = float(raw.get("abstraction_level", 0.6))
    except (TypeError, ValueError):
        abstraction = 0.6
    abstraction = max(0.0, min(1.0, abstraction))

    session_id: str | None = None
    for m in members:
        if m.session_id:
            session_id = m.session_id
            break

    salience = max((m.salience for m in members), default=0.0)

    # 2026-04-27: lift summary above cluster's idx_priority + inherit
    # cluster activation history. Two coupled changes:
    #
    # (1) idx_priority is at least max(initial_priority, cluster_max +
    #     SUMMARY_PRIORITY_BONUS), capped at 1.0. The summary is the
    #     synthesized answer for the cluster's topic; it should rank
    #     above any individual member on direct topical recall.
    # (2) access_history is the union of every member's access_history
    #     plus `now`. This gives the summary's B_i the cluster's
    #     collective rehearsal signal — without it, the summary's B_i
    #     would collapse to age-decay alone (length-1 history) and the
    #     idx_priority bump in (1) would erode within an hour as eta
    #     decays.
    #
    # See docs/dreaming-decision.md "Final validation" for the
    # motivating finding (forgetting curves showed no rehearsal).
    member_max_priority = max((m.idx_priority for m in members), default=0.0)
    ip = min(1.0, max(initial_idx_priority(params), member_max_priority + SUMMARY_PRIORITY_BONUS))
    inherited_history = sorted({h for m in members for h in m.access_history} | {now})

    source_ctx: dict[str, Any] = {
        "extracted_by": "dream_consolidate",
        "cluster_size": len(members),
        "inherited_history_len": len(inherited_history),
        "member_max_priority": round(member_max_priority, 4),
    }

    return Memory(
        id=str(ulid.new()),
        workspace_id=members[0].workspace_id,
        agent_id=_cluster_agent_id(members),
        session_id=session_id,
        created_at=now,
        content=content,
        content_embedding=None,
        role=None,
        memory_type=memory_type,
        abstraction_level=abstraction,
        access_history=inherited_history,
        last_accessed_at=now,
        salience=salience,
        emotional_weight=0.0,
        reminisced_count=0,
        idx_priority=ip,
        index_tier=idx_priority_to_tier(ip),
        derived_from=[m.id for m in members],
        derived_to=[],
        source_message_ids=[],
        source_context=source_ctx,
    )


# ─── refinements parse (P4 Refine equivalent) ──────────────────────


def _parse_refinements(
    response: dict[str, Any],
    members: list[Memory],
) -> list[Refinement]:
    raw = response.get("refinements", [])
    if not isinstance(raw, list):
        return []

    out: list[Refinement] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        index_raw = entry.get("index")
        if not isinstance(index_raw, int):
            continue
        idx0 = index_raw - 1  # convert 1-indexed LLM ref to 0-indexed
        if not (0 <= idx0 < len(members)):
            continue

        fields = ExtractionFields(
            gist=_norm_str(entry.get("gist")),
            entities=None,
            time=_parse_time(entry.get("time")),
            location=None,
            participants=None,
            level=2,
        )
        out.append(Refinement(member_index=idx0, fields=fields))
    return out


# ─── patterns parse (P6 Generalize equivalent, intra-cluster) ──────


def _parse_patterns(
    response: dict[str, Any],
    members: list[Memory],
    params: FormulaParams,
    now: datetime,
) -> list[Memory]:
    raw = response.get("patterns", [])
    if not isinstance(raw, list):
        return []

    out: list[Memory] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        refs = entry.get("derived_from", [])
        if not isinstance(refs, list):
            continue

        sources: list[Memory] = []
        for ref in refs:
            if not isinstance(ref, int):
                continue
            idx0 = ref - 1
            if 0 <= idx0 < len(members):
                sources.append(members[idx0])
        if len(sources) < 2:
            continue

        # Same idx_priority + access_history inheritance as summaries
        # (see _parse_summary). Patterns inherit only from their cited
        # `sources` (which can be a strict subset of members), not from
        # the entire cluster — patterns reference ≥2 specific members.
        sources_max_priority = max((s.idx_priority for s in sources), default=0.0)
        ip = min(
            1.0,
            max(initial_idx_priority(params), sources_max_priority + SUMMARY_PRIORITY_BONUS),
        )
        inherited_history = sorted({h for s in sources for h in s.access_history} | {now})

        out.append(
            Memory(
                id=str(ulid.new()),
                workspace_id=sources[0].workspace_id,
                agent_id=_cluster_agent_id(sources),
                session_id=None,
                created_at=now,
                content=content,
                content_embedding=None,
                role=None,
                memory_type=MemoryType.PATTERN,
                abstraction_level=PATTERN_ABSTRACTION_LEVEL,
                access_history=inherited_history,
                last_accessed_at=now,
                salience=max((s.salience for s in sources), default=0.0),
                idx_priority=ip,
                index_tier=idx_priority_to_tier(ip),
                derived_from=[s.id for s in sources],
                derived_to=[],
                source_message_ids=[],
                source_context={
                    "extracted_by": "dream_consolidate",
                    "scope": "intra_cluster_pattern",
                    "source_fact_count": len(sources),
                },
            )
        )
    return out


# ─── atomic facts parse ────────────────────────────────────────────

ATOMIC_FACT_ABSTRACTION_LEVEL = 0.4


def build_singleton_extraction_prompt(memory: Memory) -> str:
    """Render the per-memory atomic-fact extraction prompt.

    HDBSCAN labels memories with no semantic neighbours as noise
    (``cluster_id=None``). Cluster-level Consolidate skips them — but
    they often carry the most distinctive facts (the one turn where
    "Target" or "$400K mortgage" or "Wells Fargo" was named, with no
    nearby cluster of related turns to anchor the LLM's grouping).
    The LongMemEval-S 51a45a95 (Target store entity) miss is an
    instance of this gap: the relevant turn never reached the LLM
    because its embedding sat in HDBSCAN noise space.

    This prompt is tighter than the cluster prompt — single memory,
    no refinements, no patterns, no summary, just atomic facts. Same
    output shape as the cluster prompt's ``atomic_facts`` field so
    parsing reuses ``_parse_atomic_facts`` semantics.
    """

    role = memory.role or "note"
    stamp = memory.created_at.strftime("%Y-%m-%d") if memory.created_at else "????-??-??"
    return "\n".join(
        [
            "The following is a single conversational memory that did "
            "not cluster with any other memories. Extract zero or more "
            "standalone, self-contained propositional facts asserted "
            "in it. The [YYYY-MM-DD] prefix is the date the memory "
            "was created — use it to resolve relative time expressions "
            'in the content ("last Sunday", "three weeks ago") to '
            "absolute dates.",
            "",
            f"[{stamp} {role}] {memory.content}",
            "",
            "Extract any of:",
            "",
            "  - NAMED ENTITIES — facts mentioning a brand, store, "
            "model name, person, organization, or product MUST include "
            "the entity name verbatim. Right: 'User redeemed a $5 "
            "coffee creamer coupon at Target.'",
            "",
            "  - USER PREFERENCES — content revealing a preference, "
            "phrased 'User prefers X (over Y)' or 'User prefers X "
            "because Z'.",
            "",
            "  - COMMITTED FACTS — concrete claims with dates, amounts, "
            "names ('User Anna's mortgage pre-approval at Wells Fargo "
            "is $400,000.').",
            "",
            "  - EVENT DATES — when the content describes a dated event "
            "(visit, attendance, milestone, purchase), include the "
            "absolute YYYY-MM-DD date in the fact, resolving relative "
            "phrases against the [YYYY-MM-DD] prefix above. Right: "
            "'User attended the Walk for Hunger charity event on "
            "2025-03-15.'",
            "",
            "Each fact must be a single complete sentence that names "
            "its subject and object explicitly so it makes sense "
            "without the surrounding conversation. Zero is a valid "
            "answer — emit no facts when the memory carries only "
            "filler, agreement, small-talk, or repeats something "
            "trivial.",
            "",
            "Respond with a single JSON object of this exact shape:",
            "{",
            '  "atomic_facts": [',
            "    {",
            '      "content": "self-contained propositional fact, one sentence"',
            "    }",
            "  ]",
            "}",
        ]
    )


async def extract_atomic_facts_from_singleton(
    memory: Memory,
    llm: LLMClient,
    params: FormulaParams,
    *,
    now: datetime | None = None,
) -> list[Memory]:
    """Call the LLM once on a singleton and return atomic-fact Memories.

    Returns ``[]`` on LLM failure or when the LLM emits no facts
    (filler / small-talk memories). Each returned Memory has
    ``derived_from=[memory.id]`` and is otherwise shaped like the
    cluster-emitted atomic facts so downstream persistence is
    uniform.
    """

    prompt = build_singleton_extraction_prompt(memory)
    try:
        response = await llm.complete_json(prompt, max_tokens=1024)
    except Exception as e:  # pragma: no cover — provider errors vary
        log.warning("extract_atomic_facts_from_singleton: LLM failure: %s", e)
        return []

    t = now if now is not None else datetime.now(UTC)
    return _parse_atomic_facts(response, [memory], params, t)


def _parse_atomic_facts(
    response: dict[str, Any],
    members: list[Memory],
    params: FormulaParams,
    now: datetime,
) -> list[Memory]:
    """Materialise the ``atomic_facts`` list as level-2 FACT memories.

    Differs from ``_parse_patterns`` in two ways:

    - One source minimum (vs two for patterns): a single rich turn can
      assert a standalone fact ("user just got pre-approved for $400K
      mortgage at Wells Fargo"). Patterns are recurring-behaviour
      claims and need ≥2 supporting memories to be defensible; atomic
      facts only need to be self-contained.
    - Lower ``abstraction_level`` (0.4 vs 0.85): atomic facts are
      concrete propositions, not abstractions. They sit just below
      raw-turn level-1 extractions.
    """

    raw = response.get("atomic_facts", [])
    if not isinstance(raw, list):
        return []

    out: list[Memory] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        refs = entry.get("derived_from", [])
        if not isinstance(refs, list):
            continue

        sources: list[Memory] = []
        for ref in refs:
            if not isinstance(ref, int):
                continue
            idx0 = ref - 1
            if 0 <= idx0 < len(members):
                sources.append(members[idx0])
        if not sources:
            continue

        # Same idx_priority lift + access_history inheritance pattern as
        # summaries and patterns. Atomic facts deserve the lift because
        # they're the distilled answer for fact-grained queries the raw
        # turn would otherwise dominate (e.g. "what's the latest pre-
        # approval amount" should rank the fact above the conversational
        # turn that contained it).
        sources_max_priority = max((s.idx_priority for s in sources), default=0.0)
        ip = min(
            1.0,
            max(initial_idx_priority(params), sources_max_priority + SUMMARY_PRIORITY_BONUS),
        )
        inherited_history = sorted({h for s in sources for h in s.access_history} | {now})

        out.append(
            Memory(
                id=str(ulid.new()),
                workspace_id=sources[0].workspace_id,
                agent_id=_cluster_agent_id(sources),
                session_id=sources[0].session_id,
                created_at=now,
                content=content,
                content_embedding=None,
                role=None,
                memory_type=MemoryType.FACT,
                abstraction_level=ATOMIC_FACT_ABSTRACTION_LEVEL,
                access_history=inherited_history,
                last_accessed_at=now,
                salience=max((s.salience for s in sources), default=0.0),
                idx_priority=ip,
                index_tier=idx_priority_to_tier(ip),
                derived_from=[s.id for s in sources],
                derived_to=[],
                source_message_ids=[],
                source_context={
                    "extracted_by": "dream_consolidate",
                    "scope": "atomic_fact",
                    "source_member_count": len(sources),
                },
            )
        )
    return out


# ─── small parsers (mirror refine.py's originals) ──────────────────


def _norm_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt: datetime = dateutil_parser.isoparse(str(value))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
