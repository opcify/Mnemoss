"""Shared dataclasses for simulation traces.

The trace is the interop boundary between :mod:`demo.simulate` (which
writes one) and :mod:`demo.render_trace` (which reads one). Shaped
for JSON round-trip so traces can live on disk, get committed to the
repo, and be replayed by a static HTML player without a Python
runtime in the blog post's hosting environment.

Event kinds
-----------

``observe``
    One memory row was ingested. Populates ``role``, ``content``,
    ``memory_id``.

``recall``
    The simulation asked the backend to recall top-k for a query.
    Populates ``query`` and ``hits`` (list of hit dicts with
    ``memory_id``, ``rank``, ``score``, and optionally ``breakdown``
    from Mnemoss's ``ActivationBreakdown.to_dict()``).

``agent_response``
    The simulated agent returned a response. Populates ``role="assistant"``
    and ``content``. Always emitted AFTER the recall that produced the
    memories used to generate this response (if any).

Future kinds (not yet implemented)
----------------------------------

``clock_advance``
    Used by Scene 2 (decay/reconsolidation) to mark time jumps.
``dream``
    Fires at scene end if the scenario triggers consolidation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class HitSnapshot:
    """A recall hit, frozen at the moment the scenario observed it.

    Includes an optional ``breakdown`` block — the JSON form of
    Mnemoss's ``ActivationBreakdown`` — so the player can render the
    per-component bars without re-running Mnemoss.
    """

    memory_id: str
    rank: int
    score: float | None
    breakdown: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "rank": self.rank,
            "score": self.score,
            "breakdown": self.breakdown,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HitSnapshot:
        return cls(
            memory_id=d["memory_id"],
            rank=d["rank"],
            score=d.get("score"),
            breakdown=d.get("breakdown"),
        )


@dataclass
class TraceEvent:
    """One line of the simulation trace.

    ``t`` is seconds from the start of the scene (float). Not wall-
    clock; the player uses it to pace replay but can also scale it
    (fast-forward / slow-motion without changing the source trace).
    """

    t: float
    kind: str  # "observe" | "recall" | "agent_response"
    role: str | None = None
    content: str | None = None
    memory_id: str | None = None
    query: str | None = None
    hits: list[HitSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"t": self.t, "kind": self.kind}
        if self.role is not None:
            out["role"] = self.role
        if self.content is not None:
            out["content"] = self.content
        if self.memory_id is not None:
            out["memory_id"] = self.memory_id
        if self.query is not None:
            out["query"] = self.query
        if self.hits:
            out["hits"] = [h.to_dict() for h in self.hits]
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceEvent:
        hits = [HitSnapshot.from_dict(h) for h in d.get("hits", []) or []]
        return cls(
            t=float(d["t"]),
            kind=d["kind"],
            role=d.get("role"),
            content=d.get("content"),
            memory_id=d.get("memory_id"),
            query=d.get("query"),
            hits=hits,
        )


@dataclass
class ScriptedTurn:
    """One user turn in a scenario.

    - ``user`` — the user-visible message text
    - ``stub_response`` — the canned assistant response used when the
      scenario runs against :class:`demo.llm.StubLLM`. The real
      recording replaces this with a Gemini-generated string.
    - ``recall_before_response`` — if True, the simulation asks the
      backend to recall top-k for the user message BEFORE the agent
      responds. This is the default; set it False for "observe-only"
      turns where we want to populate memory without triggering a
      recall event yet (useful for scenes where recall should only
      fire on a specific later turn).
    """

    user: str
    stub_response: str = ""
    recall_before_response: bool = True


@dataclass
class Scenario:
    """A scripted conversation the simulation drives end-to-end."""

    name: str
    description: str
    turns: list[ScriptedTurn]
    recall_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recall_k": self.recall_k,
            "turns": [asdict(t) for t in self.turns],
        }


@dataclass
class Trace:
    """One full scenario run, ready to serialize."""

    scenario: Scenario
    backend: str  # backend_id, e.g. "mnemoss" / "static_file"
    llm: str  # label, e.g. "stub" / "gemini-2.5-flash"
    events: list[TraceEvent]
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "backend": self.backend,
            "llm": self.llm,
            "duration_seconds": self.duration_seconds,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        scen_dict = d["scenario"]
        scenario = Scenario(
            name=scen_dict["name"],
            description=scen_dict["description"],
            recall_k=int(scen_dict.get("recall_k", 5)),
            turns=[
                ScriptedTurn(
                    user=t["user"],
                    stub_response=t.get("stub_response", ""),
                    recall_before_response=t.get("recall_before_response", True),
                )
                for t in scen_dict["turns"]
            ],
        )
        return cls(
            scenario=scenario,
            backend=d["backend"],
            llm=d["llm"],
            duration_seconds=float(d.get("duration_seconds", 0.0)),
            events=[TraceEvent.from_dict(e) for e in d["events"]],
        )
