"""End-to-end smoke for the simulation runner.

Runs Scene 1 against both backends with ``StubLLM`` (no network). Asserts:
- Every scripted turn produces a predictable event sequence
  (observe-user → recall → agent_response → observe-assistant).
- Timestamps are monotonically non-decreasing.
- Mnemoss traces include ``breakdown`` on recall hits; StaticFileBackend
  traces do not (explain not implemented → None).
- Trace JSON round-trips through :class:`demo.types.Trace`.
"""

from __future__ import annotations

import json

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.static_file_backend import StaticFileBackend
from demo.llm import StubLLM
from demo.scenarios import SCENE1_PREFERENCE_RECALL
from demo.simulate import run_scenario
from demo.types import Trace
from mnemoss import FakeEmbedder


def _stub_from_scenario(scenario) -> StubLLM:
    return StubLLM([t.stub_response for t in scenario.turns])


# ─── event sequencing ─────────────────────────────────────────────


async def test_scene1_events_have_expected_sequence_static_file() -> None:
    """Per-turn pattern: observe(user) → recall → agent_response → observe(assistant)."""

    scenario = SCENE1_PREFERENCE_RECALL
    async with StaticFileBackend() as be:
        trace = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    # For each turn with recall_before_response=True, expect 4 events.
    expected_per_turn = sum(4 if t.recall_before_response else 3 for t in scenario.turns)
    assert len(trace.events) == expected_per_turn

    # Walk per-turn and assert the shape.
    idx = 0
    for turn in scenario.turns:
        ev = trace.events[idx]
        assert ev.kind == "observe" and ev.role == "user"
        assert ev.content == turn.user
        idx += 1
        if turn.recall_before_response:
            ev = trace.events[idx]
            assert ev.kind == "recall"
            assert ev.query == turn.user
            idx += 1
        ev = trace.events[idx]
        assert ev.kind == "agent_response" and ev.role == "assistant"
        idx += 1
        ev = trace.events[idx]
        assert ev.kind == "observe" and ev.role == "assistant"
        idx += 1


async def test_scene1_timestamps_are_monotonic_nondecreasing() -> None:
    scenario = SCENE1_PREFERENCE_RECALL
    async with StaticFileBackend() as be:
        trace = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    prev = -1.0
    for ev in trace.events:
        assert ev.t >= prev, f"timestamp went backwards at {ev}"
        prev = ev.t


# ─── backend-specific behavior ────────────────────────────────────


async def test_mnemoss_backend_populates_breakdown_on_recall_hits() -> None:
    """Mnemoss implements ``explain()`` so recall hits carry a
    per-component activation breakdown. The player uses these to draw
    the stacked bar."""

    scenario = SCENE1_PREFERENCE_RECALL
    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        trace = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    recalls = [e for e in trace.events if e.kind == "recall"]
    assert recalls, "scenario has recall-enabled turns"
    # Turn 1's recall is against an empty-ish workspace (first
    # user observe just happened), so hits may be sparse. Later
    # turns are the ones we care about — look for at least one
    # recall with a non-empty hits list and a breakdown on the top hit.
    nontrivial = [r for r in recalls if r.hits]
    assert nontrivial, "at least one recall must produce hits"
    # Every recall hit that DID come back from Mnemoss has a breakdown.
    for r in nontrivial:
        assert r.hits[0].breakdown is not None
        b = r.hits[0].breakdown
        # ActivationBreakdown.to_dict() keys (from
        # src/mnemoss/formula/activation.py): base_level, spreading,
        # matching, noise, total, idx_priority, w_f, w_s, query_bias.
        assert "base_level" in b
        assert "spreading" in b
        assert "matching" in b
        assert "noise" in b
        assert "total" in b


async def test_static_file_backend_breakdown_is_none() -> None:
    """StaticFileBackend has no ``explain()`` — breakdowns stay None."""

    scenario = SCENE1_PREFERENCE_RECALL
    async with StaticFileBackend() as be:
        trace = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    for ev in trace.events:
        if ev.kind == "recall":
            for hit in ev.hits:
                assert hit.breakdown is None


# ─── stub LLM responses ───────────────────────────────────────────


async def test_stub_llm_responses_land_in_agent_events() -> None:
    """The scripted stub responses appear verbatim as agent_response
    content, in turn order."""

    scenario = SCENE1_PREFERENCE_RECALL
    async with StaticFileBackend() as be:
        trace = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    agent_events = [e for e in trace.events if e.kind == "agent_response"]
    assert len(agent_events) == len(scenario.turns)
    for ev, turn in zip(agent_events, scenario.turns, strict=True):
        assert ev.content == turn.stub_response


# ─── trace JSON roundtrip ─────────────────────────────────────────


async def test_trace_roundtrips_through_json() -> None:
    """Write trace → JSON → read back via Trace.from_dict → equal shape."""

    scenario = SCENE1_PREFERENCE_RECALL
    async with StaticFileBackend() as be:
        original = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    as_json = json.dumps(original.to_dict())
    revived = Trace.from_dict(json.loads(as_json))

    assert revived.backend == original.backend
    assert revived.llm == original.llm
    assert len(revived.events) == len(original.events)
    for a, b in zip(original.events, revived.events, strict=True):
        assert a.t == b.t
        assert a.kind == b.kind
        assert a.role == b.role
        assert a.content == b.content
        assert a.memory_id == b.memory_id
        assert a.query == b.query
        assert len(a.hits) == len(b.hits)


# ─── scenario coverage ────────────────────────────────────────────


async def test_every_scripted_turn_produces_a_user_observe() -> None:
    """Invariant: N scripted turns → N user observes → N assistant
    observes. If this breaks, simulate.py is dropping turns."""

    scenario = SCENE1_PREFERENCE_RECALL
    async with StaticFileBackend() as be:
        trace = await run_scenario(scenario, be, _stub_from_scenario(scenario))

    user_observes = [e for e in trace.events if e.kind == "observe" and e.role == "user"]
    asst_observes = [e for e in trace.events if e.kind == "observe" and e.role == "assistant"]
    assert len(user_observes) == len(scenario.turns)
    assert len(asst_observes) == len(scenario.turns)
