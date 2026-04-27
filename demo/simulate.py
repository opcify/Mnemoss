"""Run a scripted :class:`~demo.types.Scenario` end-to-end against any
:class:`~bench.backends.base.MemoryBackend` + :class:`~demo.llm.LLM`.

Produces a :class:`~demo.types.Trace` capturing every observe, recall,
and agent response as a timestamped event. The trace is the artifact
the blog post's HTML player replays.

Flow per scripted turn
----------------------

1. Observe the user message → emit ``observe`` event (role=user).
2. (Optional, default on) Recall top-k memories for the user message
   → emit ``recall`` event with ``hits``. If the backend exposes an
   ``explain()`` method (Mnemoss does; static_file doesn't), also
   capture the per-hit ``ActivationBreakdown`` into the trace so the
   player can render the stacked-component bar.
3. Invoke the LLM with the user message + formatted memory context →
   emit ``agent_response`` event (role=assistant).
4. Observe the assistant message → emit ``observe`` event
   (role=assistant) so the next turn's recall can see it too.

Timing model
------------

Event ``t`` fields are in seconds from the start of the scene. The
simulation advances a virtual clock by a fixed ``turn_duration`` per
scripted turn (default 2.0s), so Scene 1's 8 turns become a ~16s
trace. This is NOT wall-clock recording time; it's just the frame
budget the HTML player uses to pace replay. Fast-forward / slow-mo
is a client-side multiplier on these ``t`` values.
"""

from __future__ import annotations

import time
from typing import Any

# Auto-load .env so ``GEMINI_API_KEY`` / ``OPENAI_API_KEY`` resolve
# when someone runs ``python -m demo.simulate`` directly. Safe no-op
# if the file's missing; explicit shell exports still take precedence.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — dev extras carry python-dotenv
    pass

from bench.backends.base import MemoryBackend
from demo.llm import LLM
from demo.types import HitSnapshot, Scenario, Trace, TraceEvent

DEFAULT_TURN_DURATION = 2.0


async def _maybe_explain(
    backend: MemoryBackend, query: str, memory_id: str
) -> dict[str, float] | None:
    """Best-effort breakdown capture; returns ``None`` if the backend
    doesn't support explanation (e.g. StaticFileBackend)."""

    explain = getattr(backend, "explain", None)
    if explain is None or not callable(explain):
        return None
    try:
        out = await explain(query, memory_id)
    except Exception:  # pragma: no cover — defensive, explain is best-effort
        return None
    return out


def _format_memory_context(hits: list[HitSnapshot], observed_texts: dict[str, str]) -> list[str]:
    """Turn recall hits into human-readable context strings for the LLM.

    ``observed_texts`` maps ``memory_id → text`` from earlier observe
    events in this scene, so the LLM gets the actual memory content,
    not just the id.
    """

    out: list[str] = []
    for h in hits:
        text = observed_texts.get(h.memory_id, "")
        if text:
            out.append(text)
    return out


async def run_scenario(
    scenario: Scenario,
    backend: MemoryBackend,
    llm: LLM,
    *,
    turn_duration: float = DEFAULT_TURN_DURATION,
) -> Trace:
    """Drive ``scenario`` end-to-end. Returns a trace ready to serialize.

    The backend is NOT closed by this function — the caller manages
    its lifecycle (matches the async-context-manager pattern both
    concrete backends provide).
    """

    events: list[TraceEvent] = []
    observed_texts: dict[str, str] = {}

    started = time.perf_counter()
    t = 0.0

    for _turn_index, turn in enumerate(scenario.turns):
        # 1. Observe user message
        user_mid = await backend.observe(turn.user, ts=t)
        observed_texts[user_mid] = turn.user
        events.append(
            TraceEvent(
                t=t,
                kind="observe",
                role="user",
                content=turn.user,
                memory_id=user_mid,
            )
        )

        memory_context: list[str] = []

        # 2. Optional recall before the agent speaks
        if turn.recall_before_response:
            hits = await backend.recall(turn.user, k=scenario.recall_k)
            snapshots = [
                HitSnapshot(
                    memory_id=h.memory_id,
                    rank=h.rank,
                    score=h.score,
                    breakdown=await _maybe_explain(backend, turn.user, h.memory_id),
                )
                for h in hits
            ]
            # Recall events share the same ``t`` as the user observe
            # but fire immediately after in the event list.
            events.append(
                TraceEvent(
                    t=t,
                    kind="recall",
                    query=turn.user,
                    hits=snapshots,
                )
            )
            memory_context = _format_memory_context(snapshots, observed_texts)

        # 3. Agent response
        t += turn_duration
        response = await llm.generate(turn.user, memory_context)
        events.append(
            TraceEvent(
                t=t,
                kind="agent_response",
                role="assistant",
                content=response,
            )
        )

        # 4. Observe the assistant response so next turn can recall it.
        asst_mid = await backend.observe(response, ts=t)
        observed_texts[asst_mid] = response
        events.append(
            TraceEvent(
                t=t,
                kind="observe",
                role="assistant",
                content=response,
                memory_id=asst_mid,
            )
        )

        t += turn_duration

    duration = time.perf_counter() - started
    return Trace(
        scenario=scenario,
        backend=backend.backend_id,
        llm=llm.llm_id,
        events=events,
        duration_seconds=duration,
    )


# ─── CLI for real recording ────────────────────────────────────────


def _build_backend_cli(name: str) -> Any:
    """Construct a backend by name. Separate from the bench harness's
    ``_build_backend`` so demo/ has its own default wiring without
    importing CLI-specific flags."""

    if name == "mnemoss":
        from bench.backends.mnemoss_backend import MnemossBackend

        # Real recording uses OpenAIEmbedder by default (inside MnemossBackend).
        return MnemossBackend()
    if name == "static_file":
        from bench.backends.static_file_backend import StaticFileBackend

        return StaticFileBackend()
    raise ValueError(f"Unknown backend: {name}")


def _build_llm_cli(name: str, stub_responses: list[str]) -> LLM:
    from demo.llm import GeminiLLM, StubLLM

    if name == "gemini":
        return GeminiLLM()
    if name == "stub":
        return StubLLM(stub_responses)
    raise ValueError(f"Unknown LLM: {name}")


async def _main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from demo.scenarios import get_scenario

    parser = argparse.ArgumentParser(description="Record a Mnemoss simulation trace.")
    parser.add_argument(
        "--scene",
        required=True,
        help="Scenario name (e.g. 'scene1_preference_recall').",
    )
    parser.add_argument(
        "--backend",
        choices=["mnemoss", "static_file"],
        default="mnemoss",
    )
    parser.add_argument("--llm", choices=["gemini", "stub"], default="stub")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    scenario = get_scenario(args.scene)
    backend = _build_backend_cli(args.backend)
    llm = _build_llm_cli(args.llm, [t.stub_response for t in scenario.turns])
    try:
        trace = await run_scenario(scenario, backend, llm)
    finally:
        await backend.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace.to_dict(), indent=2) + "\n")
    print(f"recorded {len(trace.events)} events ({trace.duration_seconds:.2f}s) → {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import asyncio

    return asyncio.run(_main(argv))


if __name__ == "__main__":
    import sys

    sys.exit(main())
