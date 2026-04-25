"""Scripted scenes for the live-recorded simulation.

Each scenario is framework-agnostic — the blog post's three framework
sections (OpenClaw / Hermes / Claude Code) embed the SAME recorded
traces, with section copy explaining that installing Mnemoss into that
framework gives you this behavior.

Scenes
------

``scene1_preference_recall`` (primary)
    8-turn conversation. User shares three preferences early
    (turns 1-3: vegan, morning meetings, Taipei timezone), has
    unrelated chat (turns 4-6), then asks something that requires
    those preferences (turn 7: team lunch). The simulation surfaces
    the preference memories at turn 7 with activation breakdowns,
    demonstrating the core "memory without an LLM deciding what to
    remember" thesis.

Scene 2 (decay + reconsolidation) is planned but not shipped in this
pass — see the launch design doc's cut list.
"""

from __future__ import annotations

from demo.types import Scenario, ScriptedTurn

SCENE1_PREFERENCE_RECALL = Scenario(
    name="scene1_preference_recall",
    description=(
        "A user chats across several turns. Early turns drop "
        "long-lived preferences (diet, meeting style, timezone). Later "
        "turns wander through unrelated small talk. The critical turn "
        "asks the agent to set up a team lunch — which should recall "
        "the earlier preferences and use them in the response. "
        "Demonstrates ACT-R recall on conversational memory."
    ),
    recall_k=5,
    turns=[
        ScriptedTurn(
            user=(
                "Quick heads-up — I've gone vegan as of this month, so "
                "please factor that into any food-related stuff going "
                "forward."
            ),
            stub_response=(
                "Noted — I'll keep that in mind for anything food-adjacent from here on out."
            ),
        ),
        ScriptedTurn(
            user=(
                "Also, I strongly prefer morning meetings over "
                "afternoon ones. Afternoons kill my focus."
            ),
            stub_response=(
                "Got it. I'll default to morning slots when I'm scheduling things for you."
            ),
        ),
        ScriptedTurn(
            user=(
                "And for context — I'm based in Taipei, so that's "
                "UTC+8. Worth mentioning for scheduling."
            ),
            stub_response=("Perfect, I'll use UTC+8 as your reference timezone."),
        ),
        ScriptedTurn(
            user=("Totally different topic — what's a good workflow for reviewing a long PR diff?"),
            stub_response=(
                "I usually split it into logical commits, read each "
                "one top-down, and leave inline comments as I go."
            ),
        ),
        ScriptedTurn(
            user=("Yeah, that helps. How about when the PR description is sparse?"),
            stub_response=(
                "Push back — ask the author for a 3-bullet 'what / "
                "why / risks' summary before reviewing."
            ),
        ),
        ScriptedTurn(
            user="Cool. Can we circle back to the Q3 roadmap?",
            stub_response=("Sure, what's the biggest open question on it right now?"),
        ),
        # THE critical turn — should recall turns 1-3.
        ScriptedTurn(
            user=("Set up a team lunch for tomorrow — find a slot that works for everyone."),
            stub_response=(
                "I'll aim for a morning-friendly slot (noon local "
                "Taipei time, UTC+8) and line up a restaurant with "
                "good vegan options. Sending the invite now."
            ),
        ),
        ScriptedTurn(
            user="Perfect. Send the invite.",
            stub_response="Invite out.",
        ),
    ],
)


_SCENARIOS: dict[str, Scenario] = {
    SCENE1_PREFERENCE_RECALL.name: SCENE1_PREFERENCE_RECALL,
}


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name. Raises ``KeyError`` with a helpful
    message if unknown."""

    if name not in _SCENARIOS:
        known = ", ".join(sorted(_SCENARIOS))
        raise KeyError(f"Unknown scenario {name!r}. Known: {known}")
    return _SCENARIOS[name]


def list_scenarios() -> list[Scenario]:
    return list(_SCENARIOS.values())
