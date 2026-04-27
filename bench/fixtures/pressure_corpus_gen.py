"""Synthetic accumulating-pressure corpus for Mnemoss dreaming validation.

Generates ``pressure_corpus_seed{N}.jsonl`` — ~500 memories spread
over 30 simulated days, with a 10/70/20 high/medium-utility/junk
split, plus 30 adversarial queries.

The corpus is **adversarial by design**: junk memories deliberately
share vocabulary with the queries (e.g., the queries ask about
"Phoenix release date" and the junk includes "Pixar's release date",
"Phoenix Suns released their roster", etc.). Without Dispose, junk
pollutes top-10 on most queries because the embedder sees surface
overlap. With Dispose, low-utility memories tombstone over the 30-day
accumulation and top-10 cleans up.

Determinism: ``--seed`` picks templates and slot values. The output
JSONL is committed so any corpus change is reviewable as a diff.

Usage::

    python -m bench.fixtures.pressure_corpus_gen --seed 42

The pressure corpus harness in bench/ablate_dreaming.py loads the
output and uses ``freezegun`` to inject the per-memory simulated
timestamps into Mnemoss's observe path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ─── corpus parameters (load-bearing — touch carefully) ────────────

TOTAL_MEMORIES = 500
HIGH_UTILITY_FRAC = 0.10  # 50 memories about Project Phoenix
BACKBONE_FRAC = 0.20  # 100 daily-routine memories
JUNK_FRAC = 0.70  # 350 one-shot trivia with adversarial vocab
SIMULATED_DAYS = 30
SECONDS_PER_DAY = 86400

# 30 adversarial queries: 20 want Phoenix-themed answers, 10 want
# backbone-themed answers. Negative queries (gold=empty) live in
# the topology corpus, not here.
NUM_QUERIES = 30


# ─── content templates ─────────────────────────────────────────────

# High-utility theme: Project Phoenix. Each memory mentions Phoenix
# explicitly so spreading-activation / clustering can find them.
HIGH_UTILITY_TEMPLATES = [
    "Project Phoenix v{version} release set for {weekday} the {day}",
    "Phoenix architecture review with {person} on the {component} module",
    "Sprint {n} kickoff for Project Phoenix; focus on {component}",
    "Phoenix bug fix landed in commit {sha}: {component} now handles {edge_case}",
    "Phoenix deployment plan: {weekday} morning, {component} first then {component2}",
    "Phoenix migration spec reviewed; {person} approved the {component} schema",
    "Phoenix dashboard shipped — {component} metrics now visible to ops",
    "Phoenix design doc published: {component} subsystem and {component2} interactions",
    "Phoenix release date confirmed for {month} {day}; {person} owns coordination",
    "Phoenix retro: {component} performed well, {component2} needs another iteration",
]

# Backbone: daily/weekly routine. Doesn't answer Phoenix queries
# but is part of the workflow background. Should survive Dispose
# because it accumulates regular B_i bumps.
BACKBONE_TEMPLATES = [
    "Daily standup: {person} reported on {project}, blocker on {component}",
    "Weekly all-hands at {time}; {person} presented {project} status",
    "Coffee chat with {person} about {topic} this morning",
    "Lunch meeting in conference room {room}: {project} planning",
    "{person} pinged me on Slack about {topic}",
    "1:1 with {person}: career growth, {topic}, next quarter goals",
    "Friday wrap-up: {project} on track, {project2} slipping a sprint",
    "Office hours with {person}: {topic} questions answered",
]

# JUNK: one-shot trivia with DELIBERATE vocabulary overlap with the
# queries. The queries below ask about "Phoenix release date",
# "architecture review", "deployment plan", etc. The junk uses those
# same words in entirely unrelated contexts. Without Dispose, the
# embedder will surface these on cosine similarity alone.
JUNK_TEMPLATES = [
    # "release" overlap
    "Watched {movie}, the new release from {studio}; {opinion}",
    "Pixar release schedule for {year} looks {opinion}",
    "Music release: {artist} dropped a new album titled {album_name}",
    "Phoenix Suns released their {year} roster yesterday",  # phoenix overlap too
    "Apple release event was on {weekday}; {opinion}",
    # "date" / "kickoff" overlap
    "Date night at {restaurant} was {opinion}",
    "Football kickoff at {time}; {team} won {score}",
    "Concert kickoff for {artist} tour next month",
    # "architecture" / "design" overlap
    "Visited the new {museum} — {architect} did the architecture",
    "Fashion design week showed {designer}'s spring collection",
    "Boat design magazine featured {designer} this issue",
    # "deployment" / "migration" overlap
    "Bird migration is starting; saw {bird} at the park",
    "Military deployment in the news, {region}",
    # "phoenix" non-Project overlap
    "Trip to Phoenix Arizona was {opinion}; weather was {weather}",
    "Phoenix Bar on Main Street has new {drink_type}",
    "Phoenix bird folklore: rises from {element} every {n} years",
    # "sprint" / "fix" overlap
    "Sprint workout at the gym: {minutes} minutes on the treadmill",
    "Fixed the leaky {fixture} in the {room} this weekend",
    # "dashboard" / "metrics" overlap
    "Car dashboard light came on; {opinion}",
    "Fitness metrics this week: {steps} steps, {minutes} active minutes",
    # generic noise
    "Coffee at {coffee_shop} was {opinion}",
    "Saw a {color} {bird} at the park this morning",
    "Hallway light flickered again; need to call {role}",
    "Read {chapter_count} chapters of {book_title}",
    "Trail run on {trail} was {opinion}",
    "Made {recipe} for dinner; turned out {opinion}",
    "Birthday party for {person}'s {relation} this weekend",
    "Bought new {object} from {store}",
    "Public transit was {opinion} today",
    "Weather is {weather} this week",
]


# ─── slot vocabularies ─────────────────────────────────────────────

SLOTS: dict[str, list[str]] = {
    "version": ["1.0", "1.1", "0.9-rc", "1.0.1", "2.0-alpha"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "month": ["October", "November", "December"],
    "day": ["3rd", "8th", "12th", "15th", "19th", "22nd", "27th"],
    "person": [
        "Alice",
        "Bob",
        "Carol",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Henry",
        "Ivy",
        "Jack",
    ],
    "component": [
        "auth",
        "billing",
        "ingestion",
        "ranking",
        "router",
        "scheduler",
        "storage",
        "telemetry",
    ],
    "component2": [
        "queue",
        "worker",
        "indexer",
        "cache",
        "limiter",
    ],
    "sha": ["a1b2c3d", "f4e5d6c", "0a1b2c3", "b9c8d7e", "c5d4e3f"],
    "edge_case": [
        "null inputs",
        "concurrent writes",
        "stale tokens",
        "empty batches",
        "rate-limited bursts",
    ],
    "n": [str(i) for i in range(1, 9)],
    "time": ["10am", "11am", "2pm", "3pm", "4pm"],
    "topic": ["roadmap", "headcount", "tooling", "release", "scope"],
    "project": ["Phoenix", "Atlas", "Beacon", "Cobalt", "Drake"],
    "project2": ["Atlas", "Beacon", "Cobalt", "Drake", "Echo"],
    "room": ["A", "B", "C", "Sunset", "Oak"],
    "movie": [
        "Skyline Memories",
        "The Last Architect",
        "Release Window",
        "Migration North",
        "Sprint to Sunset",
    ],
    "studio": ["Pixar", "A24", "Universal", "Warner", "Sony"],
    "year": ["2025", "2026", "2027"],
    "opinion": ["great", "decent", "underwhelming", "strange", "fine"],
    "artist": ["Mira", "Jolt", "Eclipse Set", "Marlow", "Hush Tone"],
    "album_name": ["Static", "Lowlight", "Migration Routes", "Open Doors"],
    "restaurant": ["Bistro 22", "The Daily", "Foglight", "Ember"],
    "team": ["Eagles", "Hawks", "Foxes", "Ravens"],
    "score": ["21-14", "10-7", "28-21", "17-13"],
    "museum": ["Skyhall", "Arcline", "Forge"],
    "architect": ["Veiga", "Halsten", "Mori"],
    "designer": ["Vega", "Kline", "Halo"],
    "region": ["the southern coast", "the eastern border", "the gulf"],
    "bird": ["robin", "goldfinch", "starling", "warbler"],
    "weather": ["sunny", "rainy", "windy", "humid", "clear"],
    "drink_type": ["seasonal cocktail menu", "espresso flight", "cider tasting"],
    "element": ["ash", "ember", "smoke", "fire"],
    "minutes": ["20", "30", "45", "60"],
    "fixture": ["faucet", "shower head", "doorknob", "hinge"],
    "steps": ["8200", "9500", "11400", "12800"],
    "coffee_shop": ["Ember", "Bluebird", "Foglight", "Stumptown"],
    "color": ["red", "blue", "yellow", "brown"],
    "role": ["the super", "facilities", "maintenance"],
    "chapter_count": ["two", "three", "four", "five"],
    "book_title": [
        "The Tide Turns",
        "Open Window",
        "Quiet Engines",
        "Letters from the Coast",
    ],
    "trail": [
        "the ridge loop",
        "the river path",
        "the eastern fire road",
    ],
    "recipe": ["pasta", "stew", "stir fry", "risotto"],
    "relation": ["nephew", "niece", "cousin", "sister"],
    "object": ["lamp", "rug", "chair", "shelf"],
    "store": ["the corner shop", "Marlow's", "the hardware store"],
}


# ─── adversarial queries ───────────────────────────────────────────

# Each query has:
#   - "query": the natural-language question
#   - "relevant_template_idx": which HIGH_UTILITY_TEMPLATES indices
#       answer this question (used to compute relevant_ids after
#       generation)
#   - "junk_pattern": substrings that, if present in junk content,
#       mark that junk memory as "designed to pollute" this query.
# After generation we walk the corpus and label junk_ids per query
# based on these patterns. The harness then computes top-K cleanliness
# = (junk_ids ∩ top_K is empty).

ADVERSARIAL_QUERIES: list[dict[str, Any]] = [
    {
        "query": "what is the phoenix release date",
        "relevant_template_idx": [0, 8],
        "junk_pattern": ["release", "phoenix"],
    },
    {
        "query": "phoenix architecture review notes",
        "relevant_template_idx": [1, 7],
        "junk_pattern": ["architecture", "phoenix"],
    },
    {
        "query": "phoenix sprint kickoff focus",
        "relevant_template_idx": [2],
        "junk_pattern": ["sprint", "kickoff", "phoenix"],
    },
    {
        "query": "phoenix bug fix details",
        "relevant_template_idx": [3],
        "junk_pattern": ["fix", "phoenix"],
    },
    {
        "query": "phoenix deployment plan",
        "relevant_template_idx": [4],
        "junk_pattern": ["deployment", "phoenix"],
    },
    {
        "query": "phoenix migration schema approval",
        "relevant_template_idx": [5],
        "junk_pattern": ["migration", "phoenix"],
    },
    {
        "query": "phoenix dashboard metrics",
        "relevant_template_idx": [6],
        "junk_pattern": ["dashboard", "metrics", "phoenix"],
    },
    {
        "query": "phoenix design doc components",
        "relevant_template_idx": [7],
        "junk_pattern": ["design", "phoenix"],
    },
    {
        "query": "phoenix retro outcomes",
        "relevant_template_idx": [9],
        "junk_pattern": ["retro", "phoenix"],
    },
    # Variants — same templates, different surface phrasing.
    {
        "query": "when does phoenix ship",
        "relevant_template_idx": [0, 8],
        "junk_pattern": ["release", "phoenix", "ship"],
    },
    {
        "query": "who is reviewing phoenix architecture",
        "relevant_template_idx": [1],
        "junk_pattern": ["architecture", "phoenix"],
    },
    {
        "query": "phoenix component performance feedback",
        "relevant_template_idx": [9, 1],
        "junk_pattern": ["phoenix", "component"],
    },
    {
        "query": "phoenix coordination owner",
        "relevant_template_idx": [8],
        "junk_pattern": ["phoenix"],
    },
    {
        "query": "phoenix bug edge cases handled",
        "relevant_template_idx": [3],
        "junk_pattern": ["fix", "edge", "phoenix"],
    },
    {
        "query": "phoenix v1 release week",
        "relevant_template_idx": [0, 8],
        "junk_pattern": ["release", "phoenix"],
    },
    # Backbone queries (no Phoenix).
    {
        "query": "weekly all-hands status update",
        "relevant_template_idx": [-1],  # marks "use BACKBONE_TEMPLATES idx"
        "backbone_relevant_idx": [1, 6],
        "junk_pattern": ["weekly", "status"],
    },
    {
        "query": "daily standup blockers reported",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [0],
        "junk_pattern": ["standup", "blocker"],
    },
    {
        "query": "coffee chat topics this week",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [2],
        "junk_pattern": ["coffee", "chat"],
    },
    {
        "query": "1:1 conversations with team members",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [5],
        "junk_pattern": ["1:1", "with"],
    },
    {
        "query": "office hours questions",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [7],
        "junk_pattern": ["office", "hours"],
    },
    {
        "query": "lunch meeting agenda",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [3],
        "junk_pattern": ["lunch", "meeting"],
    },
    {
        "query": "friday project wrap-up summary",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [6],
        "junk_pattern": ["friday", "wrap"],
    },
    {
        "query": "slack ping topics",
        "relevant_template_idx": [-1],
        "backbone_relevant_idx": [4],
        "junk_pattern": ["slack", "ping"],
    },
    # More Phoenix queries with sharper phrasing.
    {
        "query": "phoenix telemetry component review",
        "relevant_template_idx": [1, 9],
        "junk_pattern": ["telemetry", "phoenix"],
    },
    {
        "query": "phoenix sprint number focus",
        "relevant_template_idx": [2],
        "junk_pattern": ["sprint", "phoenix"],
    },
    {
        "query": "phoenix deployment morning order",
        "relevant_template_idx": [4],
        "junk_pattern": ["deployment", "morning"],
    },
    {
        "query": "phoenix v2 alpha release",
        "relevant_template_idx": [0, 8],
        "junk_pattern": ["release", "phoenix", "v2", "alpha"],
    },
    {
        "query": "phoenix retro lessons learned",
        "relevant_template_idx": [9],
        "junk_pattern": ["retro", "phoenix"],
    },
    {
        "query": "phoenix subsystem interactions",
        "relevant_template_idx": [7],
        "junk_pattern": ["subsystem", "phoenix"],
    },
    {
        "query": "phoenix iteration needed",
        "relevant_template_idx": [9],
        "junk_pattern": ["iteration", "phoenix"],
    },
]


# ─── generation ────────────────────────────────────────────────────


def _fill(template: str, rng: random.Random) -> str:
    out = template
    while "{" in out:
        # Find the next slot.
        start = out.index("{")
        end = out.index("}", start)
        slot = out[start + 1 : end]
        choices = SLOTS.get(slot)
        if not choices:
            raise KeyError(f"unknown slot {slot!r} in template {template!r}")
        out = out[:start] + rng.choice(choices) + out[end + 1 :]
    return out


def _spread_timestamps(n: int, rng: random.Random) -> list[int]:
    """Distribute ``n`` ts_offset_seconds across ``SIMULATED_DAYS``.

    Roughly uniform with jitter so memories don't all stack at midnight.
    Returned list is sorted ascending so the corpus is in temporal
    order.
    """

    span = SIMULATED_DAYS * SECONDS_PER_DAY
    offsets = [rng.randrange(0, span) for _ in range(n)]
    offsets.sort()
    return offsets


def _generate(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)

    n_high = int(TOTAL_MEMORIES * HIGH_UTILITY_FRAC)
    n_backbone = int(TOTAL_MEMORIES * BACKBONE_FRAC)
    n_junk = TOTAL_MEMORIES - n_high - n_backbone

    # Track which template each memory came from so we can compute
    # relevant_ids per query without re-parsing content.
    memories: list[dict[str, Any]] = []

    # High-utility (50): rotate through templates so every Phoenix
    # query gets multiple matches.
    for i in range(n_high):
        tpl_idx = i % len(HIGH_UTILITY_TEMPLATES)
        memories.append(
            {
                "id": f"h{i:03d}",
                "utility": "high",
                "template_kind": "high",
                "template_idx": tpl_idx,
                "content": _fill(HIGH_UTILITY_TEMPLATES[tpl_idx], rng),
            }
        )

    # Backbone (100): rotate through backbone templates.
    for i in range(n_backbone):
        tpl_idx = i % len(BACKBONE_TEMPLATES)
        memories.append(
            {
                "id": f"b{i:03d}",
                "utility": "medium",
                "template_kind": "backbone",
                "template_idx": tpl_idx,
                "content": _fill(BACKBONE_TEMPLATES[tpl_idx], rng),
            }
        )

    # Junk (350): rotate through junk templates so every adversarial
    # vocabulary pattern is well-represented.
    for i in range(n_junk):
        tpl_idx = i % len(JUNK_TEMPLATES)
        memories.append(
            {
                "id": f"j{i:03d}",
                "utility": "low",
                "template_kind": "junk",
                "template_idx": tpl_idx,
                "content": _fill(JUNK_TEMPLATES[tpl_idx], rng),
            }
        )

    # Shuffle then sort by assigned timestamp so the corpus is in
    # observe order. Each memory gets a single ts_offset_seconds
    # drawn uniformly across the 30-day span.
    rng.shuffle(memories)
    offsets = _spread_timestamps(len(memories), rng)
    for m, off in zip(memories, offsets, strict=True):
        m["ts_offset_seconds"] = off

    # Compute per-query relevant_ids and junk_ids.
    queries: list[dict[str, Any]] = []
    by_template = {("high", t): [] for t in range(len(HIGH_UTILITY_TEMPLATES))}
    by_template.update({("backbone", t): [] for t in range(len(BACKBONE_TEMPLATES))})
    for m in memories:
        if m["template_kind"] == "junk":
            continue
        kind = "high" if m["template_kind"] == "high" else "backbone"
        by_template[(kind, m["template_idx"])].append(m["id"])

    for q in ADVERSARIAL_QUERIES:
        relevant: list[str] = []
        if q["relevant_template_idx"] == [-1]:
            for idx in q["backbone_relevant_idx"]:
                relevant.extend(by_template[("backbone", idx)])
        else:
            for idx in q["relevant_template_idx"]:
                relevant.extend(by_template[("high", idx)])
        # Junk ids: any junk memory whose content matches AT LEAST
        # ONE of the query's junk patterns (case-insensitive
        # substring). The "any" semantics matches what the embedder
        # actually surfaces — a single shared keyword is often enough
        # for cosine similarity to pull a junk memory into top-K.
        # The harness's pre-Dispose validation step (in
        # bench/ablate_dreaming.py) then filters this candidate set
        # against the actual top-K to confirm the corpus is genuinely
        # adversarial under the chosen embedder.
        patterns = [p.lower() for p in q["junk_pattern"]]
        junk_ids = [
            m["id"]
            for m in memories
            if m["template_kind"] == "junk" and any(p in m["content"].lower() for p in patterns)
        ]
        queries.append(
            {
                "query": q["query"],
                "relevant_ids": sorted(relevant),
                "junk_ids": sorted(junk_ids),
            }
        )

    return {
        "_meta": {
            "seed": seed,
            "total_memories": TOTAL_MEMORIES,
            "high_utility": n_high,
            "backbone": n_backbone,
            "junk": n_junk,
            "simulated_days": SIMULATED_DAYS,
            "queries": len(queries),
        },
        "memories": [
            # Drop the internal template_kind / template_idx fields —
            # the harness only needs id / utility / content / ts_offset.
            {
                "id": m["id"],
                "utility": m["utility"],
                "ts_offset_seconds": m["ts_offset_seconds"],
                "content": m["content"],
            }
            for m in memories
        ],
        "queries": queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic accumulating-pressure corpus.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: bench/fixtures/pressure_corpus_seed{N}.jsonl).",
    )
    args = parser.parse_args()

    corpus = _generate(args.seed)

    if args.out is None:
        out = Path(__file__).parent / f"pressure_corpus_seed{args.seed}.jsonl"
    else:
        out = Path(args.out)

    # Single JSON document, line-delimited per top-level field for
    # diff-friendliness. We could write proper JSONL (one memory per
    # line) but a single JSON object is easier to load with
    # json.loads() in the harness.
    out.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    meta = corpus["_meta"]
    print(
        f"wrote {out}\n"
        f"  memories: {meta['total_memories']} "
        f"(high={meta['high_utility']}, backbone={meta['backbone']}, junk={meta['junk']})\n"
        f"  queries:  {meta['queries']}\n"
        f"  span:     {meta['simulated_days']} simulated days",
        flush=True,
    )

    # Quick adversariality check: how many queries got at least one
    # junk_id assigned? Few junk_ids = corpus isn't actually
    # adversarial. We don't enforce a threshold here; the harness's
    # pre-Dispose validation step in bench/ablate_dreaming.py is the
    # actual gate.
    qs_with_junk = sum(1 for q in corpus["queries"] if q["junk_ids"])
    print(
        f"  queries with junk candidates: {qs_with_junk}/{len(corpus['queries'])} "
        f"({100 * qs_with_junk // len(corpus['queries'])}%)",
        flush=True,
    )
    if qs_with_junk < int(0.7 * len(corpus["queries"])):
        print(
            "warning: <70% of queries got junk candidates assigned. "
            "Consider widening junk_pattern or adding templates.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
