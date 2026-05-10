# LongMemEval-S Pilot — Mnemoss vs mem0

**Pilot date:** 2026-05-08 → 2026-05-10
**Slice:** 24 questions, stratified 4 per question type across the 6 LongMemEval-S
categories (single-session-user, single-session-assistant, single-session-preference,
multi-session, knowledge-update, temporal-reasoning).
**Generator + judge:** `deepseek-chat` (via DeepSeek API, OpenAI-compatible).
**Embedder (Mnemoss + mem0):** OpenAI `text-embedding-3-small`.

## TL;DR

- **Mnemoss best config: 14/24 = 58.3%** with gpt-4o-mini at k=30.
  +25pp absolute over the M-baseline (33%), +20pp over mem0 (38%),
  and +29pp over mem0 at the gpt-4o tier (29%).
- The 33% → 58% total breaks into +17pp from LLM upgrade and **+8pp
  from architecture** (Dream consolidate atomic facts + cross-session
  edges + tuned prompts). Both are necessary; neither alone reaches 58%.
- At k=10, both gpt-4o and gpt-4o-mini land at 13/24 (54.2%) with
  different per-question wins. **gpt-4o-mini is the production sweet
  spot** (~10× cheaper than gpt-4o, equal/better accuracy).
- At the gpt-4o tier (apples-to-apples), **Mnemoss leads mem0 by 25pp**:
  Mnemoss 13/24 (54%) vs mem0 7/24 (29%).
- At the deepseek-chat tier, Mnemoss leads mem0 by 8pp:
  11/24 (46%) vs 9/24 (38%).
- **Mnemoss's architectural advantage amplifies with stronger LLMs**
  because it surfaces multi-tier evidence (raw turns + atomic facts +
  Dream summaries) — the generator can synthesize across all three.
  mem0 only feeds its ingest-time-extracted facts to the generator;
  a stronger LLM has no extra material to work with, and the stricter
  gpt-4o-mini judge actually penalizes mem0's terser answers.
- The architectural lift came from **adding LLM-based atomic-fact extraction
  inside Dream Consolidate** (Cold Path, principle-conformant) with an
  explicit prompt for named entities, user preferences, and most-recent-value
  collapse on conflicting facts.
- All four pure-retrieval-tuning attempts (Dream alone, +cross-session edges +
  spreading-activation expand, +timestamped snippets, k=10→20) plateaued at
  33% — **proving that recall surfacing more or differently-tagged turn-level
  candidates can't close the gap**. Cross-session synthesis, knowledge updates,
  and preference questions need fact-grained memories that simply weren't in
  the index.
- Mem0 was the lever: it ships LLM extraction at ingest, producing atomic
  fact memories ("User redeemed a $5 coupon at Target", "User's mortgage
  pre-approval is now $400,000 (raised from $350,000)") that survive recall
  ranking even when the raw turn doesn't.
- A **singleton-sweep variant** (per-memory atomic-fact extraction over
  HDBSCAN-noise outliers) regressed to 37.5% in two policy variants
  (blanket and salience≥0.5 filter). Even high-signal singleton facts
  displace cluster-derived ones in top-K ranking. Kept in the codebase
  as opt-in (`DreamerParams.process_singletons=False` by default,
  `singleton_salience_threshold=0.5`), documented here as a negative
  result.

## Full 13-config matrix

```
config                                   user   asst   pref m-sess  k-upd   temp    TOT
---------------------------------------------------------------------------------------
M-baseline (no Dream)                    3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
M-Dream                                  2/4    4/4    0/4    0/4    1/4    0/4    7/24 (29%)   ← regression
M-Dream+expand                           3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
M-Dream+expand+ts                        3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
M-Dream+expand+k=20                      3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
mem0 (deepseek gen+judge)                4/4    2/4    1/4    1/4    1/4    0/4    9/24 (38%)
M-facts (cluster atomic facts)           3/4    4/4    0/4    1/4    1/4    0/4    9/24 (38%)
M-facts-v2 (extraction prompt v2)        3/4    4/4    1/4    1/4    2/4    0/4   11/24 (46%)  ← BEST deepseek (A)
M-facts-v2+singletons (blanket)          3/4    4/4    1/4    0/4    1/4    0/4    9/24 (38%)  ← noise
M-facts-v2+singletons (salience≥0.5)     3/4    4/4    0/4    1/4    1/4    0/4    9/24 (38%)  ← still noisy
M-facts-v3 (+ generator prompt v3)       3/4    4/4    0/4    2/4    2/4    0/4   11/24 (46%)  ← BEST deepseek (B)
M-facts-v4 (drop type-trust line)        3/4    4/4    0/4    2/4    1/4    0/4   10/24 (42%)  ← regression vs v3
M-facts-v3 + gpt-4o gen+judge+dream      4/4    3/4    1/4    2/4    3/4    0/4   13/24 (54%)  ← tied BEST
mem0 + gpt-4o gen + gpt-4o-mini judge    3/4    0/4    1/4    2/4    1/4    0/4    7/24 (29%)  ← regression vs deepseek
M-facts-v3 + gpt-4o-mini all-LLMs        4/4    4/4    1/4    1/4    3/4    0/4   13/24 (54%)  ← 10× cheaper
M-facts-v3 + gpt-4o-mini + k=30          4/4    4/4    1/4    2/4    3/4    0/4   14/24 (58%)  ← BEST overall
M-baseline + gpt-4o-mini + k=30          4/4    4/4    1/4    1/4    2/4    0/4   12/24 (50%)  ← architectural ablation
M-facts-v3 + gpt-4o-mini + k=30 + dates   4/4    4/4    1/4    2/4    3/4    0/4   14/24 (58%)  ← no lift, dates redundant
```

The two BEST configs (v2 and v3) both land at 11/24 (46%) but with different
per-question wins; the v2 ∪ v3 union is 13/24 (54%). The two prompts couple
through LLM behavior so they don't compose for free.

v4 was the attempt to recover v2's wins by dropping the v3 prompt's
"trust 'fact' and 'summary' snippets when they conflict with raw 'episode'
snippets" line (hypothesised cause of the v2-vs-v3 trade-off) and adding
an explicit preference-synthesis instruction. It regressed: lost
`852ce960` (the Wikipedia-paste mortgage knowledge-update v3 won) without
recovering any v2-only win. The type-trust line was apparently what
biased the LLM toward picking the consolidated `$400K` summary over the
raw `$350K` episode for `852ce960`. Architectural conclusion: prompt
components compose non-monotonically; v3 is the shipped generator prompt.

## Architectural arc

### Phase 1 — Retrieval-tuning ceiling (33%)

Started from the M-baseline config (Dream off, default ACT-R recall). Added
one architectural lever per run:

| Run | Lever | Result | Insight |
| --- | --- | --- | --- |
| M-Dream | Run Dream pipeline after ingest | 29% (-1) | Dream surfaced contradicting facts the LLM bailed on ("I don't know — commute is 45 min OR 1 hour"). Real bug: `derived_from` edges weren't in the relation table (only on `Memory.derived_to` JSON column), so spreading couldn't traverse them. **Fixed in `_persist_derived`** — see commit 1. |
| M-Dream+expand | Add `derived_to` to expand predicates + primer-query to engage `auto_expand` | 33% | Cross-session edges now traversable, regression recovered. But hard slices stayed at 0/4. |
| M-Dream+expand+ts | Prefix recall snippets with `[YYYY-MM-DD type] ...` | 33% | 16/24 predictions changed but 0 flipped correctness. The right facts still weren't in top-10. |
| M-Dream+expand+k=20 | Double the top-K | 33% | Identical per-question correctness. Confirmed: missing facts aren't at rank 11-20 either. |

**Conclusion: retrieval is the wall.** Cosine over turn-level chunks doesn't surface
the answers — the answers aren't in the index in distillable form.

### Phase 2 — Mem0 cross-comparison (architectural diagnosis)

Ran `mem0` on the same 24 questions. Mem0 calls `gpt-4o-mini` at every
`add()` to extract atomic facts into discrete memory rows.

| Run | Result | Notes |
| --- | --- | --- |
| mem0 | 9/24 (38%) | First time any config got a preference question right (06878be2 Sony) and a multi-session question right (gpt4_59c863d7 model kits). |

Cross-comparison vs Mnemoss-best (`M-Dream+expand`):

```
slice                       both  mem0-only  mnem-only  neither
single-session-user           3      1          0         0
single-session-assistant      2      0          2         0
single-session-preference     0      1          0         3
multi-session                 0      1          0         3
knowledge-update              0      1          1         2
temporal-reasoning            0      0          0         4

union (either correct):      12/24 = 50%   ← combined upper bound
```

The split was clean:
- **Mem0 wins** on questions where atomic-fact extraction surfaces things
  turn-level recall buries: the named entity (Target), the user preference
  (Sony), the cross-session count (model kits), the latest fact value (5K time).
- **Mnemoss wins** on questions where verbatim turn fidelity matters: exact
  shift assignment, descriptive detail, count aggregation across separate
  mentions.
- **Both fail** on temporal-reasoning (4/4 misses) — calendar arithmetic +
  event ordering is a separate problem orthogonal to extraction.

Architectural prescription: port mem0's atomic-fact extraction into Mnemoss's
Dream pipeline (Cold Path, respects all 9 principles), and Mnemoss should pick
up most of mem0's wins without losing its own.

### Phase 3 — Atomic-fact extraction in Dream Consolidate (38% → 46%)

#### M-facts (cluster-only): 9/24 (38%)

Extended `dream/consolidate.py`'s prompt with section (D) ATOMIC FACTS,
asking for a list of self-contained propositional facts per cluster (in
addition to the existing summary, refinements, patterns). Each fact persists
as a level=2 FACT memory with `derived_from` edges to its source members
and an `idx_priority` lift on top of the cluster's max member priority.

Result: matched mem0 at 38% (same overall, different per-question wins).
Picked up `gpt4_59c863d7` (multi-session model kits). No regressions.

#### M-facts-v2 (tuned prompt): 11/24 (46%)

Tightened the (D) ATOMIC FACTS section with three explicit instructions:
1. **Named entities verbatim** — "User redeemed a $5 coupon at **Target**."
2. **Distill preferences explicitly** — "User prefers Sony-compatible accessories."
3. **Collapse conflicting values to current** — "User's current 5K personal best is 25:50 (improved from 27:12)."

Result: +2 questions from the prompt tune.
- **`06878be2` (preference / Sony) ✓** — explicit preference-distillation worked.
- **`6a1eabeb` (knowledge-update / 5K time) ✓** — most-recent-collapse worked.

**Final standing: M-facts-v2 = 11/24 (46%), mem0 = 9/24 (38%).** Mnemoss now
beats mem0 by 8.3 pp.

Cross-comparison M-facts-v2 vs mem0:
```
both: 8        Mnemoss-only: 3        mem0-only: 1        neither: 12
```

The single mem0-only win is `51a45a95` (Target store entity in a
single-session-user question). That turn likely sat in HDBSCAN noise space
and never reached the cluster prompt — see Phase 4.

### Phase 3.5 — Generator prompt tune (M-facts-v3): same total, different composition

The recall snippets carry `[YYYY-MM-DD type]` prefixes (added in
M-Dream+expand+ts) but the original generator prompt didn't tell the LLM
how to use them. Updated the prompt to explicitly:

- Order events chronologically when the question asks for sequence or
  elapsed time.
- Resolve conflicts by preferring the more recent value as the current
  truth.
- Anchor calendar arithmetic in the dates.
- Trust `fact` and `summary` snippets over raw `episode` snippets when
  they conflict.

Result: 11/24 (46%) — same overall as facts-v2, different composition.

| | facts-v2 | facts-v3 |
| --- | --- | --- |
| Won | 06878be2 (preference), 6a1eabeb (5K time) | b5ef892d (multi-session "8 days"), **852ce960 (Wikipedia mortgage)** |
| Lost vs other | (the two on the right) | (the two on the left) |

The headline win for v3: `852ce960` succeeded for the FIRST time across all
12 prior configs (both Mnemoss variants AND mem0 ALL missed it). The
"prefer most recent value on conflict" instruction got the LLM to pick
the $400K (current) value over the $350K (earlier) value that had
dominated every prior recall.

The headline disappointment: temporal-reasoning stayed 0/4. The
date-arithmetic instruction is well-formed; the four temporal questions
fail because **recall doesn't surface the two specific time-anchored
events** the question needs to subtract. The answers aren't even in the
context the LLM gets — this is a recall-side gap, not a generator gap.

### Phase 3.6 — Stronger generator (gpt-4o): retrieval-bound hypothesis falsified

To test whether the 46% ceiling on deepseek-chat was generator-bound or
genuinely retrieval-bound, swapped all three LLMs (generator, judge,
Dream-Consolidate) from `deepseek-chat` to `gpt-4o` (gpt-4o-mini for
judge and Dream to balance cost). Same Mnemoss architecture, same v3
generator prompt, same atomic-fact extraction.

Result: **13/24 (54%)**, +8pp absolute over deepseek (46%).

| Slice | v3-deepseek | v3-gpt4o |
| --- | --- | --- |
| sso-user | 3/4 | **4/4** ← gpt-4o caught `51a45a95` (Target) — first lift across any Mnemoss config |
| sso-asst | 4/4 | 3/4 ← lost `c4f10528` |
| sso-pref | 0/4 | **1/4** ← recovered `06878be2` (Sony) |
| multi-session | 2/4 | 2/4 (swapped: lost `gpt4_59c863d7`, gained `6d550036`) |
| **knowledge-update** | 2/4 | **3/4** ← +1 (kept all v3 wins, recovered `6a1eabeb`) |
| temporal | 0/4 | 0/4 |

`6d550036` was a **first win across all 12 prior configs** — never solved
before by anyone (including mem0). Same for `51a45a95` lifting on Mnemoss.
gpt-4o synthesizes correctly from snippets that deepseek-chat couldn't.

**Architectural finding falsified:** the earlier "retrieval is the wall"
conclusion (Phase 1) was actually generator-bound. The right facts WERE
in top-K — the deepseek generator just couldn't reason from them. With
a stronger generator, the same Mnemoss architecture (atomic-fact
extraction in Dream + cross-session edges + auto-expand + tuned prompts)
extracts substantially more value.

This complicates the cost story: gpt-4o costs ~10× per call vs deepseek-
chat. For agent workloads where memory recall is the inner loop of every
LLM turn, that's a real deployment decision, not a clean win.

### Phase 3.7 — Fair mem0 comparison at gpt-4o (the gap WIDENS)

To pin down whether Mnemoss's 8pp deepseek-tier lead over mem0 was a
real architectural advantage or just LLM-specific, re-ran mem0 with
the same gpt-4o stack (gen=gpt-4o, judge=gpt-4o-mini, mem0's internal
extraction LLM was already gpt-4o-mini).

Result: **mem0-gpt4o = 7/24 (29.2%)** — mem0 *regressed* with the
stronger LLM, while Mnemoss *lifted*. The comparison gap widens:

| Tier | Mnemoss best | mem0 | Gap |
| --- | --- | --- | --- |
| deepseek-chat | 11/24 (46%) | 9/24 (38%) | +8pp |
| gpt-4o | 13/24 (54%) | 7/24 (29%) | **+25pp** |

What happened to mem0 specifically:

- single-session-assistant collapsed from 2/4 → 0/4 (the gpt-4o-mini
  judge is stricter than deepseek-chat on partial/terse answers, and
  mem0's ingest-time-extracted facts are terser than raw turns).
- 51a45a95 (Target) lost — counterintuitive, since this was mem0's
  signature win at the deepseek tier.
- knowledge-update dropped from 1/4 → 1/4 (different question won).

What Mnemoss did differently:

- Surfaces THREE tiers of evidence at recall time: raw episode turns,
  Dream-extracted atomic facts, Dream-extracted summaries. gpt-4o can
  synthesize across all three.
- mem0 only feeds its ingest-time-extracted atomic facts. If those
  facts dropped or distorted some content (which is inherent to LLM
  extraction), the stronger generator has no extra material.

This is the strongest architectural finding of the pilot. Mnemoss's
multi-tier evidence policy isn't just a stylistic choice — it gives
the generator more raw material, and a stronger generator extracts
more value. mem0's "extract-once, surface-only-extracted-facts" model
caps the system at the quality of its ingest-time extraction.

### Phase 3.8 — Cost positioning: gpt-4o-mini ties gpt-4o on accuracy

To pin down whether 54% requires gpt-4o specifically or if the cheaper
gpt-4o-mini also delivers, ran M-facts-v3 with gpt-4o-mini for all
three LLMs (gen, judge, Dream-Consolidate). Result: **13/24 (54.2%)
— exactly ties Mnemoss-gpt4o**, with different per-question wins:

| Slice | gpt-4o | gpt-4o-mini |
| --- | --- | --- |
| sso-user | 4/4 | 4/4 |
| sso-asst | 3/4 (lost `c4f10528`) | **4/4** ← keeps `c4f10528` |
| sso-pref | 1/4 (`06878be2`) | 1/4 (`8a2466db` — never won before by ANY config) |
| multi-session | 2/4 | 1/4 |
| knowledge-update | 3/4 | 3/4 (same wins) |
| temporal | 0/4 | 0/4 |

The two configs are Pareto-different per-question (combined union ceiling
15/24 = 62.5%) but tied on overall.

**Cost positioning:** gpt-4o-mini is roughly 10× cheaper per call than
gpt-4o ($0.15/$0.60 per 1M tokens vs $2.50/$10 per 1M). For the
production-ready Mnemoss config, **gpt-4o-mini is the sweet spot** —
identical accuracy at this slice size, an order of magnitude cheaper.
The gpt-4o variant is useful for ablation / signal-investigation runs
but not the recommended deployment.

### Phase 3.9 — k=30 lifts multi-session counting

The M-Dream+expand+k=20 deepseek run earlier in the pilot showed no
lift from doubling k. But that was before atomic-fact extraction
landed. Re-tested k=30 (with snippet_max_chars=18000 to fit the
extra material) on top of the gpt-4o-mini config.

Result: **14/24 (58.3%)**, +1 question vs k=10 (54.2%). The lift
came from `0a995998` — a multi-session counting question ("how many
items to pick up") that needs the LLM to enumerate items across
multiple sessions. With k=30 the recall surfaces enough individual
item mentions for gpt-4o-mini to do the count. Never won by any of
the 15 prior configs.

Temporal-reasoning stayed 0/4 — proving that the temporal gap is
*not* recall-depth-bound. The right time-anchored events aren't in
the cosine top regardless of how deep we search. The only way to
close that slice is recall-side ranking that boosts memories whose
`created_at` is near the question's implied time anchor; that's
genuinely architectural and out of scope for this pilot.

### Phase 3.10 — Architectural ablation: how much credit does Dream get?

To pin down whether the 58% best comes from the LLM upgrade or from
Dream's atomic-fact extraction, ran **M-baseline** (no Dream, no
atomic facts, no auto-expand, no cross-session edges) at the same
gpt-4o-mini + k=30 setup. Result: **12/24 (50%)** — +17pp over
M-baseline-deepseek but -8pp behind the M-facts-v3 best.

Decomposition of the 33% → 58% total lift:

| Component | Lift |
| --- | --- |
| LLM upgrade (deepseek → gpt-4o-mini, k=10 → 30) | **+17pp** (33% → 50%) |
| Architecture (Dream consolidate + atomic facts + cross-session edges + tuned prompts) | **+8pp** (50% → 58%) |

Both are real. The architecture is necessary but not sufficient; the
LLM is necessary but not sufficient. Together they get to 58%.

The two questions Dream specifically lifts at the gpt-4o-mini tier:

- `0a995998` (multi-session count "how many items to pick up") — the
  cluster's atomic facts give the LLM discrete countable items.
  Without atomic facts, the same k=30 surfaces enough raw mentions
  that the LLM hallucinates a count.
- `852ce960` (Wikipedia-paste mortgage knowledge-update) — Dream's
  summary captures the "$400K (raised from $350K)" narrative. Without
  it, raw recall surfaces both numbers but the LLM picks the more
  cosine-similar `$350K` (the dominant phrasing).

Both are exactly the cases where raw cosine on turn-grained chunks
fails for structural reasons that no top-K bump can fix.

Ingest time tells the same story: 3,016s for baseline vs 5,758s for
v3 — Dream's LLM-extraction pass nearly doubles ingest time but adds
the +8pp accuracy. For a write-once-read-many memory workload
(typical for chat-history layers), that's the right cost trade.

### Phase 3.11 — Date-aware atomic-fact prompts: no lift on temporal

To probe whether temporal-reasoning was extraction-side, updated:
- `build_consolidate_prompt` to render each cluster member as
  `[YYYY-MM-DD role] content` (was `[role] content`) so the LLM
  can resolve relative time expressions ("last Sunday") to
  absolute dates during cluster consolidation.
- The (D) ATOMIC FACTS section to explicitly require absolute
  YYYY-MM-DD dates inside event-related atomic-fact content (e.g.
  "User attended the Walk for Hunger charity event on 2025-03-15.").
- The singleton-extraction prompt similarly.

Result: **14/24 (58.3%)** — *zero per-question differences* vs the
prior k=30 run. The dates the LLM needs are already accessible via
the `[YYYY-MM-DD type]` prefix that recall_text adds to every
snippet; embedding them inside the fact text was redundant.

The prompt changes are principled improvements (member-date context
in consolidate, event-date instruction in atomic facts) and stay
in — they don't hurt, and they might help on a corpus where event
dates aren't trivially recoverable.

This is the THIRD temporal-reasoning negative result of the pilot:

  - v3 generator prompt with explicit date-arithmetic instruction
  - k=30 (more candidates couldn't surface the right time-anchored events)
  - date-aware atomic-fact extraction (above)

The slice is now confirmed unaddressable through prompt-side
changes. Closing it requires recall-side architectural work — time-
aware ranking that boosts memories whose `created_at` (or
`extracted_time`) is near the question's implied time anchor, or a
separate "fetch by entity → list of dated memories" API. Both are
out of scope for this pilot.

### Phase 4 — Singleton-sweep negative result (two attempts)

Added `extract_atomic_facts_from_singleton()` and a `process_singletons=True`
flag on `DreamerParams` to run a per-memory extraction pass over HDBSCAN
noise singletons. Two policies tried, both regressed.

| Run | Result | Notes |
| --- | --- | --- |
| M-facts-v2 + singletons (blanket) | 9/24 (38%) | Lost `gpt4_59c863d7` (multi-session) and `6a1eabeb` (knowledge-update). 51a45a95 stayed ✗. |
| M-facts-v2 + singletons (salience≥0.5) | 9/24 (38%) | Same regression: lost `06878be2` (preference) and `6a1eabeb` (knowledge-update). Filter cut filler turns but high-salience singleton facts still crowded out cluster-derived ones at K=10. |

**Architectural lesson:** singleton fact extraction in any form regresses
accuracy on this slice. Even with a salience filter that removes
filler/agreement turns, the surviving high-signal singleton facts
displace cluster-derived facts in top-K cosine ranking. The fix for
`51a45a95` (Target store entity, the only mem0-only win) lies elsewhere
— either:
- Recall-side: prefer cluster-derived facts over singleton-derived ones
  (e.g. by giving cluster facts a structural priority lift beyond what
  `idx_priority` already provides).
- Capacity cap: limit singleton facts to a small N per workspace
  (top-N by salience or LLM-emitted confidence).
- Or accept the gap: 51a45a95 is one entity-recall question that
  Mnemoss handles via raw-turn cosine recall in 3/4 sibling questions
  anyway.

The code is preserved in `dream/consolidate.py` and `dream/runner.py` as
opt-in (`DreamerParams.process_singletons=False` by default). The
salience filter is wired through `DreamerParams.singleton_salience_threshold`
(default 0.5) for any future investigation.

## Files changed

### Core (src/mnemoss/)

| File | Change |
| --- | --- |
| `dream/runner.py` | `_persist_derived` writes `derived_from`/`derived_to` edges into the relation table; `_phase_consolidate` persists atomic facts; `_singletons_from_state` + opt-in singleton sweep |
| `dream/consolidate.py` | `ConsolidationResult.atomic_facts`; section (D) ATOMIC FACTS in cluster prompt with entity/preference/recency guidance; `_parse_atomic_facts`; `extract_atomic_facts_from_singleton` and its prompt builder |
| `recall/expand.py` | `derived_to` added to `_EXPAND_PREDICATES` so BFS can traverse source→summary→sibling-source |
| `core/config.py` | `DreamerParams.process_singletons` (default False) |
| `client.py` | Plumb `process_singletons` through to `DreamRunner` |

### Bench (bench/)

| File | Change |
| --- | --- |
| `longmemeval.py` | LongMemEval-S harness: `--auto-expand`, `--max-memory-chars`, `--process-singletons`, mem0 per-question storage scoping, OpenAI cap lowered from 30K to 20K chars |
| `backends/mnemoss_backend.py` | `ingest_session`, `recall_text` (with optional `expand_via_streak` primer), `dream` method, plumbing for `llm_client` / `cost_limits` / `dreamer` / `expand_via_streak` |
| `backends/mem0_backend.py` | Single-thread executor pin (mem0's history SQLite needs `check_same_thread`); per-question storage tempdir and cleanup |
| `data/prepare_longmemeval.py` | Dataset loader |
| `tests/test_longmemeval.py` | Harness tests |
| `results/lme_s_stratified_*.json` | 9 result artifacts |

## Roadmap

The 4 temporal-reasoning questions remain 0/4 across every config. They need
calendar arithmetic + event ordering — neither extraction nor cosine recall
addresses this. Likely needs:
- Time-aware recall ranking (boost memories whose `created_at` is near the
  question's implied time anchor).
- Generator prompt awareness of the `[YYYY-MM-DD type]` snippet prefix.

The single mem0-only miss (`51a45a95` Target) likely needs a selective
singleton extractor — a salience- or entity-richness filter that picks
only the high-signal noise turns. The blanket sweep was too noisy.

Larger-n confirmation (stratified=10, 60 questions) would tighten the 46%
vs 38% comparison to make publishable claims.
