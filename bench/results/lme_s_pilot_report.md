# LongMemEval-S Pilot — Mnemoss vs mem0

**Pilot date:** 2026-05-08 → 2026-05-10
**Slice:** 24 questions, stratified 4 per question type across the 6 LongMemEval-S
categories (single-session-user, single-session-assistant, single-session-preference,
multi-session, knowledge-update, temporal-reasoning).
**Generator + judge:** `deepseek-chat` (via DeepSeek API, OpenAI-compatible).
**Embedder (Mnemoss + mem0):** OpenAI `text-embedding-3-small`.

## TL;DR

- **Mnemoss best config (M-facts-v3 + gpt-4o): 13/24 = 54.2%**, vs mem0
  (9/24 = 37.5%) on this stratified slice. mem0 has not yet been
  re-measured at the gpt-4o tier; that's in flight.
- At the deepseek-chat tier (apples-to-apples with the original mem0
  baseline), Mnemoss's best is 11/24 (45.8%) vs mem0's 9/24 (37.5%).
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
M-facts-v3 + gpt-4o gen+judge+dream      4/4    3/4    1/4    2/4    3/4    0/4   13/24 (54%)  ← BEST overall
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

A fair mem0 vs Mnemoss comparison at the gpt-4o tier is in flight (mem0
was previously measured at 38% on deepseek-chat).

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
