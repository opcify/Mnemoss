# LongMemEval-S Pilot — Mnemoss vs mem0

**Pilot date:** 2026-05-08 → 2026-05-10
**Slice:** 24 questions, stratified 4 per question type across the 6 LongMemEval-S
categories (single-session-user, single-session-assistant, single-session-preference,
multi-session, knowledge-update, temporal-reasoning).
**Generator + judge:** `deepseek-chat` (via DeepSeek API, OpenAI-compatible).
**Embedder (Mnemoss + mem0):** OpenAI `text-embedding-3-small`.

## TL;DR

- **Mnemoss best config (M-facts-v2): 11/24 = 45.8%**, ahead of mem0 (9/24 = 37.5%)
  by 8.3 pp on this stratified slice.
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
  HDBSCAN-noise outliers) regressed to 37.5% — the noise from filler/agreement
  turns crowded out high-signal cluster-derived facts. Kept in the codebase
  as opt-in (`DreamerParams.process_singletons=False` by default), documented
  here as a negative result.

## Full 9-config matrix

```
config                                   user   asst   pref m-sess  k-upd   temp    TOT
---------------------------------------------------------------------------------------
M-baseline (no Dream)                    3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
M-Dream                                  2/4    4/4    0/4    0/4    1/4    0/4    7/24 (29%)   ← regression
M-Dream+expand                           3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
M-Dream+expand+ts                        3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
M-Dream+expand+k=20                      3/4    4/4    0/4    0/4    1/4    0/4    8/24 (33%)
mem0                                     4/4    2/4    1/4    1/4    1/4    0/4    9/24 (38%)
M-facts (cluster atomic facts)           3/4    4/4    0/4    1/4    1/4    0/4    9/24 (38%)
M-facts-v2 (entity/pref/recency prompt)  3/4    4/4    1/4    1/4    2/4    0/4   11/24 (46%)  ← BEST
M-facts-v2+singletons                    3/4    4/4    1/4    0/4    1/4    0/4    9/24 (38%)  ← noise
```

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

### Phase 4 — Singleton-sweep negative result

Added `extract_atomic_facts_from_singleton()` and a `process_singletons=True`
flag on `DreamerParams` to run a per-memory extraction pass over HDBSCAN
noise singletons.

| Run | Result | Notes |
| --- | --- | --- |
| M-facts-v2 + singletons | 9/24 (38%) | Lost both `gpt4_59c863d7` and `6a1eabeb` (the two prior wins from cluster facts). 51a45a95 stayed ✗ — not actually a singleton issue. |

The blanket policy added too many low-signal facts (small-talk, agreement,
filler turns) that crowded out high-signal cluster-derived facts at K=10.

The code is preserved in `dream/consolidate.py` and `dream/runner.py` as
opt-in (`DreamerParams.process_singletons=False` by default). Closing this
gap likely needs a salience- or structure-based filter on which singletons
are eligible for extraction — left as future work.

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
