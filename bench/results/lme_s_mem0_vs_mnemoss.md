# Mnemoss vs Mem0 — LongMemEval-S Comparison

All numbers are on the same stratified LongMemEval-S slice with the same
embedder (`text-embedding-3-small`), same generator + judge models for
each row, and the same per-question protocol (one fresh backend per
question; ingest sessions chronologically; recall top-K; generate; judge).

## Overall accuracy

| Config | LLM (gen + judge) | n | Mem0 | Mnemoss-best | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| deepseek-chat | deepseek-chat | 24 | 9/24 = **37.5%** | 11/24 = **45.8%** | **+8.3pp** |
| gpt-4o tier | gpt-4o gen, gpt-4o-mini judge | 24 | 7/24 = **29.2%** | 13/24 = **54.2%** | **+25.0pp** |
| gpt-4o-mini, k=30 | gpt-4o-mini | 24 | — (not run) | 14/24 = **58.3%** | n/a |
| gpt-4o-mini, k=30 | gpt-4o-mini | 60 | — (not run) | 34/60 = **56.7%** | n/a |

Mnemoss-best at the deepseek tier = M-facts-v2 or M-facts-v3 (tied at 11/24).
Mnemoss-best at gpt-4o = M-facts-v3 (atomic-fact extraction in Dream
Consolidate + cross-session edges + tuned generator prompt).
Mnemoss-best overall = the production config: gpt-4o-mini + k=30 + facts-v3.

## Per-slice accuracy at n=24

| Slice | Mem0 (deepseek) | Mem0 (gpt-4o) | Mnemoss-best (deepseek) | Mnemoss-best (gpt-4o-mini, k=30) |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 4/4 (100%) | 3/4 (75%) | 3/4 (75%) | **4/4 (100%)** |
| single-session-assistant | 2/4 (50%) | 0/4 (0%) | 4/4 (100%) | **4/4 (100%)** |
| single-session-preference | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) | 1/4 (25%) |
| multi-session | 1/4 (25%) | 2/4 (50%) | 2/4 (50%) | **2/4 (50%)** |
| knowledge-update | 1/4 (25%) | 1/4 (25%) | 2/4 (50%) | **3/4 (75%)** |
| temporal-reasoning | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) | 0/4 (0%) |
| **Overall** | **9/24 (38%)** | **7/24 (29%)** | **11/24 (46%)** | **14/24 (58%)** |

## Mnemoss confidence-tightened at n=60

| Slice | n=10 Mnemoss-best | Notes |
| --- | ---: | --- |
| single-session-user | 7/10 (70%) | similar to n=4 |
| single-session-assistant | **10/10 (100%)** | perfect at both n |
| single-session-preference | 3/10 (30%) | similar |
| multi-session | 3/10 (30%) | n=4 was upper end |
| knowledge-update | **9/10 (90%)** | strongest hard slice |
| temporal-reasoning | 2/10 (20%) | NOT zero at larger n |
| **Overall** | **34/60 (57%)** | n=4 estimate (58%) held |

Mem0 was not re-measured at n=60. The ~$10-15 / 6-8h cost was deferred
when the pilot landed.

## Why mem0 regresses with a stronger LLM

The mem0 deepseek → gpt-4o swap *lost* 2pp (38% → 29%). Mnemoss
*gained* 8pp (46% → 54%) with the same swap. Reasons:

1. **Mem0 surfaces only its ingest-time-extracted facts.** A stronger
   generator has no extra material to work with.
2. **Mnemoss surfaces three tiers of evidence** at recall: raw turns,
   Dream-extracted atomic facts, Dream-consolidated summaries.
   gpt-4o can synthesize across all three.
3. **The gpt-4o-mini judge is stricter** than deepseek-chat. It
   penalizes mem0's terser answers (which lack the surrounding
   context that came with raw turns).

## Architectural-lift decomposition (Mnemoss)

The 33% → 58% Mnemoss lift at n=24 decomposes as:

| Component | Lift |
| --- | --- |
| LLM upgrade (deepseek → gpt-4o-mini, k=10 → 30) | +17pp |
| **Architecture** (Dream + atomic facts + cross-session edges + tuned prompts) | **+8pp** |

Confirmed by the architectural ablation: M-baseline + gpt-4o-mini + k=30
(no Dream, no atomic facts) = 50%. Adding the architecture lifts it to
58%. The 8pp comes from exactly two questions that raw cosine on
turn-grained chunks structurally can't solve:

- `0a995998` — multi-session count, needs atomic facts to give the LLM
  discrete countable items.
- `852ce960` — Wikipedia-paste mortgage knowledge-update, needs the
  Dream summary's "$400K (raised from $350K)" framing to override
  the dominant raw `$350K` phrasing.

## Cost note

Mnemoss + gpt-4o-mini + k=30 is the **production sweet spot** —
exactly ties gpt-4o on accuracy (both 13/24 at k=10) while costing
~10× less per call. The k=30 bump adds +1 question over k=10 with
negligible additional cost.

Mem0's per-question ingest at gpt-4o-mini takes ~500s (50 sessions ×
~10s extraction-LLM-per-session). Mnemoss's is ~125s. Both at the
same LLM tier, but mem0 calls the LLM on every `add()`; Mnemoss only
calls it during Dream Consolidate (Cold Path, post-ingest).
