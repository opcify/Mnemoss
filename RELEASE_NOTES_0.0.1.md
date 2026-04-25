# Mnemoss 0.0.1 — Release Notes

**Date:** 2026-04-26
**Status:** First formal alpha release of the ACT-R memory architecture.

This release lands the architectural rework that turns Mnemoss from
"works in fresh-corpus benchmarks" into "beats raw-cosine baselines
on realistic agent workloads at 2× lower latency."

---

## TL;DR

Under five new default settings, Mnemoss now beats a flat cosine
baseline on **both** axes simultaneously:

| Workload | Mnemoss recall@10 | raw_stack recall@10 | Mnemoss p50 | raw_stack p50 |
|---|---:|---:|---:|---:|
| N=20K LoCoMo + chains, MiniLM | **0.4622** | 0.4205 | **~25 ms** | 54 ms |

That's **+4.17pp recall** at **2× faster latency**. Bench pipeline:
`bench/bench_multi_step.py` with `--locomo-scale 20000
--rebalance-each-phase --arms mnemoss_supersede`.

The architecture's **upper bound** under perfect classification
(`bench/bench_tier_oracle.py`) is recall@10 = 0.7122 (MiniLM) /
0.7683 (Nomic v2 MoE). The 25-30pp gap from realistic to ceiling
defines the headroom for post-MVP work on classification quality.

---

## What's new — five default changes

### 1. Capacity-based tier bucketing

`Rebalance` now ranks memories by `idx_priority` and fills HOT (200),
WARM (2,000), COLD (20,000), DEEP (rest) top-down by capacity, not
threshold. Replaces the old `idx_priority > 0.7 → HOT` rule which
degenerated under any aged corpus (>99% of memories collapsed into
DEEP, defeating the cascade).

Cognitively grounded: working memory is hard-capped (Miller 1956:
7±2; Cowan 2001: 4±1), not threshold-gated. A polymath knowing 10M
facts has the same ~5-item working memory as anyone else.

`TierCapacityParams` config dataclass added; pin handling baked in
(pinned memories take HOT seats regardless of rank).

### 2. Tier-cascade-pure-cosine recall path

`use_tier_cascade_recall=True` is the new default. Recall reads tier
classifications that Dream/Rebalance computed off the read path;
ranking within a tier is **pure cosine**. No per-candidate `B_i`
recompute, no spreading activation, no matching `idx_priority` gate,
no `τ` floor.

Codified as **Principle 9** in §3 of the project knowledge doc:
*`idx_priority` is for ranking, not search.* The activation formula
governs lifecycle classification (which tier, when to dispose, what
to export); cosine governs query-time relevance. Mixing the two at
recall produced the failure modes documented during the rework.

Legacy ACT-R recall remains opt-in via `use_tier_cascade_recall=False`
for workloads needing within-session chain-version supersession.

### 3. Cascade short-circuit disabled by default

`cascade_min_cosine = 0.99`. Real-world cosines rarely reach 0.99,
so the cascade exhausts every populated tier on each query. The
earlier 0.5 default caused a 4.7pp recall regression because
realistic Rebalance can't reliably put every gold answer in HOT —
short-circuit at HOT then returned popular distractors instead of
the actual answer.

May be lowered when the Rebalance signal improves (selective
reconsolidation API or query-aware classification). The
`bench_tier_oracle.py` ceiling is the canonical re-evaluation gate.

### 4. Cosine-gated reconsolidation

`reconsolidate_min_cosine = 0.7`. Only memories whose query-time
cosine clears the threshold get `access_history` bumps when
`reconsolidate=True` is passed to `recall()`. Reduces "popular
distractor" promotion in subsequent Rebalances.

Sweep on `bench/bench_rebalance_lift.py` (N=20K MiniLM):

| Threshold | Test recall@10 | p50 |
|---:|---:|---:|
| ungated (-1.0) | 0.3737 | 16 ms |
| 0.5 | 0.3737 | 16 ms |
| **0.7** (shipped) | **0.3882** | 23 ms |
| 0.8 | 0.3975 | 23 ms |

### 5. `supersede_on_observe = True` is the new default

Was opt-in. At the conservative 0.85 cosine threshold, the mechanism
filters near-duplicate memories from recall (verbatim repeats,
multi-writer races, accidental re-ingests). The 0.85 threshold has
< 1% false-positive rate on the 50-pair non-contradiction bench.

Combined with the four other changes, this is the configuration
that lifts realistic LoCoMo recall above raw_stack for the first
time (0.4622 vs 0.4205, +4.17pp).

---

## Bench evidence

All numbers reproducible from `bench/results/README.md`. Key benches:

| Bench | Question answered |
|---|---|
| `bench/bench_tier_lifecycle.py` | Does the formula produce a sane tier distribution under aged corpora? |
| `bench/bench_rebalance_lift.py` | Does running Rebalance speed up subsequent recall? |
| `bench/bench_tier_oracle.py` | What's the architectural recall ceiling if classification were perfect? |
| `bench/bench_multi_step.py` | Does the architecture handle within-session chain-version supersession? |
| `bench/bench_d_recall_sweep.py` | Where does the time-decay knee sit on the LoCoMo+supersession Pareto curve? |

The oracle ceiling test (§8 of `bench/results/README.md`) is the
diagnostic of architectural headroom — re-run it with new embedders
or after any architecture change to verify the ceiling hasn't
regressed.

---

## What this release does NOT solve

**Within-session chain-version supersession** (e.g., "junior eng at
Google" → "staff eng at Stripe" within minutes, no Rebalance fired).
The tier-cascade-pure-cosine recall path cannot recover the 50%
latest@1 win that the legacy ACT-R recall delivered on this case.
Three workarounds:

1. **`supersede_on_observe`** at 0.85 catches near-duplicate
   contradictions but not chain versions (their cosine is below
   threshold by design).
2. **Disposal during Dream** removes contradicted facts in bulk —
   slow, depends on Dream cadence.
3. **Opt-in legacy recall**: `use_tier_cascade_recall=False` for
   evolving-state agent workloads where within-session supersession
   matters more than warm-cache speed.

The fundamental tradeoff is documented in `MNEMOSS_PROJECT_KNOWLEDGE.md`
§17, gap #4.

---

## Migration

Behavior changes that may affect existing callers:

| Previous behavior | New behavior | How to restore old |
|---|---|---|
| Activation-scored recall (`compute_activation` per candidate) | Pure cosine within tier | `FormulaParams(use_tier_cascade_recall=False)` |
| Threshold-mapped tier (`idx_priority > 0.7 → HOT`) | Capacity-bucketed tier | (no opt-out — capacity is structurally better) |
| Cascade short-circuit at `cascade_min_cosine=0.5` | No short-circuit (`0.99`) | `FormulaParams(cascade_min_cosine=0.5)` |
| Indiscriminate reconsolidation | Gated at `reconsolidate_min_cosine=0.7` | `FormulaParams(reconsolidate_min_cosine=-1.0)` |
| `supersede_on_observe=False` (off) | `True` (on) | `EncoderParams(supersede_on_observe=False)` |

Existing workspaces continue to work — `index_tier` values from the
old threshold rule will be miscalibrated until first Rebalance under
the new code, after which they self-correct. No schema migration
required.

---

## Roadmap

The 25-30pp gap to the oracle ceiling defines the next milestone.
Three paths, none of them this release:

1. **Selective reconsolidation API** — `mem.reinforce([m1, m3])`
   after the agent acts on retrieved memories. Higher-quality
   classification signal than indiscriminate top-k bumping.
2. **Query-aware classification** — a fast classifier that, at
   observe or recall, marks high-likelihood-gold memories ahead of
   Rebalance.
3. **Adaptive cascade short-circuit** — re-enable at lower
   thresholds when classification confidence is high. Requires a
   confidence signal not currently available.

Closing even half the ceiling gap would shift the headline to
"recall@10 ≈ 0.55, p50 ≤ 25 ms" — meaningfully better than any
flat-cosine baseline.

---

## Engineering notes

- All 792 unit tests pass under the new defaults
- No schema version bump (`SCHEMA_VERSION=8` unchanged) — new
  defaults are runtime, not on-disk
- Bench harness extended with `--rebalance-before-scoring`,
  `--rebalance-each-phase`, `--correlated-active`,
  `--reconsolidate-min-cosine`, `--cascade-min-cosine`,
  `--hot-cap`/`--warm-cap`/`--cold-cap`, `--skip-oracle-placement`
  flags for re-validation
- New benches: `bench_tier_lifecycle.py`, `bench_rebalance_lift.py`,
  `bench_tier_oracle.py`
- `bench_tier_oracle.py` introduces a `_BatchCachedEmbedder` that
  pre-embeds in batches and caches by text — 2.5× speedup on Nomic
  v2 MoE at N=20K vs naive per-call embedding
- Repository organization consolidated: published supersession
  bundle (1066-line README + 13 cited result files) moved from
  `reports/supersession_bench/` to `bench/published/`
