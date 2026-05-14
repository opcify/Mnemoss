# Adaptive Tier Caps — Design Spec

- **Date:** 2026-05-15
- **Status:** Approved design, pre-implementation
- **Topic:** Runtime self-tuning of the multi-tier index capacity caps
  (`hot_cap` / `warm_cap` / `cold_cap`)

## Context & Problem

The multi-tier index buckets every memory into HOT / WARM / COLD / DEEP.
Rebalance (`src/mnemoss/index/rebalance.py`) ranks all memories by
`idx_priority` descending and fills the tiers top-down by **absolute
capacity caps** from `TierCapacityParams` — `hot_cap=200`,
`warm_cap=2000`, `cold_cap=20000`. DEEP absorbs the tail. This
capacity-based scheme replaced an older threshold-based one
(`HOT iff idx_priority > 0.7`) that provably degenerated at scale.

The capacity scheme is structurally sound — it gives the `O(log)`
cascade bound regardless of `N`. The weak spot is that the **cap
values themselves are guesses**: "cognitive-realistic order-of-
magnitude estimates" with a Miller 7±2 / Cowan 4±1 justification that
argues for ~7, not 200. They are fixed regardless of workload,
hardware, or corpus age.

The repo already *measures* the consequences but never *acts* on the
measurements:

- `bench/scale_cascade_audit.py` — confirms fresh `observe()` sets
  `idx_priority ≈ 0.731`, so every memory lands in HOT until Rebalance
  runs; the cascade is dead code on un-rebalanced workspaces.
- `bench/bench_tier_lifecycle.py` — backdates 6 months of usage to ask
  whether HOT/WARM/COLD carry meaningful share or HOT swallows
  everything.
- `bench/bench_tier_oracle.py` — separates classification error from
  ranking error; measured a real **−4.7pp recall gap** (mnemoss
  recall@10 = 0.3737 vs raw_stack 0.4205).
- `bench/bench_rebalance_lift.py` — measures whether Rebalance speeds
  up subsequent recall.

The caps trade off two opposing forces: bigger tiers = safer recall
(gold memory more likely inside a scanned tier), smaller tiers =
faster cascade. With `cascade_min_cosine=0.99` and `include_deep=False`
as the production default, a too-small HOT **silently misses** gold
answers. `TierCapacityParams` is plain config — changing how the caps
are *derived* is not a schema change.

## Goal

Make the tier caps **self-tuning at runtime**: a controller observes
recall telemetry from this specific workspace and nudges the effective
caps toward a workload-appropriate operating point, seeded from
today's static defaults.

This is "Method C" from the three-method proposal (the other two —
an offline Pareto sweep harness, and a hardware latency-budget probe —
were considered and deprioritized for now).

## Non-Goals

- Not building the offline calibration harness (Method A) or the
  hardware latency probe (Method B).
- Not changing the Rebalance bucketing algorithm itself
  (`_bucket_by_capacity` stays as-is).
- Not changing the activation formula, `idx_priority`, or the recall
  ranking logic.
- Not deriving `max_cap` from hardware — it is a constant for now
  (that would be Method B).

## Decisions Locked During Brainstorming

1. **Validation approach:** a synthetic aged-workspace test fixture —
   a minimal slice of `bench_tier_lifecycle.py`'s backdating machinery,
   used purely as a test rig (not shipped as a calibration tool), with
   a known-good cap target the controller must converge toward.
2. **Controller objective:** a **blended score with one knob** — a
   single weighted combination of a latency-pressure signal and a
   recall-loss signal, with one tunable parameter `λ` biasing the
   trade-off.

## Design

### New module: `src/mnemoss/index/adaptive_caps.py`

Sits next to `rebalance.py`. Its core is a **pure function**, matching
the `formula/` package philosophy so the decision logic is testable in
isolation:

```python
compute_adjusted_caps(
    current: TierCapacityParams,
    telemetry: TierTelemetry,
    params: FormulaParams,
) -> TierCapacityParams
```

The telemetry read/write and `workspace_meta` persistence live in the
store/rebalance layer; the **decision** is pure and side-effect-free.

A `TierTelemetry` dataclass carries the accumulated signal counters
consumed by the controller.

### Control mechanism

At each Rebalance the controller reads accumulated recall telemetry
and decides whether to nudge the caps.

**Two pressure signals**, derived from telemetry the engine already
mostly produces:

- `latency_pressure` rises when scans are wasteful — high
  `candidates_scored` mean, cascade rarely short-circuits at HOT.
  Pushes caps **down**.
- `recall_pressure` rises when recall is leaking — high COLD/DEEP
  winner share, frequent `reminisce_to_warm` events. Pushes caps
  **up**.

**The blended score arbitrates** when the two conflict. One knob
`λ ∈ [0, 1]`:

```
delta = (1 − λ) · recall_pressure − λ · latency_pressure
```

Positive → grow; negative → shrink; magnitude → step size
(proportional controller).

**Applied multiplicatively** (AIMD-style — stable across scales):

```
hot_cap *= (1 + clamp(delta, ±max_step))
```

then clamp to `[min_floor, max_cap]`. WARM and COLD move on the same
signal with their own bounds.

**Stability guardrails** (the part that makes or breaks this):

- **Dead-band** — `|delta|` below `adaptive_tier_deadband` → no change
  at all (kills thrashing on noise).
- **Min-dwell** — no adjustment unless ≥ `adaptive_tier_min_queries`
  queries accumulated since the last change.
- **Max-step** — bounded multiplicative change per adjustment, can't
  lurch.
- **Bounds clamp** — `[min_floor, max_cap]` preserves the `O(log)`
  structural guarantee. `max_cap` is a constant.

The cold-start seed is today's `TierCapacityParams` defaults — the
controller starts from the known-reasonable guess and adapts.

### Integration points

Two touch points, both at existing choke points:

1. **Telemetry collection — `recall/engine.py`, `_tier_cascade_recall`.**
   The engine already builds `CascadeStats` (`stopped_at`,
   `candidates_scored`). Add **winner provenance** — for the returned
   top-k, which tier's `vec_search` first surfaced each result,
   captured at scan time before reminiscence mutates `index_tier`.
   Reminiscence events are already emitted here. These increment
   **in-memory counters** on the engine — cheap, no I/O, the <50ms hot
   path is untouched. Entirely gated behind the opt-in flag.

2. **Controller run — Dream P7 Rebalance (`index/rebalance.py`).**
   After Stage 1 recompute, before Stage 2 bucketing: flush in-memory
   counters to `workspace_meta`, read accumulated `TierTelemetry`, call
   `compute_adjusted_caps(...)`, persist the new caps to
   `workspace_meta`, reset the counters. Stage 2 then buckets with the
   **effective caps** (from `workspace_meta`) instead of the static
   `TierCapacityParams`. `_bucket_by_capacity` itself is unchanged.

**Persistence.** Counters and effective caps live in `workspace_meta` —
the exact KV table and pattern `CostLedger` (`dream/cost.py`) uses.
Survives close/reopen, per-workspace. No new table. `workspace_meta`
is schema-flexible KV — to be verified during implementation, but
likely **no `SCHEMA_VERSION` bump**.

### Config surface

Added to `FormulaParams`, all with `__post_init__` validators per repo
convention:

| Param | Default | Meaning |
|---|---|---|
| `adaptive_tier_caps` | `False` | Opt-in master switch (mirrors `use_fast_index_recall`) |
| `adaptive_tier_lambda` | `0.5` | The one blend knob — 0 = pure recall-safety, 1 = pure latency |
| `adaptive_tier_min_queries` | `200` | Min-dwell before any adjustment |
| `adaptive_tier_max_step` | `0.2` | Max ±20% multiplicative move per adjustment |
| `adaptive_tier_deadband` | `0.05` | `delta` magnitude below this → no change |

Added to `TierCapacityParams`:

| Param | Default | Meaning |
|---|---|---|
| `min_floor` | `20` | Lower clamp on every tier cap |
| `max_cap` | (generous constant) | Upper clamp — preserves the `O(log)` guarantee |

`TierCapacityParams` keeps its current role as the **seed**; effective
caps diverge into `workspace_meta` once the controller runs.

### Observability

`status()` already surfaces `tier_counts` and `llm_cost`. Add an
`adaptive_caps` block — current effective caps, last adjustment delta,
queries-since-adjustment — so operators can watch the controller move.
Every value stays primitive/list/dict per the `status()` shape
invariant.

### Error handling & failure modes

The controller runs inside Dream P7, which already wraps every phase in
try/except — so a controller exception never kills a dream run; it is
recorded as `PhaseOutcome.status="error"` with `degraded_mode=True`,
and recall keeps working on whatever caps `workspace_meta` last held.

On top of that baseline:

- **Missing/corrupt telemetry** (fresh workspace, partial write, absent
  key) → fall back to the seed `TierCapacityParams`, skip the
  adjustment this pass, log a warning. The controller is *advisory* —
  absence of telemetry means "don't move," never "crash."
- **Degenerate computed caps** (NaN, negative, `hot > warm` ordering
  violation) → `TierCapacityParams.__post_init__` validators reject
  these; the controller catches the `ValueError`, **keeps the previous
  caps**, logs. A bad computation cannot poison the workspace.
- **Insufficient data** (queries-since-adjustment <
  `adaptive_tier_min_queries`) → not an error, a no-op skip. Dead-band
  and min-dwell are the *normal* resting state.
- **Flag off (`adaptive_tier_caps=False`)** → telemetry counters never
  increment, the controller is never called, effective caps = static
  `TierCapacityParams`. This path must be **byte-identical to today's
  behavior**, asserted by an explicit test.

### Named risk

The winner-provenance signal has **no ground truth** — a query
returning junk from HOT still counts as a HOT win. The dead-band +
min-dwell + max-step damp this, and the convergence fixture is
specifically designed to catch a controller that drifts on biased
signal. But it is a proxy, and the design's honest position is
**stable and conservative beats clever** — when in doubt, the
controller does nothing.

## Testing Strategy

Four layers, lightest to heaviest:

1. **Unit — `compute_adjusted_caps` pure function**
   (`tests/test_adaptive_caps.py`). Direction correctness
   (recall_pressure grows, latency_pressure shrinks), proportional
   magnitude, dead-band suppresses small deltas, max-step clamps large
   ones, bounds clamp to `[min_floor, max_cap]`, ordering invariant
   preserved.

2. **Integration — telemetry round-trip.** Counters increment in
   `_tier_cascade_recall`, flush to `workspace_meta`, survive
   close/reopen, reset after the controller consumes them. Plus the
   **no-op assertion**: with `adaptive_tier_caps=False`, recall results
   and tier distribution are identical to a run without the feature.

3. **Convergence fixture — the safety net.** A synthetic aged
   workspace (minimal slice of `bench_tier_lifecycle.py` backdating)
   built so a *known* number of memories are genuinely "hot-worthy"
   (e.g. ~120). Run the recall→Rebalance loop K times and assert:
   - **Convergence** — `hot_cap` moves from the 200 seed toward ~120
     and settles.
   - **Stability** — once settled, stays inside the dead-band; no
     oscillation across further rounds.
   - **Workload-shift tracking** — flip the synthetic workload (~400
     hot-worthy) and assert re-convergence within K Rebalances.
   - **Adversarial** — feed a deliberately noisy/biased signal and
     assert caps stay put (dead-band holds).

4. **End-to-end sanity — existing benches.** Run
   `bench_rebalance_lift.py` and `bench_tier_lifecycle.py` with
   `adaptive_tier_caps=True` to confirm recall and latency do not
   regress versus the static-cap baseline on the real LoCoMo corpus.

`mypy --strict src/mnemoss` + `ruff check`/`ruff format` throughout.

## Architectural Invariants & Principles Check

- **Principle 1 (formula drives everything, no LLM in system
  decisions)** — upheld. The controller is pure arithmetic over
  telemetry; no LLM involved.
- **Principle 9 (`idx_priority` is for ranking, not search)** —
  unaffected; the controller tunes *capacity*, not ranking, and search
  still uses pure cosine within tiers.
- **Capacity-based bucketing invariant** — preserved. The controller
  changes the *values* of the caps, not the rank-and-fill mechanism;
  the `[min_floor, max_cap]` clamp keeps the `O(log)` structural
  bound.
- **`status()` shape invariant** — the new `adaptive_caps` block keeps
  all values primitive/list/dict.
- **Hot Path <50ms, zero LLM** — telemetry is in-memory counter
  increments only; flush happens at Rebalance, not on the read path.
- **Opt-in, default off** — mirrors `use_fast_index_recall`; the
  `adaptive_tier_caps=False` path is byte-identical to current
  behavior, so this is not a breaking change.
- **`SCHEMA_VERSION`** — likely no bump (`workspace_meta` is
  schema-flexible KV); to be confirmed during implementation. If a
  migration turns out to be needed, follow the registered-chain
  framework in `store/migrations.py`.

## Open Questions for Implementation

- Confirm `workspace_meta` can hold the new counter/cap keys without a
  `SCHEMA_VERSION` bump; if not, add a `Migration` step.
- Exact functional form of `latency_pressure` and `recall_pressure`
  (normalization, which telemetry terms, weighting) — to be pinned in
  the implementation plan and tuned against the convergence fixture.
- Whether the full ACT-R recall path (not just `_tier_cascade_recall`)
  also needs telemetry instrumentation, or whether adaptive caps are
  scoped to the tier-cascade production default only.
- Default value for `max_cap` on `TierCapacityParams`.

## Effort

Medium–large. New controller module, telemetry plumbing in
`recall/engine.py` + `index/rebalance.py`, `workspace_meta` counters,
opt-in config, `status()` block, and — the load-bearing part — the
convergence fixture and control-loop stability tests.
