# TODOs

Captured follow-ups not in scope for the current PR. Each item has enough
context that someone (you or a CC agent) picking it up in 3 months can act
without re-deriving the motivation.

---

## Pre-commit hook for pre-registration discipline (dreaming-validation study)

**What:** A `.git/hooks/pre-commit` script (or `pre-commit` framework config) that refuses to commit any file matching `bench/ablate_dreaming*` if `docs/dreaming-decision.md` has zero prior commits in `git log`.

**Why:** The dreaming-validation study (see `~/.gstack/projects/opcify-Mnemoss/yangqi-worktree-expressive-brewing-cloud-design-20260427-001958.md`) relies on `docs/dreaming-decision.md` being committed *before* any harness code. Git timestamps are the pre-registration audit trail. Today this is honor-system; a hook makes it mechanical and prevents accidental discipline-breaking.

**Pros:**
- Catches the discipline violation at commit time, not during a six-months-later audit
- ~30 min to write
- The validation results are only trustworthy if the discipline held

**Cons:**
- One more local hook to maintain
- Only relevant during the validation study (could remove after weekend 4)
- Hooks are local; doesn't enforce on collaborators (though this is a solo study)

**Context:** Captured as a TODO during /plan-eng-review on 2026-04-27. The /office-hours design selected Approach B (full two-corpus study) with C's pre-registration discipline absorbed. The hook is the mechanical enforcement of that discipline. Implementation: ~10 lines of bash checking `git log --oneline docs/dreaming-decision.md | wc -l` is non-zero before allowing staged `bench/ablate_dreaming*` files.

**Depends on / blocked by:** `docs/dreaming-decision.md` must exist (weekend 1, task 1). Hook can be enabled the same day as the first decision-doc commit.

---

## Fan-out `DreamerParams` to SDK / REST / MCP / framework adapters

**What:** When `DreamerParams` lands in `core/config.py` (during the dreaming-validation study, weekend 1), it is intentionally scoped to `Mnemoss(__init__)` plumbing only. The SDK (`src/mnemoss/sdk/`), REST server schemas (`src/mnemoss/server/`), MCP wrapper (`src/mnemoss/mcp/`), TypeScript SDK (`sdks/typescript/`), and framework adapters (`adapters/hermes-agent/`, `adapters/openclaw/`) do NOT learn about it in that PR. This TODO captures the eventual fan-out.

**Why:** Per CLAUDE.md's "touched-the-API" discipline, every public API addition normally fans out across the SDK / REST / MCP / adapters. `DreamerParams` skipped that during the study because the harness doesn't need it to cross the wire — `Mnemoss(__init__)` plumbing is sufficient for an internal benchmark. But once a real consumer wants to tune `cluster_min_size` / `replay_limit` from a remote agent or a TypeScript SDK call, the fan-out becomes load-bearing.

**Pros:**
- Completes the API surface consistently with `FormulaParams` / `EncoderParams` / `SegmentationParams` patterns
- Lets remote agents (via REST) and TypeScript callers tune dream behavior
- Removes an inconsistency that would surprise future contributors

**Cons:**
- ~3-4h work across 5+ packages
- No demonstrated consumer demand yet — this is "build it because the pattern says so," not "build it because someone needs it"
- Adds maintenance surface (e.g., REST schema versioning) for params that 99% of users won't touch

**Context:** Captured during /plan-eng-review on 2026-04-27. The /office-hours session on the same day designed the dreaming-validation study; the eng review surfaced that `Mnemoss.dream()` hardcodes `cluster_min_size=3` etc. in `client.py:435-442`, which the harness can't override without `DreamerParams`. The user initially chose to fan out in this PR, then revised after outside-voice subagent flagged it as "launch-quality work for a benchmark harness." The narrower scope (config + Mnemoss only) ships in the validation PR; this TODO captures the eventual fan-out when there's a real consumer.

**Depends on / blocked by:** `DreamerParams` must land first (validation study weekend 1). Triggered by: any user request to tune dream params from REST / SDK / MCP / TypeScript / a remote agent.

---

## Adaptive tier caps — post-merge follow-ups

**What:** Five small follow-ups identified by the final whole-branch review of the adaptive-tier-caps work (Method C of the cap-calibration proposal). The feature itself is fully shipped, tested, and behind an opt-in flag (`FormulaParams.adaptive_tier_caps=False` by default). These items polish operability and ship validation; none blocks the feature.

1. **Surface `adaptive_caps` in `mnemoss-inspect` human output.** The JSON path already includes the block (since `mnemoss-inspect --json` dumps `status()` wholesale). The formatted table in `src/mnemoss/cli/inspect.py` iterates only `workspace`/`memories`/`llm cost`/`dreams`/`timestamps` sections — adaptive caps is invisible to operators who run plain `mnemoss-inspect <ws>`. ~10 lines: an `if status.get("adaptive_caps", {}).get("queries_since_adjustment", 0) > 0:` section with current effective caps + last delta + queries-since-adjustment.

2. **Document or fix the toggle-cycle resume semantic.** If a user enables the flag, lets the controller drift caps, then disables and re-enables it later, `read_effective_caps()` returns the previously persisted caps as the new starting point. There's no kill switch that clears `adaptive:caps_*` keys from `workspace_meta`. Arguably correct (resume where you left off) but could surprise an operator who expected "off" to reset state. Either document it in the CLAUDE.md invariant block, or add a `mnemoss-inspect --reset-adaptive-caps` flag.

3. **Reword "byte-identical" → "behaviorally identical" in the spec + CLAUDE.md.** The spec/invariant copy says "Default off → byte-identical no-op." The observable behavior IS identical (no telemetry written, no caps adjusted, recall results unchanged), but `recall/engine.py:_tier_cascade_recall` runs `time.perf_counter()` and allocates an empty `candidate_tier: dict` on every call regardless of the flag — sub-microsecond CPU overhead, far inside the <50ms hot-path budget but technically not zero. "Behaviorally identical" is the more accurate phrasing.

4. **Run `bench_rebalance_lift.py` and `bench_tier_lifecycle.py` with `adaptive_tier_caps=True`.** The plan's testing strategy listed end-to-end bench sanity as item 4 but deferred it (benches are standalone, not wired into pytest). Worth running post-merge against the LoCoMo corpus before recommending the flag in production — confirms recall + latency don't regress versus the static-cap baseline on a real workload.

5. **Consider Prometheus metrics for adaptive-caps telemetry** under the `[observability]` extra. The `status().adaptive_caps` block surfaces effective caps + telemetry-window state for ad-hoc inspection; exposing the same numbers as Prometheus gauges/counters would let operators watch drift in dashboards alongside the existing cost + dream metrics.

**Why:** All five came out of a careful final review and are real improvements, but none is blocking. Capturing them here so future work picks them up cleanly.

**Pros:**
- Items 1 + 2 close real operability gaps that would cause "what just happened?" support questions if anyone enables the flag.
- Item 4 is the empirical sanity check that the plan promised but deferred — needed before recommending the flag for production rollout.
- Items 3 + 5 are doc/polish that improves precision and observability.

**Cons:**
- All cosmetic / operability — none changes behavior or correctness.
- Item 5 in particular may be premature until someone actually opts into the flag in production.

**Context:** Generated by the final whole-branch reviewer for `adaptive-tier-caps` (final review at HEAD ≈ `c693917`). Spec at `docs/superpowers/specs/2026-05-15-adaptive-tier-caps-design.md`; plan at `docs/superpowers/plans/2026-05-15-adaptive-tier-caps.md`.

**Depends on / blocked by:** None. Items 1–3 are independently small. Item 4 needs an embedder + the LoCoMo corpus already in `bench/data/`. Item 5 should follow whoever first turns the flag on for a real workload.

