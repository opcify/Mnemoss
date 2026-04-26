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

