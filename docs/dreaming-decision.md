# Dreaming Validation — Decision Document

**Status:** PRE-REGISTERED (thresholds locked, harness not yet written)
**Created:** 2026-04-27
**Branch:** worktree-expressive-brewing-cloud
**Design doc:** `~/.gstack/projects/opcify-Mnemoss/yangqi-worktree-expressive-brewing-cloud-design-20260427-001958.md`

The git timestamp on this commit is the pre-registration. The harness in
`bench/ablate_dreaming.py` lands in a later commit. Any threshold change
after the harness ran must be appended to the **Threshold Amendments**
section at the bottom — never edit a numeric threshold in place.

---

## Replay — out of scope

Replay is a pass-through: it reads the most recent observe buffer plus
Working Memory's active set and emits a candidate list. Its only failure
mode ("emits an empty list when work is available") is already covered by
the existing unit tests in `tests/test_dream_*.py`. No threshold is
pre-registered for Replay; this study does not produce a verdict for it.

---

## Cluster

**Cognitive job.** Group memories that a human would call "the same topic"
so Consolidate can rewrite each group as one sharper gist. If Cluster
fails, every downstream phase's input is noise.

**Measurement.** Adjusted Rand Index (ARI) between HDBSCAN's predicted
cluster labels and hand-labeled topic labels on the topology corpus
(30 memories, 3 topics). Noise-aware: HDBSCAN's noise label
(`cluster_id=None`) is excluded from both sides before scoring.

**Cut threshold.** ARI < 0.5 → Cluster doesn't agree with human topic
intuition meaningfully better than chance. Topic groupings drive
Consolidate's prompts; if Cluster is random, Consolidate is rewriting
random groups.

**Keep threshold.** ARI > 0.7 → Cluster recovers human topic structure
substantially. Standard interpretation: 0.7+ on small corpora is "strong
agreement."

**REBUILD trigger.** 0.5 ≤ ARI ≤ 0.7 → Cluster is doing *something* but
not what human readers would call same-topic. Action: revisit corpus
embedding choice, HDBSCAN `min_cluster_size`, or whether the upstream
embedder (`text-embedding-3-small` at 1536d) is the right tool for short
conversational memories.

---

## Consolidate

**Cognitive job.** Take a cluster of related memories and emit a single
**better** gist than the level-1 heuristic produced. "Better" = more
useful for answering a downstream query about that topic.

**Measurement.** Hand-rolled pairwise LLM-as-judge in
`bench/gist_quality.py`. For each (query, cluster) pair where Consolidate
touched the gist, present the judge with [query, level-1 gist,
post-Consolidate gist] in randomized order, ask "which is more useful for
answering Q?" Win rate = fraction of comparisons where post-Consolidate
beats level-1, with bootstrapped 95% CI over queries. Judge model
(`deepseek/deepseek-v4-flash`) is in a different family from the
Consolidate model (`tencent/hy3-preview:free`) to mitigate self-preference
bias.

**Cut threshold.** Win rate CI upper bound ≤ 55% → post-Consolidate gists
are statistically indistinguishable from level-1. Consolidate is paying
LLM cost for no quality lift.

**Keep threshold.** Win rate CI lower bound ≥ 65% → post-Consolidate
gists clearly beat level-1. Consolidate earns its cost.

**REBUILD trigger.** CI overlaps both 50% and 65% → ambiguous lift.
Action: revisit the consolidation prompt template in
`src/mnemoss/dream/consolidate.py` (`build_consolidate_prompt`); the
schema may be guiding the LLM toward shape over substance.

---

## Relations

**Cognitive job.** Populate the relation graph with edges that let
spreading activation `Σ W_j·S_ji` find related memories the embedder
alone misses. The classic case: a multi-hop query like "what did I say
about X right after meeting Y" needs the X→Y edge to surface both
memories.

**Measurement.** Recall@10 delta on the topology corpus's 4 multi-hop
queries, comparing full-pipeline vs `phases={"replay","cluster","consolidate","rebalance","dispose"}`
(Relations excluded). Absolute percentage points.

**Cut threshold.** Multi-hop recall@10 delta < 3 absolute pp → Relations
graph isn't pulling its weight on multi-hop topology that requires
spreading activation. The graph is decorative.

**Keep threshold.** Multi-hop recall@10 delta > 8 absolute pp →
Relations is the load-bearing piece for multi-hop queries. Keep,
prioritize.

**REBUILD trigger.** 3 ≤ delta ≤ 8 pp → modest contribution; might be
worth a redesign of the edge-emission rules in
`src/mnemoss/dream/relations.py`.

---

## Rebalance

**Cognitive job.** Migrate memories across HOT/WARM/COLD/DEEP tiers so
recall's cascade returns relevant rows from the cheapest tier without
scanning DEEP. Read-side performance enabler; if useless, recall still
correct but slower.

**Measurement.** Recall@10 delta on the pressure corpus (~500 memories
accumulating over 30 simulated days), comparing full-pipeline vs
`phases=` (full set minus rebalance). Absolute percentage points. Also
inspect `status().tier_counts` over time to see whether HOT/WARM/COLD/
DEEP populations match the corpus's high/medium/low utility split (10/20/70).

**Cut threshold.** Recall@10 delta on pressure corpus < 2 absolute pp
**AND** tier_counts after Rebalance look indistinguishable from
tier_counts without Rebalance → tier migration changes nothing
measurable.

**Keep threshold.** Recall@10 delta > 5 absolute pp **OR** tier_counts
show clear utility-correlated migration (high-utility memories
concentrate in HOT, low-utility in COLD/DEEP) → Rebalance does what it
was designed to do.

**REBUILD trigger.** Tier migration happens but doesn't improve recall
**OR** improves recall but doesn't track utility. Action: revisit the
`idx_priority` formula and tier offset thresholds.

---

## Dispose

**Cognitive job.** Tombstone low-utility memories so recall's top-K
isn't polluted by junk that the embedder happens to surface. Adversarial
queries: those for which junk would otherwise pollute top-10.

**Measurement.** Top-K cleanliness on the pressure corpus's 30
adversarial queries — defined as: fraction of queries with **zero
junk-utility memories in top-10**. Compare full-pipeline vs `phases=`
(full set minus dispose).

**Cut threshold.** Cleanliness delta < 10% of queries → Dispose isn't
removing the junk that pollutes recall meaningfully more often than
chance. Tombstoning isn't the right intervention.

**Keep threshold.** Cleanliness delta > 30% of queries →
Dispose is load-bearing for top-K quality on accumulating corpora.

**REBUILD trigger.** 10–30% delta → Dispose helps but inconsistently.
Action: revisit the `max_A_i < τ - δ` criterion in
`src/mnemoss/dream/dispose.py`. May need a slower decay or a higher τ
to be more aggressive.

---

## Threshold Amendments

(Empty as of pre-registration commit. Any threshold revision after the
harness has run lands here, with date, original value, new value, and
the reasoning. Do not edit thresholds in place; appending here keeps the
audit trail honest.)

---

## Verdicts

(Filled in after the harness produces results. Each phase gets one of:
KEEP / CUT / REBUILD, with the cited evidence row from the harness
output and an action item for any CUT or REBUILD verdict.)

| Phase | Verdict | Evidence | Action |
|-------|---------|----------|--------|
| Cluster | TBD | TBD | — |
| Consolidate | TBD | TBD | — |
| Relations | TBD | TBD | — |
| Rebalance | TBD | TBD | — |
| Dispose | TBD | TBD | — |
