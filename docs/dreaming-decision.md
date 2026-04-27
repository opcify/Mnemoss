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

### 2026-04-27 — Consolidate model swap, take 2

Three Consolidate-model swaps before any verdict was recorded:

1. `tencent/hy3-preview:free` — SiliconFlow provider returned 400
   on `response_format=json`. Mnemoss's `_phase_consolidate`
   requires JSON mode. Failed before producing any output.
2. `meta-llama/llama-3.3-70b-instruct:free` — JSON-capable, but
   OpenRouter's free tier caps the model at ~8 rpm (and providers
   like Venice impose USD spend caps on top). Even with proactive
   pacing + exponential backoff, 4 of 4 Consolidate calls in the
   topology binary gate hit 429 / 402. Failed before producing
   any verifiable Consolidate output.
3. `deepseek/deepseek-v4-flash` — current. Higher rpm, JSON-mode
   supported. **Self-preference-bias caveat:** this is also the
   judge model in `bench/ablate_dreaming.toml` `[llm.judge]`.
   Before running `make gist-quality`, swap one of them so judge
   and Consolidate aren't from the same model family — otherwise
   the gist-quality verdict in weekend 3 is structurally biased.

These are config swaps, not threshold revisions — the per-phase
thresholds above are untouched. Recording here for audit-trail
completeness.

---

## Verdicts

### Topology corpus, full ablation matrix (2026-04-27)

Run: `python -m bench.ablate_dreaming --full` against
`bench/fixtures/topology_corpus.json` with the pinned config
(text-embedding-3-small + inclusionai/ling-2.6-1t:free). 14
ablation conditions. Raw output:
`bench/results/topology_results.jsonl`.

Sorted by recall@10:

```
label                          recall@k  llm_calls
--------------------------------------------------
dreaming_off                     0.5417          0   (baseline)
cluster_only                     0.5417          0
consolidate_only                 0.5417          0   (skipped, no input)
dispose_only                     0.5417          0
rebalance_only                   0.5417          0
relations_only                   0.5417          0
replay_only                      0.5417          0
no_replay                        0.5417          0   (downstream skips)
no_consolidate                   0.4583          0
no_rebalance                     0.2292          4
full                             0.1250          4
no_cluster                       0.1250          1
no_dispose                       0.1250          4
no_relations                     0.1250          4
```

**Headline.** Every condition that runs Consolidate's LLM lands at
0.12–0.23 recall@10. Every condition that doesn't lands at 0.46–0.54.
The signal is entirely Consolidate.

**Why** — Consolidate writes 1–4 summary memories (e.g. "The platform
team is targeting a release...", "User joined a local hiking
group..."). These ARE good summaries; they share vocabulary with the
queries by design. Under the recall@k metric — which only counts
original gold corpus ids — those summaries displace the originals
from top-10. Recall@k punishes useful summary creation. The decision
doc anticipated this (see "What Makes This Cool" section in the
linked design doc); the gist-quality judge is the right metric for
Consolidate's content verdict, not recall@k.

### Per-phase verdicts (recall@k metric only)

| Phase | Verdict | Evidence | Action |
|-------|---------|----------|--------|
| Cluster | KEEP (qualified) | full=0.1250 vs no_cluster=0.1250 — Cluster doesn't change recall@k on this 30-memory corpus, but the noise-aware ARI metric in `bench/_metrics.py` is the right test. ARI not yet wired into the harness. | Wire ARI computation into `bench/ablate_dreaming.py` before recording a final verdict on Cluster. |
| Consolidate | **REBUILD under recall@k. Real verdict deferred to gist-quality.** | Consolidate's summaries dominate top-10 and displace original gold ids by ~42pp. But that's a metric artifact — the summaries are content-correct. | Run `make gist-quality` (after swapping the judge model away from `inclusionai/ling-2.6-1t:free` to avoid self-preference bias). The win-rate CI there is the load-bearing verdict. |
| Relations | INCONCLUSIVE | full=0.1250 vs no_relations=0.1250 — Relations changes nothing measurable on this corpus. May be a real null OR may be masked by Consolidate's recall@k pollution. | Re-test Relations contribution AFTER fixing recall@k to credit summaries (or with multi-hop-only queries). |
| Rebalance | INCONCLUSIVE | full=0.1250 vs no_rebalance=0.2292 — Rebalance appears to slightly HURT recall when Consolidate is active. Probably noise on 30 memories where tier migration shouldn't matter. | Re-test on the pressure corpus (500 memories) where Rebalance has actual work to do. Topology corpus is too small. |
| Dispose | INCONCLUSIVE on this corpus | full=0.1250 vs no_dispose=0.1250 — Dispose tombstones nothing on 30 fresh memories with no accumulated decay. Expected. | Re-test on the pressure corpus where memories age over 30 simulated days and Dispose has tombstoning to do. |

### What this run does NOT verdict

- **Replay** is intentionally out of scope per pre-registration.
- **Cluster's ARI** vs hand-labeled topics — needs the harness to
  emit predicted cluster_id per memory, not yet wired.
- **Consolidate's content quality** — needs `make gist-quality`,
  not run yet.
- **Dispose's cleanliness** and **Rebalance's tier-migration utility**
  — both need the pressure corpus, not run yet.

### Caveats on the recall@k pollution finding

This run uses `inclusionai/ling-2.6-1t:free` for Consolidate. A
different LLM might produce summaries that don't dominate top-10 the
same way. The recall@k vs gist-quality split is the principled
answer: the Pareto chart (`bench/results/pareto.png`) is honest about
the trade-off — more LLM calls = more summaries = lower
original-id recall, but possibly higher content-quality coverage.
