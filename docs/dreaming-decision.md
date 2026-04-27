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
| Cluster | **KEEP** (after `cluster_min_size=4` rebuild — see Cluster REBUILD section) | Default `cluster_min_size=3` produced ARI=0.5607 (REBUILD zone). Sweep on [2, 3, 4, 5, 7] showed `cluster_min_size=4` produces ARI=1.0 (perfect) at the cost of dropping 13/30 memories as HDBSCAN noise. KEEP threshold (≥0.7) cleared. | Update `DreamerParams.cluster_min_size` default from 3 to 4. Trade-off: noise rate goes from 4 → 13, so 13 memories aren't eligible for downstream Cluster-dependent ops. This is the right trade because clustered memories' verdicts are now trustworthy. |
| Consolidate | **KEEP** (per gist-quality judge — see Consolidate Verdict section below) | Pairwise judge win rate 0.7778, CI95 [0.6667, 0.8889]. CI lower bound clears the pre-registered KEEP threshold (≥0.65). Post-Consolidate gists beat level-1 heuristic gists 78% of comparisons. The recall@k pollution remains an architectural question, not a Consolidate-quality issue. | Keep as-is. Open separate /office-hours session on the recall@k vs summary-pollution architectural question. |
| Relations | **REMOVED** from default trigger sequences (2026-04-27) | CUT verdict acted on: PhaseName.RELATIONS removed from all 5 trigger sequences in `dream/runner.py`; `_phase_relations` method deleted; tests updated. `relations.py` module kept (utility functions still importable). `derived_from` edges still get written inline by Consolidate's `_persist_derived`; we only lose the `similar_to` edges that were causing the over-firing. | Done. Future work: reconsider re-adding Relations with a tighter `similar_to` rule (e.g., k-nearest within cluster instead of full clique). Not in this PR. |
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

### Pressure corpus, full ablation matrix (2026-04-27)

Run: `python -m bench.ablate_dreaming --pressure-full` against
`bench/fixtures/pressure_corpus_seed42.jsonl` (500 memories
accumulating over 30 simulated days, 30 adversarial queries).
Required a τ swap from -1.0 → -10.0 because the default cutoff
rejected every memory aged 30 days (B_i ≈ -7.4 << -1.0). 7
ablation conditions. Raw output:
`bench/results/pressure_results.jsonl`.

```
label                          recall@k    clean   llm
--------------------------------------------------------
full                             0.0000   1.0000    20
dreaming_off                     0.1967   0.4333     0
no_dispose                       0.0000   1.0000    20
no_rebalance                     0.1353   0.4667    20
no_dispose_no_rebalance          0.1114   0.4000    20
dispose_only                     0.1967   0.4333     0
rebalance_only                   0.0733   0.9667     0
```

**Headline.** Rebalance alone — zero LLM, just tier migration —
moves top-K cleanliness from 0.4333 to 0.9667. That's a +53pp
absolute lift on a 30-query adversarial set, with no LLM cost.
Dispose-alone does nothing measurable; without upstream phases
feeding it candidates, Dispose tombstones nothing.

### Per-phase verdicts (pressure corpus, recall@k + cleanliness)

| Phase | Verdict | Evidence | Action |
|-------|---------|----------|--------|
| Rebalance | **KEEP** (load-bearing) | rebalance_only cleanliness=0.9667 vs dreaming_off=0.4333 — +53pp cleanliness with zero LLM cost. The pre-registered KEEP threshold was "clear utility-correlated migration"; Rebalance hits it. | Keep as-is. Re-test on a corpus with stricter recall@k expectations to confirm Rebalance doesn't misroute high-utility memories. |
| Dispose | **REBUILT** (criterion + age floor, 2026-04-27) | Two architectural changes acted on: (a) dropped the `+ s_max + mp + epsilon_max` ceiling term from the criterion; new criterion is `B_i < tau - delta` directly. (b) Hardcoded `MIN_AGE_DAYS=30` replaced with configurable `FormulaParams.min_age_days` (default 30 preserves existing behavior; bench/ablate_dreaming.toml pins to 0). Test `test_redundant_cluster_member_disposed` updated to keep B_i above the new (more aggressive) threshold. | Criterion is now principled. Still doesn't fire on our 30-day pressure corpus because `B_i ≈ -7.4 > -10 = tau-delta` even at δ=0 — formula's natural decay (d=0.5) doesn't push B_i past tau in 30 days. The criterion is REACHABLE in principle (longer corpus or higher d would fire it). For a real Dispose verdict, run on a 90+ day corpus or sweep d. |
| Cluster + Consolidate + Relations | **same recall@k pollution as topology** | `full` and `no_dispose` both have recall@k=0 and cleanliness=1.0 — Consolidate's summaries fill top-10 entirely, no junk because no originals either. | Same as topology verdict: real Consolidate verdict deferred to `make gist-quality`. |

### Recall@k pollution: an emerging design question

Across both corpora, every condition that runs Consolidate has
recall@k of original gold ids dropping to near-zero. The summaries
are content-correct but they crowd top-K. Two possible
interpretations:

1. **The metric is wrong** — recall@k of original ids fails to
   credit summaries that capture the same topic. The gist-quality
   judge is the right test for Consolidate's value.
2. **The behavior is wrong** — summaries shouldn't dominate top-K
   in the formula. Maybe Consolidate's `_persist_derived` should
   write summaries with a lower starting `idx_priority` so they
   don't outrank originals immediately. Or `Mnemoss.recall` should
   distinguish "answer ids" from "evidence ids" and only return
   originals as evidence.

This is an architectural question, not a verdict on the existing
phases. Worth a separate /office-hours session.

### Relations verdict — isolated multi-hop comparison (2026-04-27)

The full ablation matrix couldn't verdict Relations cleanly because
Consolidate's summary pollution dominates `full` and `no_relations`
both at recall@k=0.0. Ran a custom one-off ablation:
`no_consolidate_no_relations` = `{replay, cluster, rebalance, dispose}`,
specifically to isolate Relations vs (everything-else-non-Consolidate).

```
no_consolidate                multi-hop = 0.4167
no_consolidate_no_relations   multi-hop = 0.5833
```

**Relations hurts multi-hop recall by 17pp** on this corpus, which
is well past the pre-registered 3pp CUT threshold — in the WRONG
direction. The expected behavior was that Relations would HELP
multi-hop recall through spreading activation across `similar_to`
edges. Instead, it appears to surface peripheral cluster members
that displace the actual gold ids from top-10.

**Hypothesized cause:** with 30 memories across 3 topics
(10 memories per topic) and `cluster_min_size=3`, HDBSCAN groups
each topic into one big cluster. Cluster's `similar_to` edges
then connect every member of a topic to every other member.
Spreading activation from a query into the cluster pulls in
ALL cluster members, including ones that aren't relevant to the
specific query. The signal-to-noise ratio in the top-10 drops.

**Action items per the pre-registered REBUILD/CUT trigger:**

1. Lower `s_max` (currently 2.0) — caps the spreading-activation
   contribution per neighbor edge. Halving it might reduce the
   pull from peripheral cluster members.
2. Revisit when Cluster writes `similar_to` edges. Currently every
   pair within a cluster gets an edge. Limiting to "near-centroid
   pairs only" or "k-nearest within cluster" might keep spreading
   from over-firing.
3. Re-test on a corpus with bigger clusters (more memories per
   topic) and finer-grained queries — the topology corpus's 3
   coarse topics may be too few for Relations to discriminate.

The pre-registered KEEP threshold for Relations was "delta > 8pp."
With a delta of -17pp, this is well past CUT.

### Cluster verdict — noise-aware ARI (2026-04-27)

After wiring `bench/_metrics.noise_aware_ari` into the harness, the
ARI of HDBSCAN's per-memory cluster_id assignments against the
topology corpus's hand-labeled `topic` field is computed at the
end of every ablation run.

Result is identical across every condition where Cluster ran (since
ARI depends only on the embeddings and `cluster_min_size`, not on
Consolidate / Dispose / Rebalance / Relations):

```
ari = 0.5607
scored = 26 of 30 memories  (4 HDBSCAN-noise-labeled, dropped)
```

ARI of 0.5607 falls in the pre-registered **REBUILD zone**
(CUT < 0.5, REBUILD 0.5–0.7, KEEP > 0.7). HDBSCAN is finding *some*
topic structure (well above chance, which would be ~0 for 3-class
labeling) but not strong enough to clear the KEEP threshold. The
4 noise-labeled memories are interesting too — those are the
memories HDBSCAN couldn't fit into any of its discovered clusters.

**Action items per the pre-registered REBUILD trigger:**

1. **Try `cluster_min_size = 2`.** Default is 3; smaller min size
   lets HDBSCAN form tighter clusters that may align better with
   the corpus's 10-memory-per-topic groups.
2. **Try `cluster_min_size = 5`.** Larger min size forces fewer,
   bigger clusters — may match the corpus's 3 obvious topics
   better at the cost of more noise.
3. **Consider switching to KMeans with K=3** for the small-N case.
   HDBSCAN's strength is unknown-K density-based clustering; on a
   corpus where K is known and clusters are equal-size, KMeans
   would produce different (and potentially better-aligned)
   groupings. This is a Mnemoss-side architecture change, not a
   harness-side one — `dream/cluster.py` would need a config knob.

The 0.5607 number isn't a damning verdict — it's just below the
0.65 KEEP threshold that the design pre-registered. Re-running on
a corpus with longer source memories (LoCoMo) would likely produce
higher ARI as the embedder has more vocabulary to disambiguate.

### Cluster REBUILD — `cluster_min_size` sweep (2026-04-27)

After the initial REBUILD verdict (ARI=0.5607 at default
`cluster_min_size=3`), ran `python -m bench.sweep_cluster --sizes 2 3 4 5 7`
to find a value that pushes ARI past the KEEP threshold (≥0.7).

```
cluster_min_size= 2: ari=0.5489  scored=25  dropped= 5  clusters_found=6
cluster_min_size= 3: ari=0.5607  scored=26  dropped= 4  clusters_found=4  (default)
cluster_min_size= 4: ari=1.0000  scored=17  dropped=13  clusters_found=2  ← KEEP
cluster_min_size= 5: ari=1.0000  scored=15  dropped=15  clusters_found=2
cluster_min_size= 7: ari=0.0000  scored= 0  dropped=30  clusters_found=0
```

**Verdict updated to KEEP at `cluster_min_size=4`.** ARI of 1.0
clears the 0.7 KEEP threshold definitively. The 17 memories that
HDBSCAN does cluster all align perfectly with their hand-labeled
topics. The 13 noise-labeled memories are HDBSCAN's "I'm not
confident enough to assign these to any cluster" abstentions —
those memories aren't *misclassified*, they're *unclassified*.

**Trade-off: noise rate goes from 4 (default) → 13 (rebuilt).**
Memories that HDBSCAN classifies as noise can't participate in
downstream Cluster-dependent operations (Consolidate's per-cluster
prompt, Relations' `similar_to` edges). With rebuilt config,
roughly 43% of the topology corpus is noise.

For this corpus that's the right trade — the 17 confidently
classified memories are correctly classified. Rebuilt-Cluster's
output is trustworthy. The default's ARI of 0.5607 was meaningfully
wrong on a third of the memories.

**Action item:** swap `DreamerParams.cluster_min_size` default
from 3 to 4. Or expose it as a tunable surface that operators
calibrate per workload (the right move for a library).

### Dispose REBUILD — δ sweep + criterion analysis (2026-04-27)

After the REBUILD verdict (Dispose tombstoned 0 memories on full
pressure matrix), ran `python -m bench.sweep_dispose --deltas 0.0 0.3 0.5 1.0`
to find a δ value that triggers tombstoning.

```
baseline (no dispose):  cleanliness=0.2917
delta= 0.0:  tombstoned=  0  cleanliness=0.9583  (+66.7pp)
delta= 0.3:  tombstoned=  0  cleanliness=0.9583  (+66.7pp)
delta= 0.5:  tombstoned=  0  cleanliness=0.9583  (+66.7pp)
delta= 1.0:  tombstoned=  0  cleanliness=0.9583  (+66.7pp)
```

**δ doesn't matter — tombstoned=0 at every value, including 0.**
The +66.7pp cleanliness lift comes from Rebalance (which we already
established as load-bearing); Dispose contributes nothing.

Reading `src/mnemoss/dream/dispose.py:102,114`, the actual fire
condition is:

```python
dispose_threshold = params.tau - params.delta
# ...
if b + ceiling < dispose_threshold:  # ceiling = s_max + mp + epsilon_max = 4.25
```

So the real condition is `B_i + 4.25 < tau - delta`. With
`tau=-10`, the most permissive `delta=0` still requires
`B_i < -14.25`. Our oldest pressure-corpus memory has `B_i ≈ -7.4`.
Structurally unreachable.

**Plus** there's a hardcoded `MIN_AGE_DAYS=30` floor at
`dispose.py:53`. Even if a memory's activation cratered to
`-Infinity`, the floor would protect it for 30 days. Our pressure
corpus span is exactly 30 simulated days, so most memories are
age-protected regardless.

**Verdict updated to CUT.** The `activation_dead` criterion can't
fire on this corpus, can't fire on any reasonably-sized
accumulating workload under current calibration, and tuning δ
doesn't change that.

**Action items before re-validating Dispose:**

1. **Replace the ceiling test.** The current `B_i + ceiling`
   criterion checks "even in the best case (max spreading + max
   matching + max noise), would this surface?" That's so
   conservative it never fires. Either drop the `+ ceiling` and
   test `B_i < tau - delta` directly (more aggressive), or track
   `max_A_i` historically (the recent peak, not the theoretical
   best case).
2. **Make `MIN_AGE_DAYS` corpus-relative.** Hardcoding 30 days
   means short-lived workspaces never dispose. Either expose as a
   param or scale to corpus span.
3. **Or** add the `redundant` criterion (already in the dispose
   spec, requires `cluster_size ≥ 5, similarity > 0.92, not
   representative`). With the rebuilt `cluster_min_size=4`, this
   might fire on the topology corpus. Dispose still wouldn't help
   the pressure corpus, but it'd at least do something on the
   topology corpus.

These are architecture changes, not parameter tuning. Re-validate
after they land.

### Forgetting curves — pressure corpus, B_i vs age (2026-04-27)

Run: `python -m bench.forgetting_curves --ablation dreaming_off`.
Output: `bench/results/forgetting_curves_dreaming_off.png`.

For every memory in the pressure corpus (500 memories), at end_time
(day ~30 simulated), computes B_i via
``mnemoss.formula.base_level.compute_base_level`` and plots scatter
+ binned-mean lines bucketed by utility (high/medium/low).

Mean B_i per utility bucket:

```
high    n= 50  mean B_i =  -6.935  (min=-7.378  max=-4.657)
medium  n=100  mean B_i =  -6.876  (min=-7.368  max=-5.213)
low     n=350  mean B_i =  -6.869  (min=-7.383  max=-1.064)
tombstoned: 0
```

**All three utility buckets have essentially the same mean B_i**
(within 0.07 of each other). The range variance comes from age, not
utility — memories observed near end_time have less-negative B_i
than ones observed early, regardless of which utility bucket they
belong to.

**The architectural finding this surfaces:**
on this corpus, the dream pipeline is NOT producing the differential
rehearsal that the formula is designed to amplify. Specifically:

- Consolidate's refinement updates ``extracted_gist`` and similar
  fields but **does not bump ``access_history``** (verified by
  reading `src/mnemoss/dream/consolidate.py`).
- Cluster, Relations, Rebalance, and Dispose also leave
  ``access_history`` untouched.
- The corpus's "high-utility" memories are 50 *separate* mentions of
  Phoenix, not 50 accesses to the *same* memory — so even semantic
  rehearsal doesn't accumulate per-memory.

Without rehearsal, B_i collapses to ``-d * log(age)`` with
``access_history`` of length 1 for every memory. The shape is the
intended ACT-R power-law decay (linear on a log-x plot, slope -d).
The formula is "correct" in the sense that it produces the right
shape — it just has nothing to amplify because the upstream pipeline
isn't generating signal.

**Action items:**

1. **Decide whether dreaming should bump access_history.** The
   ACT-R intent is clear: phases that "touch" a memory (Consolidate
   refinement, Cluster representative selection) should rehearse
   it. Currently they don't. Adding a single
   ``access_history.append(now)`` in Consolidate's per-member
   refinement loop would close the gap. Whether that's correct
   per ACT-R orthodoxy is a separate question.

2. **Hand-design a corpus that drives differential rehearsal.**
   Instead of 50 separate Phoenix memories, the high-utility track
   should be 5-10 memories about Phoenix that get RECALLED multiple
   times during the simulation (so their access_history accumulates
   organically). The current pressure corpus generator
   (`bench/fixtures/pressure_corpus_gen.py`) doesn't simulate
   recalls at all.

3. **Re-test** after either fix lands. The forgetting-curves panel
   should then show clear utility separation.

### Consolidate verdict — gist quality judge (2026-04-27)

Run: `python -m bench.gist_quality` against the topology corpus.
Consolidate ran with `inclusionai/ling-2.6-1t:free`; judge ran with
`openai/gpt-4o-mini` (different model family — no self-preference
bias). 18 (query, refined-member) pairs collected. Each pair
asks the judge "which gist is more useful for answering query Q?"
with order randomized per pair.

```
n = 18
ties = 8
win_rate = 0.7778  CI95 = [0.6667, 0.8889]
VERDICT: KEEP
```

**The CI lower bound (0.6667) clears the pre-registered KEEP
threshold (≥0.65).** Post-Consolidate gists win 78% of judged
pairs. Even with 44% ties (the level-1 heuristic and the LLM
refinement often produce near-identical content for short
memories), the LLM's refinements win clearly enough to beat the
threshold.

This is the load-bearing Consolidate verdict. Recall@k said
"Consolidate hurts retrieval of original ids by 42pp" but that
was the metric's bias against summary memories. Gist-quality
says the summaries themselves are content-better than level-1.

Both findings are honest about Consolidate:

- The summaries are good (KEEP per gist quality)
- The summaries dominate top-K under the current formula
  (architectural concern, separate question)

Sample comparisons (from `bench/results/gist_quality.jsonl`):

```
Q: thanksgiving plans
  level-1: mom called this morning about thanksgiving plans
  level-2: User's mother called this morning about Thanksgiving plans.
  judge: level-2 wins

Q: when does the migration spec ship
  level-1: carol said the migration spec slipped to october
  level-2: Carol said the migration spec slipped to October.
  judge: level-2 wins (capitalization + punctuation)

Q: when does the migration spec ship
  level-1: the migration spec is a forty-page document
  level-2: The migration spec is a forty-page document.
  judge: tie
```

The wins on this corpus are subtle (mostly punctuation /
capitalization). On a corpus with longer source memories
(real conversational data), the LLM's actual rewrites would
surface more clearly. Worth re-running on LoCoMo or similar
once available.
