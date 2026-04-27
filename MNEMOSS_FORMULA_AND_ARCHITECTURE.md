# Mnemoss: Formula and Architecture

**Authoritative specification of the Mnemoss memory system.**

This document defines the mathematical core and structural design of Mnemoss.
For project philosophy, implementation plan, and development stages, see
`MNEMOSS_PROJECT_KNOWLEDGE.md`.

---

## Part I: The Formula

### 1.1 Unified Activation Equation

Every Mnemoss behavior — retrieval ranking, index tier assignment, disposal
judgment, reminiscence — emerges from a single equation.

For a memory $m_i$ at time $t$ against query $q$:

$$
A_i = B_i + \sum_{j \in \mathcal{C}} W_j \cdot S_{ji} + \text{MP} \cdot \Big[w_F(m_i, q) \cdot \tilde{s}_F + w_S(m_i, q) \cdot \tilde{s}_S\Big] + \epsilon
$$

Four components:

$$
A_i = \underbrace{B_i}_{\text{history}} + \underbrace{\sum_j W_j S_{ji}}_{\text{context}} + \underbrace{\text{MP} \cdot [\cdots]}_{\text{matching}} + \underbrace{\epsilon}_{\text{noise}}
$$

---

### 1.2 Component 1: Base-Level Activation

Historical usage strength (power-law sum over past accesses) plus an 
**encoding-grace bonus** that lifts fresh memories and decays on its own 
time scale.

$$
B_i = \underbrace{\ln\left(\sum_{k=1}^{n}(t - t_k)^{-d}\right)}_{\text{history}} + \underbrace{\eta(t)}_{\text{encoding grace}}, \quad d \approx 0.5
$$

$$
\eta(t) = \eta_0 \cdot \exp\left(-\frac{t - t_{\text{creation}}}{\tau_\eta}\right)
$$

Where:
- $t_k$: timestamp of each past access (including creation)
- $n$: total access count
- $d$: decay exponent (default 0.5 per Anderson & Schooler 1991)
- $\eta_0$: initial encoding-grace bonus (default $1.0$)
- $\tau_\eta$: grace e-folding time (default $3600$ s, i.e. 1 hour)

**Why the grace term.** Without it, a just-encoded memory has 
$\ln(\sum \ldots) = \ln(1) = 0$ (with the 1-second floor on $t - t_k$), 
giving $\text{idx}\_\text{priority} = \sigma(0) \approx 0.5$ — WARM, not HOT. 
That mismatches human memory, where freshly encoded items are maximally 
accessible before the encoding-specific trace fades. $\eta(t)$ lifts a 
brand-new memory to $B_i \approx 1.0$ (so $\text{idx}\_\text{priority} \approx 
\sigma(1.0) \approx 0.73$, HOT), then decays smoothly:

| Age | $\eta(t)$ | $B_i$ (unused memory) | idx_priority |
|---|---|---|---|
| 0 s (just encoded) | $1.00$ | $\approx 1.00$ | $\approx 0.73$ (HOT) |
| 1 hour ($\tau_\eta$) | $0.37$ | $\approx 0.37$ | $\approx 0.59$ (WARM) |
| 3 hours | $0.05$ | $\approx 0.05$ | $\approx 0.51$ (WARM) |
| 5 hours ($5\tau_\eta$) | $\approx 0$ | $\approx 0$ | $\approx 0.50$ (WARM) |

Memories that get accessed during the grace window carry the boost 
forward through their growing history term; memories that are never used 
slide gently to WARM rather than starting there by fiat. No hard 
discontinuities, no ad-hoc "new = 1.0" rule.

The history sum simultaneously encodes three phenomena:

| Phenomenon | How it emerges |
|---|---|
| Power-law forgetting | Each access's contribution decays as $t^{-d}$ |
| Frequency effect | More accesses → more terms in the sum |
| Spacing effect | Distributed accesses sum to more than clustered ones |

**Engineering notes:**
- Compute $B_i$ at retrieval time; never cache stale values
- On every retrieval: `access_history.append(now())` — but only for the 
  top-k memories actually returned to the caller (see §1.9)
- Floor $t - t_k$ at $1$ second to avoid the singularity at $t_k = t$. 
  A memory queried in the same second it was created gets history 
  $\ln(1) = 0$; the grace term $\eta(t_{\text{creation}}) = \eta_0 = 1.0$ 
  provides the freshness lift.
- $\eta(t)$ can be evaluated in $O(1)$ from `created_at`; no storage 
  overhead.
- Drives both index tier placement and disposal decisions

---

### 1.3 Component 2: Spreading Activation

Context-driven priming from currently active memories.

$$
\sum_{j \in \mathcal{C}} W_j \cdot S_{ji}, \quad S_{ji} = S_{\max} - \ln(\text{fan}_j)
$$

Where:
- $\mathcal{C}$: memory set currently active in Working Memory
- $W_j$: attention weight for each active memory (typically $1/|\mathcal{C}|$)
- $\text{fan}_j$: out-degree of memory $j$ in the relation graph
- $S_{\max}$: ceiling constant (default 2.0)

**Fan effect**: memories connected to many others provide weaker activation 
to any single relation. A memory for "my dog Max" (fan=5) primes related 
memories strongly; a memory for "thing" (fan=5000) barely primes anything.

**Engineering notes:**
- Relation graph is maintained by Dreaming phase P5 (rich relations, 
  Stage 4+); Stage 1 ships basic co-occurrence-only relations written 
  at encode time
- `fan` values are refreshed by Dreaming phase P7 (Stage 2+); Stage 1 
  recomputes `fan` on the fly at retrieval time
- The active set $\mathcal{C}$ is held in a per-agent Working Memory 
  FIFO buffer (default capacity 10), populated by `observe()` and by 
  returned `recall()` results
- Stage 1 ships full spreading activation — not a placeholder. The 
  formula's four components interact and partial implementations give 
  misleading behavior

---

### 1.4 Component 3: Dynamic Hybrid Matching

The most novel part of Mnemoss. **Fresh memories match by literal content; 
old memories match by meaning.** The weights adapt automatically per memory 
and per query.

#### Dynamic Weights

Each scoring mode (FTS, Semantic) has a raw weight computed from two 
factors — memory state and query characteristic. The two raw weights 
are then normalized so they sum to 1.

**Raw weights:**

$$
w_F^{\text{raw}}(m_i, q) = \underbrace{(0.2 + 0.6 \cdot \text{idx}\_\text{priority}(m_i))}_{\text{memory state: high for fresh memories}} \cdot \underbrace{b_F(q)}_{\text{query bias toward FTS}}
$$


$$
w_S^{\text{raw}}(m_i, q) = \underbrace{(0.8 - 0.6 \cdot \text{idx}\_\text{priority}(m_i))}_{\text{memory state: high for old memories}} \cdot \underbrace{b_F(q)^{-1}}_{\text{inverse query bias}}
$$

Note the symmetric structure:
- Memory-state factors: $(0.2 + 0.6p)$ and $(0.8 - 0.6p)$ always sum to $1.0$
- Query-bias factors: $b_F(q)$ and $b_F(q)^{-1}$ are reciprocals

**Normalized weights:**

$$
w_F = \frac{w_F^{\text{raw}}}{w_F^{\text{raw}} + w_S^{\text{raw}}}, \qquad w_S = 1 - w_F
$$

This guarantees $w_F + w_S = 1$ and keeps both in $(0, 1)$.

**Query bias function $b_F(q)$:**

Every rule is purely structural / typographic — we never inspect the
query for named entities, topics, or meaning. Memory stays query-
agnostic; $b_F$ only picks up patterns a user *types* to signal "I
mean this literally."

| Query feature | $b_F$ | Example |
|---|---|---|
| Quote chars or backtick fence | 1.5 | `"4:20 PM"`, `` `userId` ``, 「我明天要去」, 《红楼梦》 |
| URL, email, or file path | 1.4 | `https://example.com`, `src/main.py`, `foo@bar.com` |
| Time/date/number, hashtag, @-mention, CamelCase / snake_case / kebab-case, version | 1.3 | `4:20`, `2026-04-22`, `#release`, `@alice`, `iPhone`, `my_var`, `v1.2.3` |
| ALL-CAPS Latin acronym ≥3 chars | 1.2 | `USA`, `NASA`, `IBM` |
| Default | 1.0 | plain query in any script |

Notes:
- Non-Latin scripts (CJK, Arabic, Devanagari, …) trigger every rule
  that uses language-neutral regex (quotes, URLs, digits, hashtags,
  code identifiers). The acronym rule is Latin-only by construction —
  no other common script uses uppercase — so it's silent on those
  scripts rather than biased against them.
- Title-Case proper-noun detection was removed: it was English-biased
  (silent on CJK / Arabic) and duplicated work that embeddings and
  the entities FTS column already handle. Named entities flow into
  recall via Dream P3 output, not via query parsing — see §1.4.1.

#### Normalized Scores

$$
\tilde{s}_F = 1 - \exp\left(-\frac{s_{\text{BM25}}}{5}\right) \in [0, 1]
$$

$$
\tilde{s}_S = \frac{\cos(\vec{q}, \vec{m}_i) + 1}{2} \in [0, 1]
$$

- $s_{\text{BM25}}$: raw BM25 score from FTS index
- $\cos$: cosine similarity between query and memory embeddings

#### Matching Penalty

$$
\text{MP} \approx 1.5
$$

Scaling coefficient that controls matching's weight relative to $B_i$.

#### Combined Behavior

Computed directly from the normalized-weight formula above:

| Scenario | idx_priority | $b_F(q)$ | $w_F$ | Result |
|---|---|---|---|---|
| New memory + quoted query | 0.95 | 1.5 | ≈ 0.87 | FTS dominates |
| New memory + plain query | 0.95 | 1.0 | ≈ 0.74 | FTS favored by state alone |
| Old memory + quoted query | 0.20 | 1.5 | ≈ 0.51 | Roughly balanced |
| Old memory + plain query | 0.20 | 1.0 | ≈ 0.26 | Semantic dominates |

This naturally captures human memory behavior: recent events recalled by 
literal details, old events recalled by gist. The symmetric design means 
extreme idx_priority values push one mode hard in both factors — a fresh 
memory facing a precise query gets *both* a higher FTS state-factor *and* 
the FTS-leaning bias — compounding into strong preference.

**Note on NER.** Mnemoss does not perform named-entity recognition at
any stage (query, encode, or Dream). `b_F(q)` inspects typographic
structure only — never semantic content. The reliance is on the
embedder (cosine) and the trigram FTS over `content` for every
entity-like pattern. See MNEMOSS_PROJECT_KNOWLEDGE.md §9.7 for the
design rationale and DIY hooks.

---

### 1.5 Component 4: Noise

$$
\epsilon \sim \text{Logistic}(0, s), \quad s \approx 0.25
$$

Sampled fresh on each retrieval. Never cached.

**Why noise**:
- Same query sometimes succeeds, sometimes fails (human-like)
- Avoids deterministic retrieval
- Makes softmax selection meaningful

Standard deviation of Logistic(0, s) is $s\pi/\sqrt{3} \approx 0.45$ 
for $s = 0.25$.

---

### 1.6 Derived: Index Priority

A memory's index tier placement is derived from $B_i$ plus protection signals:

$$
\text{idx}\_\text{priority}(m_i) = \sigma\Big(B_i + \alpha \cdot \text{salience}_i + \beta \cdot {\text{emotional}\_\text{weight}}_i + \gamma \cdot \mathbb{1}[\text{pinned}]\Big)
$$

Where $\sigma$ is the sigmoid function.

**Parameters:**
- $\alpha \approx 0.5$: salience weight
- $\beta \approx 0.4$: emotional weight
- $\gamma \approx 2.0$: pin protection (keeps pinned memories near 1.0)

**Tier mapping:**

| idx_priority | Tier | Target latency | Default recall |
|---|---|---|---|
| > 0.7 | HOT | < 10ms | ✓ |
| 0.3 – 0.7 | WARM | < 50ms | ✓ |
| 0.1 – 0.3 | COLD | < 200ms | ✓ |
| ≤ 0.1 | DEEP | < 500ms | Only with strong cues |

Computed by Dreaming phase P7 (Rebalance), not at each retrieval.

---

### 1.7 Derived: Retrieval Threshold and Selection

**Threshold:**

$$
A_i > \tau \implies \text{enters candidate pool}, \quad \tau \approx -1.0
$$

**Softmax selection** (optional):

$$
P(\text{retrieve } i) = \frac{e^{A_i / T}}{\sum_j e^{A_j / T}}
$$

**Cascade with early stopping:**

```
scan HOT   → top_A > CONFIDENCE_HOT?   yes → stop
scan WARM  → top_A > CONFIDENCE_WARM?  yes → stop
scan COLD  → top_A > CONFIDENCE_COLD?  yes → stop
scan DEEP  → only with strong cues
```

Confidence thresholds are anchored to $\tau$:

$$
\text{CONFIDENCE}\_\text{HOT} = \tau + 2.0, \quad \text{CONFIDENCE}\_\text{WARM} = \tau + 1.0, \quad \text{CONFIDENCE}\_\text{COLD} = \tau
$$

With the default $\tau = -1.0$: HOT cuts off at $A = 1.0$, WARM at $A = 0.0$, 
COLD at $A = -1.0$ (any above-threshold candidate).

#### 1.7.1 Fast-Index Recall Mode (the async-ACT-R bet)

The cascade above evaluates the full activation formula for every 
candidate at recall time. That is **semantically faithful** but puts
linear-in-N work (the vec/FTS scans, then $O(K)$ formula evaluations) 
on the user-facing read path. 

**Mnemoss's defining bet: the activation formula is expressive enough 
that its *output* can be cached**. Everything A_i depends on that isn't 
query-dependent — history-driven base level $B_i$, spreading from the 
*persistent* relation graph, salience, emotional weight, pin status — 
collapses into the single scalar $\text{idx\_priority}$ via §1.6. 
Dream P7 Rebalance recomputes $\text{idx\_priority}$ async. Store 
writes recompute it on `reconsolidate`. The read path never does.

In `FormulaParams(use_fast_index_recall=True)` mode, the retrieval 
pipeline becomes:

$$
\text{score}(m, q) = w_{\text{sem}} \cdot \cos(q, \text{emb}(m)) + w_{\text{pri}} \cdot \text{idx\_priority}(m)
$$

1. ANN top-K via HNSW: $O(\log N)$.
2. Batch-fetch cached $\text{idx\_priority}$: one indexed SQL query, $O(K)$.
3. Linear combine + sort: $O(K \log K)$.

What's given up: query-dependent matching weights ($w_F(q)$, BM25), 
active-set spreading at recall time, noise. For conversational workloads 
where BM25 contributes <1pp to recall (cosine-dominant weighting) and 
spreading is captured in `idx_priority` via Dream, this is close to 
lossless on semantic recall while collapsing the read-path cost to 
$O(\log N)$. Verified empirically on LoCoMo through N=20,000 (see 
`docs/ROOT_CAUSE.md` Phase 2).

Default is `False` — the full ACT-R path is the right choice for 
agent workloads that want recency priors and literal-query boosts. 
`mnemoss_rocket` in `bench/launch_comparison.py` turns it on as the 
"conversational memory SDK" preset.

---

### 1.8 Derived: Disposal Criterion

A memory is eligible for disposal when it cannot be retrieved even under 
ideal conditions:

$$
\underbrace{B_i + S_{\max} + \text{MP} + \epsilon_{\max}}_{\max A_i} < \tau - \delta
$$

**Parameters:** $\delta \approx 1.0$ (safety margin), 
$\epsilon_{\max} \approx 0.75$ (a practical cap on the Logistic noise — 
roughly the 99.5th percentile of $\text{Logistic}(0, 0.25)$, i.e. about 
$3 \cdot \text{stddev}$; the distribution is formally unbounded so we 
pick a high quantile rather than a true maximum).

**Geometric auxiliary criteria** (non-formula, use clustering and cosine). A memory $m_i$ is flagged **redundant** when all three hold:

- `cluster_size` $\geq 5$
- `sim_to_centroid` $> 0.92$
- $\neg$ `is_representative`

It is flagged **fact-covered** when $\cos(\vec{m_i},\ \vec{v}_\text{facts}) > 0.85$ and $B_i < -3$, where $\vec{v}_\text{facts}$ is the mean embedding of the aggregated fact memories that cover $m_i$.

**Hard protections** (veto any disposal):
- `pinned_fields` is non-empty
- `manually_marked == "important"`
- `salience > 0.8` or `emotional_weight > 0.7`
- `age_days < 30` (minimum retention period)

**Zero LLM calls in disposal decisions.**

---

### 1.9 Derived: Reconsolidation Feedback

**Scope.** Reconsolidation fires only on the **top-k memories actually 
returned to the caller** — not on the full candidate pool, and not on 
every memory scored. Retrieval strengthens what was *used*, not what was 
merely considered. A memory that was scored, cleared the $\tau$ threshold, 
but lost to higher-$A$ peers is *not* reconsolidated.

Only metadata updates. Content and extracted fields never change.

| Field | Update | Formula effect |
|---|---|---|
| `access_history` | `.append(now())` | $B_i$ increases |
| `rehearsal_count` | `+= 1` | Future priority decay slows |
| `last_accessed_at` | `now()` | bookkeeping |
| `idx_priority` | recomputed by P7 (or on the fly in Stage 1) | may promote tier |
| If reactivated from DEEP | `reminisced_count += 1`, jump to WARM | reminiscence |

---

### 1.10 Parameter Defaults

| Parameter | Default | Source |
|---|---|---|
| $d$ (decay rate) | 0.5 | Anderson & Schooler 1991 |
| $\tau$ (retrieval threshold) | -1.0 | ACT-R 7 reference |
| $\text{MP}$ (matching penalty) | 1.5 | Anderson 2007 |
| $s$ (noise scale) | 0.25 | Cognitive model literature |
| $S_{\max}$ (spreading ceiling) | 2.0 | ACT-R manual |
| $\alpha$ (salience weight) | 0.5 | Empirical |
| $\beta$ (emotional weight) | 0.4 | Empirical |
| $\gamma$ (pin boost) | 2.0 | Ensures pinned stay HOT |
| $\delta$ (disposal margin) | 1.0 | Conservative |
| $\epsilon_{\max}$ (noise cap for disposal) | 0.75 | 99.5th pct of Logistic(0, 0.25) |
| $t_{\text{floor}}$ (age floor in $B_i$) | 1.0 s | Avoids singularity at $t_k = t$ |
| $\eta_0$ (encoding-grace bonus) | 1.0 | Lifts fresh memories into HOT ($\sigma(1.0) \approx 0.73$) |
| $\tau_\eta$ (grace e-folding time) | 3600 s | ~1 hour; negligible by $5\tau_\eta$ |
| CONFIDENCE\_HOT / WARM / COLD | $\tau{+}2$ / $\tau{+}1$ / $\tau$ | Cascade early-stop |

These are starting points. Real deployments should calibrate $d$, $\tau$, 
and $\text{MP}$ against benchmarks.

---

## Part II: Architecture

### 2.1 Top-Level View

```
╔════════════════════════════════════════════════════════════════════╗
║                         MNEMOSS v5                                 ║
║       One Formula · One Table · Three Paths · Four Tiers           ║
╚════════════════════════════════════════════════════════════════════╝

               ┌──────────────────────────────────────┐
               │       Message from Agent             │
               │  (user / assistant / tool_call /     │
               │   tool_result)                        │
               └──────────────────┬───────────────────┘
                                  │
                ┌─────────────────▼─────────────────┐
                │      HOT PATH (<50ms)             │
                │     「Encoding, no LLM」           │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
                ┌───────────────────────────────────┐
                │    UNIFIED MEMORY STORE           │
                │      (one table for all)          │
                └──────────────────┬────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │   WARM PATH  │  │   COLD PATH  │  │   READS      │
        │  (event-     │  │  (Dreaming:  │  │              │
        │   driven)    │  │   5 triggers)│  │   recall()   │
        └──────────────┘  └──────────────┘  └──────────────┘
                                   │
                                   ▼
                ┌───────────────────────────────────┐
                │      USER-FACING OUTPUTS          │
                │  · memory.md (auto-generated)     │
                │  · memory_overrides.md (user)     │
                │  · Dream Diary (audit)            │
                └───────────────────────────────────┘
```

---

### 2.2 Hot Path Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    HOT PATH (<50ms)                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   [User / Assistant / Tool Message]                               │
│            │                                                      │
│            ▼                                                      │
│   ┌──────────────────────┐                                        │
│   │ ① Raw Log            │  All messages, unconditional         │
│   │   append-only        │  Never filtered, never modified       │
│   └──────────────────────┘                                        │
│            │                                                      │
│            ▼                                                      │
│   ┌──────────────────────┐                                        │
│   │ ② Working Memory     │  Session-level buffer                │
│   │   · message buffer   │  Event boundary detection             │
│   │   · active entities  │                                       │
│   └──────────────────────┘                                        │
│            │                                                      │
│            ▼                                                      │
│   ┌──────────────────────┐                                        │
│   │ ③ Event Segmentation │  Rule-based, no LLM                  │
│   │   · turn complete?   │  Triggers encoding                    │
│   │   · topic shift?     │                                       │
│   │   · time gap?        │                                       │
│   │   · task complete?   │                                       │
│   └──────────────────────┘                                        │
│            │                                                      │
│            ▼ event detected                                       │
│   ┌──────────────────────┐                                        │
│   │ ④ Event → Memory     │  Encoding                            │
│   │   · synthesize text  │  Concatenate messages                 │
│   │   · quick salience   │  Simple heuristic (Stage 3+)          │
│   │   · embed            │  20-50ms local; ~150ms cloud (see §G) │
│   │   · create Memory    │  type=episode, abstraction=0.0       │
│   │   · idx_priority     │  σ(B_i + α·sal + β·emo + γ·pin)      │
│   │                      │  B_i includes η(t) grace bonus →      │
│   │                      │  fresh ≈ 0.73 (HOT), decays to ~0.5   │
│   │                      │  over ~3 hrs if unused; see §1.2      │
│   └──────────────────────┘                                        │
│            │                                                      │
│            ▼ async (non-blocking)                                 │
│   ┌──────────────────────┐                                        │
│   │ ⑤ Store + Index      │                                       │
│   │   Memory Store       │                                       │
│   │   + HOT index (HNSW) │                                       │
│   │   + FTS5 index       │                                       │
│   └──────────────────────┘                                        │
│                                                                   │
│   ═══════════════════════                                         │
│                                                                   │
│   [Query arrives]                                                 │
│            │                                                      │
│            ▼                                                      │
│   ┌──────────────────────┐                                        │
│   │ ⑥ Retrieval Engine   │  Cascade HOT → WARM → COLD → DEEP   │
│   │   · parallel vec+FTS │                                       │
│   │   · ACT-R scoring    │                                       │
│   │   · lazy extraction  │  Top-K candidates only               │
│   │   · reconsolidation  │  Update access_history                │
│   └──────────────────────┘                                        │
│            │                                                      │
│            ▼                                                      │
│   [Retrieved Memories]                                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Cloud embedder and the <50ms budget.** The <50ms Hot Path budget assumes 
a local embedder (~20–50ms on CPU). OpenAI's `text-embedding-3-small` adds 
~100–300ms of network round-trip and breaks the budget. Resolution:

- **Stage 1:** `observe()` blocks on embedding regardless of embedder. 
  Document cloud latency; no background pipeline yet. Acceptable because 
  Stage 1 has no concurrency story to protect.
- **Stage 2+:** `observe()` returns as soon as the Raw Log append + Memory 
  row insert land (with `content_embedding=NULL`, `index_tier="pending"`). 
  Embedding is produced in a background task; the memory becomes queryable 
  only after the embedding arrives. This restores the <50ms wire budget 
  for callers while letting cloud embedders be used.

---

### 2.3 Unified Memory Store

All memory types share one table, distinguished by `memory_type` and 
`abstraction_level`.

```
┌──────────────────────────────────────────────────────────────────┐
│                UNIFIED MEMORY STORE (single table)                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │                  Memory Table                             │   │
│   │                                                            │   │
│   │   ┌─ memory_type: episode ─────────────┐                 │   │
│   │   │  abstraction_level: 0.0            │  most concrete  │   │
│   │   │  content: synthesized event text   │                 │   │
│   │   │  source_message_ids: [...]         │                 │   │
│   │   └────────────────────────────────────┘                 │   │
│   │                                                            │   │
│   │   ┌─ memory_type: fact ────────────────┐                 │   │
│   │   │  abstraction_level: 0.5            │  mid-abstract   │   │
│   │   │  content: "Alice manages auth"     │                 │   │
│   │   │  derived_from: [ep_001, ep_042]   │  ◀ trace chain  │   │
│   │   └────────────────────────────────────┘                 │   │
│   │                                                            │   │
│   │   ┌─ memory_type: entity ──────────────┐                 │   │
│   │   │  abstraction_level: 0.8            │  high-abstract  │   │
│   │   │  content: "Alice"                  │                 │   │
│   │   │  aliases: ["Alice Chen"]            │                 │   │
│   │   │  entity_type: "Person"              │                 │   │
│   │   └────────────────────────────────────┘                 │   │
│   │                                                            │   │
│   │   ┌─ memory_type: pattern ─────────────┐                 │   │
│   │   │  abstraction_level: 0.9            │  most abstract  │   │
│   │   │  content: "meets regularly in X"    │                 │   │
│   │   │  derived_from: [10 episodes]       │                 │   │
│   │   └────────────────────────────────────┘                 │   │
│   │                                                            │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│   ┌─────── Supporting stores ──────────────────────────────┐     │
│   │  · Raw Log (append-only message log)                    │     │
│   │  · Relations Graph (derived from memory.relations)      │     │
│   │  · Tombstones (records of disposed memories)            │     │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- `memory_type` is a **label**, not a physical partition
- `abstraction_level` is **continuous**, not discrete
- `derived_from` / `derived_to` form an **abstraction chain** 
  (episode → fact → pattern)
- All types share the ACT-R formula — fact's $B_i$, entity's spreading, 
  pattern's disposal all computed identically
- `agent_id: str | None` scopes each memory to one agent in the 
  workspace (gateway); null = workspace-shared, visible to every agent. 
  Default recall for agent A is `WHERE agent_id = 'A' OR agent_id IS NULL`.

---

### 2.4 Four-Tier Index

```
┌──────────────────────────────────────────────────────────────────┐
│              FOUR INDEX TIERS (shared data, distinct metadata)    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │   HOT    (idx_priority > 0.7)          target < 10ms     │  │
│   │   ─────                                                    │  │
│   │   · HNSW dense vector index (exact)                       │  │
│   │   · Full FTS over all fields                              │  │
│   │   · Participates in default recall ✓                      │  │
│   └───────────────────────────────────────────────────────────┘  │
│                         ▼ formula-driven migration                │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │   WARM   (0.3 < idx_priority ≤ 0.7)    target < 50ms     │  │
│   │   ─────                                                    │  │
│   │   · HNSW quantized (compressed)                           │  │
│   │   · FTS over core fields                                  │  │
│   │   · Participates in default recall ✓                      │  │
│   └───────────────────────────────────────────────────────────┘  │
│                         ▼                                          │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │   COLD   (0.1 < idx_priority ≤ 0.3)    target < 200ms    │  │
│   │   ─────                                                    │  │
│   │   · IVF approximate nearest neighbor                       │  │
│   │   · Gist-only FTS                                          │  │
│   │   · Participates in default recall ✓                      │  │
│   └───────────────────────────────────────────────────────────┘  │
│                         ▼                                          │
│   ┌───────────────────────────────────────────────────────────┐  │
│   │   DEEP   (idx_priority ≤ 0.1)           target < 500ms   │  │
│   │   ─────                                                    │  │
│   │   · Sparse n-gram only                                    │  │
│   │   · Does NOT participate in default recall ✗             │  │
│   │   · Triggered only by strong cues:                        │  │
│   │     - Rare n-gram matches                                 │  │
│   │     - Queries with "long ago", "years back"               │  │
│   │     - When default tiers return nothing                   │  │
│   └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│   ⚠ Data exists once; tier migration only updates metadata       │
│   ⚠ Reminiscence: DEEP hit with high activation → jump to WARM   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Implementation: per-tier indices, shared data.** Each tier has its own 
index structure (HNSW-exact, HNSW-quantized, IVF, sparse n-gram). A 
memory's **row** — content, embedding, metadata — lives in the single 
`Memory` table regardless of tier. What moves during tier migration is 
not the data, but the memory's **registration**: Dreaming P7 drops the 
memory from tier N's vector/FTS index and inserts it into tier N+1's. The 
`Memory.index_tier` column records which tier currently owns the 
registration, so retrieval knows which index structure to scan.

Concretely: "HOT = HNSW exact" and "WARM = HNSW quantized" do not mean 
the same vector is stored twice in different precisions — the vector is 
stored once in the `Memory` row, and each tier's index holds references 
(plus whatever auxiliary structure it needs: an HNSW graph, a quantized 
codebook, an IVF posting list, an n-gram inverted index) keyed by 
`memory.id`.

**Stage 1** ships only the HOT tier, backed by sqlite-vec + FTS5. Multi-tier 
coexistence and the P7 migration step land in Stage 2+.

---

### 2.5 Dreaming: Five Triggers, Eight Phases

```
┌────────────────────────────────────────────────────────────────────┐
│             DREAMING: 5 Triggers × 6 Phases                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ═══ TRIGGERS ═══                                                  │
│                                                                     │
│   Light (frequent, cheap):                                          │
│    ├─ idle            (5min+ user inactivity)                       │
│    ├─ session_end     (session terminates)                          │
│    ├─ surprise        (conflict detected)                           │
│    └─ cognitive_load  (before context compaction)                   │
│                                                                     │
│   Deep (scheduled, comprehensive):                                  │
│    └─ nightly         (daily deep consolidation)                    │
│                                                                     │
│   ═══ PIPELINE ═══                                                  │
│                                                                     │
│   P1: Replay                                                        │
│     Select memories by ACT-R $B_i$; prioritize high-activation      │
│     un-consolidated candidates.                                     │
│                                                                     │
│   P2: Cluster                                                       │
│     HDBSCAN on embeddings. Assign cluster_id, similarity,           │
│     representative flag.                                            │
│                                                                     │
│   P3: Consolidate  (uses LLM — one call per cluster)                │
│     Merged from the former Extract + Refine + Generalize trio.     │
│     One structured call per cluster emits:                          │
│      (A) a summary memory (fact/entity/pattern) with                │
│          derived_from linking to members,                           │
│      (B) refined extracted_* fields for each member                 │
│          (gist, entities, time, location, participants),            │
│      (C) zero or more intra-cluster patterns.                       │
│     Cross-cluster generalization is dropped — clusters are the     │
│     semantic boundary.                                              │
│                                                                     │
│   P4: Relations                                                     │
│     Update memory-to-memory relations. Handle conflicts via         │
│     supersedes chains. Recompute fan values.                       │
│                                                                     │
│   P5: Rebalance  ⭐                                                 │
│     Recompute idx_priority = σ(B + α·sal + β·emo + γ) for all.     │
│     Migrate between index tiers. Metadata only, content untouched. │
│                                                                     │
│   P6: Dispose  ⭐                                                   │
│     Formula-based disposal:                                         │
│      · activation_dead: max(A_i) < τ - δ                            │
│      · redundant: cluster geometry                                  │
│      · fact_covered: cosine geometry                                │
│     Write Tombstone. Zero LLM decisions.                           │
│                                                                     │
│   ═══ TRIGGER → PHASE MAPPING ═══                                   │
│                                                                     │
│   idle:            P1, P2, P3, P4                                   │
│   session_end:     P1, P2, P3, P4                                   │
│   surprise:        P3, P4                                           │
│   cognitive_load:  P3                                               │
│   nightly:         P1–P6 (all phases)                               │
│                                                                     │
│   ═══ OUTPUT ═══                                                    │
│                                                                     │
│   Dream Diary (Markdown, human-readable, auditable):                │
│    · Trigger type, time                                             │
│    · Memories processed                                             │
│    · New facts/entities/patterns created                            │
│    · Disposed memories with reasons                                 │
│    · Conflicts resolved                                             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

### 2.6 Complete System View

```
╔══════════════════════════════════════════════════════════════════════════╗
║                       MNEMOSS COMPLETE SYSTEM                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  INTEGRATION LAYER                                                        ║
║  ═════════════════                                                        ║
║                                                                           ║
║   ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐        ║
║   │  Python API  │   │  MCP Server  │   │  Framework Plugins   │        ║
║   │ (direct use) │   │  (any agent) │   │  (OpenClaw, Hermes)  │        ║
║   └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘        ║
║          │                   │                      │                     ║
║          └───────────────────┼──────────────────────┘                     ║
║                              ▼                                            ║
║  ┌───────────────────────────────────────────────────────────────────┐   ║
║  │                    MNEMOSS CORE                                    │   ║
║  │                                                                    │   ║
║  │  ┌──────────────────────────────────────────────────────────┐    │   ║
║  │  │   HOT PATH (<50ms, no LLM)                                │    │   ║
║  │  │   Raw Log → Working Memory → Event Seg → Encode → Store  │    │   ║
║  │  │   + Retrieval Engine (cascade, ACT-R, reconsolidation)   │    │   ║
║  │  └──────────────────────┬───────────────────────────────────┘    │   ║
║  │                         │                                          │   ║
║  │                         ▼                                          │   ║
║  │  ┌──────────────────────────────────────────────────────────┐    │   ║
║  │  │   DATA LAYER                                              │    │   ║
║  │  │                                                            │    │   ║
║  │  │   ┌─ Memory Store (unified single table) ─────┐          │    │   ║
║  │  │   │  episode / fact / entity / pattern         │          │    │   ║
║  │  │   └─────────────────────────────────────────────┘          │    │   ║
║  │  │                                                            │    │   ║
║  │  │   ┌─ Raw Log (append-only) ─────────────────────┐         │    │   ║
║  │  │   │  all messages, unfiltered                    │         │    │   ║
║  │  │   └───────────────────────────────────────────────┘         │    │   ║
║  │  │                                                            │    │   ║
║  │  │   ┌─ Tombstones ────────────────────────────────┐         │    │   ║
║  │  │   │  records of disposed memories                 │         │    │   ║
║  │  │   │  traceable to Raw Log                         │         │    │   ║
║  │  │   └───────────────────────────────────────────────┘         │    │   ║
║  │  │                                                            │    │   ║
║  │  │   ┌─ Four Index Tiers ──────────────────────────┐         │    │   ║
║  │  │   │  HOT (HNSW+FTS) / WARM (HNSW-Q)             │         │    │   ║
║  │  │   │  COLD (IVF) / DEEP (sparse n-gram)          │         │    │   ║
║  │  │   │  shared data, tier via metadata             │         │    │   ║
║  │  │   └───────────────────────────────────────────────┘         │    │   ║
║  │  │                                                            │    │   ║
║  │  │   ┌─ Relations Graph ───────────────────────────┐         │    │   ║
║  │  │   │  derived from memory.relations               │         │    │   ║
║  │  │   │  supports spreading activation               │         │    │   ║
║  │  │   └───────────────────────────────────────────────┘         │    │   ║
║  │  └──────────────────────┬───────────────────────────────────┘    │   ║
║  │                         │                                          │   ║
║  │                         ▼                                          │   ║
║  │  ┌──────────────────────────────────────────────────────────┐    │   ║
║  │  │   COLD PATH: DREAMING (offline, opportunistic)            │    │   ║
║  │  │                                                            │    │   ║
║  │  │   5 Triggers:                                              │    │   ║
║  │  │    idle | session_end | surprise |                        │    │   ║
║  │  │    cognitive_load | nightly                               │    │   ║
║  │  │                                                            │    │   ║
║  │  │   6-Phase Pipeline:                                        │    │   ║
║  │  │    P1 Replay → P2 Cluster → P3 Consolidate →              │    │   ║
║  │  │    P4 Relations → P5 Rebalance → P6 Dispose               │    │   ║
║  │  │    (P3 is one LLM call per cluster —                      │    │   ║
║  │  │     summary + refinements + intra-cluster patterns)       │    │   ║
║  │  │                                                            │    │   ║
║  │  │   Output: Dream Diary (auditable Markdown)                 │    │   ║
║  │  └──────────────────────┬───────────────────────────────────┘    │   ║
║  │                         │                                          │   ║
║  │                         ▼                                          │   ║
║  │  ┌──────────────────────────────────────────────────────────┐    │   ║
║  │  │   USER-FACING VIEWS                                       │    │   ║
║  │  │   · memory.md          (auto-generated view)              │    │   ║
║  │  │   · memory_overrides.md (user-edited)                     │    │   ║
║  │  │   · Dream Diary        (audit log)                        │    │   ║
║  │  │   · Tombstones         (disposal records)                 │    │   ║
║  │  │   · explain_recall()   (debug tool)                       │    │   ║
║  │  └──────────────────────────────────────────────────────────┘    │   ║
║  │                                                                    │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                                                           ║
║  ═══════════════════════════════════════════════════════════════════     ║
║                                                                           ║
║  DRIVEN BY ONE FORMULA:                                                   ║
║                                                                           ║
║     A_i = B_i + Σ W_j·S_ji + MP·[w_F·s̃_F + w_S·s̃_S] + ε                  ║
║                                                                           ║
║     ├─→ Retrieval ranking (top-k by A_i)                                 ║
║     ├─→ Index tier assignment (via idx_priority)                         ║
║     ├─→ Disposal decision (max A_i < τ - δ)                              ║
║     └─→ Reminiscence trigger (DEEP hit with high A)                      ║
║                                                                           ║
║  LLM is used ONLY for content generation inside Dreaming (P3/P4/P6).     ║
║  NO LLM decisions anywhere in the control flow.                           ║
║                                                                           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Part III: Formula-Architecture Mapping

The Rosetta Stone of Mnemoss: where does each formula term come from, 
who maintains it, and what does it drive?

| Formula term | Data source | Computed when | Drives |
|---|---|---|---|
| $B_i$ | `Memory.access_history` | Retrieval time | Ranking, tier, disposal |
| $W_j$ | Working Memory active set | Retrieval time | Context priming |
| $S_{ji}$ | `Memory.relations` + graph | Retrieval time | Spreading activation |
| $\text{fan}_j$ | Relation graph out-degree | Dreaming P5 | Fan effect |
| $\text{idx}\_\text{priority}$ | $\sigma(B + \alpha s + \beta e + \gamma p)$ | Dreaming P7 | Tier assignment |
| $w_F, w_S$ | idx_priority × $b_F(q)$ | Retrieval time | Dynamic weighting |
| $b_F(q)$ | Query regex analysis | Retrieval time, <1ms | Query bias |
| $\tilde{s}_F$ | FTS BM25 | Recall stage | Literal matching |
| $\tilde{s}_S$ | Vector cosine | Recall stage | Semantic matching |
| $\epsilon$ | Logistic sample | Each retrieval | Probabilistic recall |
| $\tau$ | Config constant | Post-ranking filter | Candidate threshold |
| $\delta$ | Config constant | Disposal check | Safety margin |

### Feedback Loop

```
Successful retrieval
  → access_history.append(now)
  → B_i increases on next retrieval
  → Dreaming P7 recomputes idx_priority
  → May migrate to higher tier
  → Retrieved faster next time
  → Positive feedback: important things surface naturally
```

---

## Part IV: One-Page Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                           MNEMOSS                                 │
│         ACT-R based memory system for AI agents                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ONE FORMULA                                                     │
│                                                                   │
│     A_i = B_i + Σ W_j·S_ji + MP·[w_F·s̃_F + w_S·s̃_S] + ε           │
│          │       │           │                    │               │
│          │       │           │                    └─ Noise        │
│          │       │           └─ Dynamic Matching                  │
│          │       │              (literal for new, semantic        │
│          │       │               for old)                         │
│          │       └─ Spreading (context activation)                │
│          └─ Base Level (usage power-law)                          │
│                                                                   │
│   ONE TABLE                                                       │
│                                                                   │
│     Memory: episode | fact | entity | pattern                     │
│     (discriminated by memory_type + abstraction_level,            │
│      unified dynamics)                                            │
│                                                                   │
│   THREE PATHS                                                     │
│                                                                   │
│     HOT  (<50ms)   Encode + Retrieve                              │
│     WARM (<1s)     Index maintenance                              │
│     COLD (offline) Dreaming (5 triggers × 6 phases)               │
│                                                                   │
│   FOUR INDEX TIERS                                                │
│                                                                   │
│     HOT (<10ms) / WARM (<50ms) / COLD (<200ms) / DEEP (<500ms)    │
│     (one data copy, tier via metadata)                            │
│                                                                   │
│   SIX DREAMING PHASES                                             │
│                                                                   │
│     P1 Replay → P2 Cluster → P3 Consolidate →                     │
│     P4 Relations → P5 Rebalance → P6 Dispose                      │
│                                                                   │
│   DRIVEN BY FORMULA, NOT LLM                                      │
│                                                                   │
│     Retrieval ranking      ← A_i                                  │
│     Index tier             ← idx_priority = σ(B + α·s + β·e)      │
│     Disposal               ← max A_i < τ - δ                      │
│     Reminiscence           ← DEEP hit with high A                 │
│                                                                   │
│     LLM only for content generation in Dreaming                   │
│                                                                   │
│   COGNITIVE SCIENCE FOUNDATIONS                                   │
│                                                                   │
│     · ACT-R activation (Anderson & Schooler 1991)                 │
│     · Episodic continuum (Tulving, Moscovitch)                    │
│     · Dual-process theory (Yonelinas 2002)                        │
│     · Event segmentation (Zacks 2007)                             │
│     · Awake replay (Foster & Wilson 2006)                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Final Words

The formula is the heart. The architecture is the skeleton that serves it.

From raw message ingestion to memory consolidation, from retrieval to 
disposal, every decision in Mnemoss emerges from the same mathematical 
structure. This is not architectural decoration — it is the design principle.

When in doubt, return to the formula.

