"""Matching term — cosine-dominant weighted sum of FTS + cosine.

See ``MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.4`` (current revision) and
``docs/ROOT_CAUSE.md`` for the derivation history.

The matching term combines two retrieval signals:

    s̃_F = normalize_bm25(bm25_raw)     # literal / FTS trigram
    s̃_S = normalize_cosine(cos_sim)    # semantic / dense embedding

    w_F_raw = (f_base + f_slope·idx_priority) · b_F(q)
    w_S_raw = (s_base − f_slope·idx_priority) / b_F(q)
    w_F = w_F_raw / (w_F_raw + w_S_raw)
    w_S = 1 − w_F

    match = MP · [w_F · s̃_F + w_S · s̃_S]

Constants are configurable via ``FormulaParams.match_w_f_base``,
``match_w_f_slope``, and ``match_w_s_base``. Defaults
``(0.02, 0.05, 0.98)`` are tuned for modern dense embedders +
conversational text:

- **Default (plain query, fresh memory):** ``w_F ≈ 0.07, w_S ≈ 0.93``.
  Cosine carries the discriminative load. On conversational corpora
  where BM25 trigram has lots of shared-vocabulary noise (every memory
  mentioning "Caroline" matches a "Caroline" query), cosine is what
  separates the answer from the distractors — BM25 is a small
  tiebreaker, not a co-equal partner.
- **Literal-match query (b_F = 1.5):** ``w_F`` shifts to ≈0.14.
  Quoted strings, IDs, version numbers get more FTS weight but cosine
  still dominates. Use case: ``"v1.2.3"`` should prefer memories with
  that exact version string, but paraphrases remain retrievable.
- **Old memory (idx_priority ≈ 0.2):** ``w_F`` drops to ≈0.03.
  Old memories recalled by gist, not by exact wording — the ACT-R
  psychology preserved where it applies.

Why these defaults instead of a heavier BM25 share:
  - Empirical: on LoCoMo 2024, a w_F ≈ 0.70 default cost −22pp
    vs raw-stack pure cosine; w_F ≈ 0.25 still cost −22pp; w_F ≈ 0.07
    closes the gap to within ~1pp. See ``docs/ROOT_CAUSE.md``.
  - Intuition: modern dense embedders handle literal terms well — a
    unique version string has a distinctive embedding signature. BM25
    trigram mostly adds noise on conversational text where every
    memory shares vocabulary with every query.
  - Not pure-semantic: a small BM25 share still helps as a tiebreaker
    when cosine scores are tightly clustered, and scales up on literal
    queries via ``b_F(q)`` without drowning cosine.
  - An earlier revision tried ``noisy-OR`` (MP · [1 − (1−s_F)(1−s_S)])
    as the combiner — theoretically cleaner but empirically worse
    (saturates too fast when BM25 is noisy, collapsing discriminative
    power at the top of the ranking). See ``docs/ROOT_CAUSE.md``.

For workloads that reward BM25 (literal IDs dominating the queries,
code identifiers, regex-shaped searches), raise ``match_w_f_base``
and ``match_w_f_slope``. The dynamic-weight structure (memory-state
+ query-bias) is retained from the ACT-R design; only the scale
constants were recalibrated for modern embedders.

The ACT-R lineage is preserved in ``B_i`` (base-level activation),
``Σ W_j·S_ji`` (spreading), and ``η(t)`` (encoding grace) — parts of
the formula where cognitive-science priors help.
"""

from __future__ import annotations

import math

from mnemoss.core.config import FormulaParams


def normalize_bm25(bm25_raw: float) -> float:
    """Map SQLite FTS5 BM25 into ``[0, 1]``.

    SQLite FTS5 returns BM25 as a *negative* number (lower magnitude = more
    relevant in SQLite's convention; 0 means no match). We take ``abs`` and
    apply an exponential squash:

        s̃_F = 1 - exp(-|bm25| / BM25_SCALE)

    ``BM25_SCALE = 20`` is tuned for trigram FTS over conversational
    corpora, where BM25 magnitudes routinely fall in ``[15, 60]``. The
    historical default of ``5`` saturated past ``|bm25| ≈ 15`` — every
    relevant candidate collapsed to ``s̃_F ≈ 1.0`` and the BM25 signal
    stopped discriminating. See ``docs/ROOT_CAUSE.md`` for the
    diagnostic that surfaced this on LoCoMo 2024.
    """

    return 1.0 - math.exp(-abs(bm25_raw) / 20.0)


def normalize_cosine(cos_sim: float) -> float:
    """Map cosine similarity into ``[0, 1]``.

    Historical default mapped ``[-1, 1] → [0, 1]`` via ``(x + 1) / 2``,
    which assumes a symmetric cosine distribution. Modern dense
    embedders (OpenAI ``text-embedding-3-*``, sentence-transformers
    MiniLM, Gemini ``gemini-embedding-001``) rarely produce negative
    cosines in practice — typical ranges are ``[-0.1, +0.9]`` with
    semantically-related pairs clustering in ``[0.05, 0.35]``. The
    symmetric mapping wasted half the input range on a region that's
    almost never occupied.

    Clamping to ``[0, 1]`` preserves the raw discriminative signal
    above zero and treats below-zero cosines as "unrelated" (their
    natural interpretation). See ``docs/ROOT_CAUSE.md``.
    """

    if cos_sim <= 0.0:
        return 0.0
    if cos_sim >= 1.0:
        return 1.0
    return cos_sim


def compute_matching(
    idx_priority: float,
    bm25_raw: float,
    cos_sim: float,
    query_bias: float,
    params: FormulaParams,
) -> float:
    """Return ``MP · [w_F · s̃_F + w_S · s̃_S]`` — semantic-dominant
    weighted sum.

    Weights come from :func:`matching_weights` and depend on both
    memory age (``idx_priority``) and query shape (``query_bias``).
    See the module docstring for the constants' rationale.
    """

    s_f = normalize_bm25(bm25_raw)
    s_s = normalize_cosine(cos_sim)
    w_f, w_s = matching_weights(idx_priority, query_bias, params)
    return params.mp * (w_f * s_f + w_s * s_s)


def matching_weights(
    idx_priority: float,
    query_bias: float,
    params: FormulaParams | None = None,
) -> tuple[float, float]:
    """Return ``(w_F, w_S)`` for the semantic-dominant weighted sum.

    Structure:

        w_F_raw = (f_base + f_slope·idx_priority) · b_F(q)
        w_S_raw = (s_base − f_slope·idx_priority) / b_F(q)
        w_F    = w_F_raw / (w_F_raw + w_S_raw)
        w_S    = 1 − w_F

    Defaults (``f_base=0.05``, ``f_slope=0.2``, ``s_base=0.95``) give
    fresh memory + plain query → ``w_F ≈ 0.25, w_S ≈ 0.75``. A
    literal-match query shifts w_F upward; an old memory shifts w_F
    downward. See the module docstring for the rationale, and
    :class:`FormulaParams` for how to tune for BM25-heavy workloads.

    ``params`` is optional for back-compat; when omitted, defaults apply.
    """

    if params is None:
        f_base, f_slope, s_base = 0.02, 0.05, 0.98
    else:
        f_base = params.match_w_f_base
        f_slope = params.match_w_f_slope
        s_base = params.match_w_s_base

    w_f_raw = (f_base + f_slope * idx_priority) * query_bias
    w_s_raw = (s_base - f_slope * idx_priority) / query_bias
    total = w_f_raw + w_s_raw
    if total <= 0.0:
        return 0.5, 0.5
    # Guard against negative w_s_raw when idx_priority and f_slope push
    # the numerator below zero — clamp each share into [0, 1].
    w_f_raw = max(0.0, w_f_raw)
    w_s_raw = max(0.0, w_s_raw)
    total = w_f_raw + w_s_raw
    if total <= 0.0:
        return 0.5, 0.5
    w_f = w_f_raw / total
    return w_f, 1.0 - w_f


def signal_contributions(s_f: float, s_s: float) -> tuple[float, float]:
    """Decompose noisy-OR contribution into per-signal shares.

    Returns ``(contrib_F, contrib_S)`` such that both are in ``[0, 1]``
    and roughly sum to the noisy-OR total (exact decomposition uses
    inclusion-exclusion).

    Useful for explainability — lets ``explain_recall`` callers see
    "how much of this match came from FTS vs semantic."

    - BM25-only: ``s_f=0.9, s_s=0.0`` → ``(0.9, 0.0)``
    - Cosine-only: ``s_f=0.0, s_s=0.9`` → ``(0.0, 0.9)``
    - Both: ``s_f=0.7, s_s=0.7`` → ``(0.7·0.3, 0.7·0.3)`` = ``(0.21, 0.21)``
      + shared ``0.49`` — shown via the residual
    """

    only_f = s_f * (1.0 - s_s)
    only_s = s_s * (1.0 - s_f)
    return only_f, only_s
