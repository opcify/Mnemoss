"""Unit tests for the matching term.

Covers the cosine-dominant weighted-sum formulation
(``MP · [w_F · s̃_F + w_S · s̃_S]``) and its normalizations.
Default constants (``match_w_f_base=0.02, match_w_f_slope=0.05,
match_w_s_base=0.98``) tuned for modern dense embedders on
conversational corpora: fresh memory + plain query → ``w_F ≈ 0.07,
w_S ≈ 0.93``. BM25 acts as a tiebreaker; cosine carries the signal.
"""

from __future__ import annotations

import math

import pytest

from mnemoss.core.config import FormulaParams
from mnemoss.formula.matching import (
    compute_matching,
    matching_weights,
    normalize_bm25,
    normalize_cosine,
    signal_contributions,
)

PARAMS = FormulaParams()


# ─── normalize_bm25 ────────────────────────────────────────────────


def test_normalize_bm25_handles_negative_sqlite_convention() -> None:
    # SQLite FTS5 returns BM25 as negative. Normalization uses abs.
    assert normalize_bm25(0.0) == 0.0
    assert normalize_bm25(-5.0) == pytest.approx(normalize_bm25(5.0))
    assert 0.0 < normalize_bm25(-5.0) < 1.0


def test_normalize_bm25_preserves_discrimination_in_realistic_range() -> None:
    # Regression test for the /20 recalibration: in the conversational-
    # corpus BM25 range (|bm25| in [15, 60]), the normalized output must
    # visibly discriminate. Old /5 divisor collapsed everything past 15 to 1.
    low = normalize_bm25(-15.0)
    mid = normalize_bm25(-30.0)
    high = normalize_bm25(-60.0)
    assert mid - low > 0.10
    assert high - mid > 0.05
    assert high < 1.0


# ─── normalize_cosine ──────────────────────────────────────────────


def test_normalize_cosine() -> None:
    # Clamp-to-[0,1]: negatives collapse to 0, positive range passes through.
    assert normalize_cosine(-1.0) == 0.0
    assert normalize_cosine(-0.5) == 0.0
    assert normalize_cosine(0.0) == 0.0
    assert normalize_cosine(0.3) == pytest.approx(0.3)
    assert normalize_cosine(0.75) == pytest.approx(0.75)
    assert normalize_cosine(1.0) == 1.0


# ─── compute_matching: weighted-sum properties ────────────────────


def test_cosine_dominates_on_fresh_plain_query() -> None:
    """On fresh memories + plain queries, cosine carries the signal.
    BM25 is a tiebreaker, not a co-equal partner.
    """

    # Pure BM25 (high bm25, zero cos). Fresh memory, plain query →
    # w_F ≈ 0.07, so this contributes at most ~0.07 · MP.
    m_fts = compute_matching(
        idx_priority=1.0, bm25_raw=-40.0, cos_sim=0.0, query_bias=1.0, params=PARAMS
    )
    # Pure cosine (zero bm25, high cos). w_S ≈ 0.93.
    m_cos = compute_matching(
        idx_priority=1.0, bm25_raw=0.0, cos_sim=0.9, query_bias=1.0, params=PARAMS
    )
    # Cosine-only is a strong match; BM25-only is tiny.
    assert m_cos > 0.70 * PARAMS.mp, f"cosine-only should be strong: {m_cos}"
    assert m_fts < 0.15 * PARAMS.mp, (
        f"BM25-only should be a minority contribution: {m_fts}"
    )
    # And cosine-only must strictly beat BM25-only at equal raw strength.
    assert m_cos > m_fts


def test_both_weak_is_weak() -> None:
    m = compute_matching(
        idx_priority=0.7, bm25_raw=-1.0, cos_sim=0.05, query_bias=1.0, params=PARAMS
    )
    # Both signals very weak → match should be small.
    assert m < 0.20 * PARAMS.mp


def test_both_zero_is_zero() -> None:
    m = compute_matching(
        idx_priority=0.7, bm25_raw=0.0, cos_sim=0.0, query_bias=1.0, params=PARAMS
    )
    assert m == 0.0


def test_monotonic_in_bm25() -> None:
    base = compute_matching(0.7, -10.0, 0.3, 1.0, PARAMS)
    stronger = compute_matching(0.7, -30.0, 0.3, 1.0, PARAMS)
    assert stronger > base


def test_monotonic_in_cosine() -> None:
    base = compute_matching(0.7, -10.0, 0.1, 1.0, PARAMS)
    stronger = compute_matching(0.7, -10.0, 0.5, 1.0, PARAMS)
    assert stronger > base


def test_saturates_at_mp() -> None:
    # Both signals at their max → match equals MP (weighted average of 1,1).
    m = compute_matching(
        idx_priority=0.5, bm25_raw=-1000.0, cos_sim=1.0, query_bias=1.0, params=PARAMS
    )
    assert math.isclose(m, PARAMS.mp, rel_tol=1e-3)


# ─── compute_matching: query_bias shifts weight toward BM25 ───────


def test_query_bias_boosts_bm25_contribution() -> None:
    # Weak-ish BM25, zero cosine. Quoted/ID query (bias=1.5) shifts w_F
    # upward — same s_f gets more weight.
    plain = compute_matching(0.7, -10.0, 0.0, 1.0, PARAMS)
    quoted = compute_matching(0.7, -10.0, 0.0, 1.5, PARAMS)
    assert quoted > plain, (
        f"query_bias=1.5 should boost BM25 vs plain query (plain={plain:.3f}, "
        f"quoted={quoted:.3f})"
    )


def test_query_bias_reduces_cosine_share_on_literal_queries() -> None:
    # Pure-cosine match, literal-match query. Cosine weight DROPS
    # (because w_S shrinks when w_F grows), so the match value drops.
    # This is the intended behavior: on literal queries, the user wants
    # BM25 to matter more — implicitly reducing the cosine share.
    plain = compute_matching(0.7, 0.0, 0.5, 1.0, PARAMS)
    quoted = compute_matching(0.7, 0.0, 0.5, 1.5, PARAMS)
    assert quoted < plain, (
        f"query_bias=1.5 shifts weight off cosine; pure-cosine match should drop "
        f"(plain={plain:.3f}, quoted={quoted:.3f})"
    )


# ─── Caroline-LGBTQ regression: golden values from ROOT_CAUSE.md ──


def test_caroline_lgbtq_regression() -> None:
    """Real LoCoMo query: gold should decisively beat the old top-1.

    With the pre-fix weighted-sum (w_F ≈ 0.70 on fresh memories),
    gold's matching was 1.311 and the false top-1's was 1.281 — gold
    won by 0.03, recency-biased base_level flipped the ranking.

    With semantic-dominant weighted-sum (w_F ≈ 0.25 default), gold
    wins by > 0.40 because cosine now carries most of the signal —
    and gold has cos=0.71 while distractor has cos ≈ 0.
    """

    # Gold D1:3: bm25=-54, cos=0.71 (both strong)
    gold = compute_matching(0.65, -53.86, 0.71, 1.2, PARAMS)
    # Old top-1 D16:5: bm25=-20, cos=-0.003 (BM25 medium, cosine effectively absent)
    distractor = compute_matching(0.76, -20.06, -0.003, 1.2, PARAMS)

    assert gold > distractor, f"gold {gold:.4f} should beat distractor {distractor:.4f}"
    assert gold - distractor > 0.30, (
        f"semantic-dominant weighted-sum should give gold a decisive lead; "
        f"gap was only {gold - distractor:.4f}"
    )


# ─── signal_contributions diagnostic helper ───────────────────────


def test_signal_contributions_bm25_only() -> None:
    cf, cs = signal_contributions(0.9, 0.0)
    assert cf == pytest.approx(0.9)
    assert cs == pytest.approx(0.0)


def test_signal_contributions_cosine_only() -> None:
    cf, cs = signal_contributions(0.0, 0.9)
    assert cf == pytest.approx(0.0)
    assert cs == pytest.approx(0.9)


def test_signal_contributions_balanced() -> None:
    cf, cs = signal_contributions(0.7, 0.7)
    # Each "only-X" contribution = x * (1 - other) = 0.7 * 0.3 = 0.21.
    assert cf == pytest.approx(0.21)
    assert cs == pytest.approx(0.21)


# ─── matching_weights — semantic-dominant weighted-sum constants ─


@pytest.mark.parametrize(
    "idx_priority, query_bias, expected_w_f",
    [
        # Fresh memory + plain query → w_F ≈ 0.07 (cosine-dominant default)
        (1.0, 1.0, 0.07),
        # Fresh memory + quoted/literal query → w_F shifts to ~0.14
        (1.0, 1.5, 0.14),
        # Fresh memory + "anti-literal" bias query → w_F drops further
        (1.0, 0.7, 0.04),
        # Old memory + plain query → w_F ≈ 0.03 (gist-dominated recall)
        (0.2, 1.0, 0.03),
        # Old memory + literal query → modest BM25 lift (still tiny share)
        (0.2, 1.5, 0.07),
    ],
)
def test_matching_weights_cosine_dominant_defaults(
    idx_priority: float, query_bias: float, expected_w_f: float
) -> None:
    w_f, w_s = matching_weights(idx_priority, query_bias)
    assert math.isclose(w_f, expected_w_f, abs_tol=0.02), (
        f"w_f at (idx_priority={idx_priority}, bias={query_bias}) = "
        f"{w_f:.3f}, expected {expected_w_f}"
    )
    assert math.isclose(w_f + w_s, 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("idx_priority", [0.01, 0.25, 0.5, 0.75, 0.99])
@pytest.mark.parametrize("bias", [0.6, 0.7, 1.0, 1.3, 1.5])
def test_matching_weights_sum_to_one(idx_priority: float, bias: float) -> None:
    w_f, w_s = matching_weights(idx_priority, bias)
    assert math.isclose(w_f + w_s, 1.0, rel_tol=1e-9)
    assert 0.0 <= w_f <= 1.0
    assert 0.0 <= w_s <= 1.0


def test_matching_weights_are_cosine_dominant_on_default_regime() -> None:
    """w_S >> w_F for fresh memories with plain queries — this is the
    whole point of the rebalance (cosine carries the load on modern
    dense embedders + conversational text, BM25 is a tiebreaker)."""

    w_f, w_s = matching_weights(idx_priority=1.0, query_bias=1.0)
    assert w_s > w_f
    assert w_s > 0.9  # w_S ≈ 0.93 at the default
    assert w_f < 0.1  # w_F ≈ 0.07 — minority share


def test_matching_weights_respect_formula_params_override() -> None:
    """Callers with BM25-friendly workloads (literal IDs, code search)
    can configure heavier FTS weighting via ``FormulaParams``. Verify
    the override actually takes effect in the weight calculation.
    """

    from mnemoss.core.config import FormulaParams

    heavy_bm25 = FormulaParams(
        match_w_f_base=0.3,
        match_w_f_slope=0.4,
        match_w_s_base=0.5,
    )
    w_f_default, _ = matching_weights(
        idx_priority=1.0, query_bias=1.0, params=None
    )
    w_f_heavy, _ = matching_weights(
        idx_priority=1.0, query_bias=1.0, params=heavy_bm25
    )
    assert w_f_heavy > 3 * w_f_default, (
        f"BM25-heavy override should lift w_F well above default "
        f"(default={w_f_default:.3f}, heavy={w_f_heavy:.3f})"
    )
