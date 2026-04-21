"""Unit tests for dynamic hybrid matching."""

from __future__ import annotations

import math

import pytest

from mnemoss.core.config import FormulaParams
from mnemoss.formula.matching import (
    compute_matching,
    matching_weights,
    normalize_bm25,
    normalize_cosine,
)

PARAMS = FormulaParams()


@pytest.mark.parametrize(
    "idx_priority, query_bias, expected_w_f",
    [
        # Values from FORMULA_AND_ARCHITECTURE.md §1.4 Combined Behavior table.
        (0.95, 1.5, 0.88),
        (0.95, 0.7, 0.62),
        (0.20, 1.5, 0.51),
        (0.20, 0.7, 0.19),
    ],
)
def test_spec_table_values(idx_priority: float, query_bias: float, expected_w_f: float) -> None:
    w_f, w_s = matching_weights(idx_priority, query_bias)
    assert math.isclose(w_f, expected_w_f, abs_tol=0.01)
    assert math.isclose(w_f + w_s, 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("idx_priority", [0.01, 0.25, 0.5, 0.75, 0.99])
@pytest.mark.parametrize("bias", [0.6, 0.7, 1.0, 1.3, 1.5])
def test_weights_sum_to_one(idx_priority: float, bias: float) -> None:
    w_f, w_s = matching_weights(idx_priority, bias)
    assert math.isclose(w_f + w_s, 1.0, rel_tol=1e-9)
    assert 0.0 <= w_f <= 1.0
    assert 0.0 <= w_s <= 1.0


def test_normalize_bm25_handles_negative_sqlite_convention() -> None:
    # SQLite FTS5 returns BM25 as negative. Normalization uses abs.
    assert normalize_bm25(0.0) == 0.0
    assert normalize_bm25(-5.0) == pytest.approx(normalize_bm25(5.0))
    assert 0.0 < normalize_bm25(-5.0) < 1.0


def test_normalize_cosine() -> None:
    assert normalize_cosine(-1.0) == 0.0
    assert normalize_cosine(0.0) == 0.5
    assert normalize_cosine(1.0) == 1.0


def test_compute_matching_scales_with_mp() -> None:
    # With w_F + w_S = 1 and both scores = 1, matching == MP.
    result = compute_matching(
        idx_priority=0.5,
        bm25_raw=-1000.0,  # large magnitude → s_f ≈ 1
        cos_sim=1.0,
        query_bias=1.0,
        params=PARAMS,
    )
    assert math.isclose(result, PARAMS.mp, rel_tol=1e-3)
