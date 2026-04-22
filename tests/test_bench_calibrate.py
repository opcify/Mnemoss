"""Smoke tests for the calibration harness.

The real value of the harness shows up with a real labeled corpus;
here we just prove the metrics math is right and the sweep returns
one result per ``(d, tau, mp)`` combo.
"""

from __future__ import annotations

import pytest

from bench.calibrate import (
    Corpus,
    ParamGrid,
    _demo_corpus,
    _eval_params,
    _mrr,
    _recall_at_k,
    _sweep,
)

# ─── metric primitives ──────────────────────────────────────────────


def test_recall_at_k_counts_top_k_hits() -> None:
    assert _recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0
    assert _recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == 0.5
    assert _recall_at_k(["x", "y"], {"a"}, k=5) == 0.0


def test_recall_at_k_empty_relevant_returns_zero() -> None:
    assert _recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert _mrr(["a", "b", "c"], {"b"}) == 0.5
    assert _mrr(["a", "b"], {"a"}) == 1.0
    assert _mrr(["a", "b"], {"z"}) == 0.0


# ─── corpus validation ─────────────────────────────────────────────


def test_corpus_rejects_unknown_relevant_id() -> None:
    with pytest.raises(ValueError, match="unknown memory ids"):
        Corpus(
            memories=[{"id": "m1", "content": "hi"}],
            queries=[{"query": "hi?", "relevant_ids": ["m999"]}],
        )


def test_corpus_rejects_empty_memories() -> None:
    with pytest.raises(ValueError, match="no memories"):
        Corpus(memories=[], queries=[{"query": "x", "relevant_ids": []}])


def test_corpus_rejects_empty_queries() -> None:
    with pytest.raises(ValueError, match="no queries"):
        Corpus(memories=[{"id": "m1", "content": "hi"}], queries=[])


# ─── sweep ─────────────────────────────────────────────────────────


async def test_sweep_produces_one_result_per_combo() -> None:
    corpus = _demo_corpus()
    # Tiny grid: 2×2×1 = 4 combos so the test runs in <1s.
    grid = ParamGrid(
        d_values=(0.3, 0.7),
        tau_values=(-1.0, 0.0),
        mp_values=(1.5,),
    )
    results = await _sweep(corpus, grid)
    assert len(results) == 4
    # Each result has the three metrics set to valid ranges.
    for r in results:
        assert 0.0 <= r.recall_at_1 <= 1.0
        assert 0.0 <= r.recall_at_5 <= 1.0
        assert 0.0 <= r.mrr <= 1.0
        assert set(r.params.keys()) == {"d", "tau", "mp"}


async def test_eval_params_produces_per_query_entries() -> None:
    corpus = _demo_corpus()
    r = await _eval_params(corpus, d=0.5, tau=-1.0, mp=1.5)
    assert len(r.per_query) == len(corpus.queries)
    for entry in r.per_query:
        assert "query" in entry
        assert "recall_at_1" in entry
        assert "mrr" in entry
