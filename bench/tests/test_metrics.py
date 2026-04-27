"""Unit tests for ``bench/_metrics.py``.

ARI noise handling and bootstrap CI math are pure-functions; these
tests pin the specific numerical behavior that the validation harness
depends on. If a future sklearn / numpy upgrade breaks these, we want
loud failure, not silent drift.

Run with::

    pytest bench/tests/

(or ``pytest -m bench`` once we add the bench marker to pyproject).
"""

from __future__ import annotations

import pytest

from bench._metrics import bootstrap_ci, noise_aware_ari, topk_cleanliness

# ─── noise_aware_ari ───────────────────────────────────────────────


def test_ari_perfect_agreement_returns_1() -> None:
    predicted = {"m1": 0, "m2": 0, "m3": 1, "m4": 1}
    gold = {"m1": "a", "m2": "a", "m3": "b", "m4": "b"}
    ari, scored, dropped = noise_aware_ari(predicted, gold)
    assert ari == pytest.approx(1.0)
    assert scored == 4
    assert dropped == 0


def test_ari_filters_noise_label_none() -> None:
    """HDBSCAN's noise label (``None``) drops from BOTH sides — the
    metric scores only memories the clusterer was confident about."""

    predicted = {"m1": 0, "m2": None, "m3": 1, "m4": 1}
    gold = {"m1": "a", "m2": "a", "m3": "b", "m4": "b"}
    ari, scored, dropped = noise_aware_ari(predicted, gold)
    # m2 (noise) drops; remaining 3 memories perfectly agree.
    assert ari == pytest.approx(1.0)
    assert scored == 3
    assert dropped == 1


def test_ari_filters_noise_label_minus_one() -> None:
    """``-1`` is the alternative HDBSCAN noise convention."""

    predicted = {"m1": 0, "m2": -1, "m3": 1}
    gold = {"m1": "a", "m2": "a", "m3": "b"}
    _ari, scored, dropped = noise_aware_ari(predicted, gold)
    assert scored == 2
    assert dropped == 1


def test_ari_returns_zero_when_too_few_scored() -> None:
    """If filtering leaves <2 memories, ARI is undefined; we surface
    that via ``scored_count`` rather than raising."""

    predicted = {"m1": None, "m2": None, "m3": 0}
    gold = {"m1": "a", "m2": "a", "m3": "b"}
    ari, scored, dropped = noise_aware_ari(predicted, gold)
    assert ari == 0.0
    assert scored == 1
    assert dropped == 2


def test_ari_rejects_mismatched_keysets() -> None:
    predicted = {"m1": 0, "m2": 0}
    gold = {"m1": "a", "m3": "b"}  # m2 missing, m3 extra
    with pytest.raises(ValueError, match="same memory ids"):
        noise_aware_ari(predicted, gold)


# ─── bootstrap_ci ──────────────────────────────────────────────────


def test_bootstrap_empty_returns_zeros() -> None:
    mean, lo, hi = bootstrap_ci([])
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


def test_bootstrap_all_wins_gives_high_ci() -> None:
    """100 wins out of 100 → mean=1.0, CI tight near 1.0."""

    mean, lo, hi = bootstrap_ci([1.0] * 100, n_resamples=500, seed=0)
    assert mean == pytest.approx(1.0)
    # All-ones resampling gives constant 1.0 every iteration.
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_balanced_centers_on_half() -> None:
    """50 wins + 50 losses → mean=0.5, CI brackets 0.5."""

    outcomes = [1.0] * 50 + [0.0] * 50
    mean, lo, hi = bootstrap_ci(outcomes, n_resamples=500, seed=42)
    assert mean == pytest.approx(0.5)
    assert lo < 0.5 < hi
    # Sanity: CI shouldn't be wider than the [0, 1] domain.
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_bootstrap_is_seed_reproducible() -> None:
    a = bootstrap_ci([1.0, 0.0, 1.0, 0.5, 1.0], n_resamples=200, seed=7)
    b = bootstrap_ci([1.0, 0.0, 1.0, 0.5, 1.0], n_resamples=200, seed=7)
    assert a == b


# ─── topk_cleanliness ──────────────────────────────────────────────


def test_topk_clean_when_no_junk_in_topk() -> None:
    assert topk_cleanliness(["a", "b", "c"], junk_ids={"x", "y"}, k=10) is True


def test_topk_dirty_when_junk_present() -> None:
    assert topk_cleanliness(["a", "x", "c"], junk_ids={"x"}, k=10) is False


def test_topk_only_considers_first_k() -> None:
    """Junk at rank 11 doesn't pollute top-10 cleanliness."""

    predicted = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "junk"]
    assert topk_cleanliness(predicted, junk_ids={"junk"}, k=10) is True
    assert topk_cleanliness(predicted, junk_ids={"junk"}, k=11) is False
