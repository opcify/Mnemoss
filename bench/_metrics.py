"""Shared scoring functions for the dreaming-validation harness.

Two pieces:

- ``noise_aware_ari`` — Adjusted Rand Index that filters out HDBSCAN's
  noise label (cluster_id ``None`` or ``-1``) before scoring, so the
  metric only compares memories the clusterer was confident about
  against their hand-labeled topic. Standard scikit-learn ARI treats
  noise as its own cluster, which is misleading when noise is really
  an *abstention*.

- ``bootstrap_ci`` — bootstrapped two-sided percentile confidence
  interval for win-rate-style binary outcomes. Used by the gist-quality
  judge in ``bench/gist_quality.py`` so the Consolidate verdict carries
  uncertainty bars.

Pure functions, no I/O, no Mnemoss dependencies. Trivially unit-testable
in ``bench/tests/test_metrics.py``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

# Numpy + sklearn are already transitive deps via sentence-transformers.
import numpy as np
from sklearn.metrics import adjusted_rand_score


def noise_aware_ari(
    predicted: dict[str, str | int | None],
    gold: dict[str, str | int | None],
) -> tuple[float, int, int]:
    """Adjusted Rand Index after filtering HDBSCAN noise labels.

    Both ``predicted`` and ``gold`` map ``memory_id -> cluster_label``.
    A label of ``None`` or ``-1`` (HDBSCAN's noise convention) is
    treated as an abstention: those memories drop from BOTH sides
    before ARI is computed.

    Returns ``(ari, scored_count, dropped_count)``. Caller can
    interpret ``dropped_count / total`` as the noise rate.

    If after filtering fewer than 2 memories remain, ARI is
    mathematically undefined; we return ``(0.0, scored_count, dropped)``
    and let the harness surface that via ``scored_count`` rather than
    raising. The interpretation is "no signal."
    """

    if predicted.keys() != gold.keys():
        # Only score memories present in BOTH maps. Mismatched keysets
        # would silently bias ARI; force the caller to align before
        # passing in.
        raise ValueError(
            "predicted and gold must cover the same memory ids "
            f"(predicted-only: {sorted(set(predicted) - set(gold))}, "
            f"gold-only: {sorted(set(gold) - set(predicted))})"
        )

    pred_labels: list[Any] = []
    gold_labels: list[Any] = []
    dropped = 0

    for mid in sorted(predicted):
        p = predicted[mid]
        g = gold[mid]
        if p is None or p == -1:
            dropped += 1
            continue
        # Noise in gold should never happen — gold labels are hand-
        # written and have no abstention concept — but guard anyway.
        if g is None or g == -1:
            dropped += 1
            continue
        pred_labels.append(p)
        gold_labels.append(g)

    scored = len(pred_labels)
    if scored < 2:
        return 0.0, scored, dropped

    # adjusted_rand_score wants array-likes; numpy arrays are fine.
    return (
        float(adjusted_rand_score(np.array(gold_labels), np.array(pred_labels))),
        scored,
        dropped,
    )


def bootstrap_ci(
    outcomes: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Two-sided percentile bootstrap CI for the mean of ``outcomes``.

    ``outcomes`` is typically a list of {0.0, 0.5, 1.0} values where
    1.0 = post-Consolidate gist won, 0.5 = tie, 0.0 = level-1 won.
    The mean is the win rate; this function gives the CI on that mean.

    Returns ``(mean, ci_lower, ci_upper)``. Empty input returns
    ``(0.0, 0.0, 0.0)`` rather than raising — the harness surfaces
    "no judge calls produced a comparable pair" in its own report.

    Reproducible: a fixed ``seed`` makes the same outcomes produce the
    same CI across runs.
    """

    if not outcomes:
        return 0.0, 0.0, 0.0

    arr = np.array(outcomes, dtype=float)
    mean = float(arr.mean())

    rng = random.Random(seed)
    means: list[float] = []
    n = len(arr)
    for _ in range(n_resamples):
        sample = [arr[rng.randrange(n)] for _ in range(n)]
        means.append(float(np.mean(sample)))

    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return mean, means[lo_idx], means[hi_idx]


def topk_cleanliness(
    predicted: Sequence[str],
    junk_ids: set[str],
    k: int = 10,
) -> bool:
    """Return True iff zero junk-utility memories appear in top-K.

    Used by the Dispose verdict on the pressure corpus's adversarial
    queries. The cleanliness *fraction* is the rate of clean queries
    across the corpus.
    """

    return not (set(predicted[:k]) & junk_ids)
