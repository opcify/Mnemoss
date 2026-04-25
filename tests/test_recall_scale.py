"""Regression guard for default ``FormulaParams`` at bulk-ingest scale.

The April-2026 root-cause sweep (see ``docs/ROOT_CAUSE.md``) traced
catastrophic recall collapse (from ~0.54 → ~0.00 on LoCoMo at N=5K)
to two default values that were quietly mis-tuned for bulk ingest:

- ``d=0.5`` — ACT-R textbook value; made ``B_i`` a steep function of
  ingest order, so last-ingested memories always won.
- ``noise_scale=0.25`` — Gaussian noise with SD too large relative
  to matching-term differentials; scrambled rankings.

Both were fixed (``d=0.01``, ``noise_scale=0.0``). This test exists so
the next person who "restores ACT-R defaults" discovers the effect
immediately instead of shipping a silent 40-pp recall loss.

Run condition: marked as ``integration`` because it needs the
``LocalEmbedder`` (MiniLM) model weights to be present. Skipped by
``pytest -m "not integration"`` in fast CI; run before any release
with ``pytest -m integration tests/test_recall_scale.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.backends.mnemoss_backend import MnemossBackend
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)

# Floor the regression would have to smash through before we ship.
# With MiniLM default params as of the fix, N=500 LoCoMo conv-26
# scores ~0.54. Pre-fix (d=0.5, noise=0.25) scored ~0.25; setting
# the threshold at 0.40 gives a ~14pp safety margin over current
# performance and still fires hard on any regression that restores
# the buggy textbook defaults.
RECALL_FLOOR = 0.40


@pytest.mark.integration
async def test_default_formula_params_do_not_collapse_at_bulk_ingest(
    tmp_path: Path,
) -> None:
    """Default ``FormulaParams()`` must produce usable recall at N=500.

    If this test starts failing, the default ``d`` or ``noise_scale``
    has drifted back toward the original ACT-R textbook values. That
    is a 20-40pp recall regression on *any* bulk-ingested workspace
    (benchmarks, batch imports, restore-from-export). Do not relax
    this threshold — fix the defaults.

    Uses the ``mnemoss`` backend (full ACT-R recall path), not the
    ``rocket`` preset. Rocket bypasses most of the formula; this test
    specifically guards the default path that new users will hit.
    """

    scale_n = 500
    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories,
        queries,
        gold_conversation_id="conv-26",
        scale_n=scale_n,
    )

    # Construct the backend with no formula override → exercises the
    # shipped FormulaParams defaults end-to-end.
    backend = MnemossBackend(embedding_model=_resolve_embedder("local"))
    try:
        dia_to_mid: dict[str, str] = {}
        for m in padded_mems:
            dia_to_mid[m["dia_id"]] = await backend.observe(m["text"], ts=m["ts"])

        scored = 0
        hit_sum = 0.0
        for q in gold_queries:
            gold_ids = {dia_to_mid[d] for d in q["relevant_dia_ids"] if d in dia_to_mid}
            if not gold_ids:
                continue
            hits = await backend.recall(q["question"], k=10)
            returned_ids = {h.memory_id for h in hits}
            hit_sum += len(returned_ids & gold_ids) / len(gold_ids)
            scored += 1

        recall = hit_sum / scored if scored else 0.0
    finally:
        await backend.close()

    assert scored > 0, "corpus produced zero scorable queries; check LoCoMo data"
    assert recall >= RECALL_FLOOR, (
        f"default FormulaParams collapsed at N={scale_n}: "
        f"recall@10 = {recall:.4f} < floor {RECALL_FLOOR}. "
        f"Scored {scored} queries. "
        "This usually means someone restored d=0.5 or noise_scale=0.25 — "
        "see tests/test_recall_scale.py header and docs/ROOT_CAUSE.md."
    )
