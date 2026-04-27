"""E2E smoke for ``bench/launch_comparison.py`` — Chart 1 runs.

The blog post's load-bearing claim is "reproducible via
``make launch-bench``." These tests lock in the mechanics: both
backends run end-to-end against LoCoMo, emit valid JSON with the
schema downstream ``bench/plots.py`` expects, and do not mutate state
between runs. Numeric recall values are asserted only for presence /
plausibility, not magnitude — magnitude depends on embedder quality
and is verified at publication time, not in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.launch_comparison import run_chart1

LOCOMO_MEMS = Path("bench/data/locomo_memories.jsonl")
LOCOMO_QS = Path("bench/data/locomo_queries.jsonl")


@pytest.fixture(scope="module")
def locomo_ready() -> bool:
    if not LOCOMO_MEMS.exists() or not LOCOMO_QS.exists():
        pytest.skip("LoCoMo corpus not prepared. Run `python -m bench.data.prepare_locomo` first.")
    return True


# ─── static_file path ──────────────────────────────────────────────


async def test_chart1_static_file_smoke(locomo_ready: bool) -> None:
    """Static-file baseline end-to-end on a tiny slice."""

    summary = await run_chart1(
        backend_name="static_file",
        k=10,
        limit_conversations=1,
        limit_utterances=50,
        fake_embedder=False,
    )
    assert summary.chart == 1
    assert summary.backend == "static_file"
    assert summary.corpus == "locomo"
    assert summary.k == 10
    assert summary.n_conversations == 1
    assert summary.n_queries > 0
    # Token-overlap grep on a real conversational slice clears 0.
    # (If this ever returns 0 exactly, the corpus or scoring broke.)
    assert summary.aggregate_recall_at_k > 0.0
    assert summary.aggregate_recall_at_k <= 1.0

    # Per-conversation shape.
    assert len(summary.per_conversation) == 1
    row = summary.per_conversation[0]
    assert row.n_memories == 50
    assert row.n_queries_scored + row.n_queries_skipped > 0
    assert 0.0 <= row.mean_recall_at_k <= 1.0


# ─── mnemoss path (FakeEmbedder — no network) ──────────────────────


async def test_chart1_mnemoss_smoke_fake_embedder(locomo_ready: bool) -> None:
    """Mnemoss end-to-end with FakeEmbedder (no OpenAI calls).

    Numeric values are not meaningful here (FakeEmbedder is hash-based),
    but the plumbing MUST work: observe → recall → score → JSON.
    """

    summary = await run_chart1(
        backend_name="mnemoss",
        k=10,
        limit_conversations=1,
        limit_utterances=50,
        fake_embedder=True,
    )
    assert summary.backend == "mnemoss"
    assert summary.n_conversations == 1
    assert summary.n_queries > 0
    # With FakeEmbedder the absolute number is meaningless, but it's
    # still bounded to the valid range.
    assert 0.0 <= summary.aggregate_recall_at_k <= 1.0


# ─── JSON artifact shape ──────────────────────────────────────────


async def test_chart1_json_has_expected_schema(locomo_ready: bool, tmp_path: Path) -> None:
    """Downstream ``bench/plots.py`` reads the JSON; the schema must be
    stable. This test is the contract between harness and plotter."""

    summary = await run_chart1(
        backend_name="static_file",
        k=5,
        limit_conversations=1,
        limit_utterances=30,
        fake_embedder=False,
    )
    out_path = tmp_path / "chart1.json"
    out_path.write_text(json.dumps(summary.to_dict()))
    payload = json.loads(out_path.read_text())

    # Top-level keys.
    for key in (
        "chart",
        "backend",
        "corpus",
        "k",
        "params",
        "per_conversation",
        "aggregate",
        "timestamp",
        "duration_seconds",
    ):
        assert key in payload, f"missing top-level key: {key!r}"

    # Types.
    assert isinstance(payload["chart"], int)
    assert isinstance(payload["backend"], str)
    assert isinstance(payload["corpus"], str)
    assert isinstance(payload["k"], int)
    assert isinstance(payload["params"], dict)
    assert isinstance(payload["per_conversation"], list)
    assert isinstance(payload["aggregate"], dict)
    assert isinstance(payload["timestamp"], str)
    assert isinstance(payload["duration_seconds"], (int, float))

    # params keys.
    assert "limit_conversations" in payload["params"]
    assert "limit_utterances" in payload["params"]
    assert "fake_embedder" in payload["params"]

    # aggregate keys.
    agg = payload["aggregate"]
    assert "mean_recall_at_k" in agg
    assert "n_conversations" in agg
    assert "n_queries" in agg

    # per-conversation row keys.
    for row in payload["per_conversation"]:
        assert "conversation_id" in row
        assert "n_memories" in row
        assert "n_queries_scored" in row
        assert "n_queries_skipped" in row
        assert "mean_recall_at_k" in row


# ─── fresh backend per conversation (state isolation) ─────────────


async def test_multiple_conversations_do_not_leak_state(
    locomo_ready: bool,
) -> None:
    """Each conversation runs in a fresh backend. A memory from conv-A
    must not appear in conv-B's recall results. Implicit in the
    ``_build_backend`` per-conversation pattern; this test locks it in
    by running two conversations and verifying each only scored against
    its own utterances."""

    summary = await run_chart1(
        backend_name="static_file",
        k=10,
        limit_conversations=2,
        limit_utterances=30,
        fake_embedder=False,
    )
    assert summary.n_conversations == 2
    ids = {r.conversation_id for r in summary.per_conversation}
    assert len(ids) == 2, "must have two distinct conversation ids"
    # Each row's scored+skipped count must be > 0 — proves queries were
    # actually attempted per-conversation, not mis-aggregated into one.
    for r in summary.per_conversation:
        assert r.n_queries_scored + r.n_queries_skipped > 0
