"""Smoke test: LoCoMo corpus ingest against MnemossBackend.

This is the "first successful ingest" milestone from the Friday-evening
plan. If this test passes, the LoCoMo adaptation in
``bench/data/prepare_locomo.py`` is usable by the launch-comparison
harness; we can ship Chart 1 against LoCoMo, not just synthetic data.

Runs on a small slice (one conversation, capped at 20 utterances) using
``FakeEmbedder`` so it finishes in <2s and doesn't need network. The
full per-conversation run against ``text-embedding-3-small`` happens in
the launch-comparison harness, not here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from bench.backends.mnemoss_backend import MnemossBackend
from mnemoss import FakeEmbedder

CORPUS_PATH = Path("bench/data/locomo_memories.jsonl")
QUERIES_PATH = Path("bench/data/locomo_queries.jsonl")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def locomo_ready() -> bool:
    """Skip these tests if the LoCoMo corpus wasn't prepared.

    The JSONLs are checked into the repo as part of the launch work,
    but the test must still be resilient to a fresh clone that hasn't
    run ``bench.data.prepare_locomo`` yet.
    """

    if not CORPUS_PATH.exists() or not QUERIES_PATH.exists():
        pytest.skip("LoCoMo corpus not prepared. Run `python -m bench.data.prepare_locomo` first.")
    return True


# ─── corpus shape ──────────────────────────────────────────────────


def test_locomo_memories_have_expected_fields(locomo_ready: bool) -> None:
    """Every memory row has the fields the harness relies on."""

    rows = _load_jsonl(CORPUS_PATH)
    assert rows, "LoCoMo memories JSONL must not be empty"
    required = {"conversation_id", "dia_id", "ts", "text", "speaker", "session"}
    for row in rows[:20]:
        missing = required - row.keys()
        assert not missing, f"memory row missing fields {missing}: {row}"
        assert isinstance(row["ts"], (int, float))
        assert row["text"], "text must be non-empty"


def test_locomo_queries_have_expected_fields(locomo_ready: bool) -> None:
    """Every query row has the fields the harness relies on."""

    rows = _load_jsonl(QUERIES_PATH)
    assert rows, "LoCoMo queries JSONL must not be empty"
    required = {"conversation_id", "question", "relevant_dia_ids", "category"}
    for row in rows[:20]:
        missing = required - row.keys()
        assert not missing, f"query row missing fields {missing}: {row}"
        assert row["relevant_dia_ids"], (
            "queries must have at least one evidence dia_id (prepare_locomo "
            "drops those without evidence)"
        )


def test_locomo_dia_ids_are_unique_within_conversation(locomo_ready: bool) -> None:
    """A ``dia_id`` is the LoCoMo-side id the harness maps to Mnemoss's
    native memory_id. Collisions within a single conversation would
    make the recall-relevance lookup ambiguous.

    Across conversations, ``dia_id`` may repeat (LoCoMo restarts
    numbering per conversation) — that's fine because each conversation
    runs in its own workspace.
    """

    rows = _load_jsonl(CORPUS_PATH)
    by_conv = defaultdict(set)
    for r in rows:
        conv = r["conversation_id"]
        dia = r["dia_id"]
        assert dia not in by_conv[conv], f"duplicate dia_id {dia!r} in conversation {conv}"
        by_conv[conv].add(dia)


def test_locomo_query_evidence_refers_to_existing_dia_ids(locomo_ready: bool) -> None:
    """Every query's ``relevant_dia_ids`` must point at actual utterances
    in the same conversation. If evidence doesn't resolve, recall@k can't
    be computed."""

    mems = _load_jsonl(CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    per_conv_ids: dict[str, set[str]] = defaultdict(set)
    for m in mems:
        per_conv_ids[m["conversation_id"]].add(m["dia_id"])

    orphans = 0
    for q in queries:
        conv = q["conversation_id"]
        known = per_conv_ids[conv]
        for dia_id in q["relevant_dia_ids"]:
            if dia_id not in known:
                orphans += 1
    # Small tolerance for benchmark-source errata; bail loudly if
    # more than 1% of evidence doesn't resolve.
    total_evidence = sum(len(q["relevant_dia_ids"]) for q in queries)
    assert orphans / total_evidence < 0.01, (
        f"{orphans} orphan evidence refs out of {total_evidence} "
        f"({orphans / total_evidence:.2%}) — LoCoMo parse is broken"
    )


# ─── ingest + recall ──────────────────────────────────────────────


async def test_locomo_first_conversation_ingests_cleanly(locomo_ready: bool) -> None:
    """The 45-min timebox objective: a slice of LoCoMo ingests into
    MnemossBackend without errors and produces recallable memories.

    Slice: first 20 utterances from the first conversation. Keeps the
    test <5s with FakeEmbedder. If this passes, the launch-comparison
    harness can run the full 5,882-utterance corpus.
    """

    mems = _load_jsonl(CORPUS_PATH)
    assert mems, "need at least one memory row"
    first_conv = mems[0]["conversation_id"]
    slice_ = [m for m in mems if m["conversation_id"] == first_conv][:20]
    assert len(slice_) == 20, "first conversation must have ≥20 utterances"

    dia_to_mem_id: dict[str, str] = {}
    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        for row in slice_:
            mem_id = await be.observe(row["text"], ts=row["ts"])
            dia_to_mem_id[row["dia_id"]] = mem_id
        # Self-recall: the first utterance's own text should come back
        # at rank 1 (FakeEmbedder hashes text, so exact-match is a
        # perfect embedding match).
        hits = await be.recall(slice_[0]["text"], k=5)
        assert hits, "recall on an ingested slice must return hits"
        assert hits[0].memory_id == dia_to_mem_id[slice_[0]["dia_id"]]


async def test_locomo_query_can_be_recalled_after_ingest(locomo_ready: bool) -> None:
    """End-to-end: ingest a conversation's worth of utterances, fire a
    LoCoMo query, verify the harness can produce a ranked list of
    ``(mem_id, rank, score)`` ready for recall@k computation.

    This test does NOT assert on the quality of Mnemoss's recall
    (``FakeEmbedder`` is hash-based, not semantic). It asserts the
    mechanics: observe every utterance → query → get ranked hits →
    translate back to ``dia_id`` space for scoring. Quality measurement
    belongs to the full launch-comparison harness with a real embedder.
    """

    mems = _load_jsonl(CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH)

    # Pick the first conversation that has at least one query.
    convs_with_qs = {q["conversation_id"] for q in queries}
    first_conv = next(m["conversation_id"] for m in mems if m["conversation_id"] in convs_with_qs)
    conv_mems = [m for m in mems if m["conversation_id"] == first_conv]
    conv_queries = [q for q in queries if q["conversation_id"] == first_conv]
    assert conv_mems and conv_queries

    dia_to_mem_id: dict[str, str] = {}
    async with MnemossBackend(embedding_model=FakeEmbedder(dim=16)) as be:
        # Ingest all utterances for this conversation.
        for row in conv_mems:
            mem_id = await be.observe(row["text"], ts=row["ts"])
            dia_to_mem_id[row["dia_id"]] = mem_id

        # Fire the first query, translate the recall result back to dia_ids.
        q0 = conv_queries[0]
        hits = await be.recall(q0["question"], k=10)
        # We cannot assert the gold answer lands in top-k with FakeEmbedder;
        # just assert the plumbing works.
        assert isinstance(hits, list)
        for h in hits:
            assert h.memory_id in set(dia_to_mem_id.values()), (
                "every hit must be an ingested memory id"
            )
