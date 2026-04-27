"""Tests for the synthetic pressure corpus generator + committed JSONL.

The committed ``bench/fixtures/pressure_corpus_seed42.jsonl`` is the
foundation of weekend-2 verdicts. These tests pin its shape so a
template tweak that quietly halves the high-utility count gets caught
in CI rather than mid-ablation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.fixtures.pressure_corpus_gen import (
    BACKBONE_FRAC,
    HIGH_UTILITY_FRAC,
    JUNK_FRAC,
    SIMULATED_DAYS,
    TOTAL_MEMORIES,
    _generate,
)

CORPUS_PATH = Path(__file__).parent.parent / "fixtures" / "pressure_corpus_seed42.jsonl"


# ─── shape invariants on the committed corpus ──────────────────────


def test_committed_corpus_loads() -> None:
    assert CORPUS_PATH.exists(), (
        "committed pressure corpus is missing; run `make pressure-corpus-gen` to regenerate"
    )
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert data["_meta"]["total_memories"] == TOTAL_MEMORIES
    assert data["_meta"]["simulated_days"] == SIMULATED_DAYS
    assert len(data["memories"]) == TOTAL_MEMORIES
    assert data["_meta"]["queries"] == len(data["queries"])


def test_committed_corpus_distribution_matches_constants() -> None:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    counts = {"high": 0, "medium": 0, "low": 0}
    for m in data["memories"]:
        counts[m["utility"]] += 1
    assert counts["high"] == int(TOTAL_MEMORIES * HIGH_UTILITY_FRAC)
    assert counts["medium"] == int(TOTAL_MEMORIES * BACKBONE_FRAC)
    assert counts["low"] == TOTAL_MEMORIES - counts["high"] - counts["medium"]
    # Sanity: junk fraction matches the constant within 1.
    assert abs(counts["low"] - int(TOTAL_MEMORIES * JUNK_FRAC)) <= 1


def test_committed_corpus_timestamps_span_simulated_period() -> None:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    offsets = [m["ts_offset_seconds"] for m in data["memories"]]
    assert min(offsets) >= 0
    assert max(offsets) <= SIMULATED_DAYS * 86400
    # Reasonable spread: last offset is in the second half.
    assert max(offsets) > (SIMULATED_DAYS * 86400) // 2


def test_committed_corpus_queries_have_structure() -> None:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for q in data["queries"]:
        assert "query" in q and isinstance(q["query"], str) and q["query"]
        assert "relevant_ids" in q
        assert isinstance(q["relevant_ids"], list)
        # Pressure corpus queries always have junk_ids (even if empty).
        assert "junk_ids" in q
        assert isinstance(q["junk_ids"], list)


def test_committed_corpus_relevant_ids_reference_real_memories() -> None:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    real_ids = {m["id"] for m in data["memories"]}
    for q in data["queries"]:
        assert set(q["relevant_ids"]).issubset(real_ids)
        assert set(q["junk_ids"]).issubset(real_ids)


def test_adversariality_at_least_70_percent_of_queries() -> None:
    """At least 70% of queries must have at least one junk candidate.

    Without this, the corpus isn't actually adversarial — Dispose would
    have nothing to do because there's no junk pollution to remove.
    The per-query pre-Dispose validation in the harness uses 80% as a
    soft gate; we use 70% here as the absolute floor.
    """

    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    qs_with_junk = sum(1 for q in data["queries"] if q["junk_ids"])
    assert qs_with_junk / len(data["queries"]) >= 0.70


# ─── generator determinism ─────────────────────────────────────────


def test_generator_is_seed_reproducible() -> None:
    """Two runs with the same seed must produce byte-identical corpora.

    Without this, the committed JSONL would drift between contributors
    and the pre-registration audit trail would be meaningless."""

    a = _generate(seed=42)
    b = _generate(seed=42)
    assert a == b


def test_different_seeds_produce_different_corpora() -> None:
    a = _generate(seed=42)
    b = _generate(seed=43)
    assert a["memories"] != b["memories"]


@pytest.mark.parametrize("seed", [0, 1, 42, 100])
def test_generator_invariants_hold_across_seeds(seed: int) -> None:
    """Distribution + timestamp invariants must hold for any seed."""

    data = _generate(seed=seed)
    assert data["_meta"]["total_memories"] == TOTAL_MEMORIES
    counts = {"high": 0, "medium": 0, "low": 0}
    for m in data["memories"]:
        counts[m["utility"]] += 1
    assert counts["high"] == int(TOTAL_MEMORIES * HIGH_UTILITY_FRAC)
    assert counts["medium"] == int(TOTAL_MEMORIES * BACKBONE_FRAC)
