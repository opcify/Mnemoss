"""Smoke tests for the LongMemEval-S harness.

These tests exercise the harness wiring end-to-end against a fixture
dataset using the Mnemoss backend with ``FakeEmbedder`` + the mock
LLM. They don't measure recall quality — that needs the real corpus
and a real generator/judge — but they pin the contract every backend
has to satisfy and catch wiring regressions cheaply.

Run with::

    pytest bench/tests/test_longmemeval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.data.prepare_longmemeval import (
    QUESTION_TYPES,
    Question,
    Session,
    Turn,
    load_longmemeval_s,
)
from bench.longmemeval import (
    GENERATOR_PROMPT,
    JUDGE_PROMPT,
    MockLLM,
    QuestionResult,
    RunSummary,
    _build_backend,
    _eval_one,
    _format_snippets,
)

# ─── fixture builder ───────────────────────────────────────────────


def _fixture_question(qid: str, qtype: str = "single-session-user") -> Question:
    """Build a tiny synthetic question with three sessions of two turns each.

    The gold answer is ``"the capital of France is Paris"`` so the mock
    LLM's "echo last line" generator naturally emits something close
    enough for the smoke path. The mock judge is wired to always say
    ``correct`` so accuracy aggregation is exercised even though the
    answer comparison itself isn't a real check.
    """

    return Question(
        question_id=qid,
        question_type=qtype,
        question="What is the capital of France?",
        answer="the capital of France is Paris",
        sessions=(
            Session(
                session_id="s1",
                ts=1_700_000_000.0,
                date_str="2023/11/14 22:13",
                turns=(
                    Turn("user", "Hi, I'm planning a trip."),
                    Turn("assistant", "Great! Where to?"),
                ),
            ),
            Session(
                session_id="s2",
                ts=1_700_086_400.0,
                date_str="2023/11/15 22:13",
                turns=(
                    Turn("user", "Going to France next week."),
                    Turn("assistant", "Paris is wonderful in November."),
                ),
            ),
            Session(
                session_id="s3",
                ts=1_700_172_800.0,
                date_str="2023/11/16 22:13",
                turns=(
                    Turn("user", "Booked the Eiffel Tower."),
                    Turn("assistant", "Don't miss the Louvre too."),
                ),
            ),
        ),
        answer_session_ids=frozenset({"s2"}),
    )


# ─── format_snippets ───────────────────────────────────────────────


def test_format_snippets_empty() -> None:
    assert _format_snippets([]) == "(no memories surfaced)"


def test_format_snippets_numbers_in_order() -> None:
    out = _format_snippets(["alpha", "beta", "gamma"])
    assert "[1] alpha" in out
    assert "[2] beta" in out
    assert "[3] gamma" in out
    # Ordering preserved.
    assert out.index("[1]") < out.index("[2]") < out.index("[3]")


def test_format_snippets_truncates_at_budget() -> None:
    # 100-char budget, snippets way above it. First snippet fits, then
    # truncation kicks in.
    long = "x" * 200
    out = _format_snippets([long, long, long], max_chars=300)
    assert "[...truncated]" in out


# ─── prompt templates ──────────────────────────────────────────────


def test_judge_prompt_starts_with_judge_marker() -> None:
    """``MockLLM`` keys on the ``[JUDGE]`` prefix to route generator
    vs judge calls. If the template ever drops the prefix the smoke
    path silently breaks (everything turns into a generator response).
    """

    rendered = JUDGE_PROMPT.format(question="q", gold="g", pred="p")
    assert rendered.startswith("[JUDGE]")


def test_generator_prompt_does_not_start_with_judge_marker() -> None:
    rendered = GENERATOR_PROMPT.format(snippets="s", question="q")
    assert not rendered.startswith("[JUDGE]")


# ─── MockLLM ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mock_llm_routes_judge_to_canned_verdict() -> None:
    llm = MockLLM(judge_says="incorrect")
    out = await llm.complete(JUDGE_PROMPT.format(question="q", gold="g", pred="p"), max_tokens=4)
    assert out == "incorrect"


@pytest.mark.asyncio
async def test_mock_llm_echoes_last_line_for_generator() -> None:
    llm = MockLLM()
    out = await llm.complete("foo\nbar\nbaz", max_tokens=4)
    assert out == "baz"


# ─── question loader ───────────────────────────────────────────────


def test_load_longmemeval_s_round_trips_synthetic_file(tmp_path: Path) -> None:
    raw = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "what?",
            "answer": "yes",
            "haystack_session_ids": ["sa", "sb"],
            "haystack_dates": ["2023/05/03 (Wed) 10:24", "2023/05/04 (Thu) 11:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "u1"}],
                [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}],
            ],
            "answer_session_ids": ["sa"],
        }
    ]
    p = tmp_path / "lme.json"
    p.write_text(json.dumps(raw))
    questions = load_longmemeval_s(p)
    assert len(questions) == 1
    q = questions[0]
    assert q.question_id == "q1"
    assert q.question_type in QUESTION_TYPES
    assert len(q.sessions) == 2
    # Sessions sorted chronologically — sa (5/3) before sb (5/4).
    assert q.sessions[0].session_id == "sa"
    assert q.sessions[0].ts < q.sessions[1].ts
    assert q.sessions[0].turns[0].content == "u1"
    assert q.sessions[1].turns[1].role == "assistant"


def test_load_longmemeval_s_missing_file_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc_info:
        load_longmemeval_s(tmp_path / "does_not_exist.json")
    msg = str(exc_info.value)
    assert "huggingface-cli" in msg
    assert "longmemeval" in msg


def test_load_longmemeval_s_rejects_inconsistent_lengths(tmp_path: Path) -> None:
    raw = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "q?",
            "answer": "a",
            "haystack_session_ids": ["sa"],
            "haystack_dates": ["2023/05/03 10:24", "2023/05/04 11:00"],  # length mismatch
            "haystack_sessions": [[{"role": "user", "content": "u"}]],
            "answer_session_ids": ["sa"],
        }
    ]
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="lengths disagree"):
        load_longmemeval_s(p)


# ─── end-to-end smoke (mnemoss backend, FakeEmbedder, MockLLM) ─────


@pytest.mark.asyncio
async def test_eval_one_smoke_with_mnemoss_fake_embedder() -> None:
    """End-to-end: ingest → recall → generate → judge → result row.

    Uses FakeEmbedder so no network is needed; uses MockLLM so no
    OPENAI_API_KEY is needed. Confirms the harness produces a
    well-formed ``QuestionResult`` with positive timing and a
    parsed-correct verdict.
    """

    backend = _build_backend("mnemoss", smoke=True, embedder="fake", question_index=0)
    try:
        question = _fixture_question("q-smoke")
        result = await _eval_one(
            question=question,
            backend=backend,
            generator=MockLLM(gen_returns="Paris"),
            judge=MockLLM(judge_says="correct"),
            k=5,
            snippet_max_chars=2000,
            dream_after_ingest=False,
        )
    finally:
        await backend.close()

    assert isinstance(result, QuestionResult)
    assert result.question_id == "q-smoke"
    assert result.n_sessions == 3
    assert result.n_turns == 6
    assert result.ingest_seconds > 0
    assert result.recall_seconds >= 0
    assert result.correct is True
    assert result.judge_raw == "correct"
    assert result.answer_pred == "Paris"


@pytest.mark.asyncio
async def test_eval_one_tolerates_judge_quoted_punctuation() -> None:
    """The judge sometimes wraps verdicts in quotes or trailing
    punctuation. ``correct.`` and ``"correct"`` should both parse
    as correct."""

    backend = _build_backend("mnemoss", smoke=True, embedder="fake", question_index=42)
    try:
        question = _fixture_question("q-quoted")
        result = await _eval_one(
            question=question,
            backend=backend,
            generator=MockLLM(gen_returns="Paris"),
            judge=MockLLM(judge_says='"Correct."'),
            k=5,
            snippet_max_chars=2000,
            dream_after_ingest=False,
        )
    finally:
        await backend.close()

    assert result.correct is True


@pytest.mark.asyncio
async def test_eval_one_handles_incorrect_verdict() -> None:
    backend = _build_backend("mnemoss", smoke=True, embedder="fake", question_index=99)
    try:
        question = _fixture_question("q-wrong")
        result = await _eval_one(
            question=question,
            backend=backend,
            generator=MockLLM(gen_returns="London"),
            judge=MockLLM(judge_says="incorrect"),
            k=5,
            snippet_max_chars=2000,
            dream_after_ingest=False,
        )
    finally:
        await backend.close()

    assert result.correct is False
    assert result.answer_pred == "London"


# ─── RunSummary serialisation ──────────────────────────────────────


def test_run_summary_to_dict_round_trips_via_json() -> None:
    summary = RunSummary(
        backend="mnemoss",
        dataset="bench/data/longmemeval_s.json",
        k=10,
        gen_model="gpt-4o-mini",
        judge_model="gpt-4o-mini",
        n_questions=1,
        overall_accuracy=1.0,
        accuracy_by_type={"single-session-user": 1.0},
        counts_by_type={"single-session-user": 1},
        duration_seconds=3.14,
        timestamp="2026-01-01T00:00:00+00:00",
        per_question=[
            QuestionResult(
                question_id="q1",
                question_type="single-session-user",
                n_sessions=1,
                n_turns=2,
                n_recalled=3,
                ingest_seconds=0.1,
                recall_seconds=0.05,
                generate_seconds=0.2,
                judge_seconds=0.05,
                correct=True,
                judge_raw="correct",
                answer_pred="Paris",
                answer_gold="Paris",
            )
        ],
    )

    dumped = summary.to_dict()
    # Must round-trip via json.dumps without custom encoders.
    text = json.dumps(dumped)
    parsed = json.loads(text)
    assert parsed["overall_accuracy"] == 1.0
    assert parsed["accuracy_by_type"]["single-session-user"] == 1.0
    assert parsed["per_question"][0]["question_id"] == "q1"


# ─── module-level CLI sanity ───────────────────────────────────────


def test_cli_help_runs_without_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--help`` must work even with no OPENAI_API_KEY in env. This
    is a regression guard: lazy LLM construction matters for
    discoverability."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from bench.longmemeval import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
