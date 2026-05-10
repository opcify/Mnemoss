"""LongMemEval-S benchmark — Mnemoss vs Mem0 vs MemOS 2.0.

LongMemEval (Wu et al. 2024) is the standard end-to-end QA benchmark
for chat-assistant memory layers. The ``-S`` slice runs 500 questions
over per-question multi-session histories averaging ~115K tokens.

Why this bench
--------------

Existing Mnemoss benches (LoCoMo recall@k, supersession, raw_stack
parity) measure recall correctness against ground-truth memory ids.
LongMemEval-S measures the thing that actually matters at the seam
between memory and a real agent: *did the agent answer the user's
question correctly using only what its memory layer surfaced.* That
question type-classifies into six slices (single-session-user/
assistant/preference, multi-session, knowledge-update, temporal-
reasoning) where the per-type breakdown reveals architectural
strengths a single accuracy number hides — Mnemoss's ACT-R formula
is built to win the knowledge-update + temporal-reasoning slices in
particular.

Protocol (per question)
-----------------------

1. Build a fresh backend instance — workspace / cube / namespace
   is unique to this question so backends with global state don't
   leak across questions.
2. Iterate ``haystack_sessions`` in chronological order. Each
   session's turns go in via ``ingest_session(session_id, ts, turns)``
   so backends that scope by session see the right structure.
3. Recall top-K for the final ``question`` — 10 by default; matches
   the original paper's evaluation budget.
4. Compose a generator prompt from the recalled memory text. Send to
   ``--gen-model`` (default ``gpt-4o-mini``) — kept light because the
   generator's job is "answer from these memories," not reasoning.
5. Send the generator's answer + the gold ``answer`` to ``--judge-model``
   (default ``gpt-4o-mini``) which returns ``correct`` / ``incorrect``.
6. Aggregate per ``question_type`` and overall accuracy.

Embedder + judge parity
-----------------------

All three backends pin OpenAI ``text-embedding-3-small`` for the
vector side (matches Mnemoss's published-chart config — Issue 1.1A
in the launch eng review). Mem0 and MemOS additionally use OpenAI for
their internal extraction LLM; we don't override because that's part
of how they ship. The generator + judge are also shared across
backends — only the *memory layer* changes between runs, which is
what the bench is meant to compare.

Usage
-----

Mnemoss vs Mem0 vs MemOS, full 500 questions, OpenAI everywhere::

    python -m bench.longmemeval --backend mnemoss \\
        --out bench/results/lme_s_mnemoss.json
    python -m bench.longmemeval --backend mem0 \\
        --out bench/results/lme_s_mem0.json
    python -m bench.longmemeval --backend memos \\
        --out bench/results/lme_s_memos.json

Smoke check (no network — FakeEmbedder + mock judge, ~5 questions)::

    python -m bench.longmemeval --backend mnemoss --smoke \\
        --limit 5 --out /tmp/lme_smoke.json

Cost note
---------

The full bench is *expensive*. Mnemoss-only ingest is just embedder
calls (~$1 at default sizes) but Mem0 + MemOS run extraction LLMs
during ingest — expect $20-$80 for a full 500-question run on each.
The generator + judge add ~$5-$10 across all 500. Use ``--limit`` and
``--question-types`` to scope down for iteration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# .env loader is opt-in (matches launch_comparison.py).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from bench.backends.base import RecallHit
from bench.data.prepare_longmemeval import (
    CANONICAL_PATH,
    QUESTION_TYPES,
    Question,
    load_longmemeval_s,
)

# ─── backend protocol ──────────────────────────────────────────────


@runtime_checkable
class LongMemEvalBackend(Protocol):
    """Minimum surface every backend implements for this bench.

    The protocol is stricter than ``bench.backends.base.MemoryBackend``
    in two places:

    - ``ingest_session`` instead of ``observe(text, ts)``: we want to
      pass ``role`` and ``session_id`` per turn so backends that key
      on conversational structure see what they need.
    - ``recall_text`` alongside ``recall``: the harness composes the
      generator prompt from text content, not opaque ids.
    """

    backend_id: str

    async def ingest_session(
        self,
        *,
        session_id: str,
        ts: float,
        turns: list[dict[str, str]],
    ) -> None: ...

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]: ...

    async def recall_text(self, query: str, k: int = 10) -> list[str]: ...

    async def close(self) -> None: ...


# ─── result dataclasses ────────────────────────────────────────────


@dataclass
class QuestionResult:
    """Per-question outcome row, written verbatim to the JSON artifact."""

    question_id: str
    question_type: str
    n_sessions: int
    n_turns: int
    n_recalled: int
    ingest_seconds: float
    recall_seconds: float
    generate_seconds: float
    judge_seconds: float
    correct: bool
    judge_raw: str
    answer_pred: str
    answer_gold: str


@dataclass
class RunSummary:
    """Aggregate JSON payload."""

    backend: str
    dataset: str
    k: int
    gen_model: str
    judge_model: str
    n_questions: int
    overall_accuracy: float
    accuracy_by_type: dict[str, float]
    counts_by_type: dict[str, int]
    duration_seconds: float
    timestamp: str
    per_question: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dataset": self.dataset,
            "k": self.k,
            "gen_model": self.gen_model,
            "judge_model": self.judge_model,
            "n_questions": self.n_questions,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "accuracy_by_type": {k: round(v, 4) for k, v in self.accuracy_by_type.items()},
            "counts_by_type": self.counts_by_type,
            "duration_seconds": round(self.duration_seconds, 2),
            "timestamp": self.timestamp,
            "per_question": [r.__dict__ for r in self.per_question],
        }


# ─── LLM judge / generator ─────────────────────────────────────────


@runtime_checkable
class _LLM(Protocol):
    """Sync LLM interface — bench is I/O-bound on memory ingest, the
    generator + judge are small enough to call sync inside an executor."""

    async def complete(self, prompt: str, *, max_tokens: int) -> str: ...


class OpenAIChatLLM:
    """Wraps ``openai.OpenAI`` for short structured completions.

    We use the chat completions endpoint with temperature=0 for both
    generator and judge — this bench is reporting a number, not
    showcasing creative answers.

    Parameters
    ----------
    model:
        Chat-completions model name (``gpt-4o-mini``, ``deepseek-chat``,
        etc.).
    api_key:
        API key. ``None`` → falls back to the SDK's env-var default
        (``OPENAI_API_KEY``).
    base_url:
        Override the API endpoint. ``None`` → SDK default
        (``https://api.openai.com/v1``). Pass
        ``https://api.deepseek.com`` to route generator + judge calls
        through DeepSeek while keeping the OpenAI SDK on the embedder
        side untouched. This is the path the LongMemEval-S CLI uses
        when ``--llm-base-url`` is set.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # lazy import — bench[openai] extra

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model

    async def complete(self, prompt: str, *, max_tokens: int) -> str:
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        return await asyncio.get_running_loop().run_in_executor(None, _call)


class MockLLM:
    """Test stub — returns canned strings. Used by ``--smoke`` and tests.

    The judge prompt always begins with the same prefix, so we can
    parse the role from the start of the prompt and respond with
    ``correct`` for *every* judge call (so the smoke run produces a
    well-formed accuracy number without an OpenAI key) and the
    question text echoed back as the generator answer.
    """

    def __init__(self, *, judge_says: str = "correct", gen_returns: str = "") -> None:
        self._judge_says = judge_says
        self._gen_returns = gen_returns

    async def complete(self, prompt: str, *, max_tokens: int) -> str:
        if prompt.startswith("[JUDGE]"):
            return self._judge_says
        if self._gen_returns:
            return self._gen_returns
        # Echo the last line as a fallback "answer". Good enough that
        # the judge has *something* to score in tests.
        last = prompt.strip().splitlines()[-1].strip()
        return last[:200]


# ─── prompts ───────────────────────────────────────────────────────

GENERATOR_PROMPT = """You are answering a question using only the memory snippets retrieved \
for you. The snippets are excerpts of past conversations. Each snippet is \
prefixed with [YYYY-MM-DD type] indicating when the underlying conversation \
happened (date) and what kind of memory it is (episode = a raw turn, fact = a \
distilled atomic fact, summary = a Dream-consolidated summary, pattern = a \
recurring-behaviour observation).

Use the dates to:
- Order events chronologically when the question asks about sequence ("first \
X then Y") or time elapsed ("how many days between A and B").
- Resolve conflicts: when two snippets assert different values for the same \
fact (e.g. "$350K" vs "$400K"), prefer the more recent one as the current \
truth.
- Anchor calendar arithmetic: dates in snippets are real dates you can \
subtract.

Trust 'fact' and 'summary' snippets when they conflict with raw 'episode' \
snippets — they're already distilled. Answer concisely and factually using \
only what the snippets support. If the snippets don't contain the answer, \
say "I don't know."

Memory snippets:
{snippets}

Question: {question}

Answer:"""


JUDGE_PROMPT = """[JUDGE] You are evaluating whether a model answer matches a gold reference \
answer for a question about a long conversation. Respond with exactly one word: \
"correct" if the model answer conveys the same factual content as the gold answer \
(paraphrasing is fine, partial-but-correct is correct), otherwise "incorrect".

Question: {question}
Gold answer: {gold}
Model answer: {pred}

Verdict:"""


def _format_snippets(snippets: list[str], max_chars: int = 8000) -> str:
    """Join recalled memory text into a numbered list, truncate at ``max_chars``.

    Truncation keeps the generator prompt from blowing through the
    model's context window when ``--k`` is generous and individual
    memories are long. The truncation point is a budget, not a quality
    signal — it triggers identically across all three backends so the
    comparison stays fair.
    """

    if not snippets:
        return "(no memories surfaced)"
    lines: list[str] = []
    used = 0
    for i, s in enumerate(snippets, start=1):
        line = f"[{i}] {s}"
        if used + len(line) > max_chars and lines:
            lines.append("[...truncated]")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


# ─── single-question evaluation ────────────────────────────────────


async def _eval_one(
    *,
    question: Question,
    backend: LongMemEvalBackend,
    generator: _LLM,
    judge: _LLM,
    k: int,
    snippet_max_chars: int,
    dream_after_ingest: bool,
) -> QuestionResult:
    """Run the four-stage protocol against one question."""

    # Phase 1: ingest.
    t0 = time.perf_counter()
    for sess in question.sessions:
        await backend.ingest_session(
            session_id=sess.session_id,
            ts=sess.ts,
            turns=[{"role": t.role, "content": t.content} for t in sess.turns],
        )
    ingest_s = time.perf_counter() - t0

    # Phase 1b: optional dream pass (Mnemoss-specific). LongMemEval-S
    # multi-session and temporal-reasoning questions need the relations
    # graph that Dream's P4 phase populates for spreading activation
    # to engage. Backends without ``dream`` are silently skipped.
    if dream_after_ingest and hasattr(backend, "dream"):
        try:
            await backend.dream(trigger="session_end")
        except Exception as exc:  # noqa: BLE001
            # Cost cap or transient LLM failure shouldn't kill the
            # whole question — recall still works on direct embedding
            # match without the relations graph.
            print(f"  !! dream skipped: {exc!r}", flush=True)

    # Phase 2: recall.
    t0 = time.perf_counter()
    snippets = await backend.recall_text(question.question, k=k)
    recall_s = time.perf_counter() - t0

    # Phase 3: generate.
    gen_prompt = GENERATOR_PROMPT.format(
        snippets=_format_snippets(snippets, max_chars=snippet_max_chars),
        question=question.question,
    )
    t0 = time.perf_counter()
    answer_pred = await generator.complete(gen_prompt, max_tokens=256)
    gen_s = time.perf_counter() - t0

    # Phase 4: judge.
    judge_prompt = JUDGE_PROMPT.format(
        question=question.question,
        gold=question.answer,
        pred=answer_pred,
    )
    t0 = time.perf_counter()
    verdict_raw = await judge.complete(judge_prompt, max_tokens=8)
    judge_s = time.perf_counter() - t0

    # Tolerant parse — judges sometimes wrap with quotes or add
    # punctuation. ``startswith("correct")`` after stripping both
    # cases handles ``correct``, ``Correct.``, ``"correct"``, etc.
    verdict_clean = verdict_raw.strip().strip('"').strip("'").lower()
    correct = verdict_clean.startswith("correct") and not verdict_clean.startswith("incorrect")

    return QuestionResult(
        question_id=question.question_id,
        question_type=question.question_type,
        n_sessions=len(question.sessions),
        n_turns=sum(len(s.turns) for s in question.sessions),
        n_recalled=len(snippets),
        ingest_seconds=round(ingest_s, 3),
        recall_seconds=round(recall_s, 3),
        generate_seconds=round(gen_s, 3),
        judge_seconds=round(judge_s, 3),
        correct=correct,
        judge_raw=verdict_raw,
        answer_pred=answer_pred,
        answer_gold=question.answer,
    )


# ─── backend factory ───────────────────────────────────────────────


EMBEDDER_CHOICES = ("openai", "local", "nomic", "gemma", "fake")


def _embedder_max_chars(embedder: str) -> int | None:
    """Return the recommended ``EncoderParams.max_memory_chars`` for an embedder.

    OpenAI ``text-embedding-3-small`` errors with HTTP 400 once a
    single embed call exceeds 8191 tokens. The chars-per-token ratio
    varies: typical English prose is ~4, but dense text like Wikipedia
    pastes drops to ~3, so the safe cap that survives any input is
    ~8191 × 3 ≈ 24k chars — and we go below that to leave headroom
    for the BPE worst case. (LongMemEval-S contains Wikipedia copy-
    pastes inside user turns; the previous 30k cap blew past the
    token limit on those, so question 852ce960 in the stratified
    pilot crashed with 0-second ingest.)

    MiniLM and similar local models silently truncate past their max
    position embedding (~512 tokens, ~2000 chars), which degrades
    semantic recall invisibly. Both failure modes are bench-killers;
    auto-splitting here means the benchmark measures memory
    architecture, not "did Mnemoss happen to emit a chunk that
    overflowed your embedder of choice."

    Returns ``None`` for embedders without a published cap — defaults
    apply (no split).
    """

    return {
        "openai": 20000,
        "local": 2000,
        "nomic": 2000,
        "gemma": 2000,
        "fake": None,
    }.get(embedder)


def _build_backend(
    backend_name: str,
    *,
    smoke: bool,
    embedder: str,
    question_index: int,
    llm_base_url: str | None = None,
    llm_api_key: str | None = None,
    dream_model: str | None = None,
    dream_max_calls: int | None = None,
    auto_expand: bool = False,
    max_memory_chars: int | None = None,
    process_singletons: bool = False,
) -> LongMemEvalBackend:
    """Build a fresh backend per question. Workspace ids include the
    question index so failed runs don't trip up follow-on questions
    via leftover state.

    ``--smoke`` forces ``embedder="fake"`` upstream — the smoke path is
    the no-network smoke path and that's incompatible with ``openai``
    or any local model that wants to download weights.

    ``llm_*`` and ``dream_*`` are Mnemoss-only. They configure the
    LLM client used by ``mem.dream()`` for the consolidate phase
    (one LLM call per cluster) and cap how many calls a single
    dream may make, so a runaway 500-turn haystack doesn't blow
    through your bill.
    """

    if backend_name == "mnemoss":
        from bench.backends.mnemoss_backend import MnemossBackend
        from bench.launch_comparison import _resolve_embedder
        from mnemoss import DreamerParams, EncoderParams
        from mnemoss.dream.cost import CostLimits
        from mnemoss.llm.client import OpenAIClient

        effective = "fake" if smoke else embedder

        encoder_kwargs: dict[str, Any] = {}
        # Caller override wins over the per-embedder safety cap so the
        # bench can sweep ``--max-memory-chars`` without having to edit
        # ``_embedder_max_chars``. The safety cap still applies as a
        # floor when the caller doesn't override.
        max_chars = max_memory_chars if max_memory_chars is not None else _embedder_max_chars(effective)
        if max_chars is not None:
            encoder_kwargs["max_memory_chars"] = max_chars
        encoder = EncoderParams(**encoder_kwargs) if encoder_kwargs else None

        dreamer = DreamerParams(process_singletons=process_singletons) if process_singletons else None

        llm_client = None
        cost_limits = None
        if dream_model and not smoke:
            llm_client = OpenAIClient(
                model=dream_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
            if dream_max_calls is not None:
                cost_limits = CostLimits(max_llm_calls_per_run=dream_max_calls)

        return MnemossBackend(
            embedding_model=_resolve_embedder(effective),
            workspace=f"lme-q{question_index}",
            encoder=encoder,
            llm_client=llm_client,
            cost_limits=cost_limits,
            expand_via_streak=auto_expand,
            dreamer=dreamer,
        )

    if backend_name == "mem0":
        import tempfile
        from pathlib import Path as _Path

        from bench.backends.mem0_backend import Mem0Backend

        # Pin mem0's LLM to ``gpt-4o-mini`` and its embedder to
        # ``text-embedding-3-small``. mem0's default LLM is
        # ``gpt-5-mini``, which rejects the ``max_tokens`` parameter
        # mem0's own client passes — every extraction call fails with
        # HTTP 400 out of the box. Pinning the model here keeps the
        # comparison apples-to-apples with Mnemoss (same embedder) and
        # avoids the upstream-default bug.
        #
        # Per-question scoping for mem0's local stores: the default
        # qdrant path is ``/tmp/qdrant`` and the default history db
        # is ``~/.mem0/history.db``. With one Mem0 instance per
        # question, both collide across questions — qdrant raises
        # "Storage folder already accessed by another instance" on
        # every question after the first. Each question gets its own
        # tempdir for both stores, cleaned up on backend.close().
        mem0_dir = _Path(tempfile.mkdtemp(prefix=f"mem0_lme_q{question_index}_"))
        mem0_config: dict[str, Any] = {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"lme_q{question_index}",
                    "path": str(mem0_dir / "qdrant"),
                    "embedding_model_dims": 1536,
                },
            },
            "history_db_path": str(mem0_dir / "history.db"),
        }
        return Mem0Backend(
            user_id=f"lme-q{question_index}",
            config=mem0_config,
            cleanup_dir=mem0_dir,
        )

    if backend_name == "memos":
        from bench.backends.memos_backend import MemOSBackend

        return MemOSBackend(user_id=f"lme-q{question_index}")

    raise ValueError(f"Unknown backend {backend_name!r}")


# ─── main loop ─────────────────────────────────────────────────────


async def _main(args: argparse.Namespace) -> RunSummary:
    questions = load_longmemeval_s(args.dataset)

    # Optional filtering.
    if args.question_types:
        wanted = set(args.question_types)
        questions = [q for q in questions if q.question_type in wanted]
    if args.stratify is not None:
        # Take the first ``stratify`` questions of each canonical type
        # in the order they appear in the dataset. Deterministic, no
        # RNG seed needed. Types that don't appear in the corpus
        # silently contribute zero rows.
        by_type: dict[str, list[Question]] = {}
        for q in questions:
            by_type.setdefault(q.question_type, []).append(q)
        stratified: list[Question] = []
        for t in QUESTION_TYPES:
            stratified.extend(by_type.get(t, [])[: args.stratify])
        questions = stratified
    if args.limit is not None and args.limit < len(questions):
        questions = questions[: args.limit]

    llm_key: str | None = None
    if args.smoke:
        generator: _LLM = MockLLM(gen_returns="smoke")
        judge: _LLM = MockLLM(judge_says="correct")
    else:
        # Resolve the LLM key. ``--llm-api-key-env`` lets you point at
        # a non-default env var (``DEEPSEEK_API_KEY``) so the embedder
        # side can keep ``OPENAI_API_KEY`` for the real OpenAI endpoint.
        llm_key = os.environ.get(args.llm_api_key_env)
        if not llm_key:
            print(
                f"error: {args.llm_api_key_env} is not set. Use --smoke to "
                "run without an API key, or pass --llm-api-key-env <NAME> "
                "to point at a different env var.",
                file=sys.stderr,
            )
            sys.exit(2)
        # ``base_url=None`` falls back to the SDK default (real OpenAI).
        # Provider-compatible APIs (DeepSeek, Together, etc.) override
        # via ``--llm-base-url``.
        base_url = args.llm_base_url or None
        generator = OpenAIChatLLM(
            model=args.gen_model, api_key=llm_key, base_url=base_url
        )
        judge = OpenAIChatLLM(
            model=args.judge_model, api_key=llm_key, base_url=base_url
        )

    started = time.perf_counter()
    started_iso = datetime.now(timezone.utc).isoformat()

    results: list[QuestionResult] = []
    correct_by_type: dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    count_by_type: dict[str, int] = {t: 0 for t in QUESTION_TYPES}

    for i, q in enumerate(questions):
        print(
            f"[{i + 1}/{len(questions)}] {q.question_id}  "
            f"type={q.question_type}  sessions={len(q.sessions)}  "
            f"turns={sum(len(s.turns) for s in q.sessions)}",
            flush=True,
        )
        backend = _build_backend(
            args.backend,
            smoke=args.smoke,
            embedder=args.embedder,
            question_index=i,
            llm_base_url=(args.llm_base_url or None) if not args.smoke else None,
            llm_api_key=llm_key if not args.smoke else None,
            dream_model=args.dream_model if args.dream_after_ingest else None,
            dream_max_calls=args.dream_max_calls,
            auto_expand=args.auto_expand,
            max_memory_chars=args.max_memory_chars,
            process_singletons=args.process_singletons,
        )
        try:
            r = await _eval_one(
                question=q,
                backend=backend,
                generator=generator,
                judge=judge,
                k=args.k,
                snippet_max_chars=args.snippet_max_chars,
                dream_after_ingest=args.dream_after_ingest,
            )
        except Exception as exc:  # noqa: BLE001
            # Per-question crashes shouldn't take down the whole run —
            # 500-question benches are too expensive to start over for
            # a single transient embedder hiccup. Record as incorrect
            # and keep going.
            print(f"  !! exception: {exc!r}", flush=True)
            r = QuestionResult(
                question_id=q.question_id,
                question_type=q.question_type,
                n_sessions=len(q.sessions),
                n_turns=sum(len(s.turns) for s in q.sessions),
                n_recalled=0,
                ingest_seconds=0.0,
                recall_seconds=0.0,
                generate_seconds=0.0,
                judge_seconds=0.0,
                correct=False,
                judge_raw=f"<exception: {type(exc).__name__}>",
                answer_pred="",
                answer_gold=q.answer,
            )
        finally:
            await backend.close()

        results.append(r)
        # Bucket per-type. Question types outside the canonical 6
        # (older forks add subtypes) bucket under "other".
        bucket = q.question_type if q.question_type in count_by_type else "other"
        count_by_type[bucket] = count_by_type.get(bucket, 0) + 1
        if r.correct:
            correct_by_type[bucket] = correct_by_type.get(bucket, 0) + 1

        verdict = "✓" if r.correct else "✗"
        print(
            f"  {verdict}  ingest={r.ingest_seconds}s  recall={r.recall_seconds}s  "
            f"gen={r.generate_seconds}s  judge={r.judge_seconds}s  "
            f"k_returned={r.n_recalled}",
            flush=True,
        )

    duration = time.perf_counter() - started

    accuracy_by_type = {
        t: (correct_by_type[t] / count_by_type[t]) if count_by_type[t] else 0.0
        for t in count_by_type
    }
    overall_correct = sum(1 for r in results if r.correct)
    overall_accuracy = overall_correct / len(results) if results else 0.0

    return RunSummary(
        backend=args.backend,
        dataset=str(args.dataset),
        k=args.k,
        gen_model=args.gen_model,
        judge_model=args.judge_model,
        n_questions=len(results),
        overall_accuracy=overall_accuracy,
        accuracy_by_type=accuracy_by_type,
        counts_by_type=count_by_type,
        duration_seconds=duration,
        timestamp=started_iso,
        per_question=results,
    )


# ─── CLI ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["mnemoss", "mem0", "memos"],
        required=True,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=CANONICAL_PATH,
        help=f"Path to longmemeval_s.json. Default: {CANONICAL_PATH}",
    )
    p.add_argument("--k", type=int, default=10, help="Top-K memories to recall.")
    p.add_argument(
        "--gen-model",
        default="gpt-4o-mini",
        help="OpenAI model used to generate the final answer from recalled memories.",
    )
    p.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI model used to score answer vs gold.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions (after --stratify, if set). Default: all.",
    )
    p.add_argument(
        "--stratify",
        type=int,
        default=None,
        help="Take the first N questions of each of the 6 canonical "
        "question types. ``--stratify 4`` → ~24 questions covering all "
        "categories. Useful for small pilots that still exercise the "
        "knowledge-update + temporal-reasoning slices Mnemoss is "
        "architecturally tuned for.",
    )
    p.add_argument(
        "--question-types",
        nargs="+",
        choices=list(QUESTION_TYPES),
        default=None,
        help="Restrict to a subset of question types. Default: all 6.",
    )
    p.add_argument(
        "--snippet-max-chars",
        type=int,
        default=8000,
        help="Cap total recalled-snippet length in the generator prompt. "
        "Triggers identically across backends so the comparison stays fair.",
    )
    p.add_argument(
        "--embedder",
        choices=list(EMBEDDER_CHOICES),
        default="openai",
        help="Mnemoss-only: which embedder to pin. ``openai`` matches the "
        "published-chart config; ``local`` runs MiniLM-multilingual on CPU "
        "with no API calls; ``nomic`` / ``gemma`` are stronger local options. "
        "Mem0 / MemOS ignore this and pick their own (they ship their own "
        "config layer).",
    )
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="Override the OpenAI SDK base URL for the generator + judge "
        "calls. Use ``https://api.deepseek.com`` to route LLM calls through "
        "DeepSeek while leaving the embedder pointed at real OpenAI. "
        "Default: SDK default (real OpenAI).",
    )
    p.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="Env var name to read for the generator + judge API key. "
        "Set to ``DEEPSEEK_API_KEY`` when ``--llm-base-url`` points at "
        "DeepSeek so the embedder can still pick up ``OPENAI_API_KEY`` "
        "for the real OpenAI endpoint. Default: ``OPENAI_API_KEY``.",
    )
    p.add_argument(
        "--dream-after-ingest",
        action="store_true",
        help="Mnemoss-only: after ingesting all sessions, call "
        "``mem.dream(trigger='session_end')`` so the relations graph "
        "(P4 of the Dream pipeline) is populated before recall. Multi-"
        "session and temporal-reasoning questions need this for "
        "spreading activation to engage. Costs N LLM calls per question "
        "where N ≈ number of consolidated clusters.",
    )
    p.add_argument(
        "--dream-model",
        default="deepseek-chat",
        help="Chat model used by the Dream consolidate phase. Re-uses "
        "``--llm-base-url`` and ``--llm-api-key-env`` so the same key "
        "covers generator + judge + dream. Default: ``deepseek-chat``.",
    )
    p.add_argument(
        "--dream-max-calls",
        type=int,
        default=30,
        help="Cap on LLM calls per question's dream pass via "
        "``CostLimits.max_llm_calls_per_run``. Bounds the per-question "
        "dream cost. Default: 30.",
    )
    p.add_argument(
        "--auto-expand",
        action="store_true",
        help="Mnemoss-only: engage spreading-activation expansion at "
        "recall time. The bench backend issues a primer recall first so "
        "the second call sees a same-topic streak and the BFS over the "
        "relation graph fires (Mnemoss's auto_expand requires a streak "
        "and bench questions land on empty history). Pair with "
        "``--dream-after-ingest`` so Consolidate's derived_from / "
        "derived_to edges are in place — those are the only cross-"
        "session links Mnemoss creates.",
    )
    p.add_argument(
        "--max-memory-chars",
        type=int,
        default=None,
        help="Mnemoss-only: override ``EncoderParams.max_memory_chars`` "
        "for this run. Defaults to the per-embedder safety cap (20000 "
        "for OpenAI). Lower values force fact-grained chunking — a "
        "multi-paragraph turn splits into many shorter Memory rows at "
        "sentence boundaries. Trades ingest time + memory count for "
        "tighter cosine-similarity grain (helps supersession catch sub-"
        "turn fact updates, helps recall surface specific facts that "
        "are otherwise diluted by surrounding context).",
    )
    p.add_argument(
        "--process-singletons",
        action="store_true",
        help="Mnemoss-only: enable Dream Consolidate's per-memory atomic-"
        "fact extraction pass over HDBSCAN-noise singletons. Closes the "
        "gap where uniquely-named entities (e.g. LongMemEval-S 51a45a95 "
        "Target store) live in turns with no cluster neighbours and "
        "would otherwise never reach the LLM. Adds N_singleton LLM "
        "calls per dream run; bound via ``--dream-max-calls``.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use FakeEmbedder + mock LLM. Runs offline; exercises the harness wiring only.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    summary = asyncio.run(_main(args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary.to_dict(), indent=2) + "\n")

    print()
    print(f"=== {summary.backend} on {Path(summary.dataset).name} ===")
    print(f"  questions      : {summary.n_questions}")
    print(f"  overall acc    : {summary.overall_accuracy:.4f}")
    print(f"  duration       : {summary.duration_seconds:.1f}s")
    print()
    print("  accuracy by question type:")
    for t in sorted(summary.accuracy_by_type):
        n = summary.counts_by_type.get(t, 0)
        if n == 0:
            continue
        print(f"    {t:30s} {summary.accuracy_by_type[t]:.4f}  (n={n})")
    print()
    # Latency rollup for quick eyeballing — full per-question detail is in
    # the JSON artifact.
    if summary.per_question:
        ingest = [r.ingest_seconds for r in summary.per_question]
        recall = [r.recall_seconds for r in summary.per_question]
        print("  latency rollup (s):")
        print(
            f"    ingest mean={statistics.mean(ingest):.2f}  "
            f"max={max(ingest):.2f}"
        )
        print(
            f"    recall mean={statistics.mean(recall):.3f}  "
            f"max={max(recall):.3f}"
        )
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
