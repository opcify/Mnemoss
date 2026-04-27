"""Simulated-LLM evaluator for MemoryAgentBench / Conflict_Resolution.

Companion to ``bench/bench_memoryagent.py``. The retrieval-only bench
graded with ``any gold ∈ top-k retrieved`` (a contains-check on the
whole retrieved set). That measures recall but not "did the system
put the right fact at the position an LLM agent would actually cite."

This bench grades by **simulating an LLM** rather than calling one.
The simulated LLM is the simplest possible model of "an agent that
trusts the memory system": it cites the **top-1 retrieved fact** as
its answer. We grade by case-insensitive substring of any gold
string in that top-1 text.

Why a simulation?
-----------------
1. **Deterministic.** Real LLM calls add noise (sampling, model drift,
   provider load). A simulated LLM produces the exact same numbers
   on every run, so backend-vs-backend deltas are pure retrieval.
2. **Free.** Free-tier reasoning models (e.g. Tencent HY3) burn
   1500-2000 tokens deliberating per call; 200 questions × 2 backends
   on free-tier ≈ 3-4 hours total wall time and hits OpenRouter's
   20 req/min throttle. The simulation runs in seconds.
3. **Fair.** Both backends face the exact same simulated grader.
   Any differential is purely the retrieval architecture's contribution.
4. **Honest about what we're measuring.** A real LLM might dig past
   top-1 if facts conflict. The top-1 simulation answers the sharper
   question: "How often does retrieval put the right fact at #1?" —
   the position a citation-trusting agent would actually use.

We also report the original ``any gold ∈ top-k`` accuracy for
comparison, so callers can see both: did we surface it at all (top-k)
vs did we surface it at the place that matters (top-1).

Usage::

    python -m bench.bench_memoryagent_llm \\
        --backend raw_stack \\
        --embedder openai \\
        --split Conflict_Resolution \\
        --k 5 \\
        --out bench/results/memoryagent_cr_sim_raw_stack.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.raw_stack_backend import RawStackBackend
from bench.launch_comparison import _resolve_embedder

_FACT_RE = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")
_LLM_SIM_NAME = "top1-simulated"


def _parse_facts(context: str) -> list[str]:
    out = []
    for line in context.split("\n"):
        m = _FACT_RE.match(line)
        if m:
            out.append(m.group(1))
    return out


def _simulated_answer(retrieved_texts: list[str]) -> str:
    """Simulate a citation-trusting LLM: emit the top-1 retrieved fact.

    A real LLM would paraphrase, but for substring grading the raw fact
    text is the strongest signal. If retrieval returns nothing, the
    simulated agent says it doesn't know.
    """

    if not retrieved_texts:
        return "unknown"
    top1 = retrieved_texts[0].strip()
    return top1 if top1 else "unknown"


def _grade(answer: str, gold_list: list[str]) -> bool:
    """Case-insensitive substring grading."""

    if not answer or not gold_list:
        return False
    answer_lower = answer.lower()
    return any(g and g.lower() in answer_lower for g in gold_list if isinstance(g, str))


def _grade_topk(retrieved_texts: list[str], gold_list: list[str]) -> bool:
    """Did any gold string appear anywhere in the top-k retrieved set?

    Equivalent to ``bench_memoryagent.py``'s grading. Reported alongside
    the top-1 simulation so callers see both: surfaced at all vs
    surfaced at the right rank.
    """

    if not retrieved_texts or not gold_list:
        return False
    joined = "  ".join(t.lower() for t in retrieved_texts)
    return any(g and g.lower() in joined for g in gold_list if isinstance(g, str))


async def _run_row(
    *,
    backend,
    backend_name: str,
    row_idx: int,
    row: dict,
    k: int,
) -> dict:
    facts = _parse_facts(row["context"])
    questions = row["questions"]
    answers = row["answers"]

    mid_to_text: dict[str, str] = {}
    ingest_t0 = time.perf_counter()
    for fact in facts:
        mid = await backend.observe(fact, ts=time.time())
        if mid is not None:
            mid_to_text[mid] = fact
    ingest_seconds = time.perf_counter() - ingest_t0

    correct_top1 = 0
    correct_topk = 0
    total = 0
    retrieval_latencies: list[float] = []

    for q, gold in zip(questions, answers, strict=False):
        gold_list = gold if isinstance(gold, list) else [gold]

        t0 = time.perf_counter()
        hits = await backend.recall(q, k=k)
        retrieval_latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_texts = [mid_to_text.get(h.memory_id, "") for h in hits]

        sim_answer = _simulated_answer(retrieved_texts)
        if _grade(sim_answer, gold_list):
            correct_top1 += 1
        if _grade_topk(retrieved_texts, gold_list):
            correct_topk += 1
        total += 1

    return {
        "row_idx": row_idx,
        "n_facts": len(facts),
        "n_questions": total,
        "correct_top1": correct_top1,
        "correct_topk": correct_topk,
        "accuracy_top1": correct_top1 / total if total else 0.0,
        "accuracy_topk": correct_topk / total if total else 0.0,
        "ingest_seconds": round(ingest_seconds, 2),
        "retrieval_p50_ms": (
            round(sorted(retrieval_latencies)[int(0.50 * len(retrieval_latencies))], 2)
            if retrieval_latencies
            else 0
        ),
        "retrieval_mean_ms": (
            round(statistics.mean(retrieval_latencies), 2) if retrieval_latencies else 0
        ),
    }


async def _run_split(
    *,
    backend_name: str,
    embedder_choice: str,
    split: str,
    max_rows: int | None,
    k: int,
    row_concurrency: int,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("ai-hyz/MemoryAgentBench")
    if split not in ds:
        raise ValueError(f"unknown split {split!r}")
    rows = ds[split]
    if max_rows is not None:
        rows = rows.select(range(min(max_rows, len(rows))))

    print(
        f"[{backend_name}] split={split} rows={len(rows)} k={k} "
        f"embedder={embedder_choice} llm={_LLM_SIM_NAME} "
        f"row_concurrency={row_concurrency}",
        flush=True,
    )

    embedder = _resolve_embedder(embedder_choice)
    sem = asyncio.Semaphore(row_concurrency)

    async def _run_one(i: int, row: dict) -> dict:
        async with sem:
            if backend_name == "raw_stack":
                backend = RawStackBackend(embedding_model=embedder)
            elif backend_name == "mnemoss_default":
                backend = MnemossBackend(embedding_model=embedder)
            else:
                raise ValueError(f"unknown backend {backend_name!r}")
            try:
                result = await _run_row(
                    backend=backend,
                    backend_name=backend_name,
                    row_idx=i,
                    row=row,
                    k=k,
                )
            finally:
                await backend.close()
            print(
                f"  row {i + 1}/{len(rows)}: {result['n_facts']} facts, "
                f"top1={result['correct_top1']}/{result['n_questions']} "
                f"({result['accuracy_top1']:.1%}), "
                f"top{k}={result['correct_topk']}/{result['n_questions']} "
                f"({result['accuracy_topk']:.1%}), "
                f"retr_p50={result['retrieval_p50_ms']}ms",
                flush=True,
            )
            return result

    per_row_unordered = await asyncio.gather(
        *[_run_one(i, row) for i, row in enumerate(rows)]
    )
    per_row = sorted(per_row_unordered, key=lambda r: r["row_idx"])
    total_top1 = sum(r["correct_top1"] for r in per_row)
    total_topk = sum(r["correct_topk"] for r in per_row)
    total_questions = sum(r["n_questions"] for r in per_row)

    return {
        "backend": backend_name,
        "embedder": embedder_choice,
        "llm": _LLM_SIM_NAME,
        "split": split,
        "k": k,
        "n_rows": len(rows),
        "total_top1": total_top1,
        "total_topk": total_topk,
        "total_questions": total_questions,
        "accuracy_top1": (
            total_top1 / total_questions if total_questions else 0.0
        ),
        "accuracy_topk": (
            total_topk / total_questions if total_questions else 0.0
        ),
        "per_row": per_row,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["raw_stack", "mnemoss_default"],
        required=True,
    )
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "gemma", "nomic", "fake"],
        default="openai",
    )
    p.add_argument(
        "--split",
        choices=[
            "Accurate_Retrieval",
            "Test_Time_Learning",
            "Long_Range_Understanding",
            "Conflict_Resolution",
        ],
        default="Conflict_Resolution",
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--row-concurrency",
        type=int,
        default=4,
        help=(
            "Number of rows to process in parallel. Each row gets its own "
            "fresh backend, so they're independent. Bounded to avoid "
            "hammering the embedder API."
        ),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    result = asyncio.run(
        _run_split(
            backend_name=args.backend,
            embedder_choice=args.embedder,
            split=args.split,
            max_rows=args.max_rows,
            k=args.k,
            row_concurrency=args.row_concurrency,
        )
    )
    result["timestamp"] = started.isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print()
    print(
        f"=== {args.backend} on {args.split} (sim={_LLM_SIM_NAME}, k={args.k}) ==="
    )
    print(
        f"  top-1 (cited fact):  "
        f"{result['total_top1']}/{result['total_questions']}"
        f" = {result['accuracy_top1']:.2%}"
    )
    print(
        f"  top-{args.k} (any rank): "
        f"{result['total_topk']}/{result['total_questions']}"
        f" = {result['accuracy_topk']:.2%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
