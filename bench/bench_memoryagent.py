"""MemoryAgentBench evaluator — Mnemoss vs raw_stack on a standard benchmark.

``ai-hyz/MemoryAgentBench`` (https://arxiv.org/abs/2507.05257) is a unified
benchmark for LLM-agent memory systems covering four competencies:

- **Accurate Retrieval (AR)** — 22 rows of paraphrased QA over long context
- **Test-Time Learning (TTL)** — 6 rows
- **Long-Range Understanding (LRU)** — 110 rows
- **Conflict Resolution (CR)** — 8 rows (closest match to Mnemoss's
  supersession story)

We focus on ``Conflict_Resolution`` first because it directly tests
the failure mode Mnemoss was built to fix: the context contains a
numbered list of 450+ facts; some facts appear MULTIPLE TIMES with
conflicting values (e.g. "Alta IF plays association football"
followed later by "Alta IF plays baseball"). A correct memory
system answers with the LATEST version, not any earlier one. Pure
cosine can't tell them apart — it'll return whichever fact has
highest cosine to the query.

Evaluation
----------

Per row:
1. Parse the context into a list of numbered facts (one per line).
2. Ingest each fact as a separate memory, in document order
   (preserves "later = newer" semantics via observe-time clock).
3. For each question, retrieve top-k memories.
4. Score: "answer correct" = any gold answer string appears in any
   retrieved memory's text (case-insensitive substring match).

Metric: accuracy = correct / total across all (row, question) pairs.

This is a retrieval-quality proxy — the original paper grades via
an LLM reading retrieved context. Our contains-check is weaker but
sufficient to measure architectural differences between backends.
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


def _parse_facts(context: str) -> list[str]:
    """Split context into its numbered fact lines."""

    facts = []
    for line in context.split("\n"):
        m = _FACT_RE.match(line)
        if m:
            facts.append(m.group(1))
    return facts


def _any_gold_in_retrieved(gold_list: list[str], retrieved_texts: list[str]) -> bool:
    """Case-insensitive substring containment check across retrieved texts."""

    if not gold_list or not retrieved_texts:
        return False
    joined = "  ".join(t.lower() for t in retrieved_texts)
    return any(gold and gold.lower() in joined for gold in gold_list)


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

    correct = 0
    total = 0
    latencies_ms: list[float] = []
    for q, gold in zip(questions, answers, strict=False):
        gold_list = gold if isinstance(gold, list) else [gold]
        t0 = time.perf_counter()
        hits = await backend.recall(q, k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        retrieved_texts = [mid_to_text.get(h.memory_id, "") for h in hits]
        if _any_gold_in_retrieved(gold_list, retrieved_texts):
            correct += 1
        total += 1

    return {
        "row_idx": row_idx,
        "n_facts": len(facts),
        "n_questions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "ingest_seconds": round(ingest_seconds, 2),
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0,
        "latency_ms_p95": (
            round(sorted(latencies_ms)[int(0.95 * len(latencies_ms))], 2) if latencies_ms else 0
        ),
    }


async def _run_split(
    *,
    backend_name: str,
    embedder_choice: str,
    split: str,
    max_rows: int | None,
    k: int,
) -> dict:
    """Evaluate one split of MemoryAgentBench with a fresh backend per row
    (so one row's ingest doesn't contaminate the next)."""

    from datasets import load_dataset

    ds = load_dataset("ai-hyz/MemoryAgentBench")
    if split not in ds:
        raise ValueError(f"unknown split {split!r}; choices: {list(ds.keys())}")
    rows = ds[split]
    if max_rows is not None:
        rows = rows.select(range(min(max_rows, len(rows))))
    print(
        f"[{backend_name}] split={split}  rows={len(rows)}  embedder={embedder_choice}",
        flush=True,
    )

    embedder = _resolve_embedder(embedder_choice)
    per_row: list[dict] = []
    total_correct = 0
    total_questions = 0

    for i, row in enumerate(rows):
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
        per_row.append(result)
        total_correct += result["correct"]
        total_questions += result["n_questions"]
        print(
            f"  row {i + 1}/{len(rows)}: {result['n_facts']} facts, "
            f"{result['correct']}/{result['n_questions']} correct "
            f"({result['accuracy']:.1%}), ingest={result['ingest_seconds']}s, "
            f"p95={result['latency_ms_p95']}ms",
            flush=True,
        )

    return {
        "backend": backend_name,
        "embedder": embedder_choice,
        "split": split,
        "n_rows": len(rows),
        "total_correct": total_correct,
        "total_questions": total_questions,
        "overall_accuracy": total_correct / total_questions if total_questions else 0.0,
        "per_row": per_row,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["raw_stack", "mnemoss_default"], required=True)
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
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap number of rows to evaluate (default: all rows in split).",
    )
    p.add_argument("--k", type=int, default=5)
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
        )
    )
    result["timestamp"] = started.isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print()
    print(f"=== {args.backend} on MemoryAgentBench/{args.split} (k={args.k}) ===")
    print(
        f"  Overall accuracy: {result['total_correct']}/{result['total_questions']}"
        f" = {result['overall_accuracy']:.1%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
