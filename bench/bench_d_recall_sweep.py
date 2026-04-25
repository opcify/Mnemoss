"""Sweep ``d_recall`` to find the supersession/LoCoMo Pareto knee.

``d_recall=0.1`` (ship default) → 52-56% new@1, ~0.68 LoCoMo recall at N=5K.
``d_recall=0.5`` → 72-80% new@1, ~0.25 LoCoMo recall.
Points in between are unmeasured. This script sweeps five values and
reports both metrics side-by-side so the user can pick a tradeoff
point with their eyes open.

Uses OpenAI embeddings for LoCoMo (our stable recall baseline) and
OpenAI for supersession (already measured; one of three embedders
the supersession bench cross-checked).
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from bench.bench_stale_fact import PAIRS_PATH, _rank_of
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)
from mnemoss import FormulaParams


async def _locomo_recall(embedder, d_recall: float, scale_n: int) -> float:
    """LoCoMo recall@10 at a given d_recall, on conv-26 at ``scale_n``."""

    mems = _load_jsonl(MEMORIES_PATH)
    qs = _load_jsonl(QUERIES_PATH)
    padded, gold = _build_scale_corpus(mems, qs, gold_conversation_id="conv-26", scale_n=scale_n)

    backend = MnemossBackend(
        embedding_model=embedder,
        formula=FormulaParams(d_recall=d_recall),
    )
    try:
        dia_to_id: dict[str, str] = {}
        for m in padded:
            dia_to_id[m["dia_id"]] = await backend.observe(m["text"], ts=m["ts"])

        scored, hit_sum = 0, 0.0
        for q in gold:
            gold_ids = {dia_to_id[d] for d in q["relevant_dia_ids"] if d in dia_to_id}
            if not gold_ids:
                continue
            hits = await backend.recall(q["question"], k=10)
            ret_ids = {h.memory_id for h in hits}
            hit_sum += len(ret_ids & gold_ids) / len(gold_ids)
            scored += 1
        return hit_sum / max(scored, 1)
    finally:
        await backend.close()


async def _supersession(embedder, d_recall: float, n_distractors: int, sleep_s: float) -> dict:
    """Supersession: returns rates dict."""

    pairs = [json.loads(line) for line in PAIRS_PATH.open()]
    backend = MnemossBackend(
        embedding_model=embedder,
        formula=FormulaParams(d_recall=d_recall),
    )
    try:
        # Distractors
        for m in _load_jsonl(MEMORIES_PATH)[:n_distractors]:
            await backend.observe(m["text"], ts=time.time())
        # Old batch
        old_ids: dict[int, str] = {}
        for i, p in enumerate(pairs):
            old_ids[i] = await backend.observe(p["old"], ts=time.time())
        # Gap
        await asyncio.sleep(sleep_s)
        # New batch
        new_ids: dict[int, str] = {}
        for i, p in enumerate(pairs):
            new_ids[i] = await backend.observe(p["new"], ts=time.time())

        new_top1 = 0
        old_top1 = 0
        super_count = 0
        both_count = 0
        for i, p in enumerate(pairs):
            hits = await backend.recall(p["question"], k=10)
            old_r = _rank_of(hits, old_ids[i])
            new_r = _rank_of(hits, new_ids[i])
            if new_r == 0:
                new_top1 += 1
            elif old_r == 0:
                old_top1 += 1
            if new_r is not None and old_r is not None:
                both_count += 1
                if new_r < old_r:
                    super_count += 1
        n = len(pairs)
        return {
            "new_top1": new_top1 / n,
            "old_top1": old_top1 / n,
            "supersession_when_both": super_count / max(both_count, 1),
            "both_recalled": both_count / n,
        }
    finally:
        await backend.close()


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--d-recall-values",
        nargs="+",
        type=float,
        default=[0.1, 0.15, 0.2, 0.3, 0.5],
    )
    p.add_argument("--embedder", choices=["openai", "local", "gemma"], default="openai")
    p.add_argument("--locomo-n", type=int, default=5000)
    p.add_argument("--supersession-distractors", type=int, default=300)
    p.add_argument("--supersession-sleep", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    embedder = _resolve_embedder(args.embedder)
    results = []
    for d in args.d_recall_values:
        print(f"\n[sweep] d_recall={d}  embedder={args.embedder}", flush=True)
        print("  running LoCoMo…", flush=True)
        t0 = time.perf_counter()
        lrecall = await _locomo_recall(embedder, d, args.locomo_n)
        print(
            f"    LoCoMo recall@10 = {lrecall:.4f}  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

        print("  running supersession…", flush=True)
        t0 = time.perf_counter()
        sup = await _supersession(
            embedder, d, args.supersession_distractors, args.supersession_sleep
        )
        print(
            f"    new@1={sup['new_top1']:.1%}  old@1={sup['old_top1']:.1%}  "
            f"superses={sup['supersession_when_both']:.1%}  both={sup['both_recalled']:.1%}  "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        results.append({"d_recall": d, "locomo_recall_at_10": lrecall, **sup})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "d_recall_sweep",
                "embedder": args.embedder,
                "locomo_n": args.locomo_n,
                "supersession_distractors": args.supersession_distractors,
                "supersession_sleep": args.supersession_sleep,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    print()
    print(f"{'d_recall':>8}  {'LoCoMo':>10}  {'new@1':>8}  {'old@1':>8}  {'superses.':>10}")
    print("─" * 55)
    for r in results:
        print(
            f"{r['d_recall']:>8.2f}  "
            f"{r['locomo_recall_at_10']:>10.4f}  "
            f"{r['new_top1']:>7.1%}  "
            f"{r['old_top1']:>7.1%}  "
            f"{r['supersession_when_both']:>9.1%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
