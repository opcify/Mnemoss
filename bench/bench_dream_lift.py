"""Does Dream P7 actually lift recall accuracy?

The premise of Mnemoss's async-ACT-R architecture: recall bumps
``access_history``, Dream P7 Rebalance recomputes ``idx_priority`` from
that history, and subsequent recalls benefit. This benchmark tries to
prove (or disprove) the lift empirically.

Experimental setup
------------------

1. Ingest LoCoMo conv-26 padded to N with distractors.
2. Configure ``mnemoss_rocket`` with a non-zero
   ``fast_index_priority_weight`` so ``idx_priority`` actually affects
   ranking.
3. **Priming phase** — for each gold QA pair, issue a recall query using
   the *gold memory's own content* as the query. This deliberately
   puts gold memories into ``access_history``. Simulates production
   "popular memories get accessed a lot" patterns.
4. **Fire Dream P7 Rebalance.** ``idx_priority`` is recomputed from
   ``access_history``; gold memories should get higher priority.
5. **Scored phase** — run the real LoCoMo questions, measure recall@10.

Three comparison arms:
- **no_prime_no_dream:** skip the priming phase entirely. Pure
  ANN+priority recall with initial priorities everywhere (≈0.5).
- **prime_no_dream:** run the priming phase but DON'T fire Dream —
  priority stays stale between reads.
- **prime_and_dream:** run priming, fire Dream, then run scored
  queries. The target arm.

Output: per-arm recall@10. The lift from Dream is
``(prime_and_dream - prime_no_dream)``.

Usage::

    python -m bench.bench_dream_lift \\
        --embedder local \\
        --sizes 1500 3000 5000 \\
        --gold-conversation conv-26 \\
        --priority-weight 0.5 \\
        --out bench/results/dream_lift.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)
from mnemoss import FormulaParams, Mnemoss, StorageParams


async def _ingest(mem: Mnemoss, padded_mems: list[dict]) -> dict[str, str]:
    """Ingest all memories; return ``{dia_id: mnemoss_memory_id}`` map."""

    dia_to_mid: dict[str, str] = {}
    for m in padded_mems:
        mid = await mem.observe(role="user", content=m["text"], session_id="bench")
        if mid is not None:
            dia_to_mid[m["dia_id"]] = mid
    return dia_to_mid


async def _score_queries(
    mem: Mnemoss,
    gold_queries: list[dict],
    k: int = 10,
) -> float:
    """Run each gold query; return mean recall@k."""

    recalls: list[float] = []
    for q in gold_queries:
        results = await mem.recall(q["question"], k=k, reconsolidate=True)
        returned = {r.memory.id for r in results}
        # gold dia_ids were rewritten to namespaced form by build_scale_corpus;
        # we re-map via the caller's dia_to_mid. For simplicity the harness
        # re-resolves against the store's content lookup — but the caller
        # attaches a ``gold_mids`` set to each q dict before calling us.
        gold_mids = q["_gold_mids"]
        if not gold_mids:
            continue
        hit = len(returned & gold_mids)
        recalls.append(hit / len(gold_mids))
    return statistics.mean(recalls) if recalls else 0.0


async def _run_arm(
    *,
    arm: str,
    scale_n: int,
    embedder_choice: str,
    gold_conversation: str,
    k: int,
    priority_weight: float,
) -> dict:
    """Run one experimental arm.

    ``arm`` controls behaviour:
    - ``no_prime_no_dream``: skip priming, skip Dream.
    - ``prime_no_dream``:    run priming, skip Dream.
    - ``prime_and_dream``:   run priming, fire Dream P7.
    """

    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories,
        queries,
        gold_conversation_id=gold_conversation,
        scale_n=scale_n,
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="mnemoss_dreamlift_"))
    try:
        mem = Mnemoss(
            workspace="dream_lift",
            embedding_model=_resolve_embedder(embedder_choice),
            formula=FormulaParams(
                noise_scale=0.0,
                eta_0=0.0,
                d=0.01,
                use_fast_index_recall=True,
                fast_index_semantic_weight=1.0,
                fast_index_priority_weight=priority_weight,
            ),
            storage=StorageParams(root=tmpdir),
        )
        dia_to_mid = await _ingest(mem, padded_mems)

        # Resolve gold dia_ids → mnemoss ids per question.
        for q in gold_queries:
            q["_gold_mids"] = {
                dia_to_mid[d] for d in q["relevant_dia_ids"] if d in dia_to_mid
            }

        if arm != "no_prime_no_dream":
            # Priming phase: query with each gold memory's content so its
            # access_history gets a hit. This simulates a production
            # pattern where popular memories get recalled often.
            for q in gold_queries:
                for dia in q["relevant_dia_ids"]:
                    if dia not in dia_to_mid:
                        continue
                    # Pull the memory's content from the padded_mems.
                    text = next(
                        (m["text"] for m in padded_mems if m["dia_id"] == dia),
                        None,
                    )
                    if text:
                        await mem.recall(text, k=k, reconsolidate=True)

        if arm == "prime_and_dream":
            # Fire P7 Rebalance directly — recomputes idx_priority from
            # access_history. Note: dream(trigger="idle") does NOT run
            # rebalance (only nightly does; see PHASES_BY_TRIGGER in
            # dream/runner.py). Calling rebalance() isolates the
            # idx_priority effect without LLM-backed Consolidate.
            await mem.rebalance()

        # Scored phase: the actual LoCoMo recall@10 measurement.
        recall = await _score_queries(mem, gold_queries, k=k)
    finally:
        await mem.close()
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "arm": arm,
        "scale_n": scale_n,
        "priority_weight": priority_weight,
        "embedder": embedder_choice,
        "recall_at_k": round(recall, 4),
        "n_queries": len(gold_queries),
    }


async def _run_all(
    *,
    sizes: list[int],
    embedder_choice: str,
    gold_conversation: str,
    k: int,
    priority_weight: float,
) -> list[dict]:
    out: list[dict] = []
    arms = ["no_prime_no_dream", "prime_no_dream", "prime_and_dream"]
    for n in sizes:
        for arm in arms:
            print(f"[dream-lift] arm={arm}  N={n}  pri_w={priority_weight}", flush=True)
            row = await _run_arm(
                arm=arm,
                scale_n=n,
                embedder_choice=embedder_choice,
                gold_conversation=gold_conversation,
                k=k,
                priority_weight=priority_weight,
            )
            out.append(row)
            print(f"  → recall@10 = {row['recall_at_k']:.4f}", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedder", choices=["openai", "local", "nomic", "fake"], default="local")
    p.add_argument("--sizes", type=int, nargs="+", default=[1500, 3000])
    p.add_argument("--gold-conversation", default="conv-26")
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--priority-weight",
        type=float,
        default=0.5,
        help="fast_index_priority_weight — must be > 0 for Dream to matter.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.priority_weight <= 0:
        raise SystemExit("--priority-weight must be > 0 for Dream to affect ranking")

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _run_all(
            sizes=args.sizes,
            embedder_choice=args.embedder,
            gold_conversation=args.gold_conversation,
            k=args.k,
            priority_weight=args.priority_weight,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "dream_lift",
                "embedder": args.embedder,
                "sizes": args.sizes,
                "priority_weight": args.priority_weight,
                "gold_conversation": args.gold_conversation,
                "k": args.k,
                "timestamp": started.isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")

    # Terminal table.
    print()
    print(f"{'arm':>22}  {'N':>6}  {'recall@10':>10}")
    print("─" * 50)
    for r in results:
        print(f"{r['arm']:>22}  {r['scale_n']:>6}  {r['recall_at_k']:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
