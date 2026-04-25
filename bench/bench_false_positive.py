"""False-positive benchmark for `supersede_on_observe`.

Companion to ``bench_stale_fact.py``: measures how often Lever 3
*incorrectly* suppresses a memory that should have been preserved.

The supersession bench proves true positives — that Lever 3 can
supersede an old contradicting fact when a newer one arrives. This
bench probes the dual risk: does Lever 3 also supersede *non-*
contradicting memories that just happen to be topically close to a
later observe?

The dataset (``bench/data/non_contradiction_pairs.jsonl``) is 25
handcrafted (a, b, question) triples across five categories. In each
pair, ``a`` and ``b`` are topically similar (same person, same time
range, same preference cluster, sequential events of one story arc,
or different aspects of one topic) but **both memories are valid** —
neither should supersede the other. The question is phrased to be
open-ended enough that both memories are legitimate top-k
candidates.

Metrics per-threshold:
- **fp_superseded** — fraction of pairs where ``a`` was marked
  ``superseded_by = b.id`` after ingestion. Every such case is a
  false positive.
- **a_recalled** — fraction of pairs where ``a`` appears in top-10
  for the question (independent of supersession; a memory can fall
  out of top-10 because of distractors too, not just supersession).
- **b_recalled** — ditto for ``b``.
- **both_recalled** — fraction of pairs where both ``a`` and ``b``
  appear in top-10. The "correct" behavior for non-contradicting
  pairs.

Combined with ``bench_stale_fact.py`` new@1, this lets us plot the
Lever 3 precision/recall curve across thresholds: how many true
contradictions catch vs how many valid memories we lose.

Usage::

    python -m bench.bench_false_positive \\
        --embedder openai \\
        --thresholds 0.5 0.6 0.7 0.85 \\
        --distractors 300 \\
        --out bench/results/false_positive.json
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
from bench.launch_comparison import MEMORIES_PATH, _load_jsonl, _resolve_embedder
from mnemoss import EncoderParams

PAIRS_PATH = Path("bench/data/non_contradiction_pairs.jsonl")


def _load_pairs() -> list[dict]:
    return [json.loads(line) for line in PAIRS_PATH.open()]


def _rank_of(hits: list, target_id: str) -> int | None:
    for i, h in enumerate(hits):
        if h.memory_id == target_id:
            return i
    return None


async def _run_threshold(
    *,
    embedder,
    threshold: float,
    pairs: list[dict],
    n_distractors: int,
    k: int,
) -> dict:
    """Run one threshold setting against all 25 non-contradiction pairs."""

    backend = MnemossBackend(
        embedding_model=embedder,
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=threshold,
        ),
    )
    try:
        # 1. Distractor padding. Unrelated memories mixed in so the
        # test simulates a realistic workspace.
        for m in _load_jsonl(MEMORIES_PATH)[:n_distractors]:
            await backend.observe(m["text"], ts=time.time())

        # 2. Ingest each pair: a first, then b (b may or may not
        # trigger supersession of a). Zero gap — this is the
        # worst-case for Lever 3 because decay has no signal either.
        fp_count = 0
        a_recalled = 0
        b_recalled = 0
        both_recalled = 0
        per_cat: dict[str, dict[str, int]] = {}

        for pair in pairs:
            a_id = await backend.observe(pair["a"], ts=time.time())
            b_id = await backend.observe(pair["b"], ts=time.time())

            # Check whether Lever 3 marked ``a`` as superseded by
            # ``b``. We reach into the store because MnemossBackend
            # doesn't expose supersede state as a public method.
            raw_a = await backend._mem._store.get_memory(a_id)
            is_fp = raw_a is not None and raw_a.superseded_by == b_id

            # Check recall behavior: for the non-contradicting
            # question, do both memories still appear in top-k?
            hits = await backend.recall(pair["question"], k=k)
            a_rank = _rank_of(hits, a_id)
            b_rank = _rank_of(hits, b_id)

            if is_fp:
                fp_count += 1
            if a_rank is not None:
                a_recalled += 1
            if b_rank is not None:
                b_recalled += 1
            if a_rank is not None and b_rank is not None:
                both_recalled += 1

            cat = pair.get("category", "unknown")
            stats = per_cat.setdefault(cat, {"n": 0, "fp": 0, "a_recalled": 0, "both": 0})
            stats["n"] += 1
            if is_fp:
                stats["fp"] += 1
            if a_rank is not None:
                stats["a_recalled"] += 1
            if a_rank is not None and b_rank is not None:
                stats["both"] += 1
    finally:
        await backend.close()

    n = len(pairs)
    return {
        "threshold": threshold,
        "n_pairs": n,
        "fp_superseded": fp_count,
        "fp_rate": fp_count / n,
        "a_recalled": a_recalled,
        "a_recall_rate": a_recalled / n,
        "b_recalled": b_recalled,
        "b_recall_rate": b_recalled / n,
        "both_recalled": both_recalled,
        "both_recall_rate": both_recalled / n,
        "per_category": per_cat,
    }


async def _run_all(
    *,
    embedder_choice: str,
    thresholds: list[float],
    n_distractors: int,
    k: int,
) -> list[dict]:
    pairs = _load_pairs()
    embedder = _resolve_embedder(embedder_choice)
    out = []
    for thr in thresholds:
        print(f"\n[fp] threshold={thr}  embedder={embedder_choice}", flush=True)
        row = await _run_threshold(
            embedder=embedder,
            threshold=thr,
            pairs=pairs,
            n_distractors=n_distractors,
            k=k,
        )
        row["embedder"] = embedder_choice
        print(
            f"  fp_rate={row['fp_rate']:.0%}  "
            f"a_recalled={row['a_recall_rate']:.0%}  "
            f"b_recalled={row['b_recall_rate']:.0%}  "
            f"both={row['both_recall_rate']:.0%}",
            flush=True,
        )
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "nomic", "gemma", "fake"],
        default="openai",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.85],
    )
    p.add_argument("--distractors", type=int, default=300)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _run_all(
            embedder_choice=args.embedder,
            thresholds=args.thresholds,
            n_distractors=args.distractors,
            k=args.k,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "false_positive",
                "embedder": args.embedder,
                "distractors": args.distractors,
                "thresholds": args.thresholds,
                "k": args.k,
                "timestamp": started.isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")

    # Summary table.
    print()
    print(f"{'threshold':>10}  {'fp_rate':>8}  {'a_recall':>9}  {'b_recall':>9}  {'both':>6}")
    print("─" * 50)
    for r in results:
        print(
            f"{r['threshold']:>10.2f}  "
            f"{r['fp_rate']:>7.0%}  "
            f"{r['a_recall_rate']:>8.0%}  "
            f"{r['b_recall_rate']:>8.0%}  "
            f"{r['both_recall_rate']:>5.0%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
