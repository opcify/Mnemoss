"""Supersession benchmark — does the memory system prefer newer contradicting facts?

The thesis Mnemoss makes but has never tested end-to-end: a memory
system with time-aware recall (decay + access history) should prefer
a *newer* fact that contradicts an *older* one, at recall time, with
no user intervention. Pure cosine similarity has no concept of time;
a BM25 + vector-search baseline will surface whichever of the two is
cosine-closer to the query, which is usually arbitrary for
topic-matched pairs. If Mnemoss can't beat a coin flip on this, the
"time-aware memory" pitch line is aspirational.

Experimental design
-------------------

1. Load ``bench/data/supersession_pairs.jsonl`` — 25 handcrafted
   (old, new, question) triples across five categories
   (state_update, preference, relationship, fact_correction,
   goal_update). Each pair is a *contradiction* — the new fact makes
   the old fact stale.
2. Ingest ``--distractors`` LoCoMo utterances first as unrelated
   padding (realistic workspace context).
3. Ingest the 25 *old* facts.
4. Sleep ``--sleep-seconds`` (default 60s) — real wall-clock gap so
   Mnemoss's ``access_history`` timestamps see real temporal
   separation between old and new.
5. Ingest the 25 *new* facts.
6. For each pair, run ``recall(question, k=10)``. Measure:

   - ``new_top1``: is the new fact ranked first?
   - ``old_top1``: is the old (wrong) fact ranked first? (bad)
   - ``new_above_old``: when both old and new return in top-10, does
     the new one outrank the old one?

Baselines tested
----------------

- ``raw_stack`` — pure cosine, no time awareness. Expected:
  supersession ≈ 50% (coin flip between two topic-matched memories).
- ``mnemoss_default`` — shipped defaults (d=0.01, noise_scale=0).
  Expected: small lift (B_i differential ~0.04 over 60s).
- ``mnemoss_decay`` — ``FormulaParams(d=0.5)``, the classic ACT-R
  decay constant. Expected: meaningful lift since old facts' B_i
  drops noticeably below new facts' in 60s.

Output is a per-arm breakdown + per-category cut so we can see
whether supersession works uniformly or is lumpy across fact types.

Usage::

    python -m bench.bench_stale_fact \\
        --embedder openai \\
        --distractors 300 \\
        --sleep-seconds 60 \\
        --out bench/results/stale_fact.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from bench.backends.mnemoss_backend import MnemossBackend
from bench.backends.raw_stack_backend import RawStackBackend
from bench.launch_comparison import MEMORIES_PATH, _load_jsonl, _resolve_embedder
from mnemoss import FormulaParams

PAIRS_PATH = Path("bench/data/supersession_pairs.jsonl")


def _load_pairs() -> list[dict]:
    return [json.loads(line) for line in PAIRS_PATH.open()]


def _rank_of(hits: list, target_id: str) -> int | None:
    for i, h in enumerate(hits):
        if h.memory_id == target_id:
            return i
    return None


async def _run_arm(
    *,
    backend,
    pairs: list[dict],
    n_distractors: int,
    sleep_seconds: float,
    k: int,
) -> dict:
    """Ingest distractors + old batch, sleep, ingest new batch, score."""

    # 1. Distractor padding — real conversational utterances from
    # LoCoMo, unrelated to the supersession pairs. Makes the workspace
    # look like a real one (few hundred memories) instead of a
    # 50-memory toy.
    locomo_texts = [m["text"] for m in _load_jsonl(MEMORIES_PATH)[:n_distractors]]
    for text in locomo_texts:
        await backend.observe(text, ts=time.time())

    # 2. Old batch — ingest at time T0.
    old_ids: dict[int, str] = {}
    for i, pair in enumerate(pairs):
        old_ids[i] = await backend.observe(pair["old"], ts=time.time())

    # 3. Temporal separation. Real sleep so Mnemoss's observe-time
    # datetime.now() gives a genuinely newer timestamp for the new
    # batch than the old batch. The bench API's ``ts`` is accepted by
    # raw_stack but dropped by MnemossBackend (see docstring there),
    # so we can't fake this with a ts argument.
    if sleep_seconds > 0:
        await asyncio.sleep(sleep_seconds)

    # 4. New batch — ingest at time T1 = T0 + sleep_seconds.
    new_ids: dict[int, str] = {}
    for i, pair in enumerate(pairs):
        new_ids[i] = await backend.observe(pair["new"], ts=time.time())

    # 5. Score. For each pair, find where old and new land in top-k.
    stats: dict[str, Any] = {
        "n_pairs": len(pairs),
        "new_top1": 0,
        "old_top1": 0,
        "neither_top1": 0,
        "new_in_topk": 0,
        "old_in_topk": 0,
        "both_in_topk": 0,
        "new_above_old_when_both": 0,
        "per_category": {},
    }

    for i, pair in enumerate(pairs):
        cat = pair.get("category", "unknown")
        cat_stats = stats["per_category"].setdefault(
            cat, {"n": 0, "new_top1": 0, "new_above_old": 0, "both_in_topk": 0}
        )
        cat_stats["n"] += 1

        hits = await backend.recall(pair["question"], k=k)
        old_rank = _rank_of(hits, old_ids[i])
        new_rank = _rank_of(hits, new_ids[i])

        if new_rank is not None:
            stats["new_in_topk"] += 1
        if old_rank is not None:
            stats["old_in_topk"] += 1
        if new_rank == 0:
            stats["new_top1"] += 1
            cat_stats["new_top1"] += 1
        elif old_rank == 0:
            stats["old_top1"] += 1
        else:
            stats["neither_top1"] += 1

        if new_rank is not None and old_rank is not None:
            stats["both_in_topk"] += 1
            cat_stats["both_in_topk"] += 1
            if new_rank < old_rank:
                stats["new_above_old_when_both"] += 1
                cat_stats["new_above_old"] += 1

    # Derived rates (the ones that matter in the writeup).
    n = stats["n_pairs"]
    both = max(stats["both_in_topk"], 1)
    stats["rate_new_top1"] = stats["new_top1"] / n
    stats["rate_old_top1"] = stats["old_top1"] / n
    stats["rate_supersession_when_both"] = stats["new_above_old_when_both"] / both
    stats["rate_both_recalled"] = stats["both_in_topk"] / n
    return stats


def _build_raw_stack(embedder) -> RawStackBackend:
    return RawStackBackend(embedding_model=embedder)


def _build_mnemoss_default(embedder) -> MnemossBackend:
    # Shipped defaults as of April 2026 — d=0.01, noise_scale=0.
    # Included to show what out-of-the-box Mnemoss does on this task.
    return MnemossBackend(embedding_model=embedder)


def _build_mnemoss_decay(embedder) -> MnemossBackend:
    # Classic ACT-R decay — opts into time-aware ranking. This is the
    # config that *should* earn the supersession lift.
    return MnemossBackend(
        embedding_model=embedder,
        formula=FormulaParams(d=0.5, noise_scale=0.0),
    )


def _build_mnemoss_supersede(embedder) -> MnemossBackend:
    # Semantic near-duplicate dedup at ingest (`supersede_on_observe`).
    # Threshold 0.7 here is *deliberately lower* than the shipped
    # default 0.85 to make the feature fire on this bench's
    # contradiction pairs (which have cosine 0.32–0.88, median 0.57).
    # The FP bench at ``bench/bench_false_positive.py`` measured
    # threshold=0.5 catches 76-80% of contradictions but also
    # suppresses 24% of topic-similar valid memories — so this arm
    # is a research probe, not a shipping recommendation. Production
    # users should stay at 0.85 (safe dedup) and rely on ``d_recall``
    # for general supersession handling.
    from mnemoss import EncoderParams

    return MnemossBackend(
        embedding_model=embedder,
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=0.7,
        ),
    )


# Retired arm (kept only as a historical note):
#
# ``mnemoss_supersede_priority`` was an experiment to replace the
# filter-based ``supersede_on_observe`` mechanism (which sets a
# ``superseded_by`` column) with progressive ``idx_priority``
# demotion instead. The 2×2 isolation test in ``bench_multi_step``
# proved the two mechanisms are mechanistically equivalent for
# ranking — the only moving factor was ``include_deep``. The priority
# approach offered no independent benefit while losing the filter
# mechanism's explicit audit trail, so the filter shipped.


ARMS = {
    "raw_stack": _build_raw_stack,
    "mnemoss_default": _build_mnemoss_default,
    "mnemoss_decay": _build_mnemoss_decay,
    "mnemoss_supersede": _build_mnemoss_supersede,
}


async def _run_all(
    *,
    arms: list[str],
    embedder_choice: str,
    n_distractors: int,
    sleep_seconds: float,
    k: int,
) -> list[dict]:
    pairs = _load_pairs()
    embedder = _resolve_embedder(embedder_choice)

    results = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choices: {list(ARMS)}")
        print(
            f"\n[stale-fact] arm={arm}  distractors={n_distractors}  "
            f"sleep={sleep_seconds}s  pairs={len(pairs)}",
            flush=True,
        )
        backend = ARMS[arm](embedder)
        try:
            stats = await _run_arm(
                backend=backend,
                pairs=pairs,
                n_distractors=n_distractors,
                sleep_seconds=sleep_seconds,
                k=k,
            )
        finally:
            await backend.close()
        stats["arm"] = arm
        stats["embedder"] = embedder_choice
        print(
            f"  new@1={stats['rate_new_top1']:.2%}  "
            f"old@1={stats['rate_old_top1']:.2%}  "
            f"supersession={stats['rate_supersession_when_both']:.2%}  "
            f"(both_recalled={stats['rate_both_recalled']:.2%})",
            flush=True,
        )
        results.append(stats)
    return results


def _print_summary(results: list[dict]) -> None:
    print()
    print(f"{'arm':>22}  {'new@1':>8}  {'old@1':>8}  {'superses.':>10}  {'both_rec':>10}")
    print("─" * 70)
    for r in results:
        print(
            f"{r['arm']:>22}  "
            f"{r['rate_new_top1']:>7.2%}  "
            f"{r['rate_old_top1']:>7.2%}  "
            f"{r['rate_supersession_when_both']:>9.2%}  "
            f"{r['rate_both_recalled']:>9.2%}"
        )

    # Per-category breakdown for the decay arm (the interesting one).
    target = next((r for r in results if r["arm"] == "mnemoss_decay"), None)
    if target and target["per_category"]:
        print()
        print("mnemoss_decay per-category supersession (new_above_old / both_in_topk):")
        for cat, s in sorted(target["per_category"].items()):
            both = max(s["both_in_topk"], 1)
            rate = s["new_above_old"] / both
            print(f"  {cat:>18s}: {rate:.2%}  ({s['new_above_old']}/{s['both_in_topk']})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["raw_stack", "mnemoss_default", "mnemoss_decay"],
        choices=list(ARMS),
    )
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "nomic", "gemma", "fake"],
        default="openai",
    )
    p.add_argument(
        "--distractors",
        type=int,
        default=300,
        help="Unrelated LoCoMo utterances to ingest before supersession pairs.",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=60.0,
        help="Wall-clock gap between old-batch and new-batch ingestion.",
    )
    p.add_argument("--k", type=int, default=10, help="Top-k for scoring.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _run_all(
            arms=args.arms,
            embedder_choice=args.embedder,
            n_distractors=args.distractors,
            sleep_seconds=args.sleep_seconds,
            k=args.k,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "stale_fact",
                "embedder": args.embedder,
                "distractors": args.distractors,
                "sleep_seconds": args.sleep_seconds,
                "k": args.k,
                "timestamp": started.isoformat(),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
