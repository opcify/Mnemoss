"""Multi-step supersession benchmark — facts that change more than once.

The headline supersession bench (``bench_stale_fact.py``) covers the
simplest case: one old fact, one new fact, pick the new one. Real
conversational memory isn't like that — facts evolve in chains:

- "Junior engineer at Google" → "Senior at Google" → "Staff at Stripe"
- "Dating Jamie" → "Moved in" → "Engaged" → "Married"
- "Coffee is my fuel" → "Cut back to one cup" → "Switched to matcha"

A correct memory system should surface the **latest** version of a
fact, not any earlier version. Whether decay-based or content-based
supersession extends cleanly to 3-4 step chains is an empirical
question this bench answers.

Dataset: ``bench/data/multi_step_pairs.jsonl`` — 20 sequences across
7 categories (career, relationship, address, preference, project,
health, possession). Each sequence has 3-4 versions and a neutral
question that should return the *latest* version.

Ingest protocol: all v1s first, sleep, all v2s, sleep, ..., all vNs.
This gives consistent time-decay gaps between versions while
amortizing the sleep budget (N_sleeps × gap, not N_sequences × N_sleeps
× gap).

Metrics per arm:
- **latest@1**: latest version is at rank 1 (headline — correct answer)
- **any_older@1**: any older version (v1..vN-1) is at rank 1 (wrong answer)
- **latest_in_topk**: latest version appears in top-10 at all
- **mean_latest_rank**: average rank position of latest version

Usage::

    python -m bench.bench_multi_step \\
        --embedder openai \\
        --distractors 300 \\
        --gap-seconds 60 \\
        --arms raw_stack mnemoss_default mnemoss_supersede \\
        --out bench/results/multi_step_openai.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
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
from bench.launch_comparison import (
    MEMORIES_PATH,
    QUERIES_PATH,
    _build_scale_corpus,
    _load_jsonl,
    _resolve_embedder,
)
from mnemoss import EncoderParams, FormulaParams

PAIRS_PATH = Path("bench/data/multi_step_pairs.jsonl")
PAIRS_PATH_ENTITY_BOUND = Path("bench/data/multi_step_pairs_entity_bound.jsonl")


def _load_sequences(path: Path = PAIRS_PATH) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _rank_of(hits: list, target_id: str) -> int | None:
    for i, h in enumerate(hits):
        if h.memory_id == target_id:
            return i
    return None


def _build_raw_stack(embedder):
    return RawStackBackend(embedding_model=embedder)


def _build_mnemoss_default(embedder):
    return MnemossBackend(embedding_model=embedder)


def _build_mnemoss_decay(embedder):
    return MnemossBackend(
        embedding_model=embedder,
        formula=FormulaParams(d=0.5, noise_scale=0.0),
    )


def _build_mnemoss_supersede_default(embedder):
    # Ship default threshold 0.85 — dedup safety-valve.
    return MnemossBackend(
        embedding_model=embedder,
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=0.85,
        ),
    )


def _build_mnemoss_supersede_aggressive(embedder):
    # Threshold 0.7 — more aggressive, research setting (bench only).
    return MnemossBackend(
        embedding_model=embedder,
        encoder=EncoderParams(
            supersede_on_observe=True,
            supersede_cosine_threshold=0.7,
        ),
    )


# Retired arms (kept only as a historical note):
#
# ``mnemoss_supersede_priority`` / ``..._priority_0_20`` / ``..._priority_0_40``
# were an experiment to replace the filter-based ``supersede_on_observe``
# mechanism (which sets a ``superseded_by`` column) with progressive
# ``idx_priority`` demotion instead. The hypothesis: re-use existing
# schema, get equivalent ranking via tier migration.
#
# The 2×2 isolation test (mechanism × include_deep) at
# ``bench/results/ms_arm_filter_aggr_no_deep.json`` and
# ``ms_arm_priority_0_20_with_deep.json`` showed the two mechanisms are
# mechanistically identical for ranking outcome — all four cells
# aligned on mechanism, with the only moving factor being
# ``include_deep``. The priority approach offered no independent
# ranking benefit while losing the filter mechanism's explicit
# ``superseded_by`` audit trail, so we kept the filter.
#
# The isolation arms (``filter_aggressive_include_deep_false``,
# ``priority_0_20_include_deep_true``) served their one-shot purpose
# and are removed too. If ``include_deep`` behavior needs re-probing,
# flip the ``MnemossBackend(include_deep=...)`` arg in a new arm rather
# than resurrecting these.


ARMS = {
    "raw_stack": _build_raw_stack,
    "mnemoss_default": _build_mnemoss_default,
    "mnemoss_decay": _build_mnemoss_decay,
    "mnemoss_supersede": _build_mnemoss_supersede_default,
    "mnemoss_supersede_aggressive": _build_mnemoss_supersede_aggressive,
}


async def _run_arm(
    *,
    backend,
    sequences: list[dict],
    n_distractors: int,
    gap_seconds: float,
    k: int,
    locomo_scale: int = 0,
    rebalance_before_scoring: bool = False,
    rebalance_each_phase: bool = False,
) -> dict:
    """Ingest sequences phase-by-phase with consistent time gaps, then score.

    When ``locomo_scale > 0``, ingest a LoCoMo scale corpus (gold conversation
    + distractors up to ``locomo_scale`` total memories) BEFORE the multi-step
    sequences. This tests historical recall in the same workspace as the
    supersession queries — answers "does mnemoss_decay's 0% older@1 come at
    a hidden LoCoMo cost?". When ``locomo_scale == 0``, falls back to the
    original behavior (``n_distractors`` random padding, no LoCoMo scoring).
    """

    # 1. Distractor / LoCoMo-scale padding.
    locomo_dia_to_mid: dict[str, str] = {}
    locomo_gold_queries: list[dict] = []
    if locomo_scale > 0:
        memories = _load_jsonl(MEMORIES_PATH)
        queries = _load_jsonl(QUERIES_PATH)
        padded_mems, locomo_gold_queries = _build_scale_corpus(
            memories, queries, gold_conversation_id="conv-26", scale_n=locomo_scale
        )
        for m in padded_mems:
            mid = await backend.observe(m["text"], ts=time.time())
            if mid is not None:
                locomo_dia_to_mid[m["dia_id"]] = mid
    else:
        for m in _load_jsonl(MEMORIES_PATH)[:n_distractors]:
            await backend.observe(m["text"], ts=time.time())

    # 2. Find max version count so we can iterate by version index.
    max_versions = max(len(s["versions"]) for s in sequences)

    # 3. Ingest phase-by-phase. Each phase: all sequences' v_i in one
    # pass, then sleep the gap before the next version. Sequences with
    # fewer versions naturally drop out of later phases.
    # Store: ids[seq_idx] = [mid_v0, mid_v1, ...]
    ids: dict[int, list[str]] = {i: [] for i in range(len(sequences))}

    for phase_idx in range(max_versions):
        any_in_phase = False
        for seq_idx, seq in enumerate(sequences):
            if phase_idx < len(seq["versions"]):
                mid = await backend.observe(seq["versions"][phase_idx], ts=time.time())
                ids[seq_idx].append(mid)
                any_in_phase = True

        # Optional rebalance after each phase. Tests whether running
        # Rebalance after every chain version ingest lets the tier
        # system continuously re-prioritize so the latest version is
        # always in HOT for the cascade.
        if rebalance_each_phase and any_in_phase and hasattr(backend, "_mem"):
            await backend._mem.rebalance()

        # Sleep between phases, but not after the last phase.
        is_last = phase_idx == max_versions - 1
        if any_in_phase and not is_last and gap_seconds > 0:
            await asyncio.sleep(gap_seconds)

    # 3.5. Optional rebalance before scoring. Lets Dream re-bucket the
    # tier index so the latest version of each chain (most recent
    # access_history entry → highest idx_priority) lands in HOT/WARM
    # while older versions drift down. Tests whether the architectural
    # claim "Rebalance restores supersession after the read path lost
    # the per-candidate B_i tiebreak" actually holds.
    if rebalance_before_scoring and hasattr(backend, "_mem"):
        await backend._mem.rebalance()

    # 4. Score: for each sequence, query and check where each version lands.
    latest_at_1 = 0
    any_older_at_1 = 0
    latest_in_topk = 0
    latest_ranks: list[int] = []
    per_cat: dict[str, dict[str, int]] = {}

    for seq_idx, seq in enumerate(sequences):
        cat = seq.get("category", "unknown")
        cat_stats = per_cat.setdefault(cat, {"n": 0, "latest_at_1": 0})
        cat_stats["n"] += 1

        hits = await backend.recall(seq["question"], k=k)
        seq_ids = ids[seq_idx]
        latest_mid = seq_ids[-1]
        older_mids = set(seq_ids[:-1])

        latest_rank = _rank_of(hits, latest_mid)
        rank_1_id = hits[0].memory_id if hits else None

        if latest_rank is not None:
            latest_in_topk += 1
            latest_ranks.append(latest_rank)
        if rank_1_id == latest_mid:
            latest_at_1 += 1
            cat_stats["latest_at_1"] += 1
        elif rank_1_id in older_mids:
            any_older_at_1 += 1

    # 5. (Optional) Score LoCoMo questions against the same workspace.
    locomo_recall_hits: list[float] = []
    locomo_n_scored = 0
    for q in locomo_gold_queries:
        gold_ids = {
            locomo_dia_to_mid[d] for d in q["relevant_dia_ids"] if d in locomo_dia_to_mid
        }
        if not gold_ids:
            continue
        hits = await backend.recall(q["question"], k=k)
        returned = {h.memory_id for h in hits}
        locomo_recall_hits.append(len(returned & gold_ids) / len(gold_ids))
        locomo_n_scored += 1

    n = len(sequences)
    locomo_recall = (
        statistics.mean(locomo_recall_hits) if locomo_recall_hits else 0.0
    )
    return {
        "n_sequences": n,
        "max_versions": max_versions,
        "latest_at_1": latest_at_1,
        "any_older_at_1": any_older_at_1,
        "latest_in_topk": latest_in_topk,
        "rate_latest_at_1": latest_at_1 / n,
        "rate_any_older_at_1": any_older_at_1 / n,
        "rate_latest_in_topk": latest_in_topk / n,
        "mean_latest_rank": statistics.mean(latest_ranks) if latest_ranks else -1,
        "per_category": per_cat,
        "locomo_scale": locomo_scale,
        "locomo_n_scored": locomo_n_scored,
        "locomo_recall_at_k": round(locomo_recall, 4),
    }


async def _run_all(
    *,
    arms: list[str],
    embedder_choice: str,
    n_distractors: int,
    gap_seconds: float,
    k: int,
    dataset: str = "default",
    locomo_scale: int = 0,
    rebalance_before_scoring: bool = False,
    rebalance_each_phase: bool = False,
) -> list[dict]:
    path = PAIRS_PATH_ENTITY_BOUND if dataset == "entity_bound" else PAIRS_PATH
    sequences = _load_sequences(path)
    embedder = _resolve_embedder(embedder_choice)
    results = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choices: {list(ARMS)}")
        pad_label = (
            f"locomo={locomo_scale}"
            if locomo_scale > 0
            else f"distractors={n_distractors}"
        )
        print(
            f"\n[multi-step] arm={arm}  {pad_label}  "
            f"gap={gap_seconds}s  sequences={len(sequences)}",
            flush=True,
        )
        backend = ARMS[arm](embedder)
        try:
            stats = await _run_arm(
                backend=backend,
                sequences=sequences,
                n_distractors=n_distractors,
                gap_seconds=gap_seconds,
                k=k,
                locomo_scale=locomo_scale,
                rebalance_before_scoring=rebalance_before_scoring,
                rebalance_each_phase=rebalance_each_phase,
            )
        finally:
            await backend.close()
        stats["arm"] = arm
        stats["embedder"] = embedder_choice
        locomo_str = (
            f"  locomo@k={stats['locomo_recall_at_k']:.4f} "
            f"({stats['locomo_n_scored']} qs)"
            if stats.get("locomo_n_scored", 0) > 0
            else ""
        )
        print(
            f"  latest@1={stats['rate_latest_at_1']:.0%}  "
            f"older@1={stats['rate_any_older_at_1']:.0%}  "
            f"latest_in_topk={stats['rate_latest_in_topk']:.0%}  "
            f"mean_rank={stats['mean_latest_rank']:.2f}{locomo_str}",
            flush=True,
        )
        results.append(stats)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
    )
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "nomic", "gemma", "fake"],
        default="openai",
    )
    p.add_argument("--distractors", type=int, default=300)
    p.add_argument(
        "--gap-seconds",
        type=float,
        default=60.0,
        help="Wall-clock sleep between version phases.",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--dataset",
        choices=["default", "entity_bound"],
        default="default",
        help="'default' = first-person generic questions; 'entity_bound' = "
        "questions reference a unique proper noun per sequence to kill "
        "cross-sequence contamination.",
    )
    p.add_argument(
        "--locomo-scale",
        type=int,
        default=0,
        help="If > 0, ingest a LoCoMo scale corpus of this size before the "
        "multi-step sequences and score the 197 LoCoMo questions alongside. "
        "Tests historical recall in the same workspace as supersession — "
        "exposes hidden costs of aggressive recency bias (e.g., d=0.5 "
        "decay's LoCoMo cliff).",
    )
    p.add_argument(
        "--rebalance-before-scoring",
        action="store_true",
        help="Run mem.rebalance() after observing all chain versions and "
        "before scoring queries. Tests whether tier-cascade-pure-cosine "
        "recall recovers the supersession win once Dream has had a chance "
        "to rank latest > older by idx_priority and re-bucket tiers.",
    )
    p.add_argument(
        "--rebalance-each-phase",
        action="store_true",
        help="Run mem.rebalance() after every chain version phase ingest. "
        "Tests continuous re-prioritization: each phase's latest version "
        "becomes HOT-eligible immediately, so by the time scoring starts "
        "the cascade should reach the most-recent versions first.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _run_all(
            arms=args.arms,
            embedder_choice=args.embedder,
            n_distractors=args.distractors,
            gap_seconds=args.gap_seconds,
            k=args.k,
            dataset=args.dataset,
            locomo_scale=args.locomo_scale,
            rebalance_before_scoring=args.rebalance_before_scoring,
            rebalance_each_phase=args.rebalance_each_phase,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "multi_step_supersession",
                "embedder": args.embedder,
                "distractors": args.distractors,
                "gap_seconds": args.gap_seconds,
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
    has_locomo = any(r.get("locomo_n_scored", 0) > 0 for r in results)
    print()
    header = f"{'arm':>30}  {'latest@1':>10}  {'older@1':>10}  {'in_topk':>10}  {'mean_rank':>10}"
    if has_locomo:
        header += f"  {'locomo@k':>10}"
    print(header)
    print("─" * (78 + (12 if has_locomo else 0)))
    for r in results:
        line = (
            f"{r['arm']:>30}  "
            f"{r['rate_latest_at_1']:>9.0%}  "
            f"{r['rate_any_older_at_1']:>9.0%}  "
            f"{r['rate_latest_in_topk']:>9.0%}  "
            f"{r['mean_latest_rank']:>10.2f}"
        )
        if has_locomo:
            loco = r.get("locomo_recall_at_k", 0.0)
            line += f"  {loco:>10.4f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
