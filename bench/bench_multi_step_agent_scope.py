"""Multi-step bench with agent-scoped recall.

Companion experiment to ``bench_multi_step.py``. Same dataset and
scoring logic, but memories are ingested under two separate agent_ids:

- ``locomo_corpus`` — 300 LoCoMo distractors
- ``user`` — the 20 multi-step sequences

At recall time, queries go through ``mem.for_agent("user").recall()``,
which scopes the candidate pool to memories agent-owned by ``user`` or
ambient (agent_id IS NULL). Cross-agent distractors drop out of the
candidate pool entirely.

Hypothesis: the 30% latest@1 ceiling in the shared-scope bench is a
distractor-competition artifact. Under agent isolation — which is
how a real multi-agent workspace would be used — latest@1 should
jump substantially.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

import tempfile

from bench.launch_comparison import MEMORIES_PATH, _load_jsonl, _resolve_embedder
from mnemoss import FormulaParams, Mnemoss, StorageParams

PAIRS_PATH = Path("bench/data/multi_step_pairs.jsonl")


def _load_sequences() -> list[dict]:
    return [json.loads(line) for line in PAIRS_PATH.open()]


def _rank_of_in_results(results: list, target_id: str) -> int | None:
    for i, r in enumerate(results):
        if r.memory.id == target_id:
            return i
    return None


async def _run_arm(
    *,
    embedder,
    formula: FormulaParams | None,
    n_distractors: int,
    gap_seconds: float,
    k: int,
    include_deep: bool,
) -> dict:
    """Ingest distractors under ``locomo_corpus``, sequences under
    ``user``, then query via ``mem.for_agent("user").recall()``."""

    tempdir = Path(tempfile.mkdtemp(prefix="mnemoss_msagent_"))
    mem = Mnemoss(
        workspace="multi_step_agent",
        embedding_model=embedder,
        formula=formula or FormulaParams(),
        storage=StorageParams(root=tempdir),
    )
    user = mem.for_agent("user")
    corpus = mem.for_agent("locomo_corpus")

    try:
        # 1. Distractors under corpus agent.
        for m in _load_jsonl(MEMORIES_PATH)[:n_distractors]:
            await corpus.observe(role="user", content=m["text"])

        # 2. Phase-by-phase ingest of sequences under user agent.
        sequences = _load_sequences()
        max_versions = max(len(s["versions"]) for s in sequences)
        ids: dict[int, list[str]] = {i: [] for i in range(len(sequences))}

        for phase_idx in range(max_versions):
            any_in_phase = False
            for seq_idx, seq in enumerate(sequences):
                if phase_idx < len(seq["versions"]):
                    mid = await user.observe(role="user", content=seq["versions"][phase_idx])
                    if mid is not None:
                        ids[seq_idx].append(mid)
                    any_in_phase = True
            is_last = phase_idx == max_versions - 1
            if any_in_phase and not is_last and gap_seconds > 0:
                await asyncio.sleep(gap_seconds)

        # 3. Score: query through user agent so distractors are scoped out.
        latest_at_1 = 0
        any_older_at_1 = 0
        latest_in_topk = 0
        latest_ranks: list[int] = []

        for seq_idx, seq in enumerate(sequences):
            results = await user.recall(
                seq["question"],
                k=k,
                include_deep=include_deep,
                auto_expand=False,
                reconsolidate=False,
            )
            seq_ids = ids[seq_idx]
            if not seq_ids:
                continue
            latest_mid = seq_ids[-1]
            older_mids = set(seq_ids[:-1])

            latest_rank = _rank_of_in_results(results, latest_mid)
            rank_1_id = results[0].memory.id if results else None

            if latest_rank is not None:
                latest_in_topk += 1
                latest_ranks.append(latest_rank)
            if rank_1_id == latest_mid:
                latest_at_1 += 1
            elif rank_1_id in older_mids:
                any_older_at_1 += 1
    finally:
        await mem.close()
        import shutil

        shutil.rmtree(tempdir, ignore_errors=True)

    n = len(sequences)
    return {
        "n_sequences": n,
        "latest_at_1": latest_at_1,
        "any_older_at_1": any_older_at_1,
        "latest_in_topk": latest_in_topk,
        "rate_latest_at_1": latest_at_1 / n,
        "rate_any_older_at_1": any_older_at_1 / n,
        "rate_latest_in_topk": latest_in_topk / n,
        "mean_latest_rank": statistics.mean(latest_ranks) if latest_ranks else -1,
    }


async def _run_all(
    *,
    arms: list[str],
    embedder_choice: str,
    n_distractors: int,
    gap_seconds: float,
    k: int,
) -> list[dict]:
    embedder = _resolve_embedder(embedder_choice)
    results = []

    arm_configs = {
        "mnemoss_default": {"formula": None, "include_deep": False},
        "mnemoss_decay": {"formula": FormulaParams(d=0.5), "include_deep": False},
    }

    for arm in arms:
        if arm not in arm_configs:
            raise ValueError(f"unknown arm {arm!r}; choices: {list(arm_configs)}")
        cfg = arm_configs[arm]
        print(
            f"\n[ms-agent] arm={arm}  distractors={n_distractors} (under locomo_corpus)  "
            f"gap={gap_seconds}s  sequences=20",
            flush=True,
        )
        stats = await _run_arm(
            embedder=embedder,
            formula=cfg["formula"],
            n_distractors=n_distractors,
            gap_seconds=gap_seconds,
            k=k,
            include_deep=cfg["include_deep"],
        )
        stats["arm"] = arm
        stats["embedder"] = embedder_choice
        print(
            f"  latest@1={stats['rate_latest_at_1']:.0%}  "
            f"older@1={stats['rate_any_older_at_1']:.0%}  "
            f"in_topk={stats['rate_latest_in_topk']:.0%}  "
            f"mean_rank={stats['mean_latest_rank']:.2f}",
            flush=True,
        )
        results.append(stats)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=["mnemoss_default", "mnemoss_decay"])
    p.add_argument("--embedder", choices=["openai", "local", "gemma"], default="openai")
    p.add_argument("--distractors", type=int, default=300)
    p.add_argument("--gap-seconds", type=float, default=60.0)
    p.add_argument("--k", type=int, default=10)
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
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "multi_step_agent_scoped",
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
    print()
    print(f"{'arm':>30}  {'latest@1':>10}  {'older@1':>10}  {'in_topk':>10}  {'mean_rank':>10}")
    print("─" * 78)
    for r in results:
        print(
            f"{r['arm']:>30}  "
            f"{r['rate_latest_at_1']:>9.0%}  "
            f"{r['rate_any_older_at_1']:>9.0%}  "
            f"{r['rate_latest_in_topk']:>9.0%}  "
            f"{r['mean_latest_rank']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
