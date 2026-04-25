"""Does Dream P7 lift recall under a *popularity* access pattern?

The sibling ``bench_dream_lift.py`` experiment produced a null result on
LoCoMo because each question has its own disjoint gold set — there is
no popularity skew for Dream to amplify. Production memory isn't like
that. Some memories (the user's name, the current project, the
standing kitchen-is-closed rule) get pulled thousands of times; most
memories get pulled zero times. That skew is Dream's job.

This benchmark fabricates that skew by *priming*: the same hot
memory's content is fed back as a recall query ``repetitions`` times,
which walks ``access_history`` up for that memory specifically. After
Dream P7 Rebalance, hot-memory ``idx_priority`` should land near
σ(~3) ≈ 0.95, vs σ(η₀)=0.73 for cold memories. With
``fast_index_priority_weight > 0`` that gap tiebreaks hot memories
ahead of semantically similar distractors at recall time.

Experimental setup
------------------

1. Ingest LoCoMo conv-26 padded to N with distractors.
2. Configure ``mnemoss_rocket`` (fast-index recall, pure ANN +
   ``idx_priority`` lookup, no ACT-R on the read path).
3. **Priming phase** — for each gold memory, run R recall queries
   using the memory's own content as the query. Every recall
   reconsolidates, so ``access_history`` for hot memories stacks up
   to ~R hits; distractors only catch the occasional top-k residue.
4. **Call ``mem.rebalance()`` directly** — this is the standalone P7
   Rebalance entry point. ``idx_priority`` recomputes from
   ``access_history``; hot memories sigmoid-saturate near 0.95.
   (``mem.dream(trigger="idle")`` does NOT run rebalance — only the
   nightly trigger includes P7 in the phase chain. Calling rebalance
   directly isolates the effect we're measuring.)
5. **Scored phase** — run the actual LoCoMo questions (different
   phrasing than the priming queries), measure recall@10.

Three arms:
- **no_prime_no_dream:** baseline — no history, no dream, flat
  initial ``idx_priority`` everywhere.
- **prime_no_dream:** priming builds ``access_history`` but
  ``idx_priority`` stays stale (reconsolidation doesn't sync it;
  only Dream does). This is the *control* — should equal
  no_prime_no_dream modulo noise, and if it doesn't, we have a bug.
- **prime_and_dream:** priming + Dream P7. The target arm.

Output: per (arm, N, repetitions) row. Dream's lift is
``prime_and_dream - prime_no_dream``.

Usage::

    python -m bench.bench_repeated_query \\
        --embedder local \\
        --sizes 3000 \\
        --repetitions 1 5 20 \\
        --priority-weight 1.0 \\
        --gold-conversation conv-26 \\
        --out bench/results/repeated_query.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    k: int,
) -> float:
    """Run each gold query; return mean recall@k on gold."""

    recalls: list[float] = []
    for q in gold_queries:
        gold_mids = q["_gold_mids"]
        if not gold_mids:
            continue
        results = await mem.recall(q["question"], k=k, reconsolidate=True)
        returned = {r.memory.id for r in results}
        hit = len(returned & gold_mids)
        recalls.append(hit / len(gold_mids))
    return statistics.mean(recalls) if recalls else 0.0


async def _build_reference_workspace(
    *,
    scale_n: int,
    gold_conversation: str,
    embedder: Any,
) -> tuple[Path, list[dict], list[dict], dict[str, str]]:
    """Ingest the corpus once into a reference tempdir.

    Subsequent arms snapshot this dir (``shutil.copytree``) and open
    their own writable copy, so the ~2 min ingest work is paid exactly
    once per (scale_n, embedder) instead of once per arm. Returns the
    reference dir path plus the shared corpus artifacts each arm needs.
    """

    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories,
        queries,
        gold_conversation_id=gold_conversation,
        scale_n=scale_n,
    )

    ref_dir = Path(tempfile.mkdtemp(prefix="mnemoss_repeatq_ref_"))
    print(
        f"[repeated-query] building reference workspace at N={scale_n} "
        f"(one-shot ingest, shared across all arms)…",
        flush=True,
    )
    mem = Mnemoss(
        workspace="repeated_query",
        embedding_model=embedder,
        # Formula choices don't affect stored vectors/content; set
        # the rocket preset here so the reference matches the
        # schema pin arms will open with.
        formula=FormulaParams(
            noise_scale=0.0,
            eta_0=0.0,
            d=0.01,
            use_fast_index_recall=True,
            fast_index_semantic_weight=1.0,
            fast_index_priority_weight=1.0,
        ),
        storage=StorageParams(root=ref_dir),
    )
    try:
        dia_to_mid = await _ingest(mem, padded_mems)
    finally:
        # Close releases the cross-process lock so copytree sees a
        # quiescent dir. The lock file itself may persist — it's just
        # a flock target; new Mnemoss instances will re-acquire on
        # their own copies without conflict.
        await mem.close()

    # Attach _gold_mids once — dia_to_mid is shared across arms so the
    # mapping is identical everywhere.
    for q in gold_queries:
        q["_gold_mids"] = {dia_to_mid[d] for d in q["relevant_dia_ids"] if d in dia_to_mid}

    return ref_dir, padded_mems, gold_queries, dia_to_mid


async def _run_arm(
    *,
    arm: str,
    scale_n: int,
    repetitions: int,
    embedder_choice: str,
    embedder: Any,
    ref_dir: Path,
    padded_mems: list[dict],
    gold_queries: list[dict],
    dia_to_mid: dict[str, str],
    k: int,
    priming_k: int,
    priority_weight: float,
    semantic_weight: float,
) -> dict:
    """Run one experimental arm on a snapshot of the reference workspace.

    ``arm`` controls behaviour:
    - ``no_prime_no_dream``: skip priming, skip Dream.
    - ``prime_no_dream``:    run priming, skip Dream.
    - ``prime_and_dream``:   run priming, fire P7 Rebalance.
    """

    # Snapshot: fresh writable copy of the ingested workspace.
    # copytree requires the destination to NOT exist, so we mkdtemp for
    # a unique name and immediately rmtree the empty dir.
    arm_dir = Path(tempfile.mkdtemp(prefix="mnemoss_repeatq_arm_"))
    shutil.rmtree(arm_dir)
    shutil.copytree(ref_dir, arm_dir)

    try:
        mem = Mnemoss(
            workspace="repeated_query",
            embedding_model=embedder,
            formula=FormulaParams(
                noise_scale=0.0,
                eta_0=0.0,
                d=0.01,
                use_fast_index_recall=True,
                fast_index_semantic_weight=semantic_weight,
                fast_index_priority_weight=priority_weight,
            ),
            storage=StorageParams(root=arm_dir),
        )

        # Hot set: union of gold memories across all gold questions.
        # Same value for every arm (derived from shared dia_to_mid).
        hot_dia_ids: set[str] = set()
        for q in gold_queries:
            hot_dia_ids.update(q["relevant_dia_ids"])
        hot_dia_ids = {d for d in hot_dia_ids if d in dia_to_mid}

        dia_to_text = {m["dia_id"]: m["text"] for m in padded_mems}

        if arm != "no_prime_no_dream":
            # Priming phase: for each hot memory, run R recalls using
            # its content as the query. Self-similarity guarantees the
            # memory lands in top-1; reconsolidation bumps its
            # access_history by ~R. Distractors catch intermittent
            # top-k residue — net effect is a real popularity skew
            # toward the hot set.
            for dia in hot_dia_ids:
                text = dia_to_text.get(dia)
                if not text:
                    continue
                for _ in range(repetitions):
                    await mem.recall(text, k=priming_k, reconsolidate=True)

        if arm == "prime_and_dream":
            # Call P7 Rebalance directly. dream(trigger="idle") does
            # NOT run rebalance — the idle phase chain is Replay →
            # Cluster → Consolidate → Relations, with no P7. Only
            # the nightly trigger includes Rebalance. Calling
            # mem.rebalance() lets us isolate the effect we care
            # about (idx_priority recompute from access_history)
            # without also firing off LLM-backed Consolidate work.
            await mem.rebalance()

        # Scored phase: actual LoCoMo questions, measure recall@k.
        recall = await _score_queries(mem, gold_queries, k=k)
        await mem.close()
    finally:
        shutil.rmtree(arm_dir, ignore_errors=True)

    return {
        "arm": arm,
        "scale_n": scale_n,
        "repetitions": repetitions,
        "priority_weight": priority_weight,
        "semantic_weight": semantic_weight,
        "priming_k": priming_k,
        "embedder": embedder_choice,
        "hot_set_size": len(hot_dia_ids),
        "recall_at_k": round(recall, 4),
        "n_queries": len(gold_queries),
    }


async def _run_all(
    *,
    sizes: list[int],
    repetitions_list: list[int],
    embedder_choice: str,
    gold_conversation: str,
    k: int,
    priming_k: int,
    priority_weight: float,
    semantic_weight: float,
) -> list[dict]:
    out: list[dict] = []
    arms = ["no_prime_no_dream", "prime_no_dream", "prime_and_dream"]

    # Load the embedder once and reuse across every arm + reference
    # ingest. MiniLM takes ~5-10s to load; at 7+ arms the savings add
    # up. Safe to share: the embedder is a stateless wrapper around a
    # loaded model, and Mnemoss.close() does not tear it down.
    embedder = _resolve_embedder(embedder_choice)

    for n in sizes:
        ref_dir, padded_mems, gold_queries, dia_to_mid = await _build_reference_workspace(
            scale_n=n,
            gold_conversation=gold_conversation,
            embedder=embedder,
        )
        try:
            for reps in repetitions_list:
                for arm in arms:
                    # no_prime_no_dream doesn't depend on repetitions —
                    # every R value would produce the same recall on
                    # the same reference snapshot. Run it once and
                    # clone the row for remaining R values.
                    if arm == "no_prime_no_dream" and reps != repetitions_list[0]:
                        prior = next(
                            (
                                r
                                for r in out
                                if r["arm"] == arm
                                and r["scale_n"] == n
                                and r["embedder"] == embedder_choice
                            ),
                            None,
                        )
                        if prior is not None:
                            clone = dict(prior)
                            clone["repetitions"] = reps
                            out.append(clone)
                            print(
                                f"[repeated-query] arm={arm}  N={n}  R={reps}  "
                                f"(reusing R={repetitions_list[0]} result)",
                                flush=True,
                            )
                            print(
                                f"  → recall@10 = {clone['recall_at_k']:.4f}",
                                flush=True,
                            )
                            continue

                    print(
                        f"[repeated-query] arm={arm}  N={n}  R={reps}  pri_w={priority_weight}",
                        flush=True,
                    )
                    row = await _run_arm(
                        arm=arm,
                        scale_n=n,
                        repetitions=reps,
                        embedder_choice=embedder_choice,
                        embedder=embedder,
                        ref_dir=ref_dir,
                        padded_mems=padded_mems,
                        gold_queries=gold_queries,
                        dia_to_mid=dia_to_mid,
                        k=k,
                        priming_k=priming_k,
                        priority_weight=priority_weight,
                        semantic_weight=semantic_weight,
                    )
                    out.append(row)
                    print(f"  → recall@10 = {row['recall_at_k']:.4f}", flush=True)
        finally:
            shutil.rmtree(ref_dir, ignore_errors=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "nomic", "gemma", "fake"],
        default="local",
    )
    p.add_argument("--sizes", type=int, nargs="+", default=[3000])
    p.add_argument(
        "--repetitions",
        type=int,
        nargs="+",
        default=[1, 5, 20],
        help="Per-hot-memory priming count. Sweep to find where lift emerges.",
    )
    p.add_argument("--gold-conversation", default="conv-26")
    p.add_argument("--k", type=int, default=10, help="Scoring top-k.")
    p.add_argument(
        "--priming-k",
        type=int,
        default=10,
        help="Top-k for priming recalls. Lower = less distractor diffusion.",
    )
    p.add_argument(
        "--priority-weight",
        type=float,
        default=1.0,
        help="fast_index_priority_weight — must be > 0 for Dream to matter.",
    )
    p.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="fast_index_semantic_weight. Tune down to let priority dominate.",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if args.priority_weight <= 0:
        raise SystemExit("--priority-weight must be > 0 for Dream to affect ranking")
    if any(r < 1 for r in args.repetitions):
        raise SystemExit("--repetitions values must be >= 1")

    started = datetime.now(timezone.utc)
    results = asyncio.run(
        _run_all(
            sizes=args.sizes,
            repetitions_list=args.repetitions,
            embedder_choice=args.embedder,
            gold_conversation=args.gold_conversation,
            k=args.k,
            priming_k=args.priming_k,
            priority_weight=args.priority_weight,
            semantic_weight=args.semantic_weight,
        )
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "chart": "repeated_query",
                "embedder": args.embedder,
                "sizes": args.sizes,
                "repetitions": args.repetitions,
                "priority_weight": args.priority_weight,
                "semantic_weight": args.semantic_weight,
                "priming_k": args.priming_k,
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

    # Terminal table. Group by (N, R) so you can eyeball the lift
    # prime_and_dream - prime_no_dream row by row.
    print()
    print(f"{'arm':>22}  {'N':>6}  {'R':>4}  {'recall@k':>10}")
    print("─" * 52)
    for r in results:
        print(
            f"{r['arm']:>22}  {r['scale_n']:>6}  {r['repetitions']:>4}  {r['recall_at_k']:>10.4f}"
        )

    # Dream-lift summary: for each (N, R), report the delta.
    print()
    print("DREAM LIFT (prime_and_dream − prime_no_dream):")
    print(f"{'N':>6}  {'R':>4}  {'lift':>8}")
    print("─" * 24)
    by_key: dict[tuple[int, int], dict[str, float]] = {}
    for r in results:
        key = (r["scale_n"], r["repetitions"])
        by_key.setdefault(key, {})[r["arm"]] = r["recall_at_k"]
    for (n, reps), arms in sorted(by_key.items()):
        lift = arms.get("prime_and_dream", 0.0) - arms.get("prime_no_dream", 0.0)
        sign = "+" if lift >= 0 else ""
        print(f"{n:>6}  {reps:>4}  {sign}{lift:>7.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
