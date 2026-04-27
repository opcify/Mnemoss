"""Dispose REBUILD: sweep ``delta`` to find a value where Dispose
actually tombstones memories on the pressure corpus.

Dispose triggers when ``max_A_i < tau - delta``. With the pinned
tau=-10 and the shipped default delta=1.0, Dispose only triggers
below -11. The pressure-corpus full-matrix run showed this is a
no-op — no memory's max_A_i drops below -11 in 30 simulated days.

This sweep tries lower delta values (including 0.0 — fire on any
memory that crosses tau itself) to find a setting that produces a
measurable tombstone count + a top-K cleanliness lift.

No LLM, no Consolidate. Runs Replay + Cluster + Dispose so Dispose
has candidates to evaluate. Also Rebalance so tier migration is
included.

Usage::

    python -m bench.sweep_dispose
    python -m bench.sweep_dispose --deltas 0.0 0.3 0.5 1.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import tomllib
from freezegun import freeze_time

from bench._metrics import topk_cleanliness
from mnemoss import (
    DreamerParams,
    FormulaParams,
    Mnemoss,
    OpenAIEmbedder,
    StorageParams,
)


async def _measure_dispose(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    delta: float,
    run_dispose: bool,
) -> dict[str, Any]:
    """Seed pressure corpus, optionally run Dispose, measure outcomes."""

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    formula_kwargs = dict(cfg["formula"])
    formula_kwargs["delta"] = delta
    formula = FormulaParams(**formula_kwargs)
    dreamer = DreamerParams(**cfg["dreamer"])

    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )

    with tempfile.TemporaryDirectory(prefix="dispose-sweep-") as td:
        mem = Mnemoss(
            workspace=f"sweep-d{delta}",
            embedding_model=embedder,
            formula=formula,
            dreamer=dreamer,
            storage=StorageParams(root=Path(td)),
        )
        try:
            id_map: dict[str, str] = {}
            end_time = anchor
            for m in corpus["memories"]:
                target = anchor + timedelta(seconds=int(m["ts_offset_seconds"]))
                with freeze_time(target, tick=True):
                    mid = await mem.observe(role="user", content=m["content"])
                if mid is None:
                    continue
                id_map[m["id"]] = mid
                if target > end_time:
                    end_time = target
            end_time = end_time + timedelta(minutes=1)

            phases = {"replay", "cluster", "rebalance", "dispose"} if run_dispose else set()
            with freeze_time(end_time, tick=True):
                report = await mem.dream(trigger="nightly", phases=phases)

                # Count tombstones: memories that no longer materialize.
                await mem._ensure_open()
                store = mem._store
                assert store is not None
                mnemoss_ids = list(id_map.values())
                materialized = await store.materialize_memories(mnemoss_ids)
                live_ids = {m.id for m in materialized}
                tombstoned = sum(1 for mid in mnemoss_ids if mid not in live_ids)

                # Tombstoned by utility bucket.
                utility_by_corpus = {m["id"]: m["utility"] for m in corpus["memories"]}
                tomb_by_util = {"high": 0, "medium": 0, "low": 0}
                for corpus_id, mnemoss_id in id_map.items():
                    if mnemoss_id not in live_ids:
                        tomb_by_util[utility_by_corpus[corpus_id]] += 1

                # Cleanliness on the corpus's adversarial queries.
                k = int(cfg["harness"]["recall_k"])
                include_deep = bool(cfg["harness"]["include_deep"])
                auto_expand = bool(cfg["harness"]["auto_expand"])
                reverse = {v: k_ for k_, v in id_map.items()}
                clean_count = 0
                for q in corpus["queries"]:
                    if not q.get("junk_ids"):
                        continue
                    results = await mem.recall(
                        q["query"],
                        k=k,
                        include_deep=include_deep,
                        auto_expand=auto_expand,
                    )
                    predicted = []
                    for r in results:
                        cid = reverse.get(r.memory.id)
                        if cid is not None:
                            predicted.append(cid)
                    if topk_cleanliness(predicted, set(q["junk_ids"]), k=k):
                        clean_count += 1
                cleanable_n = sum(1 for q in corpus["queries"] if q.get("junk_ids"))
                cleanliness = clean_count / cleanable_n if cleanable_n else 0.0

                # Dispose's own reported counts.
                dispose_outcome = next(
                    (o for o in report.outcomes if o.phase.value == "dispose"), None
                )
                dispose_details = dispose_outcome.details if dispose_outcome else {}

                return {
                    "delta": delta,
                    "ran_dispose": run_dispose,
                    "tombstoned_total": tombstoned,
                    "tombstoned_by_utility": tomb_by_util,
                    "cleanliness": round(cleanliness, 4),
                    "dispose_phase_status": (
                        dispose_outcome.status if dispose_outcome else "<not run>"
                    ),
                    "dispose_details": {
                        k_: dispose_details.get(k_)
                        for k_ in (
                            "scanned",
                            "disposed",
                            "activation_dead",
                            "redundant",
                            "protected",
                        )
                    },
                }
        finally:
            await mem.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep delta to find a value where Dispose tombstones memories "
            "and lifts top-K cleanliness on the pressure corpus."
        ),
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 0.5, 1.0],
        help="delta values to try (default: 0.0 0.3 0.5 1.0).",
    )
    parser.add_argument(
        "--corpus",
        default="bench/fixtures/pressure_corpus_seed42.jsonl",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
    )
    parser.add_argument(
        "--out",
        default="bench/results/dispose_sweep.json",
    )
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    with Path(args.config).open("rb") as fh:
        cfg = tomllib.load(fh)

    print(f"sweeping delta across {args.deltas} on {args.corpus}", flush=True)
    print("(no_dispose baseline first for cleanliness comparison)", flush=True)

    baseline = asyncio.run(
        _measure_dispose(corpus, cfg, delta=cfg["formula"]["delta"], run_dispose=False)
    )
    print(
        f"  baseline (no dispose): cleanliness={baseline['cleanliness']:.4f}",
        flush=True,
    )

    rows: list[dict[str, Any]] = [baseline]
    for delta in args.deltas:
        row = asyncio.run(_measure_dispose(corpus, cfg, delta=delta, run_dispose=True))
        rows.append(row)
        delta_pp = (row["cleanliness"] - baseline["cleanliness"]) * 100.0
        print(
            f"  delta={delta:>4.1f}: "
            f"tombstoned={row['tombstoned_total']:>3} "
            f"(high={row['tombstoned_by_utility']['high']} "
            f"med={row['tombstoned_by_utility']['medium']} "
            f"low={row['tombstoned_by_utility']['low']})  "
            f"cleanliness={row['cleanliness']:.4f} "
            f"({delta_pp:+.1f}pp vs baseline)",
            flush=True,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)

    best = max(rows[1:], key=lambda r: r["cleanliness"])
    delta_pp = (best["cleanliness"] - baseline["cleanliness"]) * 100.0
    print(
        f"\nbest: delta={best['delta']:.1f} cleanliness={best['cleanliness']:.4f} "
        f"({delta_pp:+.1f}pp vs baseline, {best['tombstoned_total']} tombstoned)",
        flush=True,
    )
    # Pre-registered KEEP threshold per docs/dreaming-decision.md:
    # "cleanliness drops by > 30% of queries" when Dispose ablated.
    # Inverted: Dispose is KEEP if running it improves cleanliness by
    # >= 30% of queries vs not running.
    if delta_pp >= 30.0:
        print("VERDICT: KEEP threshold (≥30pp cleanliness lift) cleared.", flush=True)
    elif delta_pp >= 10.0:
        print("VERDICT: still REBUILD (10pp ≤ delta < 30pp).", flush=True)
    else:
        print("VERDICT: CUT (cleanliness lift <10pp).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
