"""Comprehensive dreaming-validation: speed + accuracy, replicated.

Final verdict run for the dreaming pipeline. Addresses two gaps from
prior validations:

  1. **Strict recall@k of original gold ids structurally pins to 0**
     when Consolidate writes summaries. Adds two new metrics that
     don't have this saturation bias:
     - **Topic-aware recall@k** — top-K hit if either a direct gold
       corpus_id is returned, OR a derived memory (summary/pattern)
       whose ``derived_from`` covers a gold member.
     - **MRR** — 1/rank of first relevant item (original or topic-
       covering derived) in top-K.
  2. **Single-shot wallclock variance** — prior runs swung topology
     speed lift by 50ms across two reps. Adds 3 reps per condition
     plus a 3-query warmup that's not timed. Reports median latency
     and per-condition std.

Four conditions × two corpora × three reps = 24 ablation cycles.

Network: requires ``OPENAI_API_KEY`` (embedder) and
``OPENROUTER_API_KEY`` (Consolidate LLM for the dreaming_on
condition only).

Usage::

    python -m bench.comprehensive_validation
    python -m bench.comprehensive_validation --corpus topology
    python -m bench.comprehensive_validation --reps 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
import tomllib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from freezegun import freeze_time

from bench._metrics import topk_cleanliness
from bench.ablate_dreaming import RetryingLLM
from mnemoss import (
    DreamerParams,
    FormulaParams,
    Mnemoss,
    OpenAIClient,
    OpenAIEmbedder,
    StorageParams,
)

N_REPS_DEFAULT = 3
N_WARMUP = 3


def _load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


# ─── metrics ──────────────────────────────────────────────────────


def _strict_recall_at_k(predicted_corpus_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(predicted_corpus_ids[:k]) & relevant)
    return hits / len(relevant)


def _topic_aware_recall_at_k(
    results: list[Any],
    reverse: dict[str, str],
    relevant: set[str],
    k: int,
) -> float:
    """Hit if either a direct gold id, OR a derived memory whose
    ``derived_from`` covers a gold member.

    Returns ``covered / |relevant|``.
    """

    if not relevant:
        return 0.0

    covered: set[str] = set()
    for r in results[:k]:
        mid = r.memory.id
        cid = reverse.get(mid)
        if cid is not None:
            if cid in relevant:
                covered.add(cid)
        else:
            # Derived memory (summary or pattern) — check derived_from.
            for parent_mid in getattr(r.memory, "derived_from", []) or []:
                parent_cid = reverse.get(parent_mid)
                if parent_cid is not None and parent_cid in relevant:
                    covered.add(parent_cid)
    return len(covered) / len(relevant)


def _mrr(
    results: list[Any],
    reverse: dict[str, str],
    relevant: set[str],
    k: int,
) -> float:
    """Mean reciprocal rank, with topic-aware matching: a derived
    memory whose ``derived_from`` covers any gold member counts as a
    hit at its rank.
    """

    if not relevant:
        return 0.0

    for rank, r in enumerate(results[:k], 1):
        mid = r.memory.id
        cid = reverse.get(mid)
        if cid is not None and cid in relevant:
            return 1.0 / rank
        if cid is None:
            for parent_mid in getattr(r.memory, "derived_from", []) or []:
                parent_cid = reverse.get(parent_mid)
                if parent_cid is not None and parent_cid in relevant:
                    return 1.0 / rank
    return 0.0


# ─── seed + run ───────────────────────────────────────────────────


def _build_llm(cfg: dict[str, Any]) -> RetryingLLM | None:
    raw = OpenAIClient(
        model=cfg["llm"]["consolidate"]["model"],
        base_url=cfg["llm"]["consolidate"]["base_url"],
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    retry_cfg = cfg["harness"].get("retry", {})
    return RetryingLLM(
        raw,
        max_attempts=int(retry_cfg.get("max_attempts", 5)),
        initial_wait=float(retry_cfg.get("initial_wait_seconds", 4.0)),
        max_wait=float(retry_cfg.get("max_wait_seconds", 60.0)),
        min_wait_between_calls=float(retry_cfg.get("min_wait_between_calls_seconds", 0.0)),
        per_call_timeout=float(retry_cfg.get("per_call_timeout_seconds", 60.0)),
    )


async def _seed(
    mem: Mnemoss, corpus: dict[str, Any], cfg: dict[str, Any], is_pressure: bool
) -> tuple[dict[str, str], datetime]:
    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )
    id_map: dict[str, str] = {}
    end_time = anchor

    if is_pressure:
        for m in corpus["memories"]:
            target = anchor + timedelta(seconds=int(m["ts_offset_seconds"]))
            with freeze_time(target, tick=True):
                mid = await mem.observe(role="user", content=m["content"])
            id_map[m["id"]] = mid
            if target > end_time:
                end_time = target
        end_time = end_time + timedelta(minutes=1)
    else:
        inc = float(cfg["harness"]["simulated_time"]["per_observe_increment_seconds"])
        for i, m in enumerate(corpus["memories"]):
            target = anchor + timedelta(seconds=i * inc)
            with freeze_time(target, tick=True):
                mid = await mem.observe(role="user", content=m["content"])
            id_map[m["id"]] = mid
        end_time = anchor + timedelta(seconds=len(corpus["memories"]) * inc + 1)

    return id_map, end_time


async def _warmup(
    mem: Mnemoss, queries: list[dict[str, Any]], k: int, include_deep: bool, auto_expand: bool
) -> None:
    """Run N_WARMUP throwaway recall calls to amortize first-call
    embedder + connection cold-start. Latencies discarded."""

    for q in queries[:N_WARMUP]:
        await mem.recall(q["query"], k=k, include_deep=include_deep, auto_expand=auto_expand)


async def _run_rep(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    condition: str,
    is_pressure: bool,
    rep_idx: int,
) -> dict[str, Any]:
    """One repetition of one condition. Fresh tempdir, fresh Mnemoss."""

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    formula = FormulaParams(**cfg["formula"])
    dreamer = DreamerParams(**cfg["dreamer"])
    llm = _build_llm(cfg) if condition == "dreaming_on" else None

    queries_with_gold = [q for q in corpus["queries"] if q["relevant_ids"]]

    with tempfile.TemporaryDirectory(prefix=f"comp-{condition}-r{rep_idx}-") as td:
        mem = Mnemoss(
            workspace=f"comp-{condition}-r{rep_idx}",
            embedding_model=embedder,
            formula=formula,
            dreamer=dreamer,
            storage=StorageParams(root=Path(td)),
            llm=llm,
        )
        try:
            print(f"    rep {rep_idx + 1} seeding...", flush=True)
            id_map, end_time = await _seed(mem, corpus, cfg, is_pressure)
            reverse = {v: k_ for k_, v in id_map.items()}

            with freeze_time(end_time, tick=True):
                if condition == "dreaming_on":
                    print(f"    rep {rep_idx + 1} dreaming...", flush=True)
                    await mem.dream(trigger="nightly")

                k = int(cfg["harness"]["recall_k"])
                include_deep = bool(cfg["harness"]["include_deep"])
                auto_expand = bool(cfg["harness"]["auto_expand"])

                # Warmup — latencies discarded.
                await _warmup(mem, queries_with_gold, k, include_deep, auto_expand)

                per_query: list[dict[str, Any]] = []
                clean_count = 0
                cleanable_n = 0
                for q in queries_with_gold:
                    t0 = time.perf_counter()
                    results = await mem.recall(
                        q["query"], k=k, include_deep=include_deep, auto_expand=auto_expand
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    relevant = set(q["relevant_ids"])
                    predicted_corpus_ids = [
                        reverse[r.memory.id] for r in results if r.memory.id in reverse
                    ]
                    strict = _strict_recall_at_k(predicted_corpus_ids, relevant, k)
                    topic = _topic_aware_recall_at_k(results, reverse, relevant, k)
                    mrr = _mrr(results, reverse, relevant, k)

                    is_clean: bool | None = None
                    if "junk_ids" in q and q["junk_ids"]:
                        cleanable_n += 1
                        is_clean = topk_cleanliness(predicted_corpus_ids, set(q["junk_ids"]), k=k)
                        if is_clean:
                            clean_count += 1

                    per_query.append(
                        {
                            "query": q["query"],
                            "kind": q.get("kind", "unknown"),
                            "strict_recall_at_k": round(strict, 4),
                            "topic_recall_at_k": round(topic, 4),
                            "mrr": round(mrr, 4),
                            "latency_ms": round(latency_ms, 2),
                            "clean": is_clean,
                            "predicted_corpus_ids": predicted_corpus_ids[:k],
                        }
                    )

                latencies = [q["latency_ms"] for q in per_query]
                return {
                    "condition": condition,
                    "rep": rep_idx,
                    "n_queries": len(per_query),
                    "mean_strict_recall": round(
                        statistics.mean(q["strict_recall_at_k"] for q in per_query), 4
                    ),
                    "mean_topic_recall": round(
                        statistics.mean(q["topic_recall_at_k"] for q in per_query), 4
                    ),
                    "mean_mrr": round(statistics.mean(q["mrr"] for q in per_query), 4),
                    "mean_latency_ms": round(statistics.mean(latencies), 2),
                    "p50_latency_ms": round(statistics.median(latencies), 2),
                    "p95_latency_ms": round(
                        sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0,
                        2,
                    ),
                    "cleanliness": (
                        round(clean_count / cleanable_n, 4) if cleanable_n > 0 else None
                    ),
                    "per_query": per_query,
                }
        finally:
            await mem.close()


def _oracle(corpus: dict[str, Any], *, kind: str) -> dict[str, Any]:
    """Synthetic oracle row.

    kind="evidence": strict_recall=1.0 (every gold id retrieved).
    kind="answer":   topic_recall=1.0 + MRR=1.0 (every query's first
                     result is a relevant item).
    Both: latency=0ms (theoretical floor), cleanliness=1.0.
    """

    queries = [q for q in corpus["queries"] if q["relevant_ids"]]
    n = len(queries)
    base = {
        "condition": f"oracle_{kind}",
        "rep": 0,
        "n_queries": n,
        "mean_latency_ms": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "cleanliness": 1.0,
        "per_query": [],
    }
    if kind == "evidence":
        base.update(mean_strict_recall=1.0, mean_topic_recall=1.0, mean_mrr=1.0)
    else:  # "answer"
        base.update(
            mean_strict_recall=0.0,  # answer-oracle doesn't return originals
            mean_topic_recall=1.0,
            mean_mrr=1.0,
        )
    return base


# ─── condition aggregation ───────────────────────────────────────


async def _run_condition(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    condition: str,
    is_pressure: bool,
    n_reps: int,
) -> dict[str, Any]:
    """Run N reps of one condition; return per-rep rows + aggregate."""

    if condition.startswith("oracle_"):
        kind = condition.split("_", 1)[1]
        return {**_oracle(corpus, kind=kind), "reps": []}

    print(f"  → condition {condition!r}", flush=True)
    reps: list[dict[str, Any]] = []
    for i in range(n_reps):
        row = await _run_rep(corpus, cfg, condition=condition, is_pressure=is_pressure, rep_idx=i)
        reps.append(row)
        print(
            f"    rep {i + 1}: strict={row['mean_strict_recall']:.3f} "
            f"topic={row['mean_topic_recall']:.3f} mrr={row['mean_mrr']:.3f} "
            f"lat={row['mean_latency_ms']:.1f}ms",
            flush=True,
        )

    # Median across reps for the headline numbers.
    def median_of(field: str) -> float:
        return round(statistics.median(r[field] for r in reps), 4)

    aggregate = {
        "condition": condition,
        "n_reps": n_reps,
        "n_queries": reps[0]["n_queries"],
        "mean_strict_recall": median_of("mean_strict_recall"),
        "mean_topic_recall": median_of("mean_topic_recall"),
        "mean_mrr": median_of("mean_mrr"),
        "mean_latency_ms": median_of("mean_latency_ms"),
        "p50_latency_ms": median_of("p50_latency_ms"),
        "p95_latency_ms": median_of("p95_latency_ms"),
        "cleanliness": (
            round(
                statistics.median(r["cleanliness"] for r in reps if r["cleanliness"] is not None),
                4,
            )
            if any(r["cleanliness"] is not None for r in reps)
            else None
        ),
        "latency_std_ms": (
            round(statistics.stdev(r["mean_latency_ms"] for r in reps), 2) if n_reps >= 2 else 0.0
        ),
        "reps": reps,
    }
    return aggregate


# ─── plotting ─────────────────────────────────────────────────────


CONDITION_ORDER = ["dreaming_off", "dreaming_on", "oracle_evidence", "oracle_answer"]
CONDITION_COLORS = {
    "dreaming_off": "#d62728",
    "dreaming_on": "#1f77b4",
    "oracle_evidence": "#7f7f7f",
    "oracle_answer": "#2ca02c",
}


def _plot(
    topology_rows: list[dict[str, Any]],
    pressure_rows: list[dict[str, Any]],
    out: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        (axes[0][0], topology_rows, "topology — accuracy", "accuracy"),
        (axes[0][1], topology_rows, "topology — latency", "latency"),
        (axes[1][0], pressure_rows, "pressure — accuracy", "accuracy"),
        (axes[1][1], pressure_rows, "pressure — latency", "latency"),
    ]

    for ax, rows, title, kind in panels:
        by_cond = {r["condition"]: r for r in rows}
        if kind == "accuracy":
            metrics = [
                ("mean_topic_recall", "topic-aware recall@10"),
                ("mean_mrr", "MRR"),
                ("mean_strict_recall", "strict recall@10"),
            ]
            x = list(range(len(CONDITION_ORDER)))
            width = 0.25
            for i, (key, label) in enumerate(metrics):
                values = [by_cond[c][key] for c in CONDITION_ORDER]
                ax.bar(
                    [xi + (i - 1) * width for xi in x],
                    values,
                    width=width,
                    label=label,
                    edgecolor="black",
                    linewidth=0.4,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(CONDITION_ORDER, rotation=15, ha="right")
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("score")
            ax.legend(loc="upper left", fontsize=9)
        else:
            values = [by_cond[c]["mean_latency_ms"] for c in CONDITION_ORDER]
            stds = [by_cond[c].get("latency_std_ms", 0.0) for c in CONDITION_ORDER]
            colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
            bars = ax.bar(
                CONDITION_ORDER,
                values,
                yerr=stds,
                color=colors,
                edgecolor="black",
                linewidth=0.4,
                capsize=8,
            )
            ax.set_ylabel("mean latency (ms)")
            ax.tick_params(axis="x", rotation=15)
            for bar, val in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + max(values) * 0.02,
                    f"{val:.0f}ms",
                    ha="center",
                    fontsize=9,
                )
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Comprehensive dreaming validation (median of 3 reps)", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# ─── main ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comprehensive validation of the dreaming pipeline."
    )
    parser.add_argument(
        "--corpus",
        choices=["topology", "pressure", "both"],
        default="both",
    )
    parser.add_argument("--reps", type=int, default=N_REPS_DEFAULT)
    parser.add_argument("--config", default="bench/ablate_dreaming.toml")
    parser.add_argument("--out-dir", default="bench/results")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    corpora_to_run: list[tuple[str, str]] = []
    if args.corpus in ("topology", "both"):
        corpora_to_run.append(("topology", "bench/fixtures/topology_corpus.json"))
    if args.corpus in ("pressure", "both"):
        corpora_to_run.append(("pressure", "bench/fixtures/pressure_corpus_seed42.jsonl"))

    by_corpus: dict[str, list[dict[str, Any]]] = {}

    for corpus_name, corpus_path in corpora_to_run:
        corpus = _load_corpus(Path(corpus_path))
        is_pressure = corpus_name == "pressure"
        n_queries_with_gold = sum(1 for q in corpus["queries"] if q["relevant_ids"])
        print(
            f"\n=== {corpus_name} corpus ({len(corpus['memories'])} memories, "
            f"{n_queries_with_gold} labeled queries × {args.reps} reps) ===",
            flush=True,
        )

        rows: list[dict[str, Any]] = []
        for cond in CONDITION_ORDER:
            agg = asyncio.run(
                _run_condition(
                    corpus, cfg, condition=cond, is_pressure=is_pressure, n_reps=args.reps
                )
            )
            rows.append(agg)

        by_corpus[corpus_name] = rows

        # Per-corpus JSONL.
        out_jsonl = Path(args.out_dir) / f"comprehensive_{corpus_name}.jsonl"
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"  wrote {out_jsonl}", flush=True)

        # Per-corpus summary table.
        print(f"\n  {corpus_name} summary (median of {args.reps} reps):", flush=True)
        print(
            f"    {'condition':<18} {'topic@10':>9} {'MRR':>7} {'strict@10':>10} "
            f"{'lat ms':>9} {'std ms':>8}",
            flush=True,
        )
        print("    " + "-" * 68, flush=True)
        for r in rows:
            print(
                f"    {r['condition']:<18} {r['mean_topic_recall']:>9.3f} "
                f"{r['mean_mrr']:>7.3f} {r['mean_strict_recall']:>10.3f} "
                f"{r['mean_latency_ms']:>9.1f} {r.get('latency_std_ms', 0.0):>8.1f}",
                flush=True,
            )

    # Combined plot.
    if "topology" in by_corpus and "pressure" in by_corpus:
        out_png = Path(args.out_dir) / "comprehensive_validation.png"
        _plot(by_corpus["topology"], by_corpus["pressure"], out_png)

    # Final verdict.
    print("\n=== FINAL VERDICT ===", flush=True)
    for corpus_name, rows in by_corpus.items():
        by_cond = {r["condition"]: r for r in rows}
        off = by_cond["dreaming_off"]
        on = by_cond["dreaming_on"]
        ans = by_cond["oracle_answer"]

        topic_lift = (on["mean_topic_recall"] - off["mean_topic_recall"]) * 100.0
        mrr_lift = (on["mean_mrr"] - off["mean_mrr"]) * 100.0
        speed_lift_ms = on["mean_latency_ms"] - off["mean_latency_ms"]
        speed_lift_pct = (
            (speed_lift_ms / off["mean_latency_ms"]) * 100.0 if off["mean_latency_ms"] else 0.0
        )
        gap_to_answer = (ans["mean_topic_recall"] - on["mean_topic_recall"]) * 100.0

        print(f"\n  {corpus_name}:", flush=True)
        print(
            f"    topic-aware recall: {topic_lift:+.2f}pp  "
            f"({off['mean_topic_recall']:.3f} → {on['mean_topic_recall']:.3f}, "
            f"oracle 1.000, gap {gap_to_answer:+.2f}pp)",
            flush=True,
        )
        print(
            f"    MRR:                {mrr_lift:+.2f}pp  "
            f"({off['mean_mrr']:.3f} → {on['mean_mrr']:.3f})",
            flush=True,
        )
        print(
            f"    latency:            {speed_lift_ms:+.1f}ms ({speed_lift_pct:+.1f}%)  "
            f"({off['mean_latency_ms']:.1f}ms → {on['mean_latency_ms']:.1f}ms, oracle 0ms)",
            flush=True,
        )
        if on.get("cleanliness") is not None:
            clean_off = off.get("cleanliness") or 0.0
            clean_on = on["cleanliness"]
            print(
                f"    cleanliness:        {(clean_on - clean_off) * 100.0:+.1f}pp  "
                f"({clean_off:.3f} → {clean_on:.3f})",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
