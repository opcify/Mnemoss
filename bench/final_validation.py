"""Final validation: does dreaming improve recall speed + accuracy?
Can it reach the oracle ceiling?

Three conditions per corpus:

  1. **dreaming_off** — observe + recall, no dream phases. The
     baseline cost / accuracy of pure embedding + ACT-R recall.
  2. **full** — observe + nightly dream + recall. Five phases now
     (Relations was removed 2026-04-27): replay → cluster →
     consolidate → rebalance → dispose.
  3. **oracle** — synthetic upper bound. Accuracy = 1.0 (every
     gold relevant_id always in top-K, by construction). Latency
     is reported as 0 — the theoretical "we already knew the
     answer" floor. Real systems can't reach this; the gap to it
     is what we want to measure.

Measures per query:

  - **recall@k** of original gold corpus_ids — Consolidate's
    summaries and Cluster's representatives are NOT counted as hits
    even if they cover the same topic. This is the strict "did we
    return the exact gold ids" metric.
  - **wall-clock latency** of ``mem.recall()`` — measured around
    the awaited call, in milliseconds. Captures the cascade across
    HOT/WARM/COLD/DEEP tiers; Rebalance's tier migration should
    show up here as faster ``full`` latencies than ``dreaming_off``.

Output:

  - ``bench/results/final_validation.jsonl`` — per-query rows
    + a summary row at the end
  - ``bench/results/final_validation.png`` — 2-panel chart, accuracy
    + latency, with oracle ceiling marked

Network: requires ``OPENAI_API_KEY`` (embedder) and
``OPENROUTER_API_KEY`` (Consolidate LLM for the ``full`` condition).

Usage::

    python -m bench.final_validation
    python -m bench.final_validation --corpus pressure
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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

from bench.ablate_dreaming import RetryingLLM
from mnemoss import (
    DreamerParams,
    FormulaParams,
    Mnemoss,
    OpenAIClient,
    OpenAIEmbedder,
    StorageParams,
)


def _load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


# ─── seed + measure ───────────────────────────────────────────────


async def _seed_and_measure(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    condition: str,
    is_pressure: bool,
) -> dict[str, Any]:
    """Seed corpus, optionally run dream(trigger='nightly'), recall
    each query timing wall-clock latency. Return per-query rows +
    aggregates.
    """

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    formula = FormulaParams(**cfg["formula"])
    dreamer = DreamerParams(**cfg["dreamer"])

    llm = None
    if condition == "full":
        raw = OpenAIClient(
            model=cfg["llm"]["consolidate"]["model"],
            base_url=cfg["llm"]["consolidate"]["base_url"],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        retry_cfg = cfg["harness"].get("retry", {})
        llm = RetryingLLM(
            raw,
            max_attempts=int(retry_cfg.get("max_attempts", 5)),
            initial_wait=float(retry_cfg.get("initial_wait_seconds", 4.0)),
            max_wait=float(retry_cfg.get("max_wait_seconds", 60.0)),
            min_wait_between_calls=float(retry_cfg.get("min_wait_between_calls_seconds", 0.0)),
            per_call_timeout=float(retry_cfg.get("per_call_timeout_seconds", 60.0)),
        )

    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )

    with tempfile.TemporaryDirectory(prefix=f"final-{condition}-") as td:
        mem = Mnemoss(
            workspace=f"final-{condition}",
            embedding_model=embedder,
            formula=formula,
            dreamer=dreamer,
            storage=StorageParams(root=Path(td)),
            llm=llm,
        )
        try:
            print(f"  [{condition}] seeding {len(corpus['memories'])} memories...", flush=True)

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

            with freeze_time(end_time, tick=True):
                # Dream conditional on the condition.
                if condition == "full":
                    print(f"  [{condition}] running dream(nightly)...", flush=True)
                    await mem.dream(trigger="nightly")
                elif condition == "dreaming_off":
                    pass  # No dream.

                # Time every query.
                k = int(cfg["harness"]["recall_k"])
                include_deep = bool(cfg["harness"]["include_deep"])
                auto_expand = bool(cfg["harness"]["auto_expand"])
                reverse = {v: k_ for k_, v in id_map.items()}

                per_query: list[dict[str, Any]] = []
                for q in corpus["queries"]:
                    if not q["relevant_ids"]:
                        # Skip negative queries — they're for cleanliness,
                        # not for accuracy/latency comparison.
                        continue
                    t0 = time.perf_counter()
                    results = await mem.recall(
                        q["query"], k=k, include_deep=include_deep, auto_expand=auto_expand
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                    predicted = []
                    for r in results:
                        cid = reverse.get(r.memory.id)
                        if cid is not None:
                            predicted.append(cid)

                    relevant = set(q["relevant_ids"])
                    hits = len(set(predicted[:k]) & relevant)
                    r_at_k = hits / len(relevant) if relevant else 0.0

                    per_query.append(
                        {
                            "query": q["query"],
                            "kind": q.get("kind", "unknown"),
                            "recall_at_k": round(r_at_k, 4),
                            "latency_ms": round(latency_ms, 2),
                            "predicted_corpus_ids": predicted[:k],
                            "relevant_ids": q["relevant_ids"],
                        }
                    )

                mean_recall = sum(q["recall_at_k"] for q in per_query) / max(len(per_query), 1)
                mean_latency = sum(q["latency_ms"] for q in per_query) / max(len(per_query), 1)
                p99_latency = (
                    sorted(q["latency_ms"] for q in per_query)[int(0.99 * len(per_query))]
                    if per_query
                    else 0.0
                )

                return {
                    "condition": condition,
                    "n_queries": len(per_query),
                    "mean_recall_at_k": round(mean_recall, 4),
                    "mean_latency_ms": round(mean_latency, 2),
                    "p99_latency_ms": round(p99_latency, 2),
                    "per_query": per_query,
                }
        finally:
            await mem.close()


def _oracle_row(corpus: dict[str, Any]) -> dict[str, Any]:
    """Synthetic oracle: recall@k = 1.0, latency = 0 ms.

    Real systems can't reach 0 ms latency — we mark it as the
    theoretical floor. Recall@k = 1.0 assumes k >= |relevant_ids|
    for every query; verify and warn if not.
    """

    n_queries = sum(1 for q in corpus["queries"] if q["relevant_ids"])
    return {
        "condition": "oracle",
        "n_queries": n_queries,
        "mean_recall_at_k": 1.0,
        "mean_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
        "per_query": [
            {
                "query": q["query"],
                "kind": q.get("kind", "unknown"),
                "recall_at_k": 1.0,
                "latency_ms": 0.0,
                "predicted_corpus_ids": q["relevant_ids"],
                "relevant_ids": q["relevant_ids"],
            }
            for q in corpus["queries"]
            if q["relevant_ids"]
        ],
    }


# ─── plotting ─────────────────────────────────────────────────────


def _plot(rows: list[dict[str, Any]], corpus_name: str, out: Path) -> None:
    by_cond = {r["condition"]: r for r in rows}
    conditions = ["dreaming_off", "full", "oracle"]
    accuracies = [by_cond[c]["mean_recall_at_k"] for c in conditions]
    latencies = [by_cond[c]["mean_latency_ms"] for c in conditions]

    fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy panel
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax_a.bar(conditions, accuracies, color=colors, edgecolor="black", linewidth=0.5)
    ax_a.axhline(1.0, linestyle="--", color="grey", alpha=0.5, label="oracle ceiling = 1.0")
    ax_a.set_ylim(0.0, 1.05)
    ax_a.set_ylabel("Mean recall@10 (strict gold-id match)", fontsize=11)
    ax_a.set_title(f"Accuracy — {corpus_name}", fontsize=12)
    ax_a.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, accuracies, strict=True):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}", ha="center", fontsize=9
        )

    # Latency panel
    bars = ax_l.bar(conditions, latencies, color=colors, edgecolor="black", linewidth=0.5)
    ax_l.set_ylabel("Mean recall latency (ms)", fontsize=11)
    ax_l.set_title(f"Speed — {corpus_name}", fontsize=12)
    ax_l.grid(True, axis="y", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, latencies, strict=True):
        ax_l.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(latencies) * 0.02,
            f"{val:.1f}ms",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


# ─── main ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Final validation: dreaming vs oracle ceiling, accuracy + latency.",
    )
    parser.add_argument(
        "--corpus",
        choices=["topology", "pressure", "both"],
        default="topology",
        help="Which corpus to validate against (default: topology).",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
    )
    parser.add_argument(
        "--out-dir",
        default="bench/results",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    corpora_to_run: list[tuple[str, str]] = []
    if args.corpus in ("topology", "both"):
        corpora_to_run.append(("topology", "bench/fixtures/topology_corpus.json"))
    if args.corpus in ("pressure", "both"):
        corpora_to_run.append(("pressure", "bench/fixtures/pressure_corpus_seed42.jsonl"))

    for corpus_name, corpus_path in corpora_to_run:
        corpus = _load_corpus(Path(corpus_path))
        is_pressure = corpus_name == "pressure"
        out_jsonl = Path(args.out_dir) / f"final_validation_{corpus_name}.jsonl"
        out_png = Path(args.out_dir) / f"final_validation_{corpus_name}.png"

        print(f"\n=== {corpus_name} corpus ({len(corpus['memories'])} memories) ===", flush=True)
        rows: list[dict[str, Any]] = []

        for condition in ("dreaming_off", "full"):
            row = asyncio.run(
                _seed_and_measure(corpus, cfg, condition=condition, is_pressure=is_pressure)
            )
            rows.append(row)
            print(
                f"  [{condition}] recall@k={row['mean_recall_at_k']:.4f}  "
                f"latency={row['mean_latency_ms']:.1f}ms  "
                f"p99={row['p99_latency_ms']:.1f}ms  "
                f"n={row['n_queries']}",
                flush=True,
            )

        rows.append(_oracle_row(corpus))
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        _plot(rows, corpus_name, out_png)

        # Summary.
        by_cond = {r["condition"]: r for r in rows}
        off = by_cond["dreaming_off"]
        full = by_cond["full"]
        ora = by_cond["oracle"]
        acc_lift = (full["mean_recall_at_k"] - off["mean_recall_at_k"]) * 100.0
        gap_to_oracle = (ora["mean_recall_at_k"] - full["mean_recall_at_k"]) * 100.0
        speed_delta_ms = full["mean_latency_ms"] - off["mean_latency_ms"]
        print("\n  ACCURACY:")
        print(f"    dreaming_off → full: {acc_lift:+.2f}pp")
        print(f"    full → oracle:        {gap_to_oracle:+.2f}pp gap remaining")
        print("  SPEED:")
        speed_label = "faster" if speed_delta_ms < 0 else "slower"
        print(f"    full vs dreaming_off: {speed_delta_ms:+.2f}ms ({speed_label})")
        print(f"    full vs oracle:        {full['mean_latency_ms']:.1f}ms above the 0ms floor")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
