"""Formula-parameter calibration harness.

``FormulaParams`` defaults come from Anderson & Schooler (1991) — a
human-memory dataset from the last millennium. Real Mnemoss
deployments have different workloads, and the right ``d`` / ``tau``
/ ``mp`` / ``alpha`` / ``beta`` / ``gamma`` values for one are not
necessarily right for another.

This harness takes a **labeled relevance corpus** and sweeps a small
parameter space, reporting recall@k and MRR per combination.

Corpus format (JSON)::

    {
      "memories": [
        {"id": "m1", "content": "alice joined the Q3 kickoff"},
        {"id": "m2", "content": "bob writes the migration spec"},
        ...
      ],
      "queries": [
        {
          "query": "when was the kickoff",
          "relevant_ids": ["m1"]
        },
        ...
      ]
    }

Usage::

    # with your own corpus
    python -m bench.calibrate path/to/corpus.json

    # with the tiny built-in demo corpus
    python -m bench.calibrate --demo

The script writes a JSON summary to stdout (one dict per parameter
combination) so external trackers can parse it.

What this harness does NOT do:

- Pick the "best" params for you. Model selection on a small corpus
  is noisy; eyeball the output + pick something reasonable, or run
  with a big enough corpus that Pareto-dominated combos are obvious.
- Sweep every combination of every parameter. The search space is
  combinatorial; default sweeps are small and focused on the three
  parameters most likely to matter (``d``, ``tau``, ``mp``).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mnemoss import (
    FakeEmbedder,
    FormulaParams,
    Mnemoss,
    StorageParams,
)

# ─── corpus loading ────────────────────────────────────────────────


@dataclass
class Corpus:
    """Parsed relevance-labeled corpus."""

    memories: list[dict[str, Any]]
    queries: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.memories:
            raise ValueError("corpus has no memories")
        if not self.queries:
            raise ValueError("corpus has no queries")
        ids = {m["id"] for m in self.memories}
        for q in self.queries:
            missing = set(q["relevant_ids"]) - ids
            if missing:
                raise ValueError(
                    f"query {q['query']!r} references unknown memory "
                    f"ids: {sorted(missing)}"
                )


def _load_corpus(path: Path) -> Corpus:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Corpus(memories=raw["memories"], queries=raw["queries"])


def _demo_corpus() -> Corpus:
    """A tiny synthetic corpus so the harness is runnable offline.

    Real calibration needs a bigger, domain-specific corpus. This is
    just enough to prove the harness works.
    """

    return Corpus(
        memories=[
            {"id": "m1", "content": "alice joined the Q3 kickoff on Tuesday"},
            {"id": "m2", "content": "bob wrote the payment migration spec"},
            {"id": "m3", "content": "carol finished the dashboard design"},
            {"id": "m4", "content": "kickoff agenda circulated ahead of tuesday"},
            {"id": "m5", "content": "dashboard review is scheduled for friday"},
            {"id": "m6", "content": "migration rollout plan shared in #platform"},
            {"id": "m7", "content": "coffee order sheet updated for the team"},
            {"id": "m8", "content": "alice mentioned Q3 roadmap priorities"},
        ],
        queries=[
            {
                "query": "when was the kickoff",
                "relevant_ids": ["m1", "m4", "m8"],
            },
            {
                "query": "who owns the migration work",
                "relevant_ids": ["m2", "m6"],
            },
            {
                "query": "dashboard review",
                "relevant_ids": ["m3", "m5"],
            },
        ],
    )


# ─── metrics ───────────────────────────────────────────────────────


def _recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(predicted[:k]) & relevant)
    return hits / len(relevant)


def _mrr(predicted: list[str], relevant: set[str]) -> float:
    for rank, mid in enumerate(predicted, 1):
        if mid in relevant:
            return 1.0 / rank
    return 0.0


# ─── parameter sweep ───────────────────────────────────────────────


@dataclass(frozen=True)
class ParamGrid:
    """Sweep ranges for the three most sensitivity-worth parameters.

    Kept tight so the default run is under a minute on the demo
    corpus. Pass ``--grid full`` to widen.
    """

    d_values: tuple[float, ...] = (0.3, 0.5, 0.7)
    tau_values: tuple[float, ...] = (-2.0, -1.0, 0.0)
    mp_values: tuple[float, ...] = (1.0, 1.5, 2.0)

    def combinations(self) -> list[tuple[float, float, float]]:
        return list(
            itertools.product(self.d_values, self.tau_values, self.mp_values)
        )


_FULL_GRID = ParamGrid(
    d_values=(0.2, 0.3, 0.5, 0.7, 0.9),
    tau_values=(-3.0, -2.0, -1.0, 0.0, 1.0),
    mp_values=(0.5, 1.0, 1.5, 2.0, 3.0),
)


@dataclass
class RunResult:
    params: dict[str, float]
    recall_at_1: float
    recall_at_5: float
    mrr: float
    per_query: list[dict[str, Any]] = field(default_factory=list)


async def _eval_params(
    corpus: Corpus, d: float, tau: float, mp: float, *, k: int = 10
) -> RunResult:
    """Build a workspace, seed it, run each query, compute metrics."""

    with tempfile.TemporaryDirectory() as td:
        mem = Mnemoss(
            workspace="calibrate",
            embedding_model=FakeEmbedder(dim=64),
            formula=FormulaParams(d=d, tau=tau, mp=mp, noise_scale=0.0),
            storage=StorageParams(root=Path(td)),
        )
        try:
            for m in corpus.memories:
                await mem.observe(role="user", content=m["content"])

            # Map content → observed memory id (observe's id is auto-
            # assigned by the segmenter; we match by content).
            await mem._ensure_open()
            assert mem._store is not None
            all_ids = await mem._store.iter_memory_ids()
            mems = await mem._store.materialize_memories(all_ids)
            content_to_id = {m.content: m.id for m in mems}
            # Relevant ids in the CORPUS are the corpus's own "m1" etc;
            # translate them to the ids Mnemoss assigned.
            id_lookup = {
                corpus_m["id"]: content_to_id[corpus_m["content"]]
                for corpus_m in corpus.memories
            }

            per_query: list[dict[str, Any]] = []
            r1_total = 0.0
            r5_total = 0.0
            mrr_total = 0.0
            for q in corpus.queries:
                results = await mem.recall(q["query"], k=k)
                predicted = [r.memory.id for r in results]
                relevant = {id_lookup[x] for x in q["relevant_ids"]}
                r1 = _recall_at_k(predicted, relevant, 1)
                r5 = _recall_at_k(predicted, relevant, 5)
                mrr = _mrr(predicted, relevant)
                per_query.append(
                    {
                        "query": q["query"],
                        "recall_at_1": r1,
                        "recall_at_5": r5,
                        "mrr": mrr,
                    }
                )
                r1_total += r1
                r5_total += r5
                mrr_total += mrr

            n = len(corpus.queries)
            return RunResult(
                params={"d": d, "tau": tau, "mp": mp},
                recall_at_1=r1_total / n,
                recall_at_5=r5_total / n,
                mrr=mrr_total / n,
                per_query=per_query,
            )
        finally:
            await mem.close()


async def _sweep(corpus: Corpus, grid: ParamGrid) -> list[RunResult]:
    results: list[RunResult] = []
    for d, tau, mp in grid.combinations():
        result = await _eval_params(corpus, d, tau, mp)
        results.append(result)
    return results


def _print_table(results: list[RunResult]) -> None:
    header = (
        f"  {'d':>5}  {'tau':>6}  {'mp':>5}  "
        f"{'R@1':>6}  {'R@5':>6}  {'MRR':>6}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in sorted(results, key=lambda r: -r.mrr):
        print(
            f"  {r.params['d']:>5.2f}  {r.params['tau']:>6.2f}  "
            f"{r.params['mp']:>5.2f}  {r.recall_at_1:>6.3f}  "
            f"{r.recall_at_5:>6.3f}  {r.mrr:>6.3f}"
        )


def _main_sync(args: argparse.Namespace) -> int:
    if args.demo:
        corpus = _demo_corpus()
    elif args.corpus is not None:
        corpus = _load_corpus(Path(args.corpus))
    else:
        print(
            "error: pass a corpus path, or --demo for the built-in tiny "
            "synthetic corpus. See docstring for the JSON format.",
            file=sys.stderr,
        )
        return 2

    grid = _FULL_GRID if args.grid == "full" else ParamGrid()
    results = asyncio.run(_sweep(corpus, grid))
    _print_table(results)

    payload = [
        {
            "params": r.params,
            "recall_at_1": round(r.recall_at_1, 4),
            "recall_at_5": round(r.recall_at_5, 4),
            "mrr": round(r.mrr, 4),
        }
        for r in results
    ]
    print("JSON_RESULTS=" + json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep FormulaParams against a labeled corpus."
    )
    parser.add_argument(
        "corpus",
        nargs="?",
        help="Path to a corpus JSON file (see module docstring).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the built-in tiny synthetic corpus.",
    )
    parser.add_argument(
        "--grid",
        choices=["default", "full"],
        default="default",
        help="'default' sweeps 3×3×3 = 27 combos; 'full' sweeps 5×5×5 = 125.",
    )
    args = parser.parse_args()
    return _main_sync(args)


if __name__ == "__main__":
    raise SystemExit(main())
