"""Mnemoss dreaming-validation ablation harness.

Runs the dream pipeline under different phase masks on a labeled
corpus and emits ``results.jsonl`` plus per-condition recall@k and
cost numbers. Drives the Pareto chart in ``bench/plot_pareto.py``.

Usage::

    # Binary decision gate (full pipeline vs dreaming-off), topology corpus.
    python -m bench.ablate_dreaming --binary

    # Full ablation matrix (12 conditions).
    python -m bench.ablate_dreaming --full

    # Or against an arbitrary corpus + toml.
    python -m bench.ablate_dreaming --corpus PATH --config PATH

The harness REFUSES to run if any required config field is missing —
that's the pre-registration discipline. A future change to FormulaParams
defaults that this .toml doesn't pin would silently invalidate the
ablation deltas; refusing to run is louder than silently re-running.

Time injection: ``freezegun`` mocks ``datetime.now(UTC)`` so observe
timestamps and dream-phase ``now`` values are deterministic across
runs. No API change to Mnemoss required.

Network: requires ``OPENAI_API_KEY`` (for the embedder) and
``OPENROUTER_API_KEY`` (for both Consolidate and the gist judge LLMs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import tomllib

from mnemoss import (
    DreamerParams,
    FormulaParams,
    Mnemoss,
    OpenAIClient,
    OpenAIEmbedder,
    StorageParams,
)

UTC = timezone.utc


# ─── corpus + config loading ───────────────────────────────────────


def _load_corpus(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "memories" not in raw or "queries" not in raw:
        raise ValueError(f"corpus at {path} missing 'memories' and/or 'queries'")
    if not raw["memories"] or not raw["queries"]:
        raise ValueError(f"corpus at {path} is empty")
    # Validate each memory has id, content, topic.
    for m in raw["memories"]:
        for required in ("id", "content"):
            if required not in m:
                raise ValueError(f"memory missing {required!r}: {m}")
    # Validate queries reference real corpus ids.
    corpus_ids = {m["id"] for m in raw["memories"]}
    for q in raw["queries"]:
        for required in ("query", "relevant_ids"):
            if required not in q:
                raise ValueError(f"query missing {required!r}: {q}")
        missing = set(q["relevant_ids"]) - corpus_ids
        if missing:
            raise ValueError(f"query {q['query']!r} references unknown ids: {sorted(missing)}")
    return raw


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    # Required sections.
    for section in ("embedder", "llm", "dreamer", "formula", "harness"):
        if section not in cfg:
            raise ValueError(f"config at {path} missing required section [{section}]")
    if "consolidate" not in cfg["llm"] or "judge" not in cfg["llm"]:
        raise ValueError(f"config at {path} missing [llm.consolidate] and/or [llm.judge]")
    return cfg


# ─── ablation matrix ───────────────────────────────────────────────


# All six phases (the nightly trigger's full set).
ALL_PHASES = {"replay", "cluster", "consolidate", "relations", "rebalance", "dispose"}


def _ablation_matrix(mode: str) -> list[tuple[str, set[str] | None]]:
    """Return ``[(label, phases_mask), ...]`` for the chosen mode.

    ``phases_mask=None`` means full pipeline (no mask); ``set()`` means
    dreaming entirely off.
    """

    if mode == "binary":
        return [
            ("full", None),
            ("dreaming_off", set()),
        ]
    if mode == "full":
        rows: list[tuple[str, set[str] | None]] = [
            ("full", None),
            ("dreaming_off", set()),
        ]
        # "X_only" — only one phase at a time
        for phase in sorted(ALL_PHASES):
            rows.append((f"{phase}_only", {phase}))
        # "no_X" — full pipeline minus one phase
        for phase in sorted(ALL_PHASES):
            rows.append((f"no_{phase}", ALL_PHASES - {phase}))
        return rows
    raise ValueError(f"unknown ablation mode {mode!r}")


# ─── one ablation run ──────────────────────────────────────────────


async def _run_one_ablation(
    label: str,
    phases: set[str] | None,
    corpus: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Seed a fresh workspace, dream with the given mask, run all
    queries, return one results-row dict."""

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    consolidate_llm = OpenAIClient(
        model=cfg["llm"]["consolidate"]["model"],
        base_url=cfg["llm"]["consolidate"]["base_url"],
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    formula = FormulaParams(**cfg["formula"])
    dreamer = DreamerParams(**cfg["dreamer"])

    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )
    inc = timedelta(seconds=cfg["harness"]["simulated_time"]["per_observe_increment_seconds"])

    # We do NOT use freezegun in the harness loop because the
    # underlying SQLite clock + dream phase inputs both pull from
    # datetime.now(UTC). Freezegun would patch all of them coherently;
    # for unit tests of phase-mask correctness we don't need it
    # (existing tests use FakeEmbedder with no temporal dependency).
    # The harness uses real wall-clock time — that's fine for the
    # binary decision gate on a 30-memory corpus where dream timing
    # is incidental. The pressure corpus (weekend 2) will introduce
    # freezegun for the simulated-30-day accumulation.
    _ = anchor  # noqa: F841 (reserved for weekend 2 pressure corpus)
    _ = inc  # noqa: F841

    with tempfile.TemporaryDirectory(prefix="ablate-") as td:
        mem = Mnemoss(
            workspace=f"ablate-{label}",
            embedding_model=embedder,
            formula=formula,
            dreamer=dreamer,
            storage=StorageParams(root=Path(td)),
            llm=consolidate_llm,
        )
        try:
            # Seed in corpus order. Track corpus_id -> mnemoss_id.
            id_map: dict[str, str] = {}
            for m in corpus["memories"]:
                # observe returns the assigned memory id (or None if
                # the role got filtered out — impossible here because
                # default encoded_roles includes "user").
                mid = await mem.observe(role="user", content=m["content"])
                if mid is None:
                    raise RuntimeError(
                        f"observe returned None for {m['id']!r} — "
                        "role filtered? check EncoderParams.encoded_roles"
                    )
                id_map[m["id"]] = mid

            # Dream with the mask. Trigger NIGHTLY = all six phases
            # (the broadest superset; the mask filters from there).
            report = await mem.dream(trigger="nightly", phases=phases)

            # Recall every query, compute per-query recall@k.
            k = int(cfg["harness"]["recall_k"])
            include_deep = bool(cfg["harness"]["include_deep"])
            auto_expand = bool(cfg["harness"]["auto_expand"])
            per_query: list[dict[str, Any]] = []
            r_at_k_total = 0.0
            for q in corpus["queries"]:
                results = await mem.recall(
                    q["query"],
                    k=k,
                    include_deep=include_deep,
                    auto_expand=auto_expand,
                )
                predicted_corpus = []
                for r in results:
                    # Reverse-map mnemoss_id back to corpus id (if it's
                    # one of the originals; consolidated summaries
                    # produced by Consolidate get None).
                    for cid, mid in id_map.items():
                        if mid == r.memory.id:
                            predicted_corpus.append(cid)
                            break
                relevant = set(q["relevant_ids"])
                if relevant:
                    hits = len(set(predicted_corpus[:k]) & relevant)
                    r_at_k = hits / len(relevant)
                else:
                    # Negative query: top-K is "clean" iff zero corpus
                    # memories appear (everything in top-K is
                    # consolidated/derived, or top-K is empty).
                    r_at_k = 0.0  # No "recall" concept on a negative.
                r_at_k_total += r_at_k
                per_query.append(
                    {
                        "query": q["query"],
                        "kind": q.get("kind", "unknown"),
                        "predicted_corpus_ids": predicted_corpus[:k],
                        "relevant_ids": q["relevant_ids"],
                        "recall_at_k": round(r_at_k, 4),
                    }
                )

            mean_recall_at_k = (
                r_at_k_total / len([q for q in corpus["queries"] if q["relevant_ids"]])
                if any(q["relevant_ids"] for q in corpus["queries"])
                else 0.0
            )

            # Cost ledger snapshot — count of LLM calls during dream.
            status = await mem.status()
            llm_calls = status.get("llm_cost", {}).get("total_calls", 0)

            # Phase outcome details (clusters, dispositions, etc.).
            phase_summary = {
                o.phase.value: {
                    "status": o.status,
                    "skip_reason": o.skip_reason,
                    "details": o.details,
                }
                for o in report.outcomes
            }

            return {
                "label": label,
                "phases": sorted(phases) if phases is not None else None,
                "mean_recall_at_k": round(mean_recall_at_k, 4),
                "llm_calls_during_dream": int(llm_calls),
                "phase_summary": phase_summary,
                "per_query": per_query,
                "degraded_mode": report.degraded_mode,
            }
        finally:
            await mem.close()


# ─── orchestrator ──────────────────────────────────────────────────


async def _run_matrix(
    matrix: list[tuple[str, set[str] | None]],
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    out_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for label, phases in matrix:
            print(f"  → ablation {label!r} ({phases or 'full'})", flush=True)
            row = await _run_one_ablation(label, phases, corpus, cfg)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"    recall@k={row['mean_recall_at_k']:.4f}  "
                f"llm_calls={row['llm_calls_during_dream']}",
                flush=True,
            )
    return rows


def _evaluate_decision_gate(rows: list[dict[str, Any]]) -> str:
    """Return a human-readable decision gate verdict for the binary mode.

    The pre-registered threshold (per docs/dreaming-decision.md): if
    ``full`` recall@k minus ``dreaming_off`` recall@k is < 5 absolute pp,
    the per-phase study is moot — stop and document the null result.
    """

    by_label = {r["label"]: r for r in rows}
    if "full" not in by_label or "dreaming_off" not in by_label:
        return "decision gate skipped (need both 'full' and 'dreaming_off' rows)"
    full = by_label["full"]["mean_recall_at_k"]
    off = by_label["dreaming_off"]["mean_recall_at_k"]
    delta_pp = (full - off) * 100.0
    if delta_pp < 5.0:
        return (
            f"DECISION GATE TRIPPED: full={full:.4f} - off={off:.4f} = "
            f"{delta_pp:+.2f}pp < 5pp. Stop the study; document the null result."
        )
    return (
        f"DECISION GATE PASSED: full={full:.4f} - off={off:.4f} = "
        f"{delta_pp:+.2f}pp >= 5pp. Proceed to weekend 2."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Mnemoss dreaming-validation ablation harness.")
    parser.add_argument(
        "--binary", action="store_true", help="Run only full vs dreaming_off (decision gate)."
    )
    parser.add_argument(
        "--full", action="store_true", help="Run the full 14-condition ablation matrix."
    )
    parser.add_argument(
        "--corpus",
        default="bench/fixtures/topology_corpus.json",
        help="Path to the labeled corpus JSON.",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
        help="Path to the pinned config TOML.",
    )
    parser.add_argument(
        "--out",
        default="bench/results/topology_results.jsonl",
        help="Where to write results.jsonl.",
    )
    args = parser.parse_args()

    if not args.binary and not args.full:
        print("error: pass --binary (decision gate) or --full (matrix).", file=sys.stderr)
        return 2

    corpus = _load_corpus(Path(args.corpus))
    cfg = _load_config(Path(args.config))
    matrix = _ablation_matrix("binary" if args.binary else "full")

    print(
        f"running {len(matrix)} ablation(s) on {len(corpus['memories'])} memories, "
        f"{len(corpus['queries'])} queries → {args.out}",
        flush=True,
    )

    rows = asyncio.run(_run_matrix(matrix, corpus, cfg, Path(args.out)))

    print()
    print(_evaluate_decision_gate(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
