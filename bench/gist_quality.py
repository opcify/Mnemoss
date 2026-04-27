"""Hand-rolled pairwise LLM-as-judge for Consolidate's gist quality.

Recall@k can't tell us whether Consolidate's gists are *better* than
the level-1 heuristic — it only measures retrieval ranking. Consolidate
changes content, not order, so a different metric is required for
the weekend-3 verdict.

Approach: pairwise comparison with a different model family from
Consolidate, to mitigate self-preference bias (per /plan-eng-review
outside-voice finding).

  - Consolidate generates new gists with ``tencent/hy3-preview:free``
    (per bench/ablate_dreaming.toml).
  - Judge compares level-1 gist vs post-Consolidate gist with
    ``deepseek/deepseek-v4-flash``.
  - Order randomized per comparison so positional bias washes out.
  - Bootstrapped 95% CI on win rate.

Pipeline:

  1. Observe the topology corpus into a fresh Mnemoss workspace.
  2. Snapshot every memory's ``extracted_gist`` (level 1).
  3. Run ``mem.dream(trigger="nightly")`` so Consolidate runs and
     refines the gists (extraction_level → 2).
  4. Snapshot every memory's ``extracted_gist`` again (level 2).
  5. For each (query, member) pair where the member was refined,
     ask the judge which gist better answers the query.
  6. Aggregate: win rate of post-Consolidate vs level-1, with
     bootstrap CI.

Output: ``bench/results/gist_quality.jsonl`` (one row per comparison)
plus a summary line.

Usage::

    python -m bench.gist_quality
    python -m bench.gist_quality --corpus PATH --config PATH

Network: requires ``OPENAI_API_KEY`` (embedder + Consolidate seed)
and ``OPENROUTER_API_KEY`` (Consolidate + judge LLMs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from bench._metrics import bootstrap_ci
from mnemoss import (
    DreamerParams,
    FormulaParams,
    LLMClient,
    Mnemoss,
    OpenAIClient,
    OpenAIEmbedder,
    StorageParams,
)

# ─── prompt ────────────────────────────────────────────────────────


JUDGE_PROMPT_TEMPLATE = """You are evaluating two candidate memory summaries for usefulness.

Query: {query}

Candidate A: {gist_a}

Candidate B: {gist_b}

Which candidate is more useful for answering the query? Consider
specificity, relevance, and whether the candidate would help someone
locate the right information. Reply with exactly one word: "A", "B",
or "tie". No explanation, no punctuation."""


def build_judge_prompt(query: str, gist_a: str, gist_b: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(query=query, gist_a=gist_a, gist_b=gist_b)


def parse_judge_response(raw: str) -> str:
    """Map a freeform judge reply to "A" | "B" | "tie".

    Defensively handles whitespace and lowercase replies. Anything we
    can't parse is normalized to ``"tie"`` rather than raising — the
    win-rate metric tolerates noise but a hard exception would break
    the harness mid-run.
    """

    raw = raw.strip().strip(".\"'`").upper()
    if raw.startswith("A") and not raw.startswith("AGAIN"):
        return "A"
    if raw.startswith("B") and not raw.startswith("BOTH"):
        return "B"
    if "TIE" in raw or "EQUAL" in raw or "BOTH" in raw or "NEITHER" in raw:
        return "tie"
    return "tie"


# ─── core comparison ───────────────────────────────────────────────


@dataclass
class Comparison:
    query: str
    member_id: str
    gist_level_1: str
    gist_level_2: str
    swap: bool  # True iff B was placed in position-A on the prompt.
    raw_judge_reply: str
    parsed: str  # "A" | "B" | "tie" (normalized)
    consolidate_won: float  # 1.0 if level-2 was picked, 0.5 tie, 0.0 level-1


async def _judge_pair(
    judge: LLMClient,
    query: str,
    member_id: str,
    gist_level_1: str,
    gist_level_2: str,
    rng: random.Random,
) -> Comparison:
    """Run one pairwise comparison with order randomization."""

    swap = rng.random() < 0.5
    if swap:
        a, b = gist_level_2, gist_level_1
    else:
        a, b = gist_level_1, gist_level_2
    prompt = build_judge_prompt(query, a, b)

    raw = await judge.complete_text(prompt, max_tokens=8, temperature=0.0)
    parsed = parse_judge_response(raw)

    # Map parsed (A/B/tie) back to "did level-2 win?".
    if parsed == "tie":
        consolidate_won = 0.5
    elif (parsed == "A" and swap) or (parsed == "B" and not swap):
        # Position A held level-2 if swapped; position B held level-2 otherwise.
        consolidate_won = 1.0
    else:
        consolidate_won = 0.0

    return Comparison(
        query=query,
        member_id=member_id,
        gist_level_1=gist_level_1,
        gist_level_2=gist_level_2,
        swap=swap,
        raw_judge_reply=raw,
        parsed=parsed,
        consolidate_won=consolidate_won,
    )


# ─── orchestration ─────────────────────────────────────────────────


def _load_corpus(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


async def collect_pairs(
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    *,
    workspace_root: Path,
) -> list[tuple[str, str, str, str]]:
    """Run observe → snapshot level-1 → dream → snapshot level-2.

    Returns ``[(query, member_id, gist_level_1, gist_level_2), ...]``
    for every (query, member) pair where the member was refined.
    """

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

    mem = Mnemoss(
        workspace="gist_quality",
        embedding_model=embedder,
        formula=formula,
        dreamer=dreamer,
        storage=StorageParams(root=workspace_root),
        llm=consolidate_llm,
    )
    try:
        # Observe all memories. Track corpus_id -> mnemoss_id.
        id_map: dict[str, str] = {}
        for m in corpus["memories"]:
            mid = await mem.observe(role="user", content=m["content"])
            if mid is None:
                raise RuntimeError(f"observe returned None for {m['id']!r}")
            id_map[m["id"]] = mid
        # Reverse for later lookups.
        reverse = {v: k for k, v in id_map.items()}

        # Snapshot level-1 gists by reading the store directly.
        await mem._ensure_open()
        store = mem._store
        assert store is not None
        all_ids = list(id_map.values())
        memories_before = await store.materialize_memories(all_ids)
        level_1_by_id = {m.id: (m.extracted_gist or m.content) for m in memories_before}

        # Run dream — Consolidate refines gists in place.
        report = await mem.dream(trigger="nightly")
        # Sanity: Consolidate must have run, otherwise no gists changed.
        consolidate_outcome = next(
            (o for o in report.outcomes if o.phase.value == "consolidate"), None
        )
        if consolidate_outcome is None or consolidate_outcome.status != "ok":
            print(
                f"warning: Consolidate did not run cleanly "
                f"(status={consolidate_outcome.status if consolidate_outcome else 'absent'!r}); "
                "no comparisons will be produced",
                file=sys.stderr,
            )
            return []

        # Snapshot level-2 gists.
        memories_after = await store.materialize_memories(all_ids)
        level_2_by_id = {m.id: (m.extracted_gist or m.content) for m in memories_after}

        # Build comparison list. Only include refined members where
        # the level-2 gist actually differs from level-1 — comparing
        # identical text against itself is judge-noise, not signal.
        pairs: list[tuple[str, str, str, str]] = []
        for q in corpus["queries"]:
            for corpus_id in q["relevant_ids"]:
                mnemoss_id = id_map[corpus_id]
                level_1 = level_1_by_id[mnemoss_id]
                level_2 = level_2_by_id[mnemoss_id]
                if level_1 == level_2:
                    continue
                pairs.append((q["query"], corpus_id, level_1, level_2))
        return pairs
    finally:
        await mem.close()
    _ = reverse  # noqa: F841 (reserved for richer per-query reports)


async def run_judge(
    pairs: list[tuple[str, str, str, str]],
    judge: LLMClient,
    *,
    seed: int = 42,
) -> list[Comparison]:
    rng = random.Random(seed)
    out: list[Comparison] = []
    for query, member_id, level_1, level_2 in pairs:
        cmp = await _judge_pair(judge, query, member_id, level_1, level_2, rng)
        out.append(cmp)
    return out


def summarize(comparisons: list[Comparison]) -> dict[str, Any]:
    outcomes = [c.consolidate_won for c in comparisons]
    mean, lo, hi = bootstrap_ci(outcomes, n_resamples=2000, confidence=0.95, seed=42)

    # Verdict against pre-registered thresholds (docs/dreaming-decision.md):
    #   - CUT if win-rate CI upper bound <= 0.55
    #   - KEEP if win-rate CI lower bound >= 0.65
    #   - REBUILD otherwise
    if hi <= 0.55:
        verdict = "CUT"
    elif lo >= 0.65:
        verdict = "KEEP"
    else:
        verdict = "REBUILD"

    return {
        "num_comparisons": len(outcomes),
        "win_rate": round(mean, 4),
        "ci_lower": round(lo, 4),
        "ci_upper": round(hi, 4),
        "ties": sum(1 for c in comparisons if c.parsed == "tie"),
        "verdict": verdict,
    }


def _write_results(comparisons: list[Comparison], summary: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in comparisons:
            fh.write(
                json.dumps(
                    {
                        "query": c.query,
                        "member_id": c.member_id,
                        "gist_level_1": c.gist_level_1,
                        "gist_level_2": c.gist_level_2,
                        "swap": c.swap,
                        "raw": c.raw_judge_reply,
                        "parsed": c.parsed,
                        "consolidate_won": c.consolidate_won,
                    }
                )
                + "\n"
            )
        fh.write(json.dumps({"_summary": summary}) + "\n")


async def _amain(args: argparse.Namespace) -> int:
    corpus = _load_corpus(Path(args.corpus))
    cfg = _load_config(Path(args.config))

    judge_cfg = cfg["llm"]["judge"]
    judge = OpenAIClient(
        model=judge_cfg["model"],
        base_url=judge_cfg["base_url"],
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    with tempfile.TemporaryDirectory(prefix="gist-quality-") as td:
        print("seeding workspace + running Consolidate...", flush=True)
        pairs = await collect_pairs(corpus, cfg, workspace_root=Path(td))
        print(f"  {len(pairs)} (query, member) pairs to judge", flush=True)

        if not pairs:
            print(
                "no pairs to judge — Consolidate produced no refined gists "
                "or every refinement matched its level-1 baseline",
                flush=True,
            )
            return 0

        print(f"running judge ({judge.model})...", flush=True)
        comparisons = await run_judge(pairs, judge)
        summary = summarize(comparisons)

        out = Path(args.out)
        _write_results(comparisons, summary, out)
        print(f"wrote {out}", flush=True)
        print(
            f"\n  win_rate = {summary['win_rate']:.4f}  "
            f"CI95 = [{summary['ci_lower']:.4f}, {summary['ci_upper']:.4f}]  "
            f"n = {summary['num_comparisons']}  "
            f"ties = {summary['ties']}",
            flush=True,
        )
        print(f"  VERDICT: {summary['verdict']}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pairwise LLM-as-judge for Consolidate gist quality.",
    )
    parser.add_argument(
        "--corpus",
        default="bench/fixtures/topology_corpus.json",
        help="Labeled corpus path (defaults to topology corpus).",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
        help="Pinned config TOML.",
    )
    parser.add_argument(
        "--out",
        default="bench/results/gist_quality.jsonl",
    )
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
