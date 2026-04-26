"""Mnemoss dreaming-validation ablation harness.

Runs the dream pipeline under different phase masks on a labeled
corpus and emits ``results.jsonl`` plus per-condition recall@k and
cost numbers. Drives the Pareto chart in ``bench/plot_pareto.py``.

Two corpus types supported:

- **Topology** (``bench/fixtures/topology_corpus.json``) — 30 hand-
  labeled memories across 3 topics, 12 queries (single-hop, multi-hop,
  negative). Tests Cluster, Consolidate, Relations. No simulated time;
  observes happen at real wall-clock now.
- **Pressure** (``bench/fixtures/pressure_corpus_seed42.jsonl``) —
  ~500 synthetic memories accumulating over 30 simulated days, 30
  adversarial queries designed so junk pollutes top-10 BEFORE Dispose
  runs. Tests Dispose + Rebalance. Uses ``freezegun`` to inject
  simulated timestamps.

The harness detects which corpus you passed by checking for
``_meta.simulated_days`` and picks the right path automatically.

Usage::

    # Topology decision gate: full vs dreaming-off.
    python -m bench.ablate_dreaming --binary

    # Topology full matrix (14 conditions).
    python -m bench.ablate_dreaming --full

    # Pressure decision gate (Dispose + Rebalance combined effect).
    python -m bench.ablate_dreaming --pressure-binary

    # Pressure full matrix.
    python -m bench.ablate_dreaming --pressure-full

The harness REFUSES to run if any required config field is missing —
that's the pre-registration discipline. Network: requires
``OPENAI_API_KEY`` (embedder) and ``OPENROUTER_API_KEY`` (LLMs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import tomllib

from bench._metrics import topk_cleanliness
from mnemoss import (
    DreamerParams,
    Embedder,
    FakeEmbedder,
    FormulaParams,
    LLMClient,
    Mnemoss,
    MockLLMClient,
    OpenAIClient,
    OpenAIEmbedder,
    StorageParams,
)

# Module-level toggle set by --mock-llm. Read inside _build_clients.
USE_MOCK_LLM = False


_PROMPT_MEMBER_RE = re.compile(r"^\s*\d+\.\s*\[[^\]]+\]\s*(.+?)$", re.MULTILINE)


def _canned_consolidate_response(prompt: str) -> dict[str, Any]:
    """Deterministic canned response for the mock Consolidate LLM.

    Extracts cluster-member content from the prompt and produces a
    summary that concatenates the first three members' content. This
    is dumb but at least makes the synthetic summary share vocabulary
    with the cluster — without it, the summary is pure noise that
    pollutes top-K and torpedoes the harness's recall@k metric.

    NOT a real Consolidate verdict — see bench/gist_quality.py for
    that. The mock here is for harness validation only; the decision
    gate is suppressed under --mock-llm.
    """

    members = _PROMPT_MEMBER_RE.findall(prompt)
    # Prompt embeds an extracted-fields line per member; that line
    # starts with "current extraction:". Filter those out.
    real = [m.strip() for m in members if not m.strip().startswith("current extraction:")]
    if real:
        summary_text = " | ".join(real[:3])
    else:
        summary_text = "synthetic summary placeholder for harness mock mode"

    return {
        "summary": {
            "memory_type": "fact",
            "content": summary_text[:300],  # cap at ~300 chars to stay sensible
            "abstraction_level": 0.7,
            "aliases": [],
        },
        "refinements": [],
        "patterns": [],
    }


# ─── retry-on-rate-limit LLM wrapper ───────────────────────────────


class RetryingLLM:
    """Wraps an ``LLMClient`` with exponential backoff on 429 / 5xx.

    OpenRouter free models cap at ~8 req/min, and Mnemoss's
    Consolidate phase fires one LLM call per cluster — the topology
    corpus's 3 topics produce 3 calls, the pressure corpus's 50 high-
    utility memories produce ~10-15 clusters → ~10-15 calls. Without
    retry, almost every call after the first burst fails.

    Strategy:

      - On exception whose ``str()`` contains "429" or "rate limit",
        wait ``initial * 2^attempt + jitter`` seconds and retry.
      - On 5xx similarly retried.
      - Other exceptions (auth, schema) re-raised immediately —
        retrying a 401 just wastes time.
      - Up to ``max_attempts`` retries; final failure re-raises so
        ``DreamRunner`` records ``status="error"`` for the phase.
    """

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = 5,
        initial_wait: float = 4.0,
        max_wait: float = 60.0,
        min_wait_between_calls: float = 0.0,
    ) -> None:
        self._inner = inner
        self._max_attempts = max_attempts
        self._initial_wait = initial_wait
        self._max_wait = max_wait
        # Proactive pacing — never start a call if less than this many
        # seconds have elapsed since the previous one. Prevents 429s on
        # free tiers with hard rpm caps (OpenRouter free is 8 rpm =
        # 7.5s/call). 0 = no pacing.
        self._min_wait_between_calls = min_wait_between_calls
        self._last_call_at: float = 0.0

    @property
    def model(self) -> str:
        return self._inner.model

    @staticmethod
    def _is_retryable(err: BaseException) -> bool:
        msg = str(err).lower()
        return (
            "429" in msg
            or "rate limit" in msg
            or "rate-limit" in msg
            or "503" in msg
            or "502" in msg
            or "504" in msg
            or "500" in msg
            or "temporarily" in msg
            or "try again" in msg
        )

    async def _retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        import asyncio as _asyncio
        import random as _random
        import time as _time

        # Proactive pace: if a previous call landed less than
        # min_wait_between_calls seconds ago, sleep until the gap is met.
        if self._min_wait_between_calls > 0 and self._last_call_at > 0:
            elapsed = _time.monotonic() - self._last_call_at
            need = self._min_wait_between_calls - elapsed
            if need > 0:
                await _asyncio.sleep(need)

        last_err: BaseException | None = None
        for attempt in range(self._max_attempts):
            try:
                result = await fn(*args, **kwargs)
                self._last_call_at = _time.monotonic()
                return result
            except BaseException as e:  # noqa: BLE001 (re-raise below if non-retryable)
                if not self._is_retryable(e):
                    raise
                last_err = e
                if attempt == self._max_attempts - 1:
                    break
                wait = min(self._initial_wait * (2**attempt), self._max_wait)
                jitter = _random.uniform(0, 0.5 * wait)
                total = wait + jitter
                print(
                    f"    LLM 429/5xx on attempt {attempt + 1}, "
                    f"sleeping {total:.1f}s before retry...",
                    flush=True,
                )
                await _asyncio.sleep(total)
        assert last_err is not None
        raise last_err

    async def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        return await self._retry(
            self._inner.complete_text,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return await self._retry(
            self._inner.complete_json,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )


UTC = timezone.utc


# ─── corpus + config loading ───────────────────────────────────────


def _load_corpus(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "memories" not in raw or "queries" not in raw:
        raise ValueError(f"corpus at {path} missing 'memories' and/or 'queries'")
    if not raw["memories"] or not raw["queries"]:
        raise ValueError(f"corpus at {path} is empty")
    for m in raw["memories"]:
        for required in ("id", "content"):
            if required not in m:
                raise ValueError(f"memory missing {required!r}: {m}")
    corpus_ids = {m["id"] for m in raw["memories"]}
    for q in raw["queries"]:
        for required in ("query", "relevant_ids"):
            if required not in q:
                raise ValueError(f"query missing {required!r}: {q}")
        missing = set(q["relevant_ids"]) - corpus_ids
        if missing:
            raise ValueError(f"query {q['query']!r} references unknown ids: {sorted(missing)}")
        for jid in q.get("junk_ids", []):
            if jid not in corpus_ids:
                raise ValueError(f"query {q['query']!r} references unknown junk id {jid!r}")
    return raw


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        cfg = tomllib.load(fh)
    for section in ("embedder", "llm", "dreamer", "formula", "harness"):
        if section not in cfg:
            raise ValueError(f"config at {path} missing required section [{section}]")
    if "consolidate" not in cfg["llm"] or "judge" not in cfg["llm"]:
        raise ValueError(f"config at {path} missing [llm.consolidate] and/or [llm.judge]")
    return cfg


def _is_pressure_corpus(corpus: dict[str, Any]) -> bool:
    """Pressure corpora carry ``_meta.simulated_days``."""

    return bool(corpus.get("_meta", {}).get("simulated_days"))


def _jsonable(value: Any) -> Any:
    """Best-effort coerce a value to JSON-serializable.

    Keeps primitives, lists, and dicts; drops keys whose values can't
    be serialized (``Memory`` objects from ``_phase_replay``'s details
    are the typical culprit). Used to keep ``results.jsonl`` rows
    self-describing without leaking ORM-shaped objects.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                # Try to recurse into list/dict; otherwise drop.
                if isinstance(v, list | dict):
                    out[k] = _jsonable(v)
                else:
                    out[k] = f"<{type(v).__name__}>"
        return out
    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value:
            try:
                json.dumps(item)
                out_list.append(item)
            except TypeError:
                out_list.append(_jsonable(item))
        return out_list
    return f"<{type(value).__name__}>"


# ─── ablation matrix ───────────────────────────────────────────────


ALL_PHASES = {"replay", "cluster", "consolidate", "relations", "rebalance", "dispose"}


def _ablation_matrix(mode: str) -> list[tuple[str, set[str] | None]]:
    """Return ``[(label, phases_mask), ...]`` for the chosen mode."""

    if mode in ("binary", "pressure_binary"):
        return [
            ("full", None),
            ("dreaming_off", set()),
        ]
    if mode == "pressure_full":
        # Pressure focuses on Dispose + Rebalance (the pressure-mediated
        # phases). The structural phases (Replay/Cluster/Consolidate/
        # Relations) get their own answer from the topology corpus.
        return [
            ("full", None),
            ("dreaming_off", set()),
            ("no_dispose", ALL_PHASES - {"dispose"}),
            ("no_rebalance", ALL_PHASES - {"rebalance"}),
            ("no_dispose_no_rebalance", ALL_PHASES - {"dispose", "rebalance"}),
            ("dispose_only", {"dispose"}),
            ("rebalance_only", {"rebalance"}),
        ]
    if mode == "full":
        rows: list[tuple[str, set[str] | None]] = [
            ("full", None),
            ("dreaming_off", set()),
        ]
        for phase in sorted(ALL_PHASES):
            rows.append((f"{phase}_only", {phase}))
        for phase in sorted(ALL_PHASES):
            rows.append((f"no_{phase}", ALL_PHASES - {phase}))
        return rows
    raise ValueError(f"unknown ablation mode {mode!r}")


# ─── helpers ───────────────────────────────────────────────────────


def _build_clients(cfg: dict[str, Any]) -> tuple[Embedder, LLMClient]:
    if USE_MOCK_LLM:
        # Mock mode: fully local. FakeEmbedder is deterministic
        # (hash-based pseudo-embeddings), MockLLMClient returns the
        # canned response. The ablation comparison stays meaningful
        # because every condition uses the same embedder.
        # Real Consolidate verdicts cannot come from this mode —
        # see bench/gist_quality.py for that.
        return FakeEmbedder(dim=64), MockLLMClient(callback=_canned_consolidate_response)

    embedder = OpenAIEmbedder(
        model=cfg["embedder"]["model"],
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    raw_llm = OpenAIClient(
        model=cfg["llm"]["consolidate"]["model"],
        base_url=cfg["llm"]["consolidate"]["base_url"],
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    retry_cfg = cfg["harness"].get("retry", {})
    llm = RetryingLLM(
        raw_llm,
        max_attempts=int(retry_cfg.get("max_attempts", 5)),
        initial_wait=float(retry_cfg.get("initial_wait_seconds", 4.0)),
        max_wait=float(retry_cfg.get("max_wait_seconds", 60.0)),
        min_wait_between_calls=float(retry_cfg.get("min_wait_between_calls_seconds", 0.0)),
    )
    return embedder, llm


async def _seed_topology(
    mem: Mnemoss, corpus: dict[str, Any], cfg: dict[str, Any]
) -> tuple[dict[str, str], datetime]:
    """Seed the topology corpus with freezegun-anchored timestamps.

    The topology corpus has no per-memory ts_offset_seconds, so every
    observe gets ``anchor + i`` seconds. This is just deterministic
    spacing — the formula's decay barely registers across 30 seconds
    — but it removes wall-clock variance between ablation conditions.
    Without this, a slow ablation (e.g. one retrying 429s) lets
    memories age between observe and recall and produces different
    recall scores even with noise_scale=0.

    Returns ``(id_map, end_time)`` so the harness can run dream + recall
    inside a freeze_time(end_time) context too.
    """

    from freezegun import freeze_time

    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )
    inc = float(cfg["harness"]["simulated_time"]["per_observe_increment_seconds"])

    id_map: dict[str, str] = {}
    for i, m in enumerate(corpus["memories"]):
        target = anchor + timedelta(seconds=i * inc)
        with freeze_time(target):
            mid = await mem.observe(role="user", content=m["content"])
            if mid is None:
                raise RuntimeError(
                    f"observe returned None for {m['id']!r}; check EncoderParams.encoded_roles"
                )
            id_map[m["id"]] = mid

    end_time = anchor + timedelta(seconds=len(corpus["memories"]) * inc + 1)
    return id_map, end_time


async def _seed_pressure(
    mem: Mnemoss, corpus: dict[str, Any], cfg: dict[str, Any]
) -> tuple[dict[str, str], datetime]:
    """Seed using freezegun-injected timestamps. Returns (id_map, end_time)."""

    from freezegun import freeze_time  # Lazy import — optional dep.

    anchor = datetime.fromisoformat(
        cfg["harness"]["simulated_time"]["anchor"].replace("Z", "+00:00")
    )

    id_map: dict[str, str] = {}
    end_time = anchor
    for m in corpus["memories"]:
        target = anchor + timedelta(seconds=int(m["ts_offset_seconds"]))
        with freeze_time(target):
            mid = await mem.observe(role="user", content=m["content"])
            if mid is None:
                raise RuntimeError(f"observe returned None for {m['id']!r}")
            id_map[m["id"]] = mid
        if target > end_time:
            end_time = target
    # Add a one-minute buffer past the last observe so dream's
    # "now" is unambiguously after the latest memory.
    return id_map, end_time + timedelta(minutes=1)


# ─── one ablation run ──────────────────────────────────────────────


async def _run_one_ablation(
    label: str,
    phases: set[str] | None,
    corpus: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Seed a fresh workspace, dream with the given mask, run all
    queries, return one results-row dict.

    Detects pressure vs topology corpus from ``_meta.simulated_days``
    and picks the right seeding/dream-time path.
    """

    embedder, consolidate_llm = _build_clients(cfg)
    formula = FormulaParams(**cfg["formula"])
    dreamer = DreamerParams(**cfg["dreamer"])
    is_pressure = _is_pressure_corpus(corpus)

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
            from freezegun import freeze_time  # Lazy import.

            if is_pressure:
                id_map, end_time = await _seed_pressure(mem, corpus, cfg)
            else:
                id_map, end_time = await _seed_topology(mem, corpus, cfg)
            # Dream + recall happen at end_time so wall-clock variance
            # (e.g. long LLM retries) doesn't perturb the formula's
            # B_i values between ablation conditions.
            with freeze_time(end_time):
                report = await mem.dream(trigger="nightly", phases=phases)
                per_query = await _score_queries(mem, corpus, cfg, id_map)

            # Aggregates.
            recallable = [q for q in per_query if q["relevant_ids"]]
            mean_recall_at_k = (
                sum(q["recall_at_k"] for q in recallable) / len(recallable) if recallable else 0.0
            )
            cleanable = [q for q in per_query if "cleanliness" in q]
            mean_cleanliness = (
                sum(q["cleanliness"] for q in cleanable) / len(cleanable) if cleanable else None
            )

            # Cost ledger snapshot.
            status = await mem.status()
            llm_calls = int(status.get("llm_cost", {}).get("total_calls", 0))

            phase_summary = {
                o.phase.value: {
                    "status": o.status,
                    "skip_reason": o.skip_reason,
                    "details": _jsonable(o.details),
                }
                for o in report.outcomes
            }

            return {
                "label": label,
                "corpus_kind": "pressure" if is_pressure else "topology",
                "phases": sorted(phases) if phases is not None else None,
                "mean_recall_at_k": round(mean_recall_at_k, 4),
                "mean_cleanliness": (
                    round(mean_cleanliness, 4) if mean_cleanliness is not None else None
                ),
                "llm_calls_during_dream": llm_calls,
                "phase_summary": phase_summary,
                "per_query": per_query,
                "degraded_mode": report.degraded_mode,
            }
        finally:
            await mem.close()


async def _score_queries(
    mem: Mnemoss,
    corpus: dict[str, Any],
    cfg: dict[str, Any],
    id_map: dict[str, str],
) -> list[dict[str, Any]]:
    k = int(cfg["harness"]["recall_k"])
    include_deep = bool(cfg["harness"]["include_deep"])
    auto_expand = bool(cfg["harness"]["auto_expand"])

    # Reverse map for translating mnemoss_id back to corpus_id.
    reverse = {v: k for k, v in id_map.items()}

    per_query: list[dict[str, Any]] = []
    for q in corpus["queries"]:
        results = await mem.recall(
            q["query"],
            k=k,
            include_deep=include_deep,
            auto_expand=auto_expand,
        )
        predicted_corpus = []
        for r in results:
            cid = reverse.get(r.memory.id)
            if cid is not None:
                predicted_corpus.append(cid)

        relevant = set(q["relevant_ids"])
        if relevant:
            hits = len(set(predicted_corpus[:k]) & relevant)
            r_at_k = hits / len(relevant)
        else:
            r_at_k = 0.0

        row: dict[str, Any] = {
            "query": q["query"],
            "kind": q.get("kind", "unknown"),
            "predicted_corpus_ids": predicted_corpus[:k],
            "relevant_ids": q["relevant_ids"],
            "recall_at_k": round(r_at_k, 4),
        }
        # Cleanliness — only meaningful for queries that have a
        # junk_ids set (pressure corpus).
        if "junk_ids" in q:
            junk = set(q["junk_ids"])
            row["junk_ids"] = q["junk_ids"]
            row["cleanliness"] = float(topk_cleanliness(predicted_corpus, junk, k=k))
        per_query.append(row)
    return per_query


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
            extras = (
                f"  cleanliness={row['mean_cleanliness']:.4f}"
                if row["mean_cleanliness"] is not None
                else ""
            )
            print(
                f"    recall@k={row['mean_recall_at_k']:.4f}  "
                f"llm_calls={row['llm_calls_during_dream']}{extras}",
                flush=True,
            )
    return rows


def _evaluate_topology_gate(rows: list[dict[str, Any]]) -> str:
    by_label = {r["label"]: r for r in rows}
    if "full" not in by_label or "dreaming_off" not in by_label:
        return "decision gate skipped (need both 'full' and 'dreaming_off')"
    full = by_label["full"]["mean_recall_at_k"]
    off = by_label["dreaming_off"]["mean_recall_at_k"]
    delta_pp = (full - off) * 100.0
    if delta_pp < 5.0:
        return (
            f"DECISION GATE TRIPPED (topology): full={full:.4f} - "
            f"off={off:.4f} = {delta_pp:+.2f}pp < 5pp. Stop the study."
        )
    return (
        f"DECISION GATE PASSED (topology): full={full:.4f} - "
        f"off={off:.4f} = {delta_pp:+.2f}pp >= 5pp. Proceed to weekend 2."
    )


def _evaluate_pressure_gate(rows: list[dict[str, Any]]) -> str:
    """Per ``docs/dreaming-decision.md`` weekend 2 decision gate:
    if Dispose+Rebalance combined contribute < 2pp recall AND < 10%
    cleanliness delta on pressure corpus, mark both for CUT.
    """

    by_label = {r["label"]: r for r in rows}
    if "full" not in by_label or "no_dispose_no_rebalance" not in by_label:
        return "decision gate skipped (need 'full' + 'no_dispose_no_rebalance')"
    full = by_label["full"]
    minus = by_label["no_dispose_no_rebalance"]
    recall_delta_pp = (full["mean_recall_at_k"] - minus["mean_recall_at_k"]) * 100.0
    clean_delta_pct = (
        (full["mean_cleanliness"] - minus["mean_cleanliness"]) * 100.0
        if full["mean_cleanliness"] is not None and minus["mean_cleanliness"] is not None
        else None
    )
    parts = [
        f"recall@k delta = {recall_delta_pp:+.2f}pp",
    ]
    if clean_delta_pct is not None:
        parts.append(f"cleanliness delta = {clean_delta_pct:+.2f}%")

    if recall_delta_pp < 2.0 and (clean_delta_pct is None or clean_delta_pct < 10.0):
        return f"PRESSURE GATE TRIPPED: {', '.join(parts)} → CUT Dispose + Rebalance."
    return f"PRESSURE GATE PASSED: {', '.join(parts)} → keep Dispose / Rebalance."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mnemoss dreaming-validation ablation harness.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Topology decision gate: full vs dreaming_off.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Topology full ablation matrix (14 conditions).",
    )
    parser.add_argument(
        "--pressure-binary",
        action="store_true",
        help="Pressure decision gate: full vs dreaming_off on pressure corpus.",
    )
    parser.add_argument(
        "--pressure-full",
        action="store_true",
        help="Pressure full matrix (Dispose + Rebalance focus, 7 conditions).",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Override the corpus path. Default: topology or pressure based on --pressure-*.",
    )
    parser.add_argument(
        "--config",
        default="bench/ablate_dreaming.toml",
        help="Path to the pinned config TOML.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write results.jsonl. Default: bench/results/{kind}_results.jsonl.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help=(
            "Use MockLLMClient instead of OpenRouter. Consolidate "
            "returns a canned synthetic summary so the harness runs "
            "deterministically end-to-end without network. "
            "Useful for validating the harness when free-tier rate "
            "limits make real runs infeasible. The Consolidate "
            "verdict is fictional under --mock-llm; only structural "
            "phases (Cluster, Relations, Rebalance, Dispose) "
            "produce trustworthy ablation deltas."
        ),
    )
    args = parser.parse_args()

    global USE_MOCK_LLM
    USE_MOCK_LLM = args.mock_llm
    if USE_MOCK_LLM:
        print(
            "warning: --mock-llm active. Consolidate's results are SYNTHETIC. "
            "Use only to validate the harness end-to-end.",
            flush=True,
        )

    selected = [args.binary, args.full, args.pressure_binary, args.pressure_full]
    if sum(selected) != 1:
        print(
            "error: pass exactly one of --binary / --full / --pressure-binary / --pressure-full.",
            file=sys.stderr,
        )
        return 2

    is_pressure = args.pressure_binary or args.pressure_full
    if args.corpus is None:
        args.corpus = (
            "bench/fixtures/pressure_corpus_seed42.jsonl"
            if is_pressure
            else "bench/fixtures/topology_corpus.json"
        )
    if args.out is None:
        args.out = (
            "bench/results/pressure_results.jsonl"
            if is_pressure
            else "bench/results/topology_results.jsonl"
        )

    corpus = _load_corpus(Path(args.corpus))
    cfg = _load_config(Path(args.config))

    # Sanity: pressure flags + topology corpus (or vice versa) is a
    # user error worth catching loudly.
    if is_pressure and not _is_pressure_corpus(corpus):
        print(
            f"error: --pressure-* requires a pressure corpus "
            f"(missing _meta.simulated_days in {args.corpus})",
            file=sys.stderr,
        )
        return 2
    if not is_pressure and _is_pressure_corpus(corpus):
        print(
            f"error: pressure corpus passed without --pressure-* flag ({args.corpus})",
            file=sys.stderr,
        )
        return 2

    if args.pressure_full:
        mode = "pressure_full"
    elif args.pressure_binary:
        mode = "pressure_binary"
    elif args.full:
        mode = "full"
    else:
        mode = "binary"

    matrix = _ablation_matrix(mode)
    print(
        f"running {len(matrix)} ablation(s) on "
        f"{len(corpus['memories'])} memories, {len(corpus['queries'])} queries → {args.out}",
        flush=True,
    )

    rows = asyncio.run(_run_matrix(matrix, corpus, cfg, Path(args.out)))

    print()
    if USE_MOCK_LLM:
        print(
            "DECISION GATE SUPPRESSED: --mock-llm active. The recall@k "
            "delta under mock mode reflects synthetic summary content, "
            "not real Consolidate behavior. Re-run without --mock-llm "
            "(real LLM) to record a verdict in docs/dreaming-decision.md.",
        )
    elif is_pressure:
        print(_evaluate_pressure_gate(rows))
    else:
        print(_evaluate_topology_gate(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
