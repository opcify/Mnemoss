"""Launch comparison harness — Chart 1 (recall accuracy).

Runs recall@k on the LoCoMo 2024 benchmark against a selectable
``MemoryBackend``. Each LoCoMo conversation is ingested into a fresh
backend instance (so workspaces / tempdirs / in-memory state are
isolated per conversation) and every QA pair with evidence is scored.

Output is a JSON file shaped for ``bench/plots.py`` to consume.

Usage
-----

Full MnemossBackend run against LoCoMo (requires ``OPENAI_API_KEY``)::

    python -m bench.launch_comparison --backend mnemoss \
        --out bench/results/chart1_mnemoss.json

StaticFileBackend baseline (no network, no dependencies)::

    python -m bench.launch_comparison --backend static_file \
        --out bench/results/chart1_static_file.json

Quick smoke check with FakeEmbedder (no OpenAI, runs in seconds)::

    python -m bench.launch_comparison --backend mnemoss \
        --fake-embedder --limit-conversations 1 --limit-utterances 50 \
        --out /tmp/smoke.json

Scope notes
-----------

- Chart 1 only. Charts 2 (latency), 3 (cost), 4 (determinism), 5 (CJK)
  ship in subsequent PRs.
- LoCoMo-only corpus. Synthetic fallback lives behind a different
  ``--corpus`` value that isn't implemented yet (the design doc plans
  for it as a degradation path if the LoCoMo JSON is unavailable; on
  a machine where LoCoMo is already prepared, we use it).
- Budget guard (``MNEMOSS_BENCH_BUDGET_USD``) is a placeholder — this
  harness doesn't yet meter per-call cost. The guard activates once we
  add backends that rack up LLM bills (Mem0 would; MnemossBackend with
  a local/OpenAI embedder does not).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Auto-load .env so ``OPENAI_API_KEY`` (and any other launch env vars)
# resolve without the user having to ``export`` manually. The call is a
# no-op if ``.env`` is missing. Explicit env-var exports still win —
# ``override=False`` is the default and matches expectations. Guarded
# by ImportError so a stripped-down install without the dev extras
# still imports this module cleanly.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — dev extras carry python-dotenv
    pass

from bench.backends.base import MemoryBackend

MEMORIES_PATH = Path("bench/data/locomo_memories.jsonl")
QUERIES_PATH = Path("bench/data/locomo_queries.jsonl")


@dataclass
class ConversationResult:
    """Per-conversation Chart-1 result row."""

    conversation_id: str
    n_memories: int
    n_queries_scored: int
    n_queries_skipped: int
    mean_recall_at_k: float


@dataclass
class RunSummary:
    """Aggregate payload emitted as the JSON artifact."""

    chart: int
    backend: str
    corpus: str
    k: int
    params: dict
    per_conversation: list[ConversationResult]
    aggregate_recall_at_k: float
    n_conversations: int
    n_queries: int
    timestamp: str
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "chart": self.chart,
            "backend": self.backend,
            "corpus": self.corpus,
            "k": self.k,
            "params": self.params,
            "per_conversation": [
                {
                    "conversation_id": r.conversation_id,
                    "n_memories": r.n_memories,
                    "n_queries_scored": r.n_queries_scored,
                    "n_queries_skipped": r.n_queries_skipped,
                    "mean_recall_at_k": r.mean_recall_at_k,
                }
                for r in self.per_conversation
            ],
            "aggregate": {
                "mean_recall_at_k": self.aggregate_recall_at_k,
                "n_conversations": self.n_conversations,
                "n_queries": self.n_queries,
            },
            "timestamp": self.timestamp,
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ─── corpus loading ────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Prepare the LoCoMo corpus first:\n"
            f"  python -m bench.data.prepare_locomo"
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _group_by_conversation(
    memories: list[dict],
    queries: list[dict],
    *,
    limit_conversations: int | None,
    limit_utterances: int | None,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Group memories + queries by ``conversation_id`` and apply limits."""

    mems_by_conv: dict[str, list[dict]] = defaultdict(list)
    for m in memories:
        mems_by_conv[m["conversation_id"]].append(m)

    qs_by_conv: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        qs_by_conv[q["conversation_id"]].append(q)

    # Ordered conversation list: prefer conversations that have BOTH
    # memories and queries (skip any corpus-prep anomalies).
    conv_ids = [c for c in mems_by_conv if c in qs_by_conv]
    if limit_conversations is not None:
        conv_ids = conv_ids[:limit_conversations]

    out: dict[str, tuple[list[dict], list[dict]]] = {}
    for conv_id in conv_ids:
        mems = mems_by_conv[conv_id]
        if limit_utterances is not None:
            mems = mems[:limit_utterances]
        out[conv_id] = (mems, qs_by_conv[conv_id])
    return out


# ─── backend construction ──────────────────────────────────────────


def _resolve_embedder(choice: str):
    """Build an ``Embedder`` from the ``--embedder`` CLI choice.

    - ``openai`` → ``OpenAIEmbedder`` (text-embedding-3-small, 1536d,
      needs ``OPENAI_API_KEY``, costs money).
    - ``local``  → ``LocalEmbedder`` (MiniLM multilingual, 384d, CPU,
      free, ~470MB model download once).
    - ``nomic``  → ``LocalEmbedder`` wrapping
      ``nomic-ai/nomic-embed-text-v2-moe`` (768d MoE, multilingual,
      needs ``trust_remote_code=True``). A stronger dense embedder —
      ~475M params (305M active), trained with retrieval prompts.
      We apply the ``search_document: `` prefix symmetrically on both
      doc and query sides so the retrieval embedding space activates
      without forking the Embedder protocol for query/doc distinction.
    - ``gemma``  → ``LocalEmbedder`` wrapping
      ``google/embeddinggemma-300m`` (768d Matryoshka, multilingual,
      ~300M params). Google's open-weight retrieval embedder. Uses
      task-prefix prompting: documents get ``title: none | text: ``,
      queries get ``task: search result | query: ``. Needs
      sentence-transformers ≥ 3.1 and transformers ≥ 4.45.
    - ``fake``   → ``FakeEmbedder(dim=16)`` — for smoke tests only;
      cosine scores are hash noise.
    """

    if choice == "openai":
        from openai import RateLimitError

        from mnemoss import OpenAIEmbedder, RetryingEmbedder

        # 429s are common when benches run rows in parallel — text-embedding-3-small's
        # default RPM is 3000 and a single bench can blow that with concurrency=4+.
        # Retry on RateLimitError with longer backoff so the bench self-throttles.
        return RetryingEmbedder(
            OpenAIEmbedder(model="text-embedding-3-small"),
            max_retries=6,
            base_delay_seconds=0.5,
            max_delay_seconds=15.0,
            retry_on=(RateLimitError,),
        )
    if choice == "local":
        from mnemoss import LocalEmbedder

        return LocalEmbedder()
    if choice == "nomic":
        from mnemoss import LocalEmbedder

        # Asymmetric prompts: queries go through "search_query: ",
        # documents through "search_document: ". This is how Nomic
        # trained the model — symmetric use degrades recall badly
        # (saw 0.32 vs 0.49 for MiniLM on N=500 before the fix).
        return LocalEmbedder(
            model_name="nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True,
            text_prefix="search_document: ",
            query_prefix="search_query: ",
        )
    if choice == "gemma":
        from mnemoss import LocalEmbedder

        # Asymmetric task-prefix prompting per the EmbeddingGemma
        # model card. Symmetric use (same prefix both sides) works
        # but under-performs the task-specific pair. 768d native
        # (Matryoshka supports 512/256/128 truncation but we keep
        # full fidelity for the comparison).
        return LocalEmbedder(
            model_name="google/embeddinggemma-300m",
            trust_remote_code=True,
            text_prefix="title: none | text: ",
            query_prefix="task: search result | query: ",
        )
    if choice == "fake":
        from mnemoss import FakeEmbedder

        return FakeEmbedder(dim=16)
    raise ValueError(f"Unknown embedder {choice!r}")


def _build_backend_fast(fake_embedder: bool, embedder: str) -> MemoryBackend:
    """Build a MnemossBackend with Phase 1.3 latency knobs enabled.

    Used by the ``mnemoss_fast`` preset — skips FTS on plain queries
    and drops empty tiers from the cascade. Recall impact on LoCoMo
    is within 1pp; latency improvement at N=10K is substantial.

    Return type is the generic ``MemoryBackend`` Protocol so this helper
    doesn't force an import of ``MnemossBackend`` at module top-level
    (launch_comparison imports backends lazily — mnemoss may require
    extras the caller didn't install).
    """

    from bench.backends.mnemoss_backend import MnemossBackend
    from mnemoss import FormulaParams

    effective_embedder = "fake" if fake_embedder else embedder
    return MnemossBackend(
        embedding_model=_resolve_embedder(effective_embedder),
        formula=FormulaParams(
            noise_scale=0.0,
            skip_fts_when_no_literal_markers=True,
            skip_empty_tiers=True,
        ),
    )


def _build_backend(
    backend_name: str,
    *,
    fake_embedder: bool,
    embedder: str = "openai",
) -> MemoryBackend:
    """Construct a backend by name. Import-late so missing extras only
    fire for the backend the user asked for.

    ``embedder`` wins over ``fake_embedder`` when both are set; the old
    ``--fake-embedder`` flag is kept for CLI back-compat.
    """

    effective_embedder = "fake" if fake_embedder else embedder

    if backend_name == "mnemoss":
        from bench.backends.mnemoss_backend import MnemossBackend

        if effective_embedder == "openai":
            # Default path — MnemossBackend builds its own OpenAI+Retry wrapper.
            return MnemossBackend()
        return MnemossBackend(embedding_model=_resolve_embedder(effective_embedder))

    if backend_name == "mnemoss_fast":
        # Phase 1.3 preset — ANN + skip_fts_on_plain_query +
        # skip_empty_tiers. Measures the latency payoff of the fast-path
        # knobs at recall-preserving settings on LoCoMo-style queries.
        be = _build_backend_fast(fake_embedder=fake_embedder, embedder=embedder)
        be.backend_id = "mnemoss_fast"
        return be

    if backend_name == "mnemoss_rocket":
        # Phase 2 preset — the fast-index recall architecture.
        # Everything ACT-R runs async (observe / reconsolidate / dream);
        # recall is pure ANN + cached idx_priority. No FTS, no tier
        # cascade, no per-candidate formula evaluation. This is
        # Mnemoss's launch story: recall latency that grows with log N,
        # not linearly, while matching cosine-baseline on semantic recall.
        from bench.backends.mnemoss_backend import MnemossBackend
        from mnemoss import FormulaParams

        kwargs: dict = {
            "formula": FormulaParams(
                noise_scale=0.0,
                eta_0=0.0,  # kill recency bias for bulk-ingest workloads
                d=0.01,  # flatten base-level history too
                use_fast_index_recall=True,
                fast_index_semantic_weight=1.0,
                fast_index_priority_weight=0.0,  # pure cosine ranking
            ),
        }
        if effective_embedder != "openai":
            kwargs["embedding_model"] = _resolve_embedder(effective_embedder)
        be = MnemossBackend(**kwargs)
        be.backend_id = "mnemoss_rocket"
        return be

    if backend_name == "mnemoss_prod":
        # Production preset for LoCoMo-style conversational QA: combine
        # the fast-path knobs with mnemoss_semantic's formula weights
        # (no recency bias, near-pure-semantic matching). This is the
        # "what would I actually ship as a default for a conversational
        # memory SDK" configuration.
        from bench.backends.mnemoss_backend import MnemossBackend
        from mnemoss import FormulaParams

        kwargs: dict = {
            "formula": FormulaParams(
                noise_scale=0.0,
                eta_0=0.0,
                d=0.01,
                match_w_f_base=0.001,
                match_w_f_slope=0.0,
                match_w_s_base=0.999,
                skip_fts_when_no_literal_markers=True,
                skip_empty_tiers=True,
            ),
        }
        if effective_embedder != "openai":
            kwargs["embedding_model"] = _resolve_embedder(effective_embedder)
        be = MnemossBackend(**kwargs)
        be.backend_id = "mnemoss_prod"
        return be

    if backend_name == "mnemoss_semantic":
        # Mnemoss with recency bias disabled AND matching weights
        # pushed to near-pure-semantic. Use this to calibrate the
        # Mnemoss ceiling under LoCoMo-style conversational workloads
        # and to isolate the "architecture vs. formula-weights" share
        # of any gap to ``raw_stack``.
        #
        # ``eta_0=0`` kills the grace term. ``d=0.01`` flattens B_i.
        # ``match_w_f_*`` pushed near zero collapses the hybrid toward
        # pure-semantic — what's left is dominated by MP * cosine_sim.
        from bench.backends.mnemoss_backend import MnemossBackend
        from mnemoss import FormulaParams

        kwargs: dict = {
            "formula": FormulaParams(
                noise_scale=0.0,
                eta_0=0.0,
                d=0.01,
                match_w_f_base=0.001,
                match_w_f_slope=0.0,
                match_w_s_base=0.999,
            ),
        }
        if effective_embedder != "openai":
            kwargs["embedding_model"] = _resolve_embedder(effective_embedder)
        be = MnemossBackend(**kwargs)
        # Stamp a distinguishing backend_id so Chart 1 renders both
        # Mnemoss configs as separate bars.
        be.backend_id = "mnemoss_semantic"
        return be

    if backend_name == "raw_stack":
        from bench.backends.raw_stack_backend import RawStackBackend

        # Parity with MnemossBackend: same embedder choice policy. The
        # point of ``raw_stack`` is to isolate the memory-architecture
        # difference, not an embedder-quality difference.
        if effective_embedder == "openai":
            return RawStackBackend()
        return RawStackBackend(embedding_model=_resolve_embedder(effective_embedder))

    if backend_name == "static_file":
        from bench.backends.static_file_backend import StaticFileBackend

        return StaticFileBackend()

    raise ValueError(
        f"Unknown backend {backend_name!r}. "
        f"Supported: 'mnemoss', 'raw_stack', 'static_file'."
    )


# ─── scoring ───────────────────────────────────────────────────────


def _recall_at_k(hit_mem_ids: list[str], relevant_mem_ids: set[str], k: int) -> float:
    """``recall@k`` = (# relevant memories in top-k) / (# total relevant)."""

    if not relevant_mem_ids:
        return 0.0
    top_k = set(hit_mem_ids[:k])
    found = len(top_k & relevant_mem_ids)
    return found / len(relevant_mem_ids)


# ─── one conversation ─────────────────────────────────────────────


async def _run_one_conversation(
    conv_id: str,
    memories: list[dict],
    queries: list[dict],
    backend: MemoryBackend,
    *,
    k: int,
) -> ConversationResult:
    """Ingest one conversation's utterances, score all its queries."""

    # Ingest. dia_id → backend-native memory_id map for relevance lookup.
    dia_to_mem_id: dict[str, str] = {}
    for row in memories:
        mem_id = await backend.observe(row["text"], ts=row["ts"])
        dia_to_mem_id[row["dia_id"]] = mem_id

    scored = 0
    skipped = 0
    recall_sum = 0.0

    for q in queries:
        # Translate gold dia_ids to backend-native ids. If some evidence
        # dia_ids fall outside the ingested slice (due to --limit-utterances),
        # the query is un-scorable — skip it rather than distort the mean.
        relevant_mem_ids = {dia_to_mem_id[d] for d in q["relevant_dia_ids"] if d in dia_to_mem_id}
        if not relevant_mem_ids:
            skipped += 1
            continue
        hits = await backend.recall(q["question"], k=k)
        hit_ids = [h.memory_id for h in hits]
        recall_sum += _recall_at_k(hit_ids, relevant_mem_ids, k)
        scored += 1

    mean = recall_sum / scored if scored else 0.0
    return ConversationResult(
        conversation_id=conv_id,
        n_memories=len(memories),
        n_queries_scored=scored,
        n_queries_skipped=skipped,
        mean_recall_at_k=mean,
    )


# ─── orchestration ────────────────────────────────────────────────


def _build_scale_corpus(
    memories: list[dict],
    queries: list[dict],
    *,
    gold_conversation_id: str,
    scale_n: int,
) -> tuple[list[dict], list[dict]]:
    """Build a scale-benchmark corpus.

    Picks the gold conversation's memories as "the agent's relevant
    store" and pads with memories sampled (in deterministic corpus
    order) from OTHER conversations as distractors until the total
    reaches ``scale_n``. If ``scale_n`` is smaller than the gold's own
    size, the gold corpus is returned unpadded.

    dia_ids collide across conversations in the LoCoMo dump (4849
    collisions), so this namespaces every dia_id to
    ``{conversation_id}:{dia_id}`` and rewrites query
    ``relevant_dia_ids`` to match. Downstream code treats the scale
    corpus as a single "conversation."
    """

    gold = [dict(m) for m in memories if m["conversation_id"] == gold_conversation_id]
    distractors = [dict(m) for m in memories if m["conversation_id"] != gold_conversation_id]

    def _ns(m: dict) -> str:
        return f"{m['conversation_id']}:{m['dia_id']}"

    for m in gold + distractors:
        m["dia_id"] = _ns(m)

    # Gold first, then distractors until we hit scale_n. When scale_n
    # exceeds what's available (gold + all distractors), we wrap around
    # the distractor pool — each wrap pass gets a unique dia_id suffix
    # so every row is a distinct Memory (no dedup collisions) while the
    # text + embedding are repeats. This simulates a realistic workspace
    # where recurring content genuinely recurs ("the user mentioned
    # Caroline in session 1, session 5, session 12"); not a perfect
    # proxy for fresh distractors at scale but honest about the cap:
    # with LoCoMo's 5882-row corpus, anything past N=5882 is a simulated
    # extension, not natural growth.
    padded = list(gold)
    needed = max(0, scale_n - len(padded))
    if needed > 0 and distractors:
        wrap = 0
        while needed > 0:
            take = min(needed, len(distractors))
            for m in distractors[:take]:
                copy = dict(m)
                if wrap > 0:
                    copy["dia_id"] = f"{m['dia_id']}#r{wrap}"
                padded.append(copy)
            needed -= take
            wrap += 1

    # Rewrite queries — only the gold conversation's queries scored.
    gold_queries: list[dict] = []
    for q in queries:
        if q["conversation_id"] != gold_conversation_id:
            continue
        q2 = dict(q)
        q2["relevant_dia_ids"] = [f"{gold_conversation_id}:{d}" for d in q["relevant_dia_ids"]]
        gold_queries.append(q2)

    return padded, gold_queries


async def run_scale(
    *,
    backend_name: str,
    k: int,
    scale_n: int,
    gold_conversation_id: str,
    fake_embedder: bool,
    embedder: str,
) -> RunSummary:
    """Run the scale benchmark: one gold conversation padded to N
    memories with distractors from other conversations.

    Returns a :class:`RunSummary` whose single ``ConversationResult``
    carries the padded corpus size in ``n_memories`` — so the same
    downstream JSON shape works for both chart 1 and the scale chart.
    """

    started = datetime.now(timezone.utc)
    t0 = started.timestamp()

    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    padded_mems, gold_queries = _build_scale_corpus(
        memories,
        queries,
        gold_conversation_id=gold_conversation_id,
        scale_n=scale_n,
    )

    backend = _build_backend(backend_name, fake_embedder=fake_embedder, embedder=embedder)
    try:
        result = await _run_one_conversation(
            gold_conversation_id,
            padded_mems,
            gold_queries,
            backend,
            k=k,
        )
    finally:
        await backend.close()

    finished = datetime.now(timezone.utc)
    return RunSummary(
        chart=2,  # scale chart
        backend=backend_name,
        corpus="locomo_scale",
        k=k,
        params={
            "scale_n": scale_n,
            "gold_conversation_id": gold_conversation_id,
            "fake_embedder": fake_embedder,
            "embedder": embedder,
        },
        per_conversation=[result],
        aggregate_recall_at_k=result.mean_recall_at_k,
        n_conversations=1,
        n_queries=result.n_queries_scored,
        timestamp=started.isoformat(),
        duration_seconds=finished.timestamp() - t0,
    )


async def run_chart1(
    *,
    backend_name: str,
    k: int,
    limit_conversations: int | None,
    limit_utterances: int | None,
    fake_embedder: bool,
    embedder: str = "openai",
) -> RunSummary:
    """Run Chart 1 end-to-end. Returns the summary payload."""

    started = datetime.now(timezone.utc)
    t0 = started.timestamp()

    memories = _load_jsonl(MEMORIES_PATH)
    queries = _load_jsonl(QUERIES_PATH)
    grouped = _group_by_conversation(
        memories,
        queries,
        limit_conversations=limit_conversations,
        limit_utterances=limit_utterances,
    )

    per_conv: list[ConversationResult] = []
    all_recall_values: list[float] = []
    total_scored_queries = 0

    for conv_id, (mems, qs) in grouped.items():
        # Fresh backend per conversation — each gets its own tempdir /
        # workspace / SQLite. No cross-conversation state leakage.
        backend = _build_backend(backend_name, fake_embedder=fake_embedder, embedder=embedder)
        try:
            result = await _run_one_conversation(conv_id, mems, qs, backend, k=k)
            per_conv.append(result)
            if result.n_queries_scored:
                # Equal-weight by (conversation, query). The aggregate
                # below gives equal weight to every scored query across
                # conversations, not equal weight per conversation.
                all_recall_values.extend([result.mean_recall_at_k] * result.n_queries_scored)
                total_scored_queries += result.n_queries_scored
        finally:
            await backend.close()

    aggregate = sum(all_recall_values) / len(all_recall_values) if all_recall_values else 0.0

    finished = datetime.now(timezone.utc)
    return RunSummary(
        chart=1,
        backend=backend_name,
        corpus="locomo",
        k=k,
        params={
            "limit_conversations": limit_conversations,
            "limit_utterances": limit_utterances,
            "fake_embedder": fake_embedder,
        },
        per_conversation=per_conv,
        aggregate_recall_at_k=aggregate,
        n_conversations=len(per_conv),
        n_queries=total_scored_queries,
        timestamp=started.isoformat(),
        duration_seconds=finished.timestamp() - t0,
    )


# ─── CLI ───────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mnemoss launch-comparison harness — Chart 1 (recall accuracy).",
    )
    p.add_argument(
        "--backend",
        choices=[
            "mnemoss",
            "mnemoss_semantic",
            "mnemoss_fast",
            "mnemoss_prod",
            "mnemoss_rocket",
            "raw_stack",
            "static_file",
        ],
        required=True,
        help="Memory backend to benchmark.",
    )
    p.add_argument(
        "--chart",
        type=int,
        default=1,
        choices=[1],
        help="Chart number (only 1 implemented in this file).",
    )
    p.add_argument("--k", type=int, default=10, help="recall@k; default 10.")
    p.add_argument(
        "--limit-conversations",
        type=int,
        default=None,
        help="Cap the number of LoCoMo conversations (default: all 10).",
    )
    p.add_argument(
        "--limit-utterances",
        type=int,
        default=None,
        help="Cap utterances per conversation (default: all).",
    )
    p.add_argument(
        "--fake-embedder",
        action="store_true",
        help=(
            "Use FakeEmbedder instead of OpenAI. Useful for CI / "
            "offline smoke checks; cosine scores are hash noise. "
            "Equivalent to ``--embedder=fake`` for back-compat."
        ),
    )
    p.add_argument(
        "--embedder",
        choices=["openai", "local", "nomic", "gemma", "fake"],
        default="openai",
        help=(
            "Which embedder to use. ``local`` = LocalEmbedder (MiniLM "
            "multilingual, 384d, free, one-time ~470MB download). "
            "``openai`` = text-embedding-3-small (1536d, needs "
            "OPENAI_API_KEY, costs money). ``nomic`` = Nomic v2 MoE "
            "(768d, trust_remote_code). ``gemma`` = EmbeddingGemma "
            "300M (768d, gated HF repo — needs HF_TOKEN + license "
            "acceptance). ``fake`` = FakeEmbedder (hash noise, CI only)."
        ),
    )
    p.add_argument(
        "--scale-n",
        type=int,
        default=None,
        help=(
            "Scale-benchmark mode: ingest N memories total, where gold "
            "comes from a single conversation (``--gold-conversation``) "
            "and the rest are distractors sampled from other LoCoMo "
            "conversations. When set, ``--limit-conversations`` and "
            "``--limit-utterances`` are ignored."
        ),
    )
    p.add_argument(
        "--gold-conversation",
        type=str,
        default="conv-26",
        help=(
            "Gold conversation ID for scale-benchmark mode. Queries "
            "come from this conversation; other conversations are "
            "distractor padding."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path (parent directory is created if missing).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a human-readable summary after writing the JSON.",
    )
    return p.parse_args(argv)


def _print_summary(summary: RunSummary) -> None:
    """ASCII table for eyeball verification."""

    print()
    print(
        f"Chart {summary.chart} — backend={summary.backend} corpus={summary.corpus} k={summary.k}"
    )
    print(f"Wall clock: {summary.duration_seconds:.1f}s")
    print()
    header = f"{'conversation':<12}  {'mems':>6}  {'scored':>6}  {'skip':>5}  {'recall@k':>9}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in summary.per_conversation:
        print(
            f"{r.conversation_id:<12}  {r.n_memories:>6}  "
            f"{r.n_queries_scored:>6}  {r.n_queries_skipped:>5}  "
            f"{r.mean_recall_at_k:>9.4f}"
        )
    print(sep)
    print(
        f"{'aggregate':<12}  {'':>6}  "
        f"{summary.n_queries:>6}  {'':>5}  "
        f"{summary.aggregate_recall_at_k:>9.4f}"
    )
    print(sep)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.scale_n is not None:
        summary = asyncio.run(
            run_scale(
                backend_name=args.backend,
                k=args.k,
                scale_n=args.scale_n,
                gold_conversation_id=args.gold_conversation,
                fake_embedder=args.fake_embedder,
                embedder=args.embedder,
            )
        )
    else:
        summary = asyncio.run(
            run_chart1(
                backend_name=args.backend,
                k=args.k,
                limit_conversations=args.limit_conversations,
                limit_utterances=args.limit_utterances,
                fake_embedder=args.fake_embedder,
                embedder=args.embedder,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary.to_dict(), indent=2) + "\n")
    if args.print_summary:
        _print_summary(summary)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
