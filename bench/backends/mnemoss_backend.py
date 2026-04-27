"""Mnemoss adapter for the launch-comparison benchmark.

Default config for published-chart runs
---------------------------------------

- Embedder: ``text-embedding-3-small`` via ``OpenAIEmbedder`` wrapped
  in ``RetryingEmbedder`` (Issue 1.1A + 2.b from eng review). Pinning
  this way gives fair numbers against Chroma, which uses the same
  model, and keeps transient network hiccups from tanking a multi-
  hour Chart 1 run.
- Recall: ``auto_expand=False, include_deep=True``. Benchmark wants
  the primary recall path (Mem0 and Chroma have no auto-expand
  analogue) at Mnemoss's full tier cascade (HOT → WARM → COLD → DEEP).
  The simulation in ``demo/simulate.py`` flips these — see Issue 2.e.
- Noise: ``FormulaParams.noise_scale=0.0`` by default. Deterministic
  ranking across benchmark runs so Chart 4 Kendall tau is 1.0 by
  construction on the Mnemoss side. Override to a non-zero value if
  you're deliberately measuring noise-sensitivity.

Gap A cleanup (eng review)
--------------------------

Each instance gets a fresh ``tempfile.mkdtemp`` workspace root. A
leftover workspace from a crashed previous run would re-trigger
``WorkspaceLockError`` on re-run; fresh tempdirs per instance and
unconditional cleanup on ``close()`` close that gap.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from bench.backends.base import RecallHit
from mnemoss import (
    Embedder,
    FormulaParams,
    Mnemoss,
    OpenAIEmbedder,
    RetryingEmbedder,
    StorageParams,
    TierCapacityParams,
)


class MnemossBackend:
    """``MemoryBackend`` adapter around ``Mnemoss`` for cross-system benchmarks.

    Parameters
    ----------
    formula:
        Override the full ``FormulaParams``. If supplied, ``noise_scale``
        is ignored (use the field on the supplied dataclass). Use this
        to benchmark non-default activation configs — e.g. a
        "semantic-only" preset that kills recency bias for bulk-ingest
        workloads like LoCoMo where all memories were just created at
        once and the recency term otherwise drowns out matching.

        Example::

            FormulaParams(noise_scale=0.0, eta_0=0.0, d=0.01)

        ``eta_0=0`` disables the grace term; ``d=0.01`` (can't be zero
        per the validator) nearly flattens the history term. What's
        left is dominated by ``MP * cosine_sim``, which is essentially
        raw-stack cosine scoring with Mnemoss's candidate-selection +
        query-bias layer on top.
    """

    backend_id = "mnemoss"

    def __init__(
        self,
        *,
        embedding_model: Embedder | None = None,
        workspace: str = "bench",
        noise_scale: float = 0.0,
        include_deep: bool = True,
        formula: FormulaParams | None = None,
        encoder: Any | None = None,
        tier_capacity: TierCapacityParams | None = None,
    ) -> None:
        # Tempdir first, before Mnemoss() — stored on self so close()
        # can rmtree it even if the constructor below raises. This is
        # what closes Gap A from the eng review.
        self._tempdir = Path(tempfile.mkdtemp(prefix="mnemoss_bench_"))
        self._closed = False

        if embedding_model is None:
            embedding_model = RetryingEmbedder(
                OpenAIEmbedder(model="text-embedding-3-small"),
                max_retries=3,
            )

        if formula is None:
            formula = FormulaParams(noise_scale=noise_scale)

        kwargs: dict[str, Any] = {
            "workspace": workspace,
            "embedding_model": embedding_model,
            "formula": formula,
            "storage": StorageParams(root=self._tempdir),
        }
        if encoder is not None:
            kwargs["encoder"] = encoder
        if tier_capacity is not None:
            kwargs["tier_capacity"] = tier_capacity

        self._mem = Mnemoss(**kwargs)
        self._include_deep = include_deep

    async def observe(self, text: str, ts: float) -> str:
        """Observe ``text`` as a user message. Returns Mnemoss's memory id.

        ``ts`` is accepted for protocol uniformity but not passed to
        Mnemoss — Mnemoss uses its own ``datetime.now(UTC)`` inside
        ``observe()`` (see ``mnemoss/client.py:171``). Benchmarks that
        need replay-able timestamps should run against a corpus whose
        natural ordering matches ingestion order.

        ``turn_id`` is intentionally NOT supplied. Per ``client.py:173``,
        a missing ``turn_id`` sets ``auto_close=True`` so each observe
        produces exactly one Memory row that flushes before the call
        returns. Matches ``bench/bench_recall.py`` and gives us the
        predictable "one observe = one memory" semantics the benchmark
        needs.
        """

        # ``role="user"`` passes ``EncoderParams.encoded_roles`` by default.
        memory_id = await self._mem.observe(
            role="user",
            content=text,
            session_id="bench",
        )
        if memory_id is None:
            # Role filter would have to be actively reconfigured to drop
            # ``user`` for this to trigger — treat as a loud misconfig,
            # not a silent skip. If it fires, the caller's Mnemoss
            # construction has an encoder override we didn't expect.
            raise RuntimeError(
                "MnemossBackend.observe: Mnemoss returned None for "
                "role='user'. Check EncoderParams.encoded_roles — "
                "benchmark runs expect all roles enabled."
            )
        return memory_id

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Recall top-k with Mnemoss's full cascade, no auto-expand.

        ``reconsolidate=False`` is critical for benchmark fairness:
        Mnemoss's default reconsolidation appends ``now`` to each
        returned memory's ``access_history``, growing ``B_i`` for the
        next recall. Across hundreds of benchmark queries, the
        earliest-recalled memories get massive B_i boosts regardless
        of actual relevance, biasing the remaining queries' rankings.
        See ``docs/ROOT_CAUSE.md``.
        """

        results = await self._mem.recall(
            query,
            k=k,
            include_deep=self._include_deep,
            auto_expand=False,
            reconsolidate=False,
        )
        return [
            RecallHit(
                memory_id=r.memory.id,
                rank=i + 1,
                score=float(r.score),
            )
            for i, r in enumerate(results)
        ]

    async def explain(self, query: str, memory_id: str) -> dict | None:
        """Return Mnemoss's ``ActivationBreakdown`` for one memory as a
        JSON-safe dict, or ``None`` if the memory isn't in the workspace.

        This is the hook :mod:`demo.simulate` uses to capture the per-
        component scores (``B_i``, spreading, matching, noise) into
        trace events so the player can render the stacked-bar
        breakdown on recall events. Not part of the generic
        ``MemoryBackend`` protocol — Mnemoss-specific.
        """

        breakdown = await self._mem.explain_recall(query, memory_id)
        return breakdown.to_dict() if breakdown is not None else None

    async def close(self) -> None:
        """Close the Mnemoss handle and rmtree the tempdir. Idempotent."""

        if self._closed:
            return
        self._closed = True
        try:
            await self._mem.close()
        finally:
            # Unconditional cleanup — Gap A. ignore_errors so a
            # stuck lock file doesn't mask a real close() failure.
            shutil.rmtree(self._tempdir, ignore_errors=True)

    # Context-manager sugar so tests can ``async with MnemossBackend(...)``.
    async def __aenter__(self) -> MnemossBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
