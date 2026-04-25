"""Common async interface for launch-comparison backends.

Three shipped adapters (Mnemoss, Mem0, Chroma) all expose the same
minimal protocol: observe one piece of text with a timestamp, recall
a ranked list for a query, close. That's everything the benchmark
harness needs to produce apples-to-apples numbers across systems with
radically different internals.

Intentionally minimal — no tier migrations, no agent scoping, no
spreading activation. Each backend runs its own native pipeline; the
protocol only describes the boundary.

Issue 1.1A (eng review): fair benchmarks require embedder parity.
Every implementation must pin ``text-embedding-3-small`` so Chart 1
measures memory-architecture effects, not embedder-quality effects.
Mnemoss and Chroma pin explicitly via constructor; Mem0 picks it up
via its OpenAI-provider config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RecallHit:
    """One ranked result from ``MemoryBackend.recall``.

    The score slot is backend-native: Mnemoss emits activation ``A_i``,
    Mem0 emits its internal relevance score, Chroma emits cosine
    similarity. Comparable only within one backend's output; across
    backends, trust rank (1-indexed) and the memory id.

    Attributes
    ----------
    memory_id:
        The backend's native id for this memory row. Opaque to the
        harness — used only for cross-checking against the corpus'
        ``relevant_memory_ids`` gold labels.
    rank:
        Position in the ranked result list, 1-indexed. Rank 1 is the
        top hit.
    score:
        Backend-native score, or ``None`` if the backend doesn't expose
        one. Do not compare across backends.
    """

    memory_id: str
    rank: int
    score: float | None


@runtime_checkable
class MemoryBackend(Protocol):
    """Async interface every comparison backend implements.

    Lifecycle: construct → observe(text, ts) many times → recall(q, k)
    many times → close(). Construction is allowed to open file handles
    or HTTP clients; ``close()`` must release them all.

    Attributes
    ----------
    backend_id:
        Short, stable label used in result JSON (``"mnemoss"``,
        ``"mem0"``, ``"chroma"``). Lowercase, no spaces.
    """

    backend_id: str

    async def observe(self, text: str, ts: float) -> str:
        """Ingest one piece of text. Returns the backend's memory id.

        ``ts`` is a Unix timestamp the backend may use for recency
        weighting. Mnemoss ignores it (uses its own clock); Mem0 and
        Chroma get passed through as metadata but don't weight by it.
        Accepting a uniform corpus timestamp format keeps the harness
        simple.
        """

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Return up to ``k`` ranked hits. Empty list if nothing matches.

        Implementations must not raise on a zero-result recall — the
        caller expects ``[]``, not an exception, mirroring Mnemoss's
        own invariant (see project CLAUDE.md).
        """

    async def close(self) -> None:
        """Release all resources (DB handles, tempdirs, HTTP clients).

        Must be idempotent: calling ``close()`` twice is not an error.
        Tempdir cleanup runs even if an inner client raises during
        close (Gap A from eng review — leftover workspace tempdirs
        would re-trigger ``WorkspaceLockError`` on the next run).
        """
