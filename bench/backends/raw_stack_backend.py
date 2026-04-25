"""The "stack you'd build in an afternoon" baseline.

This is the memory layer a competent builder would throw together if
asked to "add memory to an agent" with no library research: a markdown
file that logs everything (human-readable, inspectable), an SQLite
table that stores each memory plus its embedding, and cosine-similarity
ranking on recall via a linear scan of all vectors.

No ACT-R, no activation formula, no decay, no spreading activation, no
tier cascade, no dreaming. Just embed + cosine. It's what Hermes's
``MEMORY.md`` points at if you wanted real retrieval instead of a
static prompt prefix; it's what a Claude Code user might cobble
together with `claude mcp add` and a notebook; it's what OpenClaw's
bundled SQLite memory essentially is under the hood.

The comparison that matters — Chart 1 in the blog post — is Mnemoss
against this, not against SaaS-branded "memory products." Every
builder has either built this stack or thought about building it.

Implementation choices
----------------------

- **Embedder parity:** defaults to ``OpenAIEmbedder`` with
  ``text-embedding-3-small``, wrapped in ``RetryingEmbedder`` — same
  as Mnemoss's published-chart config (Issue 1.1A from eng review).
  Override via ``embedding_model=`` for tests.
- **Vector storage:** raw ``numpy.float32`` bytes in a SQLite BLOB
  column. No ``sqlite-vec`` or similar extensions — this is the naive
  stack. On recall we load everything into RAM and dot-product.
- **Cosine similarity:** embeddings are unit-normalized on write and
  on query, so dot product == cosine. If the embedder returns
  already-normalized vectors the second normalization is a no-op.
- **Markdown log:** each observe appends one line
  ``- [ts] text`` to ``memory.md`` in the tempdir. The file is part
  of the baseline's identity — "my memory system is a markdown file"
  is a real design you see in the wild.
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import ulid

from bench.backends.base import RecallHit
from mnemoss import Embedder, OpenAIEmbedder, RetryingEmbedder


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector so cosine similarity == dot product.

    Zero vectors pass through unchanged — division by the unit norm
    would raise; the caller gets a zero similarity to everything,
    which is the right behavior for a degenerate input.
    """

    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class RawStackBackend:
    """``MemoryBackend`` implementing the markdown + SQLite + vectors baseline."""

    backend_id = "raw_stack"

    def __init__(
        self,
        *,
        embedding_model: Embedder | None = None,
    ) -> None:
        # Tempdir first so ``close()`` can cleanup even if later init raises.
        self._tempdir = Path(tempfile.mkdtemp(prefix="raw_stack_bench_"))
        self._md_path = self._tempdir / "memory.md"
        self._md_path.touch()
        self._db_path = self._tempdir / "memory.sqlite"
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            """
            CREATE TABLE memories (
                id   TEXT PRIMARY KEY,
                ts   REAL NOT NULL,
                text TEXT NOT NULL,
                vec  BLOB NOT NULL
            )
            """
        )
        self._conn.commit()
        self._closed = False

        if embedding_model is None:
            embedding_model = RetryingEmbedder(
                OpenAIEmbedder(model="text-embedding-3-small"),
                max_retries=3,
            )
        self._embedder = embedding_model
        # Cached at observe time and reused on recall to short-circuit the
        # SELECT when the workspace is empty.
        self._count = 0

    async def observe(self, text: str, ts: float) -> str:
        """Embed ``text``, append to the markdown log, insert into SQLite."""

        memory_id = str(ulid.new())
        vecs = await asyncio.to_thread(self._embedder.embed, [text])
        vec = _normalize(vecs[0])

        # Markdown log — the "human-readable memory" half of the stack.
        with self._md_path.open("a") as f:
            f.write(f"- [{ts:.0f}] {text}\n")

        # Embeddings table — the "actually queryable" half.
        self._conn.execute(
            "INSERT INTO memories (id, ts, text, vec) VALUES (?, ?, ?, ?)",
            (memory_id, ts, text, vec.tobytes()),
        )
        self._conn.commit()
        self._count += 1
        return memory_id

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Linear-scan cosine similarity over every stored embedding.

        Intentionally NOT a vector index (no HNSW, no IVF, no sqlite-vec).
        The naive stack scales O(N·dim) per query, which is what "I
        built this in an afternoon" actually means in practice.
        """

        if self._count == 0:
            return []

        # Use the embedder's query-side path when available (e.g. Nomic
        # v2 MoE uses ``search_query: `` for queries, ``search_document: ``
        # for docs — asymmetric prompts are how the retrieval training
        # was actually optimized). Symmetric embedders like MiniLM map
        # ``embed_query`` to ``embed`` transparently.
        from mnemoss.encoder.embedder import embed_query_or_embed

        qvecs = await asyncio.to_thread(embed_query_or_embed, self._embedder, [query])
        qvec = _normalize(qvecs[0])

        # Load everything. This is the honest shape of the baseline —
        # builders doing markdown+sqlite+embedding do not, in general,
        # implement approximate-nearest-neighbor themselves.
        rows = self._conn.execute("SELECT id, vec FROM memories").fetchall()
        ids = [row[0] for row in rows]
        # Stack into (N, dim). frombuffer is zero-copy.
        mat = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])

        # cosine == dot, since both sides were unit-normalized at write/query.
        sims = mat @ qvec
        top_idx = np.argsort(-sims)[:k]

        return [
            RecallHit(
                memory_id=ids[int(i)],
                rank=rank + 1,
                score=float(sims[int(i)]),
            )
            for rank, i in enumerate(top_idx)
        ]

    async def close(self) -> None:
        """Close the SQLite handle + rmtree the tempdir. Idempotent."""

        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        finally:
            shutil.rmtree(self._tempdir, ignore_errors=True)

    async def __aenter__(self) -> RawStackBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
