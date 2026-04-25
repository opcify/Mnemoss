"""Static-file memory baseline.

Represents how Hermes (``MEMORY.md`` / ``USER.md``) and Claude Code
(``CLAUDE.md``) handle "memory" today: append text to a file, recall
by reading the file back and keyword-matching. No embeddings, no
ranking formula, no decay. Just a log.

This is the *honest baseline* for the launch's framework sections —
what your tools ship with out of the box. Chart 1 shows recall@10 of
Mnemoss vs this baseline on LoCoMo, demonstrating what installing
Mnemoss buys you over the default.

Scoring
-------

Intentionally simple:

1. Tokenize query and each memory on whitespace + lowercase + strip
   punctuation.
2. Score = count of query tokens that appear in the memory's tokens.
3. Ties broken by recency (newer memory wins — matches what "read the
   file from the end" effectively does in practice).
4. Return top-k by score, dropping zero-score memories entirely (a
   grep-style tool returns nothing for queries with no keyword hits).

This is deliberately not BM25 or TF-IDF. The point is "a static file
with grep" — the actual behavior of tools like Hermes and Claude Code
when they read ``MEMORY.md`` / ``CLAUDE.md`` at prompt-construction
time. If we added stemming, IDF weighting, or fuzzy matching, we'd be
re-implementing Mnemoss's ``build_trigram_query`` in disguise and
inflating the baseline into something no framework actually uses.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import ulid

from bench.backends.base import RecallHit

# Keep only word characters (covers ASCII + unicode letters, drops punctuation).
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(s: str) -> set[str]:
    """Lowercase + extract word tokens as a set (duplicates don't boost score)."""
    return {t.lower() for t in _TOKEN_RE.findall(s)}


class StaticFileBackend:
    """``MemoryBackend`` that stores everything in a line-delimited file.

    One JSONL row per observe: ``{"id": <ulid>, "ts": <float>, "text": <str>}``.
    The path defaults to a fresh tempfile per instance so benchmark runs
    can't collide.
    """

    backend_id = "static_file"

    def __init__(
        self,
        *,
        path: Path | None = None,
    ) -> None:
        # Tempdir first so close() can clean up even if init raises later.
        self._tempdir: Path | None = None
        if path is None:
            self._tempdir = Path(tempfile.mkdtemp(prefix="static_file_bench_"))
            path = self._tempdir / "memory.jsonl"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        # In-memory mirror of the file so recall() doesn't re-read on every
        # call. The file is authoritative for persistence; the list is a
        # cache keyed by write order.
        self._rows: list[dict] = []
        self._closed = False
        # Truncate the file on open so repeated runs against the same
        # fixed path start clean (tempdir path gets a fresh file by
        # construction, so this is belt-and-braces).
        self._path.write_text("")

    async def observe(self, text: str, ts: float) -> str:
        """Append one JSONL line. Returns the generated memory id."""

        memory_id = str(ulid.new())
        row = {"id": memory_id, "ts": ts, "text": text}
        # Append-mode open per observe — slow at scale, but the whole
        # point of this baseline is that it IS slow and dumb.
        with self._path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self._rows.append(row)
        return memory_id

    async def recall(self, query: str, k: int = 10) -> list[RecallHit]:
        """Score every stored row by token overlap, return top-k.

        Zero-score rows are dropped entirely so a query with no keyword
        matches returns ``[]`` (matching grep-with-no-hits behavior).
        """

        q_tokens = _tokenize(query)
        if not q_tokens or not self._rows:
            return []

        # (overlap_count, write_index) for stable tiebreak by recency.
        scored: list[tuple[int, int, dict]] = []
        for i, row in enumerate(self._rows):
            mem_tokens = _tokenize(row["text"])
            overlap = len(q_tokens & mem_tokens)
            if overlap > 0:
                scored.append((overlap, i, row))

        # Sort by (-overlap, -write_index) → highest overlap first,
        # newer tied memories before older tied memories.
        scored.sort(key=lambda x: (-x[0], -x[1]))

        top = scored[:k]
        return [
            RecallHit(
                memory_id=row["id"],
                rank=i + 1,
                score=float(overlap),
            )
            for i, (overlap, _, row) in enumerate(top)
        ]

    async def close(self) -> None:
        """Release the tempdir (if we created it). Idempotent."""

        if self._closed:
            return
        self._closed = True
        if self._tempdir is not None:
            shutil.rmtree(self._tempdir, ignore_errors=True)

    async def __aenter__(self) -> StaticFileBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
