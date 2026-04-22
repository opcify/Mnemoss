"""Content chunker for long observes.

Mnemoss stores messages verbatim in the Raw Log regardless of length,
but the Memory store has two soft upper bounds:

- Embedder token limits (LocalEmbedder / MiniLM ≈ 512 tokens; OpenAI
  text-embedding-3-small ≈ 8k tokens). Content past the limit is
  silently dropped by the embedder — semantic recall degrades
  without any warning.
- Dream P3 Consolidate prompt size. A cluster of very long memories
  can exceed an LLM's context window.

The chunker here splits one over-long string into N pieces at the
most natural boundary available, so each Memory row is bounded in
size. The Raw Log stays 1-to-1 with the original ``observe()``
call; only the Memory table sees the chunks.

Boundary preference, in order:

1. **paragraph** — double-newline (``\\n\\n``) is the clearest reader
   boundary.
2. **line** — single newline, for content that uses ``\\n`` as a soft
   separator (logs, code).
3. **sentence** — any of ``.!?…`` followed by whitespace. Handles
   prose; stays linguistically neutral (no English-only regex).
4. **hard split** — last resort, chunks at exactly ``max_chars``.

Every chunk is at most ``max_chars``. The function never produces an
empty chunk, never drops content, and preserves the relative order.
"""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…。！？])\s+")


def split_content(content: str, max_chars: int) -> list[str]:
    """Split ``content`` into chunks of length ≤ ``max_chars``.

    Returns ``[content]`` unchanged when it already fits. Raises
    ``ValueError`` on a nonsensical cap (≤ 0); callers should pass
    ``None`` through a higher-level check if they want "no split."
    """

    if max_chars <= 0:
        raise ValueError(f"max_chars must be > 0 (got {max_chars!r})")
    if len(content) <= max_chars:
        return [content]

    # Try paragraph split first, then line, then sentence — each time
    # regrouping small pieces into chunks that don't exceed the cap.
    for splitter in (_split_paragraphs, _split_lines, _split_sentences):
        candidate = splitter(content, max_chars)
        if candidate is not None:
            return candidate

    # Nothing else worked (single super-long run of non-whitespace).
    return _hard_split(content, max_chars)


# ─── boundary-aware splitters ──────────────────────────────────────


def _split_paragraphs(content: str, max_chars: int) -> list[str] | None:
    if "\n\n" not in content:
        return None
    return _regroup(content.split("\n\n"), sep="\n\n", max_chars=max_chars)


def _split_lines(content: str, max_chars: int) -> list[str] | None:
    if "\n" not in content:
        return None
    return _regroup(content.split("\n"), sep="\n", max_chars=max_chars)


def _split_sentences(content: str, max_chars: int) -> list[str] | None:
    sentences = _SENTENCE_BOUNDARY_RE.split(content)
    if len(sentences) <= 1:
        return None
    return _regroup(sentences, sep=" ", max_chars=max_chars)


# ─── helpers ───────────────────────────────────────────────────────


def _regroup(pieces: list[str], *, sep: str, max_chars: int) -> list[str] | None:
    """Greedy-pack pieces into chunks of length ≤ ``max_chars``.

    If any single piece exceeds ``max_chars``, we fall through to the
    next splitter (caller handles this by checking for a return of
    None). Returns the full list of chunks on success.
    """

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > max_chars:
            # A single piece overflows the budget — caller falls back
            # to a finer-grained splitter. Bail out so we don't emit
            # over-sized chunks.
            return None
        candidate = (current + sep + piece) if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _hard_split(content: str, max_chars: int) -> list[str]:
    """Last-resort char-offset split. Used when no natural boundary
    produces chunks under ``max_chars`` (e.g. a 100KB single-run
    tool output with no newlines or sentence terminators)."""

    return [content[i : i + max_chars] for i in range(0, len(content), max_chars)]
