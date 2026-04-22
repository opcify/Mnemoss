"""Tests for ``encoder.chunking.split_content``.

Covers the three natural boundaries (paragraph, line, sentence) +
the hard-split fallback, and a few invariants that every caller
relies on: no empty chunks, every chunk under the cap, content
preserved + in order.
"""

from __future__ import annotations

import pytest

from mnemoss.encoder.chunking import split_content

# ─── pass-through ──────────────────────────────────────────────────


def test_short_content_is_returned_unchanged() -> None:
    assert split_content("hello", max_chars=100) == ["hello"]


def test_content_at_cap_is_returned_unchanged() -> None:
    content = "a" * 100
    assert split_content(content, max_chars=100) == [content]


# ─── paragraph boundary ───────────────────────────────────────────


def test_splits_on_double_newline() -> None:
    content = "paragraph one\n\nparagraph two\n\nparagraph three"
    chunks = split_content(content, max_chars=20)
    # Three paragraphs, each ≤ 20 chars, so each is its own chunk.
    assert chunks == ["paragraph one", "paragraph two", "paragraph three"]


def test_paragraph_regroup_packs_when_possible() -> None:
    content = "aa\n\nbb\n\ncc"
    # Cap comfortably fits all three (10 chars total with separators).
    chunks = split_content(content, max_chars=10)
    assert chunks == [content]  # all fit in one chunk


# ─── line boundary (single newline) ───────────────────────────────


def test_line_split_when_no_paragraphs() -> None:
    """Content without ``\\n\\n`` but with single newlines falls
    back to line-level splits."""

    content = "line one\nline two\nline three"
    chunks = split_content(content, max_chars=12)
    # Each line is short; they regroup until the cap.
    for c in chunks:
        assert len(c) <= 12
    assert "line one" in "".join(chunks)
    assert "line three" in "".join(chunks)


# ─── sentence boundary ───────────────────────────────────────────


def test_sentence_split_when_no_newlines() -> None:
    content = (
        "The kickoff is tomorrow. It starts at four. "
        "Everyone needs to attend."
    )
    chunks = split_content(content, max_chars=30)
    for c in chunks:
        assert len(c) <= 30
    assert "kickoff" in chunks[0]
    # Preserves content across chunks (modulo whitespace-handling).
    rejoined = " ".join(chunks)
    for token in ("kickoff", "four", "attend"):
        assert token in rejoined


def test_cjk_sentence_boundary_works() -> None:
    """The sentence splitter respects CJK terminators (。！？) so
    Chinese / Japanese content splits at natural boundaries."""

    content = "今天开会。明天交付。后天休息。"  # 3 sentences
    chunks = split_content(content, max_chars=5)
    for c in chunks:
        assert len(c) <= 5
    # All three sentence verbs survived.
    rejoined = "".join(chunks)
    for char in ("开", "交", "休"):
        assert char in rejoined


# ─── hard split fallback ──────────────────────────────────────────


def test_hard_split_when_no_natural_boundary() -> None:
    """A single super-long run of non-whitespace, non-terminator
    characters (think: a minified JSON blob) falls through every
    boundary splitter. Hard-split kicks in as the last resort."""

    content = "x" * 250
    chunks = split_content(content, max_chars=100)
    assert chunks == ["x" * 100, "x" * 100, "x" * 50]


def test_hard_split_preserves_all_bytes() -> None:
    """No character ever drops from a hard split — even at
    pathological content sizes."""

    content = "abcdefghijklmnopqrstuvwxyz" * 1000  # 26KB
    chunks = split_content(content, max_chars=1000)
    assert "".join(chunks) == content
    for c in chunks:
        assert len(c) <= 1000


# ─── invariants every splitter must preserve ──────────────────────


@pytest.mark.parametrize(
    "content, cap",
    [
        ("short enough", 50),
        ("paragraph one.\n\nparagraph two.", 20),
        ("line a\nline b\nline c", 10),
        ("First. Second! Third? Fourth…", 15),
        ("x" * 500, 50),
        ("今天开会。明天交付。后天休息。", 10),
    ],
)
def test_no_chunk_exceeds_cap(content: str, cap: int) -> None:
    chunks = split_content(content, max_chars=cap)
    assert all(
        len(c) <= cap for c in chunks
    ), [len(c) for c in chunks]


@pytest.mark.parametrize(
    "content, cap",
    [
        ("paragraph one.\n\nparagraph two.", 20),
        ("line a\nline b\nline c", 10),
        ("First. Second! Third? Fourth…", 15),
        ("x" * 500, 50),
    ],
)
def test_no_empty_chunks(content: str, cap: int) -> None:
    chunks = split_content(content, max_chars=cap)
    assert all(c for c in chunks), chunks


# ─── constructor validation ───────────────────────────────────────


def test_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_content("anything", max_chars=0)
    with pytest.raises(ValueError, match="max_chars"):
        split_content("anything", max_chars=-1)
