"""Query-bias function b_F(q) and related query heuristics.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.4.

``b_F(q)`` returns a scalar that tilts the hybrid matching weights toward
FTS (literal) or semantic (embedding) scoring. Every rule here is
**structural / typographic**, not semantic — we never inspect the query
for named entities, topics, or meaning. The memory system is
query-agnostic; only patterns a user *types* to signal "I mean this
literally" (quotes, digits, URLs, hashtags, code identifiers,
acronyms) count as cues.

Rule ladder (strongest cue wins, first-match):

- **1.5** — quote chars or a backtick-fenced span → verbatim phrase
- **1.4** — URL, email, or file path with extension → exact literal token
- **1.3** — time/date/number, hashtag, @-mention, CamelCase / snake_case
  / kebab-case identifier → concrete structural token
- **1.2** — ALL-CAPS Latin token ≥3 chars → acronym
- **1.0** — none of the above → neutral

Non-Latin scripts (CJK, Arabic, Devanagari, …) trigger every rule that
uses language-neutral regex (quotes, URLs, digits, hashtags, code
identifiers embedded in sentences). The acronym rule is structurally
Latin-only because no other common script uses uppercase — it's silent
on those scripts, not biased against them.

``has_deep_cue`` detects multilingual temporal-distance markers that
tell cascade retrieval it's worth scanning the DEEP tier even without
an explicit ``include_deep=True`` from the caller. That's a separate
signal from ``b_F`` and is already multilingual by design.
"""

from __future__ import annotations

import re

# ─── Quote characters ──────────────────────────────────────────────
# Multiple language conventions; book-title and corner brackets count as
# quotes because that's how CJK marks verbatim phrases.
_QUOTE_CHARS = (
    '"',
    "'",
    "“",
    "”",
    "「",
    "」",
    "『",
    "』",
    "«",
    "»",
    "《",
    "》",
    "【",
    "】",
    "〈",
    "〉",
)

# ─── Regex patterns — language-neutral structural cues ─────────────

# Backtick fence: `foo`, ``foo``. Matches an odd or even number of
# opening backticks to keep the rule robust to minor typos.
_BACKTICK_RE = re.compile(r"`[^`\n]+`")

# URL (with or without scheme), email, file path with extension, version.
# Kept intentionally loose — we just need "does this look structurally
# literal", not a full grammar.
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+|"  # http(s):// or www.
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",  # email
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?:(?:\.{1,2}|~)?/)?"  # optional leading ./ ../ ~/ /
    r"[\w.-]+/[\w./-]*\.[A-Za-z0-9]{1,6}\b"
)
_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?\b")

# Time patterns like 4:20, 16:00.
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
# ISO-ish dates like 2026-04-22, or day-month-year forms.
_DATE_RE = re.compile(r"\b\d{2,4}[-/]\d{1,2}[-/]\d{1,4}\b")
# Bare digit runs of 2+ characters (excluding single digits to avoid noise).
_NUMBER_RE = re.compile(r"\b\d{2,}\b")

# Hashtag: #tag (Latin, digits, underscore; must start with #letter).
_HASHTAG_RE = re.compile(r"(?:^|\s)#[A-Za-z_]\w{1,}")
# @-mention: @name (Latin, digits, underscore, hyphen, dot).
_MENTION_RE = re.compile(r"(?:^|\s)@[A-Za-z_][\w.\-]{1,}")

# Code identifiers: CamelCase (≥1 lowercase → ≥1 uppercase), snake_case,
# kebab-case with a letter on each side of the separator. All require a
# letter boundary so plain sentences ("I said") don't match.
_CAMEL_RE = re.compile(r"\b[A-Za-z]+(?:[a-z]+[A-Z]|[A-Z][a-z])\w*\b")
_SNAKE_RE = re.compile(r"\b[A-Za-z]\w*_\w+\b")
_KEBAB_RE = re.compile(r"\b[A-Za-z]\w*-\w+\b")

# ALL-CAPS Latin acronym (≥3 chars). Structurally typographic: no
# vocabulary check, no language lookup. Silent on non-Latin scripts by
# construction.
_ACRONYM_RE = re.compile(r"\b[A-Z]{3,}\b")

# ─── Deep-cue multilingual markers ─────────────────────────────────
_DEEP_CUES: tuple[str, ...] = (
    # English
    "long ago",
    "long time ago",
    "years ago",
    "years back",
    "ages ago",
    "ages back",
    "way back",
    "back then",
    "originally",
    "ancient",
    "a long time",
    # Spanish
    "hace mucho",
    "hace años",
    "hace tiempo",
    "antiguamente",
    "érase una vez",
    # French
    "il y a longtemps",
    "autrefois",
    "jadis",
    # German
    "vor langer zeit",
    "einst",
    # Italian
    "tanto tempo fa",
    "molto tempo fa",
    # Portuguese
    "há muito tempo",
    # Chinese
    "很久以前",
    "多年前",
    "从前",
    "当初",
    "当年",
    "早年",
    # Japanese
    "ずっと前",
    "何年も前",
    "昔々",
    "大昔",
    # Korean
    "오래 전에",
    "옛날에",
    # Arabic
    "منذ زمن طويل",
    "في الماضي البعيد",
)


def compute_query_bias(query: str) -> float:
    """Return b_F(q) ∈ {1.0, 1.2, 1.3, 1.4, 1.5}.

    Ordered strongest-first so mixed queries pick the most literal cue.
    All rules are regex-based and structural — no NER, no vocabulary,
    no language detection.
    """

    q = query.strip()
    if not q:
        return 1.0

    # 1.5 — verbatim phrase markers.
    if any(c in q for c in _QUOTE_CHARS):
        return 1.5
    if _BACKTICK_RE.search(q):
        return 1.5

    # 1.4 — exact literal tokens (URLs, emails, file paths).
    if _URL_RE.search(q) or _PATH_RE.search(q):
        return 1.4

    # 1.3 — concrete structural tokens: digits, tags, code identifiers,
    # version strings. Version check sits here because "v1.2.3" is a
    # concrete literal but not as strong a cue as a full URL.
    if (
        _TIME_RE.search(q)
        or _DATE_RE.search(q)
        or _NUMBER_RE.search(q)
        or _HASHTAG_RE.search(q)
        or _MENTION_RE.search(q)
        or _CAMEL_RE.search(q)
        or _SNAKE_RE.search(q)
        or _KEBAB_RE.search(q)
        or _VERSION_RE.search(q)
    ):
        return 1.3

    # 1.2 — acronym-style ALL-CAPS token (Latin-script only by structure).
    if _ACRONYM_RE.search(q):
        return 1.2

    return 1.0


def has_deep_cue(query: str) -> bool:
    """Return True when the query contains a multilingual temporal-distance
    marker ("long ago", 很久以前, 昔々, ...).

    Used by the cascade so that clearly-historical queries auto-include
    the DEEP tier without requiring callers to flag ``include_deep``
    manually. Matching is case-insensitive for Latin-script markers; CJK
    and Arabic strings are matched literally.
    """

    if not query:
        return False
    lowered = query.lower()
    return any(cue in lowered or cue in query for cue in _DEEP_CUES)
