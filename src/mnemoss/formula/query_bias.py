"""Query-bias function b_F(q) and related query heuristics.

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.4.

Stage 2 expansion:
- CJK quote brackets include 《》 『』 【】 alongside the Stage-1 set
  (corner brackets 「」, straight, curly, French guillemets).
- Latin-script proper-noun detection — a Title Case token outside the
  common-stopword list biases b_F upward to 1.2 (between neutral and
  time/number).
- ``has_deep_cue`` detects multilingual temporal-distance markers that
  tell cascade retrieval it's worth scanning the DEEP tier even without
  an explicit ``include_deep=True`` from the caller.

Stage 3+ will replace the Title Case proper-noun rule with a proper
multilingual NER pass and add intent classifiers for vague-query bias.
"""

from __future__ import annotations

import re

# Quote characters from multiple language conventions. Stage 1 had the
# Latin quotes + corner brackets; Stage 2 adds book/title brackets used
# in Chinese and Japanese quotation.
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

# Time patterns like 4:20, 16:00.
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
# ISO-ish dates like 2026-04-22, or day-month-year forms.
_DATE_RE = re.compile(r"\b\d{2,4}[-/]\d{1,2}[-/]\d{1,4}\b")
# Bare digit runs of 2+ characters (excluding single digits to avoid noise).
_NUMBER_RE = re.compile(r"\b\d{2,}\b")

# Latin-script word: ≥ 3 characters, allows accented letters so French /
# Spanish / German proper nouns (Élise, Señor, Müller) still match.
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][a-zà-öø-ÿ]{2,}")

# Stopwords that are commonly Title Cased at sentence start and would
# otherwise trigger false proper-noun positives. Lowercase comparison.
_EN_TITLECASE_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "could", "did", "do", "does", "for", "from", "had",
        "has", "have", "his", "how", "i", "if", "in", "is", "it", "its",
        "may", "might", "must", "my", "no", "not", "of", "on", "or", "our",
        "should", "so", "such", "than", "that", "the", "their", "them",
        "these", "they", "this", "those", "to", "was", "we", "were", "what",
        "when", "where", "which", "who", "whom", "whose", "why", "will",
        "with", "would", "you", "your",
    }
)

# Multilingual markers for "this is about the distant past" — triggers
# DEEP cascade inclusion even without an explicit opt-in.
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
    """Return b_F(q) ∈ {1.0, 1.2, 1.3, 1.5}.

    Higher values bias matching toward literal (FTS) mode; lower values
    bias toward semantic. The checks are ordered strongest-first so a
    quoted query that also contains a time still returns 1.5.
    """

    q = query.strip()

    if any(c in q for c in _QUOTE_CHARS):
        return 1.5

    if _TIME_RE.search(q) or _DATE_RE.search(q) or _NUMBER_RE.search(q):
        return 1.3

    if _has_latin_proper_noun(q):
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


def _has_latin_proper_noun(query: str) -> bool:
    """Heuristic Title Case detector for Latin-script proper nouns.

    Returns True if the query contains a Latin-script token of ≥ 3 letters
    that is capitalized and not on the stopword list. The first-token
    position is *not* exempted because users often write fragments like
    "Alice meeting?" where the proper noun is the first token.
    """

    for token in _WORD_RE.findall(query):
        # Skip all-upper sequences so acronyms (USA, NASA) don't flip the
        # bias — they're better served by FTS anyway, but they already
        # route through number/FTS semantics more often than not.
        if token.isupper():
            continue
        if not token[0].isupper():
            continue
        if token.lower() in _EN_TITLECASE_STOPWORDS:
            continue
        return True
    return False
