"""Heuristic lazy extraction (Stage 3).

Fills ``extracted_*`` fields on a Memory using rule-based tools, not
LLMs. Stage 4's Dreaming P4 (Refine) will later replace these values
with LLM output and bump ``extraction_level`` to 2.

Current heuristics:

- ``gist``: first sentence (split on .!?…。！？), capped at 100 chars.
- ``entities``: de-duplicated Latin-script Title-Case tokens that aren't
  common stopwords.
- ``time``: first parseable date/time found anywhere in the content,
  via ``dateparser.search.search_dates``. Handles 200+ languages.
- ``location`` / ``participants``: Stage 3 leaves these ``None`` —
  distinguishing a person from a place or finding locations without NER
  is the kind of thing Stage 5's multilingual NER pass will do
  properly.

All signals are pure and cheap; the whole extraction finishes in a
millisecond or two on typical content. Heuristic failures stay silent
(return ``None``) so a partial fill still counts as ``level=1``.

If ``dateparser`` isn't installed (e.g. lightweight test env), time
extraction degrades to ``None`` without raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from mnemoss.formula.query_bias import _EN_TITLECASE_STOPWORDS, _WORD_RE

UTC = timezone.utc

_SENTENCE_SPLIT_RE = re.compile(r"[.!?…。！？]+")
_DATEPARSER_CONTENT_LIMIT = 1000  # dateparser regex scan cost grows with length


@dataclass
class ExtractionFields:
    gist: str | None = None
    entities: list[str] | None = None
    time: datetime | None = None
    location: str | None = None
    participants: list[str] | None = None
    level: int = 1


def extract_heuristic(content: str) -> ExtractionFields:
    """Run every heuristic signal; return a partial-fill ExtractionFields.

    level is always 1 for this function (heuristic); Stage 4's LLM
    refinement sets level=2.
    """

    text = content.strip()
    if not text:
        return ExtractionFields(level=1)

    return ExtractionFields(
        gist=_extract_gist(text),
        entities=_extract_entities(text) or None,
        time=_extract_time(text),
        location=None,
        participants=None,
        level=1,
    )


def _extract_gist(text: str) -> str | None:
    if not text:
        return None
    parts = _SENTENCE_SPLIT_RE.split(text, maxsplit=1)
    first = parts[0].strip() if parts and parts[0].strip() else text
    return first[:100]


def _extract_entities(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _WORD_RE.findall(text):
        if token.isupper():
            continue
        if not token[0].isupper():
            continue
        if token.lower() in _EN_TITLECASE_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _extract_time(text: str) -> datetime | None:
    try:
        from dateparser.search import search_dates
    except ImportError:  # pragma: no cover - defensive for slim envs
        return None

    clipped = text[:_DATEPARSER_CONTENT_LIMIT]
    try:
        results = search_dates(
            clipped,
            settings={
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": "UTC",
                "PREFER_DATES_FROM": "future",
            },
        )
    except Exception:  # pragma: no cover - dateparser occasionally raises
        return None
    if not results:
        return None
    # Filter out dateparser's natural-language hallucinations: the matched
    # source substring must contain at least one digit. "no date" /
    # "the plan" / other English phrases that dateparser sometimes maps
    # to dates would otherwise leak through.
    for original, dt in results:
        if not any(c.isdigit() for c in original):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    return None
