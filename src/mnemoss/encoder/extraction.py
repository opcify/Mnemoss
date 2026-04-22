"""Heuristic lazy extraction (level=1).

Fills ``extracted_*`` fields on a Memory using rule-based tools, not
LLMs. Dream P3 Consolidate also refines ``gist`` and ``time`` at
level=2 using the LLM.

What level-1 heuristics fill:

- ``gist``: first sentence (split on .!?…。！？), capped at 100 chars.
- ``time``: first parseable date/time found anywhere in the content,
  via ``dateparser.search.search_dates``. Handles 200+ languages.

What stays ``None`` everywhere:

- ``entities`` / ``location`` / ``participants``. NER is intentionally
  not wired into Mnemoss at any stage (see
  MNEMOSS_PROJECT_KNOWLEDGE.md §9.7 for rationale). The fields exist
  on the Memory dataclass so callers can populate them manually if
  they already have entity information.

All heuristic signals are pure and cheap; the whole extraction
finishes in a millisecond or two on typical content. Failures stay
silent (return ``None``) so a partial fill still counts as ``level=1``.

If ``dateparser`` isn't installed (e.g. lightweight test env), time
extraction degrades to ``None`` without raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

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
    """Run the level-1 heuristic signals and return a partial fill.

    level is always 1 for this function; Dream P3 Consolidate sets
    level=2 with LLM-grade entities / location / participants.
    """

    text = content.strip()
    if not text:
        return ExtractionFields(level=1)

    return ExtractionFields(
        gist=_extract_gist(text),
        entities=None,
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
    for original, dt_untyped in results:
        dt: datetime = dt_untyped
        if not any(c.isdigit() for c in original):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    return None
