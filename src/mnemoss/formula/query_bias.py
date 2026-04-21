"""Query-bias function b_F(q).

See MNEMOSS_FORMULA_AND_ARCHITECTURE.md §1.4. Stage 1 ships only the
language-agnostic rules: explicit quotes, numbers/dates, time patterns.
English-specific proper-noun/pronoun/vague detection is deferred to Stage 2,
where it would need multilingual NER.
"""

from __future__ import annotations

import re

# Quote characters from multiple language conventions.
_QUOTE_CHARS = ('"', "'", "“", "”", "「", "」", "『", "』", "«", "»")

# Time patterns like 4:20, 16:00.
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
# ISO-ish dates like 2026-04-22, or day-month-year forms.
_DATE_RE = re.compile(r"\b\d{2,4}[-/]\d{1,2}[-/]\d{1,4}\b")
# Bare digit runs of 2+ characters (excluding single digits to avoid noise).
_NUMBER_RE = re.compile(r"\b\d{2,}\b")


def compute_query_bias(query: str) -> float:
    """Return b_F(q) ∈ {0.6, 0.7, 1.0, 1.2, 1.3, 1.5}.

    Higher values bias matching toward literal (FTS) mode; lower values
    bias toward semantic. See the bias table in §1.4.
    """

    q = query.strip()

    if any(c in q for c in _QUOTE_CHARS):
        return 1.5

    if _TIME_RE.search(q) or _DATE_RE.search(q) or _NUMBER_RE.search(q):
        return 1.3

    return 1.0
