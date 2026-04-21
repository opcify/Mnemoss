"""P4 Refine — LLM upgrade of the lazy extraction fields.

Inputs a Memory (typically one with ``extraction_level < 2`` after P1
replay has surfaced it) and its current heuristic fields, asks the LLM
to improve them, and writes back ``extraction_level=2`` values.

Unlike P3 Extract, P4 doesn't create new memory rows — it only updates
the existing one's ``extracted_*`` fields. ``content`` is never touched.
LLM failures never kill the dream run; they return ``None`` and the
phase records the skip count in its outcome.
"""

from __future__ import annotations

import json
import logging
from datetime import timezone

from dateutil import parser as dateutil_parser

from mnemoss.core.types import Memory
from mnemoss.encoder.extraction import ExtractionFields
from mnemoss.llm.client import LLMClient

UTC = timezone.utc
log = logging.getLogger(__name__)


def build_refine_prompt(memory: Memory) -> str:
    existing = {
        "gist": memory.extracted_gist,
        "entities": memory.extracted_entities,
        "time": (
            memory.extracted_time.isoformat() if memory.extracted_time else None
        ),
        "location": memory.extracted_location,
        "participants": memory.extracted_participants,
    }
    role = memory.role or "note"
    return (
        "Memory content (verbatim — do not modify):\n"
        f'"""{memory.content}"""\n\n'
        f"Role: {role}\n"
        "\n"
        "Current heuristic extraction (improve or correct; fields can be "
        "null if not applicable):\n"
        f"{json.dumps(existing, indent=2, default=str)}\n\n"
        "Return a single JSON object in this exact shape:\n"
        "{\n"
        '  "gist": "concise one-sentence summary",\n'
        '  "entities": ["named entities mentioned"],\n'
        '  "time": "ISO-8601 timestamp or null",\n'
        '  "location": "place name or null",\n'
        '  "participants": ["people involved"]\n'
        "}\n\n"
        "Rules:\n"
        "- Keep the summary in the same language as the content.\n"
        "- 'time' must be an ISO-8601 timestamp (with timezone) or null.\n"
        "- Use [] for empty lists, not null.\n"
        "- Do not invent facts not implied by the content.\n"
    )


async def refine_memory_fields(
    memory: Memory, llm: LLMClient
) -> ExtractionFields | None:
    """Return ``ExtractionFields(level=2)`` or ``None`` if the call fails."""

    if not memory.content.strip():
        return None

    prompt = build_refine_prompt(memory)
    try:
        response = await llm.complete_json(prompt)
    except Exception as e:  # pragma: no cover - LLM errors vary by provider
        log.warning("refine_memory_fields: LLM failure: %s", e)
        return None

    return ExtractionFields(
        gist=_norm_str(response.get("gist")),
        entities=_norm_list(response.get("entities")),
        time=_parse_time(response.get("time")),
        location=_norm_str(response.get("location")),
        participants=_norm_list(response.get("participants")),
        level=2,
    )


def _norm_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _norm_list(value) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out = [str(v).strip() for v in value if v is not None]
    out = [v for v in out if v]
    return out or None


def _parse_time(value):
    if not value:
        return None
    try:
        dt = dateutil_parser.isoparse(str(value))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
