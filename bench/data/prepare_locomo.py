"""Adapt the LoCoMo 2024 conversational-memory benchmark for the
launch-comparison harness.

Source: https://github.com/snap-research/locomo/blob/main/data/locomo10.json
Paper: Maharana et al., "Evaluating Very Long-Term Conversational Memory
of LLM Agents" (EMNLP 2024).

LoCoMo structure (per sample = one conversation):

- ``conversation.session_N`` — list of utterances with ``speaker``,
  ``dia_id`` (e.g. ``"D1:3"`` = dialog 1, turn 3), and ``text``.
- ``conversation.session_N_date_time`` — human-readable timestamp
  like ``"1:56 pm on 8 May, 2023"``.
- ``qa`` — list of ``{question, answer, evidence, category}`` where
  ``evidence`` is a list of ``dia_id`` strings pointing at the
  utterances that contain the gold answer.

Output (two JSONL files, shared across all 10 conversations):

- ``bench/data/locomo_memories.jsonl``:
  ``{conversation_id, dia_id, ts, text, speaker, session}``
  One utterance per row. The harness ingests these via
  ``backend.observe(text, ts)`` and maintains a
  ``dia_id → native_memory_id`` mapping for relevance lookup.

- ``bench/data/locomo_queries.jsonl``:
  ``{conversation_id, question, relevant_dia_ids, answer, category}``
  One QA per row. 4 rows with empty evidence are dropped (can't score
  without gold labels).

Why two flat JSONLs instead of per-conversation directories: keeps the
harness CLI simple (``--corpus locomo`` loads everything), grouping is
one ``collections.defaultdict`` at run time. Conversations are
isolated at ingest (one workspace per ``conversation_id``) so Mnemoss's
workspace lock isn't a concern.

Run: ``python -m bench.data.prepare_locomo``

Expected output counts (LoCoMo 10-conversation bundle):
- ~5,882 utterances
- ~1,982 QAs with evidence (4 dropped)
- 10 conversation_ids (conv-26, conv-29, conv-30, etc.)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path("bench/data/locomo10.json")
OUT_MEMORIES = Path("bench/data/locomo_memories.jsonl")
OUT_QUERIES = Path("bench/data/locomo_queries.jsonl")

# Matches "1:56 pm on 8 May, 2023". We intentionally do not handle
# timezone — LoCoMo itself doesn't specify one, so we treat every
# timestamp as UTC. Downstream Mnemoss only uses ts for relative
# recency; absolute TZ doesn't affect benchmark outcomes.
_DATE_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ampm>am|pm)\s+on\s+"
    r"(?P<day>\d{1,2})\s+(?P<month>\w+),?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_locomo_date(s: str) -> float:
    """Parse LoCoMo's ``"1:56 pm on 8 May, 2023"`` format to Unix seconds.

    Returns 0.0 for unparseable strings — the benchmark cares about
    relative ordering, so a zero floor is safer than raising and
    losing whole sessions.
    """

    m = _DATE_RE.search(s)
    if m is None:
        return 0.0
    hour = int(m.group("h"))
    minute = int(m.group("m"))
    ampm = m.group("ampm").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    month = _MONTHS.get(m.group("month").lower())
    if month is None:
        return 0.0
    dt = datetime(
        year=int(m.group("year")),
        month=month,
        day=int(m.group("day")),
        hour=hour,
        minute=minute,
        tzinfo=timezone.utc,
    )
    return dt.timestamp()


def utterance_timestamp(session_base_ts: float, utt_index: int) -> float:
    """Assign each utterance within a session a monotonic timestamp.

    LoCoMo only gives us a session-level date_time, so we synthesize
    per-utterance timestamps by spacing them 60 seconds apart starting
    at the session base. Preserves ingest order, keeps ts monotonically
    increasing within a session.
    """

    return session_base_ts + utt_index * 60.0


def prepare(source: Path = SOURCE, out_mem: Path = OUT_MEMORIES, out_q: Path = OUT_QUERIES) -> dict:
    """Parse LoCoMo, emit two JSONL files. Returns run stats."""

    if not source.exists():
        raise FileNotFoundError(
            f"LoCoMo source not found at {source}. Download with:\n"
            f"  curl -sL -o {source} https://raw.githubusercontent.com/"
            f"snap-research/locomo/main/data/locomo10.json"
        )

    with source.open() as f:
        data = json.load(f)

    stats = {
        "conversations": len(data),
        "utterances": 0,
        "queries_kept": 0,
        "queries_dropped_no_evidence": 0,
    }

    out_mem.parent.mkdir(parents=True, exist_ok=True)

    with out_mem.open("w") as mf, out_q.open("w") as qf:
        for sample in data:
            conv_id = sample["sample_id"]
            conv = sample["conversation"]

            # Walk sessions in order. LoCoMo uses session_1, session_2, ...
            # up to session_N where N can be up to 35; only emit sessions
            # that have both a date_time AND a non-empty utterance list.
            session_keys = [
                k for k in conv if k.startswith("session_") and not k.endswith("_date_time")
            ]
            # Sort by session number so ingest order is correct.
            session_keys.sort(key=lambda k: int(k.split("_")[1]))

            for skey in session_keys:
                session_num = int(skey.split("_")[1])
                utterances = conv.get(skey)
                if not isinstance(utterances, list) or not utterances:
                    continue
                date_str = conv.get(f"session_{session_num}_date_time", "")
                base_ts = parse_locomo_date(date_str)

                for i, utt in enumerate(utterances):
                    speaker = utt.get("speaker", "")
                    text = utt.get("text", "")
                    dia_id = utt.get("dia_id", "")
                    if not dia_id or not text:
                        continue
                    # Prepend the speaker to the text so Mnemoss sees a
                    # conversational turn, not an orphan utterance. Mem0
                    # and Chroma will also see the same surface form,
                    # keeping the embedding parity story honest.
                    full_text = f"{speaker}: {text}"
                    row = {
                        "conversation_id": conv_id,
                        "dia_id": dia_id,
                        "ts": utterance_timestamp(base_ts, i),
                        "text": full_text,
                        "speaker": speaker,
                        "session": session_num,
                    }
                    mf.write(json.dumps(row) + "\n")
                    stats["utterances"] += 1

            # QA pairs → queries with relevant_dia_ids from evidence
            for qa in sample.get("qa", []):
                question = qa.get("question", "")
                evidence = qa.get("evidence") or []
                if not question or not evidence:
                    stats["queries_dropped_no_evidence"] += 1
                    continue
                row = {
                    "conversation_id": conv_id,
                    "question": question,
                    "relevant_dia_ids": list(evidence),
                    "answer": qa.get("answer"),
                    "category": qa.get("category"),
                }
                qf.write(json.dumps(row, default=str) + "\n")
                stats["queries_kept"] += 1

    return stats


def main() -> None:
    stats = prepare()
    print(
        f"LoCoMo prepared: {stats['conversations']} conversations, "
        f"{stats['utterances']} utterances, {stats['queries_kept']} queries "
        f"(dropped {stats['queries_dropped_no_evidence']} without evidence)."
    )
    print(f"  memories → {OUT_MEMORIES}")
    print(f"  queries  → {OUT_QUERIES}")


if __name__ == "__main__":
    sys.exit(main() or 0)
