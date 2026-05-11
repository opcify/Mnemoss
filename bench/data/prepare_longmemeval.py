"""Load + validate the LongMemEval-S corpus for the long-term-memory bench.

LongMemEval (Wu et al. 2024, https://arxiv.org/abs/2410.10813) is the
standard end-to-end QA benchmark for chat-assistant memory layers. The
``-S`` slice is 500 questions over per-question multi-session histories
averaging ~115K tokens — the right size to compare Mnemoss against
Mem0 and MemOS 2.0 without burning a multi-day budget.

Per-question shape (one entry in the official ``longmemeval_s.json``)::

    {
      "question_id": "...",
      "question_type": one of QUESTION_TYPES,
      "question": "...",
      "answer": "...",
      "haystack_session_ids": ["sess_001", ...],
      "haystack_dates": ["2023/05/03 (Wed) 10:24", ...],
      "haystack_sessions": [
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}, ...],
        ...
      ],
      "answer_session_ids": ["sess_017", ...]   # subset of haystack_session_ids
    }

Eval protocol (replicated by ``bench/longmemeval.py``):

1. For each question, ingest every haystack session in *chronological*
   order — i.e. the order ``haystack_session_ids`` lists them in. Each
   turn becomes one observe; ``role``, ``session_id``, and a parsed
   timestamp travel with the call so backends that scope by session
   or weight by recency see the right signal.
2. After all sessions are ingested, query the memory layer with the
   final ``question`` and pull the top-K hits.
3. The bench harness composes those hits into a generator prompt,
   gets an answer, and an LLM judge scores it against ``answer``.
4. Accuracy is reported per ``question_type`` and overall.

This module owns step 0: locate the JSON, validate the shape, and
yield :class:`Question` rows. The dataset itself is *not* checked into
the repo — it's ~50 MB and lives behind a Hugging Face gate. We
expect callers to download it once and point us at the file::

    huggingface-cli download xiaowu0162/longmemeval \\
        --local-dir bench/data/longmemeval --repo-type dataset
    python -m bench.data.prepare_longmemeval \\
        bench/data/longmemeval/longmemeval_s.json

After the prep call succeeds, ``bench/data/longmemeval_s.json``
becomes the canonical path used by ``bench.longmemeval``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

QUESTION_TYPES: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)
"""The six categories the original paper reports per-type accuracy on.

Mnemoss vs Mem0 vs MemOS comparisons should always carry the per-type
breakdown — the ``knowledge-update`` and ``temporal-reasoning`` slices
are where memory architecture (vs vanilla embed+cosine) actually pays
off, and a single overall accuracy number hides that.
"""


CANONICAL_PATH = Path("bench/data/longmemeval_s.json")
"""Where ``bench.longmemeval`` looks for the dataset by default.

Either copy the official file here or pass ``--dataset`` on the CLI.
"""


@dataclass(frozen=True)
class Turn:
    """One message inside a haystack session."""

    role: str
    content: str


@dataclass(frozen=True)
class Session:
    """One multi-turn session inside a question's haystack.

    ``ts`` is parsed from ``haystack_dates`` into a UTC unix timestamp
    so backends that key on ``ts`` (raw stack, optional Mem0 metadata)
    see a consistent number, not the raw localized string.
    """

    session_id: str
    ts: float
    date_str: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Question:
    """One LongMemEval-S evaluation instance."""

    question_id: str
    question_type: str
    question: str
    answer: str
    sessions: tuple[Session, ...]
    answer_session_ids: frozenset[str]


def _parse_date(raw: str) -> float:
    """Parse a haystack date string to a UTC unix timestamp.

    The official dataset ships dates like ``"2023/05/03 (Wed) 10:24"``.
    We strip the day-of-week parenthetical and parse the rest with the
    formats we've actually observed in the released file. Anything
    that fails to parse falls back to a stable epoch — the bench
    treats failed parses as "no recency signal" rather than crashing
    a 500-question run on a single malformed row.
    """

    cleaned = raw
    if "(" in cleaned and ")" in cleaned:
        before, _, rest = cleaned.partition("(")
        _, _, after = rest.partition(")")
        cleaned = (before + after).strip()
    cleaned = " ".join(cleaned.split())
    for fmt in ("%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


def _validate_question(idx: int, raw: dict) -> Question:
    """Convert one raw dict into a typed :class:`Question`. Strict on shape.

    Strict because LongMemEval's released file is stable and version-pinned;
    a missing field at this layer is much more likely a wrong file than a
    schema drift. Loud failures here save us debugging weird harness
    fall-through later.
    """

    required = (
        "question_id",
        "question_type",
        "question",
        "answer",
        "haystack_session_ids",
        "haystack_dates",
        "haystack_sessions",
        "answer_session_ids",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(
            f"longmemeval entry #{idx} ({raw.get('question_id', '<no id>')}) "
            f"missing required fields: {missing}"
        )

    qtype = raw["question_type"]
    if qtype not in QUESTION_TYPES:
        # Don't reject — older internal forks add subtypes like
        # ``temporal-reasoning-explicit``. Strip a trailing modifier and
        # match the prefix; if still no match, keep the original string
        # so the harness can report it under "other".
        for canonical in QUESTION_TYPES:
            if qtype.startswith(canonical):
                qtype = canonical
                break

    sids = raw["haystack_session_ids"]
    dates = raw["haystack_dates"]
    sessions_raw = raw["haystack_sessions"]
    if not (len(sids) == len(dates) == len(sessions_raw)):
        raise ValueError(
            f"longmemeval entry #{idx} ({raw['question_id']}): "
            f"haystack_session_ids ({len(sids)}), haystack_dates ({len(dates)}), "
            f"and haystack_sessions ({len(sessions_raw)}) lengths disagree"
        )

    sessions: list[Session] = []
    for sid, date_str, turns_raw in zip(sids, dates, sessions_raw, strict=True):
        turns = tuple(Turn(role=t["role"], content=t["content"]) for t in turns_raw)
        sessions.append(
            Session(
                session_id=sid,
                ts=_parse_date(date_str),
                date_str=date_str,
                turns=turns,
            )
        )

    # Sort sessions chronologically. The released file is already in
    # order, but we don't trust upstream ordering invariants on the
    # input side of a bench — a stable sort here eliminates one class
    # of "why did this question score differently across runs" bug.
    sessions.sort(key=lambda s: s.ts)

    return Question(
        question_id=raw["question_id"],
        question_type=qtype,
        question=raw["question"],
        answer=raw["answer"],
        sessions=tuple(sessions),
        answer_session_ids=frozenset(raw["answer_session_ids"]),
    )


def load_longmemeval_s(path: Path | None = None) -> list[Question]:
    """Read and validate the LongMemEval-S JSON file.

    Parameters
    ----------
    path:
        Path to the released ``longmemeval_s.json``. Defaults to
        :data:`CANONICAL_PATH` (``bench/data/longmemeval_s.json``).

    Raises
    ------
    FileNotFoundError
        With a download hint if the file isn't on disk.
    ValueError
        If the file shape doesn't match the documented schema.
    """

    p = path if path is not None else CANONICAL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download LongMemEval-S then prep:\n\n"
            "    huggingface-cli download xiaowu0162/longmemeval \\\n"
            "        --local-dir bench/data/longmemeval --repo-type dataset\n"
            f"    python -m bench.data.prepare_longmemeval \\\n"
            f"        bench/data/longmemeval/longmemeval_s.json\n\n"
            "Source: https://huggingface.co/datasets/xiaowu0162/longmemeval"
        )

    with p.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a JSON array of questions, got {type(raw).__name__}")

    out: list[Question] = []
    for idx, entry in enumerate(raw):
        out.append(_validate_question(idx, entry))
    return out


def _summarize(questions: list[Question]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(QUESTION_TYPES, 0)
    other = 0
    for q in questions:
        if q.question_type in counts:
            counts[q.question_type] += 1
        else:
            other += 1
    if other:
        counts["other"] = other
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "source",
        type=Path,
        help="Path to the downloaded longmemeval_s.json.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=CANONICAL_PATH,
        help=f"Where to copy the validated file. Default: {CANONICAL_PATH}",
    )
    args = p.parse_args(argv)

    if not args.source.exists():
        print(f"error: {args.source} does not exist", file=sys.stderr)
        return 2

    # Validate by loading from the source path before copying so we
    # don't ever land an unreadable file at CANONICAL_PATH.
    questions = load_longmemeval_s(args.source)
    counts = _summarize(questions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() != args.out.resolve():
        shutil.copy(args.source, args.out)

    print(f"validated {len(questions)} questions, copied → {args.out}")
    for k, v in counts.items():
        print(f"  {k:30s} {v:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
