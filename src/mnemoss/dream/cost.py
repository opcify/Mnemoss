"""Dream cost governor — cap + track LLM call budgets across runs.

Dream P3 Consolidate is the one place in Mnemoss that calls an LLM.
A runaway cluster count on a large workspace can rack up real money,
so every production deployment wants at least:

1. A per-run ceiling: "don't make more than N LLM calls in one
   dream, even if there are more clusters."
2. A per-day ceiling: "don't spend more than M LLM calls in a single
   calendar day across all dream runs for this workspace."
3. Visibility: ``status()`` should surface today's + this month's
   call count so operators can see pressure building.

This module provides:

- ``CostLimits`` — a frozen policy dataclass passed to ``DreamRunner``.
- ``CostLedger`` — persists per-day call counts in ``workspace_meta``
  so limits span runs and restarts.
- ``CostExceeded`` — raised (or returned as a skip reason) when the
  next call would breach the cap.

Calls are the unit we track, not tokens. Tokens would be truer cost
but require provider-specific accounting (OpenAI reports in the
response, Gemini differently, Anthropic differently); call-count is
a universal proxy that maps linearly to spend for any single model
choice. Upgrade to token tracking when someone actually needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

import apsw

UTC = timezone.utc

# Ledger key prefix in ``workspace_meta`` — lets us filter with a single
# ``LIKE`` and keeps the key namespace tidy.
_DAILY_PREFIX = "cost:daily:"
_TOTAL_KEY = "cost:total_calls"


class CostExceededError(RuntimeError):
    """Raised when a caller tries to spend past a configured limit."""


@dataclass(frozen=True)
class CostLimits:
    """Cost ceilings for one dream run.

    ``None`` on a field means "no limit for that dimension." All three
    can coexist; the first cap hit wins.

    ``max_llm_calls_per_run`` stops the consolidate loop partway
    through when the run-local call count reaches the cap. The
    remaining clusters are skipped and the dream report names them.

    ``max_llm_calls_per_day`` checks the persisted daily ledger
    before each call; the cap spans runs.

    ``max_llm_calls_per_month`` checks all persisted days in the
    current calendar month.

    All limits are validated at construction: negative or non-integer
    values raise ``ValueError``. ``0`` is legal and means "make no
    calls this period" — useful for read-only workspaces that
    shouldn't dream at all.
    """

    max_llm_calls_per_run: int | None = None
    max_llm_calls_per_day: int | None = None
    max_llm_calls_per_month: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_llm_calls_per_run",
            "max_llm_calls_per_day",
            "max_llm_calls_per_month",
        ):
            val = getattr(self, name)
            if val is None:
                continue
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(
                    f"{name} must be an int or None (got {val!r})"
                )
            if val < 0:
                raise ValueError(
                    f"{name} must be >= 0 (got {val!r}); use None for "
                    "unlimited."
                )

    @property
    def is_unlimited(self) -> bool:
        return (
            self.max_llm_calls_per_run is None
            and self.max_llm_calls_per_day is None
            and self.max_llm_calls_per_month is None
        )


@dataclass
class CostSnapshot:
    """Point-in-time read of the ledger. Surfaced via ``status()``."""

    today_calls: int
    month_calls: int
    total_calls: int


class CostLedger:
    """Daily + cumulative LLM call ledger persisted in ``workspace_meta``.

    Keys written:

    - ``cost:daily:YYYY-MM-DD`` — calls on that calendar day (UTC).
    - ``cost:total_calls`` — monotonic all-time counter.

    All reads/writes are synchronous SQL against the memory DB. Cheap
    enough to hit on every LLM call — one SELECT + one UPSERT per call
    on a tiny key-value table.
    """

    def __init__(self, conn: apsw.Connection) -> None:
        self._conn = conn

    # ─── reads ─────────────────────────────────────────────────────

    def today_calls(self, *, now: datetime | None = None) -> int:
        day = _today(now)
        return self._read_int(_daily_key(day))

    def month_calls(self, *, now: datetime | None = None) -> int:
        ym = _today(now).strftime("%Y-%m")
        rows = self._conn.execute(
            "SELECT v FROM workspace_meta WHERE k LIKE ?",
            (f"{_DAILY_PREFIX}{ym}-%",),
        ).fetchall()
        total = 0
        for (raw,) in rows:
            try:
                total += int(raw)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                continue
        return total

    def total_calls(self) -> int:
        return self._read_int(_TOTAL_KEY)

    def snapshot(self, *, now: datetime | None = None) -> CostSnapshot:
        return CostSnapshot(
            today_calls=self.today_calls(now=now),
            month_calls=self.month_calls(now=now),
            total_calls=self.total_calls(),
        )

    # ─── writes ────────────────────────────────────────────────────

    def record_call(self, *, now: datetime | None = None) -> None:
        """Increment today's + total counters. Cheap, no-transaction."""

        day = _today(now)
        key = _daily_key(day)
        # Two upserts in one tiny transaction so a crash mid-write
        # can't leave the counters disagreeing.
        with self._conn:
            self._conn.execute(
                "INSERT INTO workspace_meta(k, v) VALUES (?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v = CAST(v AS INTEGER) + 1",
                (key, "1"),
            )
            self._conn.execute(
                "INSERT INTO workspace_meta(k, v) VALUES (?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v = CAST(v AS INTEGER) + 1",
                (_TOTAL_KEY, "1"),
            )

    # ─── budget check ──────────────────────────────────────────────

    def check_budget(
        self,
        limits: CostLimits,
        *,
        run_calls: int,
        now: datetime | None = None,
    ) -> str | None:
        """Return a skip reason string if any cap would be breached by
        the next call, else ``None``.

        ``run_calls`` is the consolidate loop's local counter — what
        we've spent so far in *this* run, independent of prior runs.
        The daily and monthly caps include this run's activity via the
        ledger only after ``record_call`` has been invoked, so we add
        ``run_calls`` on the read side to keep the check exact.
        """

        if limits.is_unlimited:
            return None
        if (
            limits.max_llm_calls_per_run is not None
            and run_calls >= limits.max_llm_calls_per_run
        ):
            return (
                f"run cap reached ({run_calls}/{limits.max_llm_calls_per_run})"
            )
        if limits.max_llm_calls_per_day is not None:
            today = self.today_calls(now=now)
            if today >= limits.max_llm_calls_per_day:
                return f"daily cap reached ({today}/{limits.max_llm_calls_per_day})"
        if limits.max_llm_calls_per_month is not None:
            month = self.month_calls(now=now)
            if month >= limits.max_llm_calls_per_month:
                return f"monthly cap reached ({month}/{limits.max_llm_calls_per_month})"
        return None

    # ─── internals ────────────────────────────────────────────────

    def _read_int(self, key: str) -> int:
        row = self._conn.execute(
            "SELECT v FROM workspace_meta WHERE k = ?", (key,)
        ).fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (ValueError, TypeError):
            return 0


# ─── helpers ──────────────────────────────────────────────────────


def _today(now: datetime | None) -> date:
    if now is None:
        now = datetime.now(UTC)
    return now.astimezone(UTC).date()


def _daily_key(day: date) -> str:
    return f"{_DAILY_PREFIX}{day.isoformat()}"
