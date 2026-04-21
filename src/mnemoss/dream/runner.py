"""Dream dispatcher: trigger → phase sequence → DreamReport.

Each trigger picks a subset of the available phases per §6.3. The
runner records a ``PhaseOutcome`` for every phase it attempts, so the
report always tells the caller what happened even when a phase was
skipped (e.g. no LLM configured, empty replay set).

Stage 4 implements P1 here. P2 Cluster, P3 Extract, P5 Relations land
in Checkpoint N; until then the runner records them as ``skipped`` so
the whole dispatch path is already exercised by tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mnemoss.core.config import FormulaParams
from mnemoss.core.types import Memory
from mnemoss.dream.replay import select_replay_candidates
from mnemoss.dream.types import (
    DreamReport,
    PhaseName,
    PhaseOutcome,
    TriggerType,
)
from mnemoss.llm.client import LLMClient
from mnemoss.store.sqlite_backend import SQLiteBackend

UTC = timezone.utc


PHASES_BY_TRIGGER: dict[TriggerType, list[PhaseName]] = {
    # Per §6.3. P4 Refine is in the spec for session_end/nightly but is
    # deferred to Stage 5 (D10).
    TriggerType.IDLE: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.EXTRACT,
        PhaseName.RELATIONS,
    ],
    TriggerType.SESSION_END: [
        PhaseName.REPLAY,
        PhaseName.CLUSTER,
        PhaseName.EXTRACT,
        PhaseName.RELATIONS,
    ],
    TriggerType.TASK_COMPLETION: [
        PhaseName.REPLAY,
        PhaseName.EXTRACT,
        PhaseName.RELATIONS,
    ],
}


class DreamRunner:
    """Runs one dream cycle for a trigger.

    Stateless w.r.t. the store / params / LLM dependencies — construct
    fresh for each call, or reuse if the config is stable.
    """

    def __init__(
        self,
        store: SQLiteBackend,
        params: FormulaParams,
        *,
        llm: LLMClient | None = None,
        replay_limit: int = 100,
        replay_min_base_level: float | None = None,
    ) -> None:
        self._store = store
        self._params = params
        self._llm = llm
        self._replay_limit = replay_limit
        self._replay_min_base_level = replay_min_base_level

    async def run(
        self,
        trigger: TriggerType,
        *,
        agent_id: str | None = None,
        now: datetime | None = None,
    ) -> DreamReport:
        t0 = now if now is not None else datetime.now(UTC)
        report = DreamReport(
            trigger=trigger,
            started_at=t0,
            finished_at=t0,
            agent_id=agent_id,
        )

        replay_set: list[Memory] = []
        for phase in PHASES_BY_TRIGGER.get(trigger, []):
            outcome = await self._run_phase(phase, replay_set, agent_id, t0)
            report.outcomes.append(outcome)
            if phase is PhaseName.REPLAY:
                replay_set = outcome.details.get("memories", [])

        report.finished_at = datetime.now(UTC)
        return report

    async def _run_phase(
        self,
        phase: PhaseName,
        replay_set: list[Memory],
        agent_id: str | None,
        now: datetime,
    ) -> PhaseOutcome:
        if phase is PhaseName.REPLAY:
            memories = await select_replay_candidates(
                self._store,
                agent_id,
                self._params,
                now=now,
                limit=self._replay_limit,
                min_base_level=self._replay_min_base_level,
            )
            return PhaseOutcome(
                phase=PhaseName.REPLAY,
                status="ok",
                details={
                    "selected": len(memories),
                    "memories": memories,
                    "memory_ids": [m.id for m in memories],
                },
            )

        if phase is PhaseName.CLUSTER:
            return PhaseOutcome(
                phase=PhaseName.CLUSTER,
                status="skipped",
                details={"reason": "deferred to Checkpoint N"},
            )

        if phase is PhaseName.EXTRACT:
            if self._llm is None:
                return PhaseOutcome(
                    phase=PhaseName.EXTRACT,
                    status="skipped",
                    details={"reason": "no llm configured"},
                )
            return PhaseOutcome(
                phase=PhaseName.EXTRACT,
                status="skipped",
                details={"reason": "deferred to Checkpoint N"},
            )

        if phase is PhaseName.RELATIONS:
            return PhaseOutcome(
                phase=PhaseName.RELATIONS,
                status="skipped",
                details={"reason": "deferred to Checkpoint N"},
            )

        # Unreachable — PhaseName is exhaustive.
        raise RuntimeError(f"unknown phase {phase}")
