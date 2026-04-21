"""Cold-path consolidation (Dreaming).

Stage 4 ships phases P1 (Replay), P2 (Cluster), P3 (Extract), P5
(Relations) and the three "light" triggers (idle, session_end,
task_completion). P4 Refine, P6 Generalize, P7 Rebalance, P8 Dispose,
and the deep triggers land in Stage 5 (P7 is already available
standalone via ``Mnemoss.rebalance``).
"""

from mnemoss.dream.runner import DreamRunner
from mnemoss.dream.types import (
    DreamReport,
    PhaseName,
    PhaseOutcome,
    TriggerType,
)

__all__ = [
    "DreamReport",
    "DreamRunner",
    "PhaseName",
    "PhaseOutcome",
    "TriggerType",
]
