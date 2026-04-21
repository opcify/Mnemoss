"""Cold-path consolidation (Dreaming).

Stage 4 shipped phases P1 (Replay), P2 (Cluster), P3 (Extract), P5
(Relations) and the two "light" triggers (idle, session_end). Stage 5
added P4 Refine, P6 Generalize, P8 Dispose, and the deep triggers
(surprise, cognitive_load, nightly). P7 Rebalance is available
standalone via ``Mnemoss.rebalance`` and is also dispatched by the
nightly trigger.
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
