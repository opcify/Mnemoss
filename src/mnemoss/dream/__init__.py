"""Cold-path consolidation (Dreaming).

Six phases — Replay, Cluster, Consolidate, Relations, Rebalance,
Dispose. Consolidate collapses the former Extract / Refine / Generalize
trio into one LLM call per cluster (see ``consolidate.py``). P7
Rebalance is also available standalone via ``Mnemoss.rebalance``.
"""

from mnemoss.dream.consolidate import (
    ConsolidationResult,
    Refinement,
    build_consolidate_prompt,
    consolidate_cluster,
)
from mnemoss.dream.cost import CostLedger, CostLimits
from mnemoss.dream.dispose import DisposalStats
from mnemoss.dream.runner import DreamRunner
from mnemoss.dream.types import (
    DreamReport,
    PhaseName,
    PhaseOutcome,
    TriggerType,
)

__all__ = [
    "ConsolidationResult",
    "CostLedger",
    "CostLimits",
    "DisposalStats",
    "DreamReport",
    "DreamRunner",
    "PhaseName",
    "PhaseOutcome",
    "Refinement",
    "TriggerType",
    "build_consolidate_prompt",
    "consolidate_cluster",
]
