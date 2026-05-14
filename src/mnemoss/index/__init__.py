"""Index-management utilities.

Stage 2 ships the P7 Rebalance pass here. Stage 4's dream()/P7 pipeline
will wrap ``rebalance`` unchanged, so the decision logic has exactly one
home. The adaptive-caps controller (Method C) also lives here.
"""

from mnemoss.index.adaptive_caps import (
    TierTelemetry,
    TierTelemetryLedger,
    compute_adjusted_caps,
    maybe_adjust_caps,
)
from mnemoss.index.rebalance import RebalanceStats, rebalance

__all__ = [
    "RebalanceStats",
    "TierTelemetry",
    "TierTelemetryLedger",
    "compute_adjusted_caps",
    "maybe_adjust_caps",
    "rebalance",
]
