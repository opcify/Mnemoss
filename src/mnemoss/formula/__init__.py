"""Formula module — the heart of Mnemoss.

All functions here are pure: no I/O, no randomness without an explicit RNG,
no side effects. They can be unit-tested in full isolation. Integration with
the store/encoder/recall layers happens in ``recall.engine``.
"""

from mnemoss.formula.activation import ActivationBreakdown, compute_activation
from mnemoss.formula.base_level import compute_base_level
from mnemoss.formula.idx_priority import compute_idx_priority
from mnemoss.formula.matching import compute_matching, matching_weights
from mnemoss.formula.noise import sample_noise
from mnemoss.formula.query_bias import compute_query_bias
from mnemoss.formula.spreading import compute_spreading

__all__ = [
    "ActivationBreakdown",
    "compute_activation",
    "compute_base_level",
    "compute_idx_priority",
    "compute_matching",
    "compute_query_bias",
    "compute_spreading",
    "matching_weights",
    "sample_noise",
]
