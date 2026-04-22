"""Unit tests for spreading activation."""

from __future__ import annotations

import math

from mnemoss.core.config import FormulaParams
from mnemoss.formula.spreading import compute_spreading

PARAMS = FormulaParams()


def test_empty_active_set_returns_zero() -> None:
    assert compute_spreading("m1", [], {}, {}, PARAMS) == 0.0


def test_no_connection_returns_zero() -> None:
    # j is active but doesn't point at target.
    assert compute_spreading("target", ["j"], {"j": {"other"}}, {"j": 1}, PARAMS) == 0.0


def test_single_connection_uniform_weight() -> None:
    # One active memory with fan=1 → S = S_max - ln(1) = S_max; W = 1.
    result = compute_spreading("target", ["j"], {"j": {"target"}}, {"j": 1}, PARAMS)
    assert math.isclose(result, PARAMS.s_max, rel_tol=1e-9)


def test_fan_effect_reduces_signal() -> None:
    """Higher fan → weaker per-relation signal."""
    low_fan = compute_spreading("target", ["j"], {"j": {"target"}}, {"j": 1}, PARAMS)
    high_fan = compute_spreading("target", ["j"], {"j": {"target"}}, {"j": 100}, PARAMS)
    assert high_fan < low_fan


def test_fan_zero_treated_as_one() -> None:
    """Stage 1 convention: missing/zero fan → 1 (avoids ln(0) = -inf)."""
    result = compute_spreading("target", ["j"], {"j": {"target"}}, {}, PARAMS)
    assert math.isfinite(result)
    assert math.isclose(result, PARAMS.s_max, rel_tol=1e-9)


def test_uniform_weights_scale_with_active_set() -> None:
    """Two active memories both pointing at target → each contributes 1/2 · S_max."""
    result = compute_spreading(
        "target",
        ["j1", "j2"],
        {"j1": {"target"}, "j2": {"target"}},
        {"j1": 1, "j2": 1},
        PARAMS,
    )
    # Sum: (1/2)·S_max + (1/2)·S_max = S_max.
    assert math.isclose(result, PARAMS.s_max, rel_tol=1e-9)


def test_custom_attention_weights() -> None:
    result = compute_spreading(
        "target",
        ["j1", "j2"],
        {"j1": {"target"}, "j2": {"target"}},
        {"j1": 1, "j2": 1},
        PARAMS,
        attention_weights={"j1": 1.0, "j2": 0.0},
    )
    # Only j1 contributes.
    assert math.isclose(result, PARAMS.s_max, rel_tol=1e-9)
