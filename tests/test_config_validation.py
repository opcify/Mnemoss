"""Configuration-validation tests.

Each params dataclass rejects obviously-wrong values at construction
time so a misconfigured deployment fails fast (at import/startup)
rather than producing subtle wrong-answer bugs at recall time.

The tests here deliberately don't enumerate every valid combination
— they hit the boundaries (negative, zero-where-forbidden, out-of-
domain, ordering-violations) and verify the error messages name the
offending field so operators can find the problem without a
debugger.
"""

from __future__ import annotations

import pytest

from mnemoss import (
    CostLimits,
    DreamerParams,
    EncoderParams,
    FormulaParams,
    MnemossConfig,
    SegmentationParams,
)

# ─── FormulaParams ─────────────────────────────────────────────────


def test_formula_params_defaults_pass_validation() -> None:
    # Sanity: the shipped defaults don't trip their own validator.
    FormulaParams()


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"d": 0}, "d"),
        ({"d": -0.1}, "d"),
        ({"mp": 0}, "mp"),
        ({"mp": -1.5}, "mp"),
        ({"s_max": 0}, "s_max"),
        ({"t_floor_seconds": 0}, "t_floor_seconds"),
        ({"eta_tau_seconds": 0}, "eta_tau_seconds"),
    ],
)
def test_formula_rejects_non_positive_scalars(kwargs: dict, offending_field: str) -> None:
    with pytest.raises(ValueError, match=offending_field):
        FormulaParams(**kwargs)


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"noise_scale": -0.01}, "noise_scale"),
        ({"alpha": -0.5}, "alpha"),
        ({"beta": -1.0}, "beta"),
        ({"gamma": -2.0}, "gamma"),
        ({"delta": -0.1}, "delta"),
        ({"eta_0": -0.5}, "eta_0"),
        ({"epsilon_max": -0.1}, "epsilon_max"),
        ({"streak_reset_seconds": -1.0}, "streak_reset_seconds"),
    ],
)
def test_formula_rejects_negative_non_negatives(kwargs: dict, offending_field: str) -> None:
    with pytest.raises(ValueError, match=offending_field):
        FormulaParams(**kwargs)


def test_formula_rejects_out_of_order_tier_offsets() -> None:
    with pytest.raises(ValueError, match="hot >= warm >= cold"):
        FormulaParams(
            confidence_hot_offset=0.0,
            confidence_warm_offset=1.0,
            confidence_cold_offset=2.0,
        )


def test_formula_accepts_equal_tier_offsets() -> None:
    # Equal is legal; strict ordering not required.
    FormulaParams(
        confidence_hot_offset=1.0,
        confidence_warm_offset=1.0,
        confidence_cold_offset=1.0,
    )


@pytest.mark.parametrize("bad_cosine", [-0.1, 1.01, 2.0])
def test_formula_rejects_cosine_out_of_unit_interval(
    bad_cosine: float,
) -> None:
    with pytest.raises(ValueError, match="same_topic_cosine"):
        FormulaParams(same_topic_cosine=bad_cosine)


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"expand_hops_max": 0}, "expand_hops_max"),
        ({"expand_hops_max": -1}, "expand_hops_max"),
        ({"expand_candidates_max": 0}, "expand_candidates_max"),
        ({"expand_candidates_max": -5}, "expand_candidates_max"),
    ],
)
def test_formula_rejects_non_positive_graph_caps(kwargs: dict, offending_field: str) -> None:
    with pytest.raises(ValueError, match=offending_field):
        FormulaParams(**kwargs)


# ─── EncoderParams ─────────────────────────────────────────────────


def test_encoder_params_defaults_pass() -> None:
    EncoderParams()


def test_encoder_rejects_empty_roles() -> None:
    with pytest.raises(ValueError, match="encoded_roles"):
        EncoderParams(encoded_roles=set())


def test_encoder_rejects_zero_working_capacity() -> None:
    with pytest.raises(ValueError, match="working_memory_capacity"):
        EncoderParams(working_memory_capacity=0)


def test_encoder_rejects_negative_cooccurrence_window() -> None:
    with pytest.raises(ValueError, match="session_cooccurrence_window"):
        EncoderParams(session_cooccurrence_window=-1)


# ─── SegmentationParams ───────────────────────────────────────────


def test_segmentation_params_defaults_pass() -> None:
    SegmentationParams()


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"time_gap_seconds": 0}, "time_gap_seconds"),
        ({"time_gap_seconds": -1.0}, "time_gap_seconds"),
        ({"max_event_messages": 0}, "max_event_messages"),
        ({"max_event_characters": 0}, "max_event_characters"),
    ],
)
def test_segmentation_rejects_non_positive(kwargs: dict, offending_field: str) -> None:
    with pytest.raises(ValueError, match=offending_field):
        SegmentationParams(**kwargs)


# ─── MnemossConfig ─────────────────────────────────────────────────


def test_mnemoss_config_rejects_empty_workspace() -> None:
    with pytest.raises(ValueError, match="workspace"):
        MnemossConfig(workspace="")


def test_mnemoss_config_rejects_whitespace_only_workspace() -> None:
    with pytest.raises(ValueError, match="workspace"):
        MnemossConfig(workspace="   ")


# ─── CostLimits ────────────────────────────────────────────────────


def test_cost_limits_defaults_are_unlimited() -> None:
    cl = CostLimits()
    assert cl.is_unlimited


def test_cost_limits_accepts_zero_cap() -> None:
    """Zero is legal and means "no calls this period" — a dry-run
    or read-only workspace uses this."""

    cl = CostLimits(max_llm_calls_per_run=0)
    assert cl.max_llm_calls_per_run == 0
    assert not cl.is_unlimited


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"max_llm_calls_per_run": -1}, "max_llm_calls_per_run"),
        ({"max_llm_calls_per_day": -10}, "max_llm_calls_per_day"),
        ({"max_llm_calls_per_month": -100}, "max_llm_calls_per_month"),
    ],
)
def test_cost_limits_rejects_negative(kwargs: dict, offending_field: str) -> None:
    with pytest.raises(ValueError, match=offending_field):
        CostLimits(**kwargs)


def test_cost_limits_rejects_non_integer_type() -> None:
    with pytest.raises(ValueError, match="max_llm_calls_per_run"):
        CostLimits(max_llm_calls_per_run=3.5)  # type: ignore[arg-type]


def test_cost_limits_rejects_bool_as_int() -> None:
    """``True`` is an ``int`` subclass, which would quietly pass as
    "make at most 1 call per run" if we relied on ``isinstance(x, int)``
    alone. We treat bools as a type error."""

    with pytest.raises(ValueError, match="max_llm_calls_per_run"):
        CostLimits(max_llm_calls_per_run=True)  # type: ignore[arg-type]


# ─── DreamerParams ─────────────────────────────────────────────────


def test_dreamer_params_defaults_match_runner_hardcodes() -> None:
    """Defaults must match the historical hardcoded values in
    ``DreamRunner.__init__`` so adding ``DreamerParams`` is a pure
    refactor with no behavior change for callers that didn't pass it."""

    p = DreamerParams()
    assert p.cluster_min_size == 3
    assert p.replay_limit == 100
    assert p.replay_min_base_level is None


@pytest.mark.parametrize(
    "kwargs, offending_field",
    [
        ({"cluster_min_size": 0}, "cluster_min_size"),
        ({"cluster_min_size": -1}, "cluster_min_size"),
        ({"replay_limit": 0}, "replay_limit"),
        ({"replay_limit": -100}, "replay_limit"),
    ],
)
def test_dreamer_rejects_non_positive(kwargs: dict, offending_field: str) -> None:
    with pytest.raises(ValueError, match=offending_field):
        DreamerParams(**kwargs)


def test_dreamer_accepts_negative_replay_min_base_level() -> None:
    """Negative base-level floors are legitimate ('only memories above
    near-dead activation'); the validator must not reject them."""

    p = DreamerParams(replay_min_base_level=-2.0)
    assert p.replay_min_base_level == -2.0


def test_dreamer_accepts_none_replay_min_base_level() -> None:
    p = DreamerParams(replay_min_base_level=None)
    assert p.replay_min_base_level is None


def test_mnemoss_config_default_dreamer_present() -> None:
    """``MnemossConfig`` gets a default ``DreamerParams`` so existing
    callers keep working without naming the new field."""

    cfg = MnemossConfig(workspace="test")
    assert cfg.dreamer == DreamerParams()
