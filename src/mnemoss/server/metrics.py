"""Prometheus metrics for the REST server.

Soft-optional: if ``prometheus_client`` is not installed (the
``[observability]`` extra wasn't picked), every record_* function is
a no-op and the server skips mounting ``/metrics``. Nothing else
changes.

Label design deliberately keeps cardinality bounded — no ``agent_id``
labels because agent counts can grow unboundedly in multi-tenant
deployments. Per-agent breakdowns belong in logs, not metrics.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover — tested via HAS_PROMETHEUS toggling
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    HAS_PROMETHEUS = False
    CONTENT_TYPE_LATEST = "text/plain"


if HAS_PROMETHEUS:
    # One private registry so tests can reason about isolated state and
    # so ``mnemoss_*`` metrics don't collide with unrelated default
    # registry series in the host process.
    REGISTRY = CollectorRegistry()

    OBSERVES_TOTAL = Counter(
        "mnemoss_observes_total",
        "Total observe() calls by workspace and whether the message was encoded.",
        ["workspace", "encoded"],
        registry=REGISTRY,
    )
    OBSERVE_DURATION = Histogram(
        "mnemoss_observe_duration_seconds",
        "Wall-clock duration of observe() calls.",
        ["workspace"],
        registry=REGISTRY,
    )
    RECALLS_TOTAL = Counter(
        "mnemoss_recalls_total",
        "Total recall() calls by workspace.",
        ["workspace"],
        registry=REGISTRY,
    )
    RECALL_DURATION = Histogram(
        "mnemoss_recall_duration_seconds",
        "Wall-clock duration of recall() calls.",
        ["workspace"],
        registry=REGISTRY,
    )
    DREAMS_TOTAL = Counter(
        "mnemoss_dreams_total",
        "Total dream cycles run by trigger.",
        ["workspace", "trigger"],
        registry=REGISTRY,
    )
    DREAM_DURATION = Histogram(
        "mnemoss_dream_duration_seconds",
        "Wall-clock duration of dream cycles.",
        ["workspace", "trigger"],
        registry=REGISTRY,
    )
    DISPOSALS_TOTAL = Counter(
        "mnemoss_disposals_total",
        "Total memories disposed, labelled by reason.",
        ["workspace", "reason"],
        registry=REGISTRY,
    )
    MEMORY_COUNT = Gauge(
        "mnemoss_memory_count",
        "Number of live memories per tier. Refreshed on each /metrics scrape.",
        ["workspace", "tier"],
        registry=REGISTRY,
    )


def record_observe(workspace: str, *, encoded: bool, duration: float) -> None:
    if not HAS_PROMETHEUS:
        return
    OBSERVES_TOTAL.labels(workspace=workspace, encoded=str(encoded).lower()).inc()
    OBSERVE_DURATION.labels(workspace=workspace).observe(duration)


def record_recall(workspace: str, *, duration: float) -> None:
    if not HAS_PROMETHEUS:
        return
    RECALLS_TOTAL.labels(workspace=workspace).inc()
    RECALL_DURATION.labels(workspace=workspace).observe(duration)


def record_dream(workspace: str, *, trigger: str, duration: float) -> None:
    if not HAS_PROMETHEUS:
        return
    DREAMS_TOTAL.labels(workspace=workspace, trigger=trigger).inc()
    DREAM_DURATION.labels(workspace=workspace, trigger=trigger).observe(duration)


def record_disposal(workspace: str, *, activation_dead: int, redundant: int) -> None:
    if not HAS_PROMETHEUS:
        return
    if activation_dead:
        DISPOSALS_TOTAL.labels(workspace=workspace, reason="activation_dead").inc(activation_dead)
    if redundant:
        DISPOSALS_TOTAL.labels(workspace=workspace, reason="redundant").inc(redundant)


async def refresh_memory_gauges(pool: Any) -> None:
    """Refresh per-workspace memory gauges. Called from the /metrics
    endpoint so values are fresh at scrape time.

    Skips silently on any per-workspace error — one misbehaving
    workspace shouldn't 500 the whole scrape.
    """

    if not HAS_PROMETHEUS:
        return
    instances = dict(getattr(pool, "_instances", {}))
    for workspace_id, mem in instances.items():
        try:
            tiers = await mem.tier_counts()
        except Exception:  # noqa: BLE001
            continue
        for tier, count in tiers.items():
            MEMORY_COUNT.labels(workspace=workspace_id, tier=tier).set(count)


def latest_metrics() -> tuple[bytes, str]:
    """Render the Prometheus text exposition for the current registry."""

    if not HAS_PROMETHEUS:
        return b"", CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
