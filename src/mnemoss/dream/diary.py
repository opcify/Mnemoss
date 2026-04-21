"""Dream Diary — per-workspace Markdown audit log of dream runs.

Every call to ``Mnemoss.dream(...)`` appends one entry to
``{storage_root}/workspaces/{workspace}/dreams/YYYY-MM-DD.md``. The
entry captures trigger, agent, duration, and per-phase outcomes, so a
developer can read the file cold and reconstruct what changed.

Format is strict plain Markdown — no LLM and no rendering-layer
dependencies. Future stages add more phases; the renderer walks
``DreamReport.outcomes`` and emits a section per phase regardless of
which ones fired.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mnemoss.dream.types import DreamReport, PhaseName

log = logging.getLogger(__name__)


def dream_diary_path(storage_root: Path, workspace: str) -> Path:
    """Return the per-workspace, per-day diary path (doesn't create dirs)."""

    return (
        storage_root
        / "workspaces"
        / workspace
        / "dreams"
        / "diary.md"
    )


def render_dream_entry(report: DreamReport) -> str:
    """Render a single dream run as a Markdown section."""

    started = report.started_at.isoformat(timespec="seconds")
    agent = report.agent_id or "(ambient)"
    lines = [
        f"## Dream run · {started}",
        "",
        f"- **Trigger:** `{report.trigger.value}`",
        f"- **Agent:** {agent}",
        f"- **Duration:** {report.duration_seconds():.3f}s",
        "",
    ]

    for outcome in report.outcomes:
        heading = f"### {outcome.phase.value.upper()} · {outcome.status}"
        lines.append(heading)
        if outcome.details:
            lines.extend(_render_details(outcome.phase, outcome.details))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_details(phase: PhaseName, details: dict) -> list[str]:
    """Format phase-specific details as a Markdown list.

    The replay phase's ``memories`` list is noisy — emit only the count
    via ``selected`` / ``memory_ids`` and hide the full dataclass dump.
    """

    out: list[str] = []
    for key, value in details.items():
        if key == "memories":  # Memory objects — too much for a diary.
            continue
        if key == "memory_ids" and len(value) > 6:
            out.append(f"- **{key}:** {len(value)} (first 3: {value[:3]})")
            continue
        if key == "ids" and isinstance(value, list) and len(value) > 6:
            out.append(f"- **{key}:** {len(value)} (first 3: {value[:3]})")
            continue
        out.append(f"- **{key}:** {value}")
    _ = phase  # currently unused; here for future phase-specific formatting
    return out


def append_entry(
    path: Path,
    report: DreamReport,
    *,
    separator: str = "\n\n---\n\n",
) -> None:
    """Write ``render_dream_entry(report)`` to ``path``, creating parents
    if needed. Appends a horizontal rule between successive entries.

    Swallows I/O errors with a warning — the dream run itself has
    already succeeded and the diary is a best-effort audit log.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.exists() and path.stat().st_size > 0
        entry = render_dream_entry(report)
        with path.open("a", encoding="utf-8") as f:
            if existing:
                f.write(separator)
            f.write(entry)
    except OSError as e:  # pragma: no cover - platform-specific
        log.warning("dream diary write failed for %s: %s", path, e)
