"""``mnemoss inspect`` — print a human-readable workspace snapshot.

Usage::

    mnemoss inspect <workspace>                 # default: ~/.mnemoss
    mnemoss inspect <workspace> --root /path    # explicit storage root
    mnemoss inspect <workspace> --json          # machine-readable
    mnemoss inspect <workspace> --tombstones    # also list recent drops

Thin wrapper around the existing ``Mnemoss.status()`` / ``.tombstones()``
APIs — just saves everyone from writing the same script twenty times
when they need to peek at a live workspace.

The default output is a fixed-width table grouped by section (schema
/ memory / tiers / cost / dreams), tuned to fit an 80-column
terminal. ``--json`` emits one JSON object on stdout so ``jq`` works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from mnemoss import (
    FakeEmbedder,
    Mnemoss,
    StorageParams,
)
from mnemoss.store.sqlite_backend import SchemaMismatchError


def _format_row(label: str, value: Any, width: int = 30) -> str:
    return f"  {label:<{width}} {value}"


def _format_table(title: str, rows: list[tuple[str, Any]]) -> str:
    out = [f"── {title} " + "─" * (60 - len(title))]
    for label, value in rows:
        out.append(_format_row(label, value))
    return "\n".join(out)


def _render_human(status: dict[str, Any], tombstones: list[Any]) -> str:
    sections: list[str] = []

    sections.append(
        _format_table(
            "workspace",
            [
                ("name", status["workspace"]),
                ("schema_version", status["schema_version"]),
                ("embedder", f"{status['embedder']['id']} (dim={status['embedder']['dim']})"),
            ],
        )
    )

    tier_counts = status["tier_counts"]
    sections.append(
        _format_table(
            "memories",
            [
                ("total", status["memory_count"]),
                ("HOT", tier_counts.get("hot", 0)),
                ("WARM", tier_counts.get("warm", 0)),
                ("COLD", tier_counts.get("cold", 0)),
                ("DEEP", tier_counts.get("deep", 0)),
                ("tombstones", status["tombstone_count"]),
            ],
        )
    )

    cost = status["llm_cost"]
    limits = cost["limits"]
    sections.append(
        _format_table(
            "llm cost",
            [
                ("today_calls", cost["today_calls"]),
                ("month_calls", cost["month_calls"]),
                ("total_calls", cost["total_calls"]),
                ("cap per_run", limits["max_llm_calls_per_run"] or "unlimited"),
                ("cap per_day", limits["max_llm_calls_per_day"] or "unlimited"),
                ("cap per_month", limits["max_llm_calls_per_month"] or "unlimited"),
            ],
        )
    )

    dreams = status["dreams"]
    dream_rows: list[tuple[str, Any]] = [
        ("recent", dreams["recent_count"]),
        ("degraded in window", dreams["recent_degraded_count"]),
    ]
    if status.get("last_dream_at"):
        dream_rows.append(("last_dream_at", status["last_dream_at"]))
        dream_rows.append(("last_dream_trigger", status.get("last_dream_trigger")))
    sections.append(_format_table("dreams", dream_rows))

    timestamps = [
        ("last_observe_at", status.get("last_observe_at")),
        ("last_rebalance_at", status.get("last_rebalance_at")),
        ("last_dispose_at", status.get("last_dispose_at")),
    ]
    sections.append(
        _format_table(
            "timestamps",
            [(label, value or "(never)") for label, value in timestamps],
        )
    )

    if tombstones:
        tomb_lines = [f"── recent tombstones ({len(tombstones)}) " + "─" * 30]
        for t in tombstones[:20]:
            when = t.dropped_at.isoformat(timespec="seconds")
            tomb_lines.append(
                f"  {t.original_id}  {when}  {t.reason:<18}  "
                f"{(t.gist_snapshot or '')[:40]!r}"
            )
        sections.append("\n".join(tomb_lines))

    return "\n\n".join(sections) + "\n"


async def _run(args: argparse.Namespace) -> int:
    # We only need to open the workspace to call status() + tombstones().
    # The embedder choice doesn't matter — we won't embed anything — so
    # we use FakeEmbedder for zero model-load cost. The workspace's
    # pinned embedder_id / dim are validated on open, which is exactly
    # the failure mode we want to surface early.
    root = Path(args.root).expanduser() if args.root else None
    mem = Mnemoss(
        workspace=args.workspace,
        embedding_model=FakeEmbedder(dim=1),  # will mismatch on purpose
        storage=StorageParams(root=root),
    )
    try:
        try:
            status = await mem.status()
        except SchemaMismatchError as e:
            # Expected when the workspace was created with a different
            # embedder: our FakeEmbedder(dim=1) intentionally won't
            # match. Fall back to a minimal file-only peek.
            print(
                f"note: cannot open workspace under this embedder "
                f"({e}). Showing file metadata only.",
                flush=True,
            )
            _print_filesystem_only(args.workspace, root)
            return 0

        tombstones = (
            await mem.tombstones(limit=50) if args.tombstones else []
        )
    finally:
        await mem.close()

    if args.json:
        payload: dict[str, Any] = dict(status)
        if args.tombstones:
            payload["tombstones"] = [
                {
                    "original_id": t.original_id,
                    "dropped_at": t.dropped_at.isoformat(),
                    "reason": t.reason,
                    "gist_snapshot": t.gist_snapshot,
                    "b_at_drop": t.b_at_drop,
                }
                for t in tombstones
            ]
        print(json.dumps(payload, indent=2))
    else:
        print(_render_human(status, tombstones))
    return 0


def _print_filesystem_only(workspace: str, root: Path | None) -> None:
    """Best-effort disk peek when the caller's embedder doesn't match.

    We don't open the DB (we'd trip the dim pin), but we CAN list the
    workspace directory and show file sizes. Enough to confirm "yes
    the workspace exists" and point at the next step.
    """

    effective_root = root or Path.home() / ".mnemoss"
    ws_dir = effective_root / "workspaces" / workspace
    if not ws_dir.exists():
        print(f"  workspace directory does not exist: {ws_dir}")
        return
    print(f"  directory:  {ws_dir}")
    for name in ("memory.sqlite", "raw_log.sqlite", ".mnemoss.lock"):
        p = ws_dir / name
        if p.exists():
            size = p.stat().st_size
            print(f"  {name:<18} {size:>12,} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mnemoss inspect",
        description="Print a snapshot of a Mnemoss workspace.",
    )
    parser.add_argument("workspace", help="Workspace name.")
    parser.add_argument(
        "--root",
        default=None,
        help="Storage root (default: ~/.mnemoss).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    parser.add_argument(
        "--tombstones",
        action="store_true",
        help="Also list the 50 most recent tombstones.",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
