"""Schema migration framework.

Each migration upgrades a workspace DB from ``from_version`` to
``to_version`` by exactly one step. On open, if the stored version is
older than ``SCHEMA_VERSION``, we find the chain of registered
migrations and apply them sequentially inside a single transaction —
either every step lands or none of them do.

If the stored version is *newer* than the code, we raise: users must
upgrade their Mnemoss install, not roll data back. Going backwards on
a vector store risks silent corruption (e.g., different embedder
output shape) that a migration can't safely undo.

Adding a new migration:

1. Bump ``SCHEMA_VERSION`` in ``src/mnemoss/core/config.py``.
2. Write the DDL/backfill logic as a plain function that takes an
   ``apsw.Connection`` and mutates the DB in place.
3. Append a ``Migration`` to ``MIGRATIONS`` below. Keep each step a
   single-version bump so chains are predictable.
4. Cover it in ``tests/test_migrations.py``.

Two separate files share the same ``SCHEMA_VERSION`` pin:

- ``memory.sqlite`` goes through the chain in ``MIGRATIONS``.
- ``raw_log.sqlite`` has had no schema changes beyond v6's creation,
  so ``migrate_raw_log`` just bumps the stored marker. When a real
  raw-log schema change lands, register it in ``RAW_LOG_MIGRATIONS``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import apsw

MigrationFn = Callable[[apsw.Connection], None]


class MigrationError(RuntimeError):
    """Raised when a migration can't be applied (missing chain link,
    data newer than code, or a step failed mid-transaction)."""


@dataclass(frozen=True)
class Migration:
    """A one-step schema upgrade.

    ``fn`` receives the memory DB connection *inside* the transaction
    the caller opened. It should not open its own transaction and
    should not touch ``workspace_meta.schema_version`` — the framework
    bumps that marker once all steps succeed.
    """

    from_version: int
    to_version: int
    description: str
    fn: MigrationFn


# ─── memory-DB migrations ────────────────────────────────────────


def _rebuild_memory_fts_single_column(conn: apsw.Connection) -> None:
    """Rebuild ``memory_fts`` with only the ``content`` column.

    Used by v7 → v8 to drop the short-lived ``entities`` column.
    FTS5 doesn't support ``ALTER TABLE ... DROP COLUMN``, so the only
    portable option is drop + recreate + repopulate from the
    authoritative ``memory.content``.
    """

    rows = conn.execute("SELECT id, content FROM memory").fetchall()
    conn.execute("DROP TABLE IF EXISTS memory_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE memory_fts USING fts5("
        "memory_id UNINDEXED, content, tokenize='trigram')"
    )
    for mid, content in rows:
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
            (mid, content),
        )


def _rebuild_memory_fts_with_entities(conn: apsw.Connection) -> None:
    """Rebuild ``memory_fts`` with an ``entities`` column.

    Used by v6 → v7 to introduce the short-lived entities column (an
    experiment that was reverted in v7 → v8). Entities are sourced
    from ``memory.extracted_entities`` (JSON list, space-joined).
    """

    rows = conn.execute(
        "SELECT id, content, extracted_entities FROM memory"
    ).fetchall()
    conn.execute("DROP TABLE IF EXISTS memory_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE memory_fts USING fts5("
        "memory_id UNINDEXED, content, entities, tokenize='trigram')"
    )
    for mid, content, entities_json in rows:
        entities_text = _entities_text_from_json(
            entities_json if entities_json is None else str(entities_json)
        )
        conn.execute(
            "INSERT INTO memory_fts(memory_id, content, entities) VALUES (?, ?, ?)",
            (mid, content, entities_text),
        )


def _entities_text_from_json(entities_json: str | None) -> str:
    if not entities_json:
        return ""
    try:
        entities = json.loads(entities_json)
    except (ValueError, TypeError):
        return ""
    if not entities:
        return ""
    return " ".join(e for e in entities if isinstance(e, str) and e)


def _add_superseded_columns(conn: apsw.Connection) -> None:
    """Add ``superseded_by`` / ``superseded_at`` columns to ``memory``.

    Used by v8 → v9 to support contradiction-aware observe. SQLite
    supports ``ALTER TABLE ADD COLUMN`` in-place (O(1) schema change,
    rows keep their existing values and new columns default to
    NULL), so the migration is cheap even on a multi-million-row
    workspace. Also creates the partial index that recall uses to
    skip superseded rows cheaply.
    """

    conn.execute("ALTER TABLE memory ADD COLUMN superseded_by TEXT")
    conn.execute("ALTER TABLE memory ADD COLUMN superseded_at REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_superseded "
        "ON memory(superseded_by) WHERE superseded_by IS NOT NULL"
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        from_version=6,
        to_version=7,
        description="Add `entities` column to memory_fts (short-lived NER experiment).",
        fn=_rebuild_memory_fts_with_entities,
    ),
    Migration(
        from_version=7,
        to_version=8,
        description="Drop `entities` column from memory_fts (NER removed).",
        fn=_rebuild_memory_fts_single_column,
    ),
    Migration(
        from_version=8,
        to_version=9,
        description="Add superseded_by/superseded_at to memory (contradiction-aware observe).",
        fn=_add_superseded_columns,
    ),
)


def apply_migrations(
    conn: apsw.Connection,
    current_version: int,
    target_version: int,
) -> list[Migration]:
    """Apply every registered migration from ``current_version`` up to
    ``target_version`` inside one transaction.

    Returns the list of migrations applied, oldest first. No-op and
    returns ``[]`` if already at target. Raises :class:`MigrationError`
    on a missing chain link or on a DB newer than the code.
    """

    if current_version == target_version:
        return []
    if current_version > target_version:
        raise MigrationError(
            f"Workspace DB is at schema v{current_version}, but this "
            f"version of Mnemoss is older (v{target_version}). Don't "
            "downgrade the data file — vector shape, FTS tokenizer, "
            "and column semantics may have diverged silently. Upgrade "
            "your Mnemoss install to match (or newer than) the DB's "
            "schema version, or open with a separate workspace."
        )

    applied: list[Migration] = []
    v = current_version
    with conn:
        while v < target_version:
            step = next(
                (m for m in MIGRATIONS if m.from_version == v), None
            )
            if step is None:
                raise MigrationError(
                    f"Schema v{v}→v{target_version} migration chain is "
                    f"broken: no registered step with from_version={v}. "
                    "This usually means the DB was written by an older "
                    "build that has since been removed from the "
                    "migration registry, or someone hand-edited the "
                    "stored schema_version. Fix: check out a Mnemoss "
                    f"release that supports opening v{v} DBs directly "
                    "and let it migrate forward."
                )
            step.fn(conn)
            v = step.to_version
            applied.append(step)

        conn.execute(
            "UPDATE workspace_meta SET v = ? WHERE k = 'schema_version'",
            (str(target_version),),
        )
    return applied


# ─── raw-log migrations ──────────────────────────────────────────

RAW_LOG_MIGRATIONS: tuple[Migration, ...] = ()


def apply_raw_log_migrations(
    conn: apsw.Connection,
    current_version: int,
    target_version: int,
) -> list[Migration]:
    """Same contract as :func:`apply_migrations`, but for the raw-log DB.

    No real migrations exist yet — raw_log.sqlite has been
    structurally stable since v6. A version gap triggers a version
    bump only, since the schema hasn't actually changed.
    """

    if current_version == target_version:
        return []
    if current_version > target_version:
        raise MigrationError(
            f"Raw-log DB is at schema v{current_version}, but this "
            f"Mnemoss install is older (v{target_version}). Upgrade "
            "Mnemoss — don't downgrade a live raw log."
        )

    applied: list[Migration] = []
    v = current_version
    with conn:
        while v < target_version:
            step = next(
                (m for m in RAW_LOG_MIGRATIONS if m.from_version == v), None
            )
            if step is None:
                # No registered migration and no real schema delta —
                # just walk the version marker forward one step.
                v += 1
                continue
            step.fn(conn)
            v = step.to_version
            applied.append(step)

        conn.execute(
            "UPDATE raw_log_meta SET v = ? WHERE k = 'schema_version'",
            (str(target_version),),
        )
    return applied
