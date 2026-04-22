"""Schema-migration framework tests.

Covers the registered memory-DB chain (v6 → v7 → v8), the raw-log
no-op chain, and the error cases (forward-skip, future DB, version
already current).
"""

from __future__ import annotations

import json

import apsw
import pytest

from mnemoss.store.migrations import (
    MIGRATIONS,
    Migration,
    MigrationError,
    apply_migrations,
    apply_raw_log_migrations,
)

# ─── helpers ───────────────────────────────────────────────────────


def _mem_db_at_v6() -> apsw.Connection:
    """Build an in-memory workspace DB shaped like schema v6.

    v6 had ``memory_fts`` with a single ``content`` column (same as v8
    today). Good enough to exercise the v6 → v7 → v8 chain.
    """

    conn = apsw.Connection(":memory:")
    # Minimal memory + workspace_meta tables — just enough columns to
    # satisfy the migrations we want to exercise.
    conn.execute(
        """
        CREATE TABLE workspace_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)
        """
    )
    conn.execute(
        """
        CREATE TABLE memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            extracted_entities TEXT
        )
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE memory_fts USING fts5("
        "memory_id UNINDEXED, content, tokenize='trigram')"
    )
    conn.execute(
        "INSERT INTO workspace_meta(k, v) VALUES (?, ?)",
        ("schema_version", "6"),
    )
    return conn


def _seed_memory(
    conn: apsw.Connection,
    mid: str,
    content: str,
    entities: list[str] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO memory(id, content, extracted_entities) VALUES (?, ?, ?)",
        (mid, content, json.dumps(entities) if entities is not None else None),
    )
    conn.execute(
        "INSERT INTO memory_fts(memory_id, content) VALUES (?, ?)",
        (mid, content),
    )


def _fts_columns(conn: apsw.Connection) -> list[str]:
    # apsw doesn't populate cursor.description on completed statements,
    # so use the FTS5 schema view instead.
    sql_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_fts'"
    ).fetchone()
    sql = sql_row[0] if sql_row else ""
    # e.g. "CREATE VIRTUAL TABLE memory_fts USING fts5(memory_id UNINDEXED,
    # content, tokenize='trigram')"
    inside = sql[sql.find("(") + 1 : sql.rfind(")")]
    cols: list[str] = []
    for part in inside.split(","):
        token = part.strip().split()[0] if part.strip() else ""
        # filter out option-only arguments like tokenize='trigram'
        if token and "=" not in token:
            cols.append(token)
    return cols


def _stored_version(conn: apsw.Connection) -> int:
    row = conn.execute(
        "SELECT v FROM workspace_meta WHERE k = 'schema_version'"
    ).fetchone()
    return int(row[0]) if row else -1


# ─── happy-path chain ──────────────────────────────────────────────


def test_v6_to_v8_applies_both_steps() -> None:
    conn = _mem_db_at_v6()
    _seed_memory(conn, "m1", "Alice ordered coffee", entities=["Alice"])
    _seed_memory(conn, "m2", "just a note", entities=None)

    applied = apply_migrations(conn, current_version=6, target_version=8)
    assert [m.from_version for m in applied] == [6, 7]
    assert [m.to_version for m in applied] == [7, 8]
    assert _stored_version(conn) == 8

    # Final FTS shape: single content column (entities dropped at v8).
    cols = _fts_columns(conn)
    assert "content" in cols
    assert "entities" not in cols

    # Data survived the rebuild.
    hits = conn.execute(
        "SELECT memory_id FROM memory_fts WHERE memory_fts MATCH 'Alice'"
    ).fetchall()
    assert ("m1",) in hits


def test_v7_to_v8_drops_entities_column_only() -> None:
    conn = _mem_db_at_v6()
    _seed_memory(conn, "m1", "Alice", entities=["Alice"])
    apply_migrations(conn, current_version=6, target_version=7)

    cols_v7 = _fts_columns(conn)
    assert "entities" in cols_v7

    applied = apply_migrations(conn, current_version=7, target_version=8)
    assert len(applied) == 1
    assert applied[0].from_version == 7
    assert applied[0].to_version == 8
    assert "entities" not in _fts_columns(conn)
    assert _stored_version(conn) == 8


def test_no_op_when_already_at_target() -> None:
    conn = _mem_db_at_v6()
    conn.execute("UPDATE workspace_meta SET v = '8' WHERE k = 'schema_version'")
    applied = apply_migrations(conn, current_version=8, target_version=8)
    assert applied == []


# ─── error cases ───────────────────────────────────────────────────


def test_raises_when_db_newer_than_code() -> None:
    conn = _mem_db_at_v6()
    with pytest.raises(MigrationError, match="older"):
        apply_migrations(conn, current_version=9, target_version=8)


def test_raises_when_chain_has_gap() -> None:
    """If a caller asks for a version gap no registered migration covers,
    we should fail loudly rather than silently skip."""

    # Pretend there's a v10 we can't reach — the chain stops at v8.
    conn = _mem_db_at_v6()
    with pytest.raises(MigrationError, match=r"chain is broken"):
        apply_migrations(conn, current_version=8, target_version=10)


def test_registered_chain_is_contiguous() -> None:
    """Every registered migration should advance by exactly one version
    and cover the full chain without gaps."""

    for m in MIGRATIONS:
        assert m.to_version == m.from_version + 1

    froms = [m.from_version for m in MIGRATIONS]
    tos = [m.to_version for m in MIGRATIONS]
    # sorted, contiguous from min to max
    assert froms == sorted(froms)
    assert tos == [f + 1 for f in froms]
    for earlier, later in zip(MIGRATIONS, MIGRATIONS[1:], strict=False):
        assert earlier.to_version == later.from_version


# ─── raw-log chain (currently no-op) ───────────────────────────────


def test_raw_log_no_op_version_bump() -> None:
    """v6 → v8 raw_log is a pure version-marker bump today; no
    registered migrations."""

    conn = apsw.Connection(":memory:")
    conn.execute("CREATE TABLE raw_log_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    conn.execute("INSERT INTO raw_log_meta(k, v) VALUES ('schema_version', '6')")

    applied = apply_raw_log_migrations(conn, current_version=6, target_version=8)
    assert applied == []  # no real migrations registered
    row = conn.execute("SELECT v FROM raw_log_meta WHERE k = 'schema_version'").fetchone()
    assert row[0] == "8"  # marker still advanced


def test_raw_log_raises_when_db_newer() -> None:
    conn = apsw.Connection(":memory:")
    conn.execute("CREATE TABLE raw_log_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    with pytest.raises(MigrationError, match="older"):
        apply_raw_log_migrations(conn, current_version=99, target_version=8)


# ─── transaction atomicity ─────────────────────────────────────────


def test_failing_migration_rolls_back_version_bump() -> None:
    """If a migration step raises mid-chain, the transaction should
    abort and the stored version marker should stay put."""

    conn = _mem_db_at_v6()

    # Inject a bad migration into the chain by shadowing the module list.
    from mnemoss.store import migrations as mig

    original = mig.MIGRATIONS
    try:
        def boom(_conn: apsw.Connection) -> None:
            raise RuntimeError("synthetic failure")

        mig.MIGRATIONS = (
            Migration(6, 7, "inject-fail", boom),
            *mig.MIGRATIONS[1:],
        )
        with pytest.raises(RuntimeError, match="synthetic failure"):
            apply_migrations(conn, current_version=6, target_version=8)
    finally:
        mig.MIGRATIONS = original

    # Version marker should still be v6 — transaction rolled back.
    assert _stored_version(conn) == 6
