"""SQL DDL for Mnemoss workspaces.

Schema v8 drops the short-lived ``entities`` FTS column — NER was
removed from the system entirely; see MNEMOSS_PROJECT_KNOWLEDGE.md
§9.7 for the rationale. Schema v6 split the Raw Log into its own
SQLite file. Two DBs per workspace:

- ``memory.sqlite`` — ``MEMORY_DDL_STATEMENTS`` below (memory, relations,
  pins, tombstones, workspace_meta) plus the ``vec0`` and FTS5
  virtual tables built separately.
- ``raw_log.sqlite`` — ``RAW_LOG_DDL_STATEMENTS`` below. A single
  ``raw_message`` table plus a minimal ``raw_log_meta`` for the schema
  version pin.

The split lets Raw Log (pure append, unbounded, rarely read) carry its
own retention and backup policy without dragging the Memory Store
(bounded, read-heavy) along. Queries never JOIN across the two files,
so keeping them as independent connections is cleaner than ATTACH.
"""

from __future__ import annotations

MIN_SQLITE_VERSION = (3, 34, 0)  # First bundled-trigram release

MEMORY_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS workspace_meta (
      k TEXT PRIMARY KEY,
      v TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory (
      id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL,
      agent_id TEXT,
      session_id TEXT,
      created_at REAL NOT NULL,
      content TEXT NOT NULL,
      role TEXT,
      memory_type TEXT NOT NULL,
      abstraction_level REAL NOT NULL,
      access_history TEXT NOT NULL,
      last_accessed_at REAL,
      rehearsal_count INTEGER NOT NULL DEFAULT 0,
      salience REAL NOT NULL DEFAULT 0.0,
      emotional_weight REAL NOT NULL DEFAULT 0.0,
      reminisced_count INTEGER NOT NULL DEFAULT 0,
      index_tier TEXT NOT NULL DEFAULT 'hot',
      idx_priority REAL NOT NULL DEFAULT 0.5,
      extracted_gist TEXT,
      extracted_entities TEXT,
      extracted_time REAL,
      extracted_location TEXT,
      extracted_participants TEXT,
      extraction_level INTEGER NOT NULL DEFAULT 0,
      cluster_id TEXT,
      cluster_similarity REAL,
      is_cluster_representative INTEGER NOT NULL DEFAULT 0,
      derived_from TEXT NOT NULL DEFAULT '[]',
      derived_to TEXT NOT NULL DEFAULT '[]',
      source_message_ids TEXT NOT NULL DEFAULT '[]',
      source_context TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(index_tier)",
    "CREATE INDEX IF NOT EXISTS idx_memory_cluster ON memory(cluster_id)",
    """
    CREATE TABLE IF NOT EXISTS relation (
      src_id TEXT NOT NULL,
      dst_id TEXT NOT NULL,
      predicate TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 1.0,
      created_at REAL NOT NULL,
      PRIMARY KEY (src_id, dst_id, predicate)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_relation_src ON relation(src_id)",
    """
    CREATE TABLE IF NOT EXISTS pin (
      memory_id TEXT NOT NULL,
      agent_id TEXT,
      pinned_at REAL NOT NULL,
      PRIMARY KEY (memory_id, agent_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tombstone (
      original_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL,
      agent_id TEXT,
      dropped_at REAL NOT NULL,
      reason TEXT NOT NULL,
      gist_snapshot TEXT NOT NULL,
      b_at_drop REAL NOT NULL,
      source_message_ids TEXT NOT NULL DEFAULT '[]'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_tombstone_agent ON tombstone(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_tombstone_dropped_at ON tombstone(dropped_at)",
]

RAW_LOG_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS raw_log_meta (
      k TEXT PRIMARY KEY,
      v TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS raw_message (
      id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL,
      agent_id TEXT,
      session_id TEXT NOT NULL,
      turn_id TEXT NOT NULL,
      parent_id TEXT,
      timestamp REAL NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_raw_session ON raw_message(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_raw_agent ON raw_message(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_message(timestamp)",
]


def vec_ddl(dim: int) -> str:
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec "
        f"USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{dim}])"
    )


FTS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
    "USING fts5(memory_id UNINDEXED, content, tokenize='trigram')"
)
