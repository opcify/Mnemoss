"""SQL DDL for Mnemoss workspaces.

Stage 1 schema: single file per workspace, all tables colocated. The Raw Log
lives in the same DB file as the Memory Store — the Principle 3 separation
is a *layer* distinction, not a *file* distinction. Splitting files is
deferred to Stage 2+ if operationally useful.
"""

from __future__ import annotations

MIN_SQLITE_VERSION = (3, 34, 0)  # First bundled-trigram release

DDL_STATEMENTS = [
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
      source_message_ids TEXT NOT NULL DEFAULT '[]',
      source_context TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memory_agent ON memory(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_session ON memory(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_tier ON memory(index_tier)",
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
