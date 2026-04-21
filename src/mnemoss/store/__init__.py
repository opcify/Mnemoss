"""Store layer — SQLite-backed persistence for Memory, Raw Log, Relations.

Two DB files per workspace (schema v6+): ``memory.sqlite`` for the Memory
Store (memory, vectors, FTS, relations, pins, tombstones) and
``raw_log.sqlite`` for the append-only Raw Log. The split lets each file
carry its own retention and backup policy — see ``paths.py`` for details.

Uses apsw (not stdlib ``sqlite3``) because stock Python's ``sqlite3`` has
loadable-extension support compiled out on several common platforms
(macOS pyenv builds in particular), and sqlite-vec requires extension
loading.
"""

from mnemoss.store.paths import raw_log_db_path, workspace_db_path
from mnemoss.store.sqlite_backend import SchemaMismatchError, SQLiteBackend

__all__ = [
    "SchemaMismatchError",
    "SQLiteBackend",
    "raw_log_db_path",
    "workspace_db_path",
]
