"""Store layer — SQLite-backed persistence for Memory, Raw Log, Relations.

One DB file per workspace. Uses apsw (not stdlib ``sqlite3``) because stock
Python's ``sqlite3`` has loadable-extension support compiled out on several
common platforms (macOS pyenv builds in particular), and sqlite-vec requires
extension loading.
"""

from mnemoss.store.paths import workspace_db_path
from mnemoss.store.sqlite_backend import SchemaMismatchError, SQLiteBackend

__all__ = ["SchemaMismatchError", "SQLiteBackend", "workspace_db_path"]
