"""Sync SQL operations against the Raw Log DB.

The Raw Log is append-only — no updates, no deletes — and lives in its
own SQLite file (``raw_log.sqlite``). It's a minimal surface: write a
message, that's all we need today.
"""

from __future__ import annotations

import json

import apsw

from mnemoss.core.types import RawMessage
from mnemoss.store._sql_helpers import json_safe


def write_raw_message(conn: apsw.Connection, msg: RawMessage) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO raw_message (
                id, workspace_id, agent_id, session_id, turn_id, parent_id,
                timestamp, role, content, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.id,
                msg.workspace_id,
                msg.agent_id,
                msg.session_id,
                msg.turn_id,
                msg.parent_id,
                msg.timestamp.timestamp(),
                msg.role,
                msg.content,
                json.dumps(json_safe(msg.metadata)),
            ),
        )
