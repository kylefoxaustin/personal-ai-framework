#!/usr/bin/env python3
"""
Reminder Service — SQLite-backed scheduled reminders (per-user).

No background process needed; `get_due_unacked()` is polled by the UI every
30s. Firing = row becomes due and unacked; UI ack marks it delivered.
"""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

from user_paths import reminders_db


def _parse_iso(s: str) -> datetime:
    """Parse ISO 8601. Accept 'Z', '+HHMM' (no colon), and naive as UTC."""
    import re as _re
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = _re.sub(r"([+-])(\d{2})(\d{2})$", r"\1\2:\3", s)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ReminderStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    acked_at TEXT,
                    cancelled_at TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_due ON reminders(due_at)")
            conn.commit()

    def schedule(self, text: str, due_at_iso: str) -> Dict:
        due = _parse_iso(due_at_iso)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO reminders (text, due_at, created_at) VALUES (?, ?, ?)",
                (text, due.isoformat(), now_iso),
            )
            conn.commit()
            return {"id": cur.lastrowid, "text": text, "due_at": due.isoformat()}

    def list_upcoming(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE cancelled_at IS NULL AND acked_at IS NULL
                   ORDER BY due_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_due_unacked(self) -> List[Dict]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM reminders
                   WHERE due_at <= ? AND acked_at IS NULL AND cancelled_at IS NULL
                   ORDER BY due_at ASC""",
                (now_iso,),
            ).fetchall()
            return [dict(r) for r in rows]

    def ack(self, reminder_id: int) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE reminders SET acked_at = ? WHERE id = ? AND acked_at IS NULL",
                (now_iso, reminder_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def cancel(self, reminder_id: int) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE reminders SET cancelled_at = ? WHERE id = ? AND cancelled_at IS NULL AND acked_at IS NULL",
                (now_iso, reminder_id),
            )
            conn.commit()
            return cur.rowcount > 0


_stores: dict[str, ReminderStore] = {}


def get_reminder_store(username: str = None) -> ReminderStore:
    if username is None:
        from auth_ctx import get_current_username
        username = get_current_username()
    if username not in _stores:
        _stores[username] = ReminderStore(reminders_db(username))
    return _stores[username]
