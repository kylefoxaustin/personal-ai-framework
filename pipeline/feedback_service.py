#!/usr/bin/env python3
"""
Feedback Service — 👍/👎 preference signals on assistant messages.

Stored in the same SQLite DB as conversations (fewer files, same backup path).
One row per message_id; re-rating overwrites. Downstream use:
  - Training selection: 👎 = exclude from LoRA data, 👍 = prefer.
  - Eval dashboard: 👍 rate over time as the simplest quality metric.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

DB_PATH = Path.home() / ".personal-ai" / "conversations.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                message_id INTEGER PRIMARY KEY,
                rating TEXT NOT NULL CHECK (rating IN ('up','down')),
                comment TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)")
        conn.commit()


_init()


def rate(message_id: int, rating: str, comment: Optional[str] = None) -> Dict:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO feedback (message_id, rating, comment, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(message_id) DO UPDATE SET
                 rating=excluded.rating,
                 comment=COALESCE(excluded.comment, feedback.comment),
                 updated_at=excluded.updated_at""",
            (message_id, rating, comment, now, now),
        )
        conn.commit()
    return {"message_id": message_id, "rating": rating}


def clear(message_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM feedback WHERE message_id = ?", (message_id,))
        conn.commit()
        return cur.rowcount > 0


def get(message_id: int) -> Optional[Dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM feedback WHERE message_id = ?", (message_id,)
        ).fetchone()
        return dict(row) if row else None


def get_many(message_ids: List[int]) -> Dict[int, Dict]:
    if not message_ids:
        return {}
    placeholders = ",".join("?" * len(message_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM feedback WHERE message_id IN ({placeholders})",
            message_ids,
        ).fetchall()
        return {row["message_id"]: dict(row) for row in rows}


def stats() -> Dict:
    with _connect() as conn:
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
        total = up + down
    return {
        "up": up,
        "down": down,
        "total": total,
        "up_rate": (up / total) if total else None,
    }


def history(days: int = 30) -> List[Dict]:
    """Daily buckets: [{date, up, down, total}, ...] for the last `days` days."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   SUM(CASE WHEN rating='up' THEN 1 ELSE 0 END) AS up,
                   SUM(CASE WHEN rating='down' THEN 1 ELSE 0 END) AS down
            FROM feedback
            WHERE created_at >= date('now', ?)
            GROUP BY day
            ORDER BY day ASC
            """,
            (f"-{days} days",),
        ).fetchall()
    return [
        {"date": r["day"], "up": r["up"], "down": r["down"], "total": r["up"] + r["down"]}
        for r in rows
    ]
