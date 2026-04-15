"""
Conversation Store - SQLite-based chat history (per-user).

Each user gets their own conversations.db at
~/.personal-ai/users/<username>/conversations.db. Call
`get_conversation_store(username)` to get a cached instance.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from user_paths import conversations_db


def _init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            summary TEXT,
            ingested_to_rag BOOLEAN DEFAULT FALSE,
            excluded_from_training BOOLEAN DEFAULT FALSE
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tokens_used INTEGER,
            context_used TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
    ''')
    cols = [r[1] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    if "excluded_from_training" not in cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN excluded_from_training BOOLEAN DEFAULT FALSE")
    conn.commit()
    conn.close()


class ConversationStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        _init_db(db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_conversation(self, title: str = None) -> str:
        conv_id = str(uuid.uuid4())[:8]
        conn = self._conn()
        conn.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        )
        conn.commit()
        conn.close()
        return conv_id

    def add_message(self, conversation_id: str, role: str, content: str,
                    tokens_used: int = None, context_used: List[str] = None) -> int:
        conn = self._conn()
        cur = conn.execute(
            """INSERT INTO messages (conversation_id, role, content, tokens_used, context_used)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, role, content, tokens_used,
             json.dumps(context_used) if context_used else None)
        )
        message_id = cur.lastrowid
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        conn.commit()
        conn.close()
        return message_id

    def get_conversation(self, conversation_id: str) -> Dict:
        conn = self._conn()
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)
        ).fetchone()

        if not conv:
            conn.close()
            return None

        messages = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,)
        ).fetchall()

        conn.close()

        return {
            "id": conv["id"],
            "title": conv["title"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "summary": conv["summary"],
            "ingested_to_rag": bool(conv["ingested_to_rag"]),
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "timestamp": m["timestamp"],
                    "tokens_used": m["tokens_used"]
                }
                for m in messages
            ]
        }

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        conn = self._conn()
        convs = conn.execute(
            """SELECT c.*, COUNT(m.id) as message_count
               FROM conversations c
               LEFT JOIN messages m ON c.id = m.conversation_id
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT ? OFFSET ?""",
            (limit, offset)
        ).fetchall()
        conn.close()

        return [
            {
                "id": c["id"],
                "title": c["title"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
                "message_count": c["message_count"],
                "ingested_to_rag": bool(c["ingested_to_rag"]),
                "excluded_from_training": bool(c["excluded_from_training"]) if "excluded_from_training" in c.keys() else False,
            }
            for c in convs
        ]

    def set_excluded(self, conversation_id: str, excluded: bool):
        conn = self._conn()
        conn.execute(
            "UPDATE conversations SET excluded_from_training = ? WHERE id = ?",
            (1 if excluded else 0, conversation_id),
        )
        conn.commit()
        conn.close()

    def update_title(self, conversation_id: str, title: str):
        conn = self._conn()
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id)
        )
        conn.commit()
        conn.close()

    def update_summary(self, conversation_id: str, summary: str):
        conn = self._conn()
        conn.execute(
            "UPDATE conversations SET summary = ? WHERE id = ?",
            (summary, conversation_id)
        )
        conn.commit()
        conn.close()

    def mark_ingested(self, conversation_id: str):
        conn = self._conn()
        conn.execute(
            "UPDATE conversations SET ingested_to_rag = TRUE WHERE id = ?",
            (conversation_id,)
        )
        conn.commit()
        conn.close()

    def delete_conversation(self, conversation_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()

    def get_uningest_conversations(self) -> List[Dict]:
        conn = self._conn()
        convs = conn.execute(
            """SELECT * FROM conversations
               WHERE ingested_to_rag = FALSE
               ORDER BY updated_at"""
        ).fetchall()
        conn.close()

        return [self.get_conversation(c["id"]) for c in convs]

    def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        conn = self._conn()
        results = conn.execute(
            """SELECT DISTINCT c.*, COUNT(m.id) as message_count
               FROM conversations c
               JOIN messages m ON c.id = m.conversation_id
               WHERE m.content LIKE ?
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT ?""",
            (f"%{query}%", limit)
        ).fetchall()
        conn.close()

        return [
            {
                "id": c["id"],
                "title": c["title"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
                "message_count": c["message_count"],
                "ingested_to_rag": bool(c["ingested_to_rag"]),
            }
            for c in results
        ]


# Per-user cached instances
_stores: dict[str, ConversationStore] = {}


def get_conversation_store(username: str = None) -> ConversationStore:
    if username is None:
        from auth_ctx import get_current_username
        username = get_current_username()
    if username not in _stores:
        _stores[username] = ConversationStore(conversations_db(username))
    return _stores[username]
