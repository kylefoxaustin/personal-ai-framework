"""
Conversation Store - SQLite-based chat history
Stores full conversations for later retrieval and RAG ingestion
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import uuid

# Database location
DB_PATH = Path.home() / ".personal-ai" / "conversations.db"

def get_db():
    """Get database connection"""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database schema"""
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            summary TEXT,
            ingested_to_rag BOOLEAN DEFAULT FALSE
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
    conn.commit()
    conn.close()

# Initialize on import
init_db()


class ConversationStore:
    """Manage conversation storage and retrieval"""
    
    def create_conversation(self, title: str = None) -> str:
        """Create a new conversation, return ID"""
        conv_id = str(uuid.uuid4())[:8]
        conn = get_db()
        conn.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        )
        conn.commit()
        conn.close()
        return conv_id
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                    tokens_used: int = None, context_used: List[str] = None):
        """Add a message to a conversation"""
        conn = get_db()
        conn.execute(
            """INSERT INTO messages (conversation_id, role, content, tokens_used, context_used)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, role, content, tokens_used, 
             json.dumps(context_used) if context_used else None)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,)
        )
        conn.commit()
        conn.close()
    
    def get_conversation(self, conversation_id: str) -> Dict:
        """Get full conversation with messages"""
        conn = get_db()
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
        """List recent conversations"""
        conn = get_db()
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
                "ingested_to_rag": bool(c["ingested_to_rag"])
            }
            for c in convs
        ]
    
    def update_title(self, conversation_id: str, title: str):
        """Update conversation title"""
        conn = get_db()
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id)
        )
        conn.commit()
        conn.close()
    
    def update_summary(self, conversation_id: str, summary: str):
        """Update conversation summary"""
        conn = get_db()
        conn.execute(
            "UPDATE conversations SET summary = ? WHERE id = ?",
            (summary, conversation_id)
        )
        conn.commit()
        conn.close()
    
    def mark_ingested(self, conversation_id: str):
        """Mark conversation as ingested to RAG"""
        conn = get_db()
        conn.execute(
            "UPDATE conversations SET ingested_to_rag = TRUE WHERE id = ?",
            (conversation_id,)
        )
        conn.commit()
        conn.close()
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and its messages"""
        conn = get_db()
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
    
    def get_uningest_conversations(self) -> List[Dict]:
        """Get conversations not yet ingested to RAG"""
        conn = get_db()
        convs = conn.execute(
            """SELECT * FROM conversations 
               WHERE ingested_to_rag = FALSE 
               ORDER BY updated_at"""
        ).fetchall()
        conn.close()
        
        return [self.get_conversation(c["id"]) for c in convs]
    
    def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """Basic text search in conversations"""
        conn = get_db()
        results = conn.execute(
            """SELECT DISTINCT c.* FROM conversations c
               JOIN messages m ON c.id = m.conversation_id
               WHERE m.content LIKE ?
               ORDER BY c.updated_at DESC
               LIMIT ?""",
            (f"%{query}%", limit)
        ).fetchall()
        conn.close()
        
        return [
            {
                "id": c["id"],
                "title": c["title"],
                "updated_at": c["updated_at"]
            }
            for c in results
        ]


# Singleton instance
_store = None

def get_conversation_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store


if __name__ == "__main__":
    # Test
    store = get_conversation_store()
    
    # Create a test conversation
    conv_id = store.create_conversation("Test Chat")
    print(f"Created conversation: {conv_id}")
    
    store.add_message(conv_id, "user", "Hello, what is the i.MX 93?")
    store.add_message(conv_id, "assistant", "The i.MX 93 is a processor from NXP...", tokens_used=50)
    
    # List
    print("\nConversations:")
    for c in store.list_conversations():
        print(f"  {c['id']}: {c['title']} ({c['message_count']} messages)")
    
    # Get full
    full = store.get_conversation(conv_id)
    print(f"\nFull conversation:")
    for m in full['messages']:
        print(f"  [{m['role']}]: {m['content'][:50]}...")
