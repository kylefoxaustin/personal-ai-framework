"""
Memory Service — per-user conversation memory (ChromaDB collection per user).

Each user gets their own ChromaDB collection named `conversation_memory_<username>`.
Call `get_memory_service(username)` to get the cached instance.
"""
import os
from typing import List, Dict, Optional, Tuple

from conversation_store import get_conversation_store
from rag_service import RAGService
from user_paths import memory_collection_name


class MemoryService:
    def __init__(self, username: str, rag: RAGService):
        self.username = username
        self.store = get_conversation_store(username)
        self.rag = rag

    def ingest_conversation(self, conversation_id: str) -> int:
        conv = self.store.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        if conv["ingested_to_rag"]:
            return 0

        chunks = self._conversation_to_chunks(conv)
        chunks_added = 0
        for chunk in chunks:
            self.rag.add_document(
                content=chunk["content"],
                metadata={
                    "source_type": "conversation_memory",
                    "conversation_id": conversation_id,
                    "conversation_title": conv["title"],
                    "timestamp": chunk["timestamp"],
                    "participants": "user,assistant"
                }
            )
            chunks_added += 1

        self.store.mark_ingested(conversation_id)
        return chunks_added

    def _conversation_to_chunks(self, conv: Dict) -> List[Dict]:
        chunks = []
        messages = conv["messages"]
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i] if i < len(messages) else None
            asst_msg = messages[i + 1] if i + 1 < len(messages) else None
            if user_msg and user_msg["role"] == "user":
                content = f"User asked: {user_msg['content']}\n\n"
                timestamp = user_msg["timestamp"]
                if asst_msg and asst_msg["role"] == "assistant":
                    content += f"Assistant answered: {asst_msg['content']}"
                chunks.append({"content": content, "timestamp": timestamp})
        if conv.get("summary"):
            chunks.append({
                "content": f"Conversation summary ({conv['title']}): {conv['summary']}",
                "timestamp": conv["updated_at"]
            })
        return chunks

    def ingest_all_pending(self) -> Dict:
        pending = self.store.get_uningest_conversations()
        results = {"total": len(pending), "ingested": 0, "chunks_added": 0, "errors": []}
        for conv in pending:
            try:
                chunks = self.ingest_conversation(conv["id"])
                results["ingested"] += 1
                results["chunks_added"] += chunks
            except Exception as e:
                results["errors"].append({"conversation_id": conv["id"], "error": str(e)})
        return results

    def search_memory(self, query: str, k: int = 3) -> List[Dict]:
        results = self.rag.search(query, k=k * 2)
        memory_results = [
            r for r in results
            if r.get("metadata", {}).get("source_type") == "conversation_memory"
        ][:k]
        return memory_results

    # Past responses containing any of these phrases are Skippy's own
    # prior refusals. Retrieving them as "memory context" causes the model
    # to parrot the refusal on similar queries, even when the underlying
    # retrieval issue has been fixed. Filter them out of memory context.
    _REFUSAL_MARKERS = (
        "excerpts don't cover",
        "excerpts do not cover",
        "retrieved excerpts don't",
        "i don't have information",
        "i do not have information",
    )

    def get_memory_context(self, query: str, k: int = 2) -> Tuple[List[str], List[Dict]]:
        results = self.search_memory(query, k=k)
        # Drop prior refusals so poisoned memory doesn't override corrected
        # retrieval. See _REFUSAL_MARKERS above.
        results = [
            r for r in results
            if not any(m in r["content"].lower() for m in self._REFUSAL_MARKERS)
        ]
        contexts = [r["content"] for r in results]
        citations = [
            {
                "id": i + 1,
                "source_file": f"Memory: {r.get('metadata', {}).get('conversation_title', 'Past chat')}",
                "conversation_id": r.get("metadata", {}).get("conversation_id"),
                "timestamp": r.get("metadata", {}).get("timestamp"),
                "score": r.get("score", 0)
            }
            for i, r in enumerate(results)
        ]
        return contexts, citations

    def clear_memory(self):
        conn = self.store._conn()
        conn.execute("UPDATE conversations SET ingested_to_rag = FALSE")
        conn.commit()
        conn.close()


_services: dict[str, MemoryService] = {}


def get_memory_service(username: str = None) -> MemoryService:
    if username is None:
        from auth_ctx import get_current_username
        username = get_current_username()
    if username not in _services:
        chroma_host = os.getenv("CHROMA_HOST", "vectordb")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        rag = RAGService(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=memory_collection_name(username),
        )
        _services[username] = MemoryService(username, rag)
    return _services[username]
