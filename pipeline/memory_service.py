"""
Memory Service - Ingest conversations into ChromaDB for RAG-based memory
Allows the AI to "remember" past discussions
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from conversation_store import get_conversation_store
from rag_service import get_rag_service

# Memory collection name (separate from knowledge base)
MEMORY_COLLECTION = "conversation_memory"


class MemoryService:
    """Manage conversation memory in ChromaDB"""
    
    def __init__(self):
        self.store = get_conversation_store()
        self.rag = get_rag_service()
    
    def ingest_conversation(self, conversation_id: str) -> int:
        """Ingest a conversation into RAG memory"""
        conv = self.store.get_conversation(conversation_id)
        if not conv:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        if conv["ingested_to_rag"]:
            print(f"Conversation {conversation_id} already ingested")
            return 0
        
        # Build document from conversation
        chunks = self._conversation_to_chunks(conv)
        
        # Add to RAG with memory metadata
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
        
        # Mark as ingested
        self.store.mark_ingested(conversation_id)
        
        return chunks_added
    
    def _conversation_to_chunks(self, conv: Dict) -> List[Dict]:
        """Convert conversation to searchable chunks"""
        chunks = []
        messages = conv["messages"]
        
        # Strategy: Create chunks of Q&A pairs for better context
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i] if i < len(messages) else None
            asst_msg = messages[i + 1] if i + 1 < len(messages) else None
            
            if user_msg and user_msg["role"] == "user":
                content = f"User asked: {user_msg['content']}\n\n"
                timestamp = user_msg["timestamp"]
                
                if asst_msg and asst_msg["role"] == "assistant":
                    content += f"Assistant answered: {asst_msg['content']}"
                
                chunks.append({
                    "content": content,
                    "timestamp": timestamp
                })
        
        # Also create a summary chunk if conversation has summary
        if conv.get("summary"):
            chunks.append({
                "content": f"Conversation summary ({conv['title']}): {conv['summary']}",
                "timestamp": conv["updated_at"]
            })
        
        return chunks
    
    def ingest_all_pending(self) -> Dict:
        """Ingest all conversations not yet in RAG"""
        pending = self.store.get_uningest_conversations()
        
        results = {
            "total": len(pending),
            "ingested": 0,
            "chunks_added": 0,
            "errors": []
        }
        
        for conv in pending:
            try:
                chunks = self.ingest_conversation(conv["id"])
                results["ingested"] += 1
                results["chunks_added"] += chunks
            except Exception as e:
                results["errors"].append({
                    "conversation_id": conv["id"],
                    "error": str(e)
                })
        
        return results
    
    def search_memory(self, query: str, k: int = 3) -> List[Dict]:
        """Search conversation memory for relevant past discussions"""
        # Search with filter for conversation memory
        results = self.rag.search(query, k=k * 2)  # Get extra to filter
        
        # Filter to just memory results
        memory_results = [
            r for r in results 
            if r.get("metadata", {}).get("source_type") == "conversation_memory"
        ][:k]
        
        return memory_results
    
    def get_memory_context(self, query: str, k: int = 2) -> Tuple[List[str], List[Dict]]:
        """Get memory context for a query (like RAG but for past conversations)"""
        results = self.search_memory(query, k=k)
        
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
        """Clear all conversation memory from RAG (keeps SQLite intact)"""
        # This would need a custom method in rag_service to delete by metadata
        # For now, we can mark all as not ingested
        conn = self.store.get_db()
        conn.execute("UPDATE conversations SET ingested_to_rag = FALSE")
        conn.commit()
        conn.close()
        print("Memory cleared - conversations marked for re-ingestion")


# Singleton
_memory_service = None

def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


if __name__ == "__main__":
    from conversation_store import get_conversation_store
    
    # Test with existing conversation
    store = get_conversation_store()
    memory = get_memory_service()
    
    # Create a test conversation
    conv_id = store.create_conversation("Test Memory Chat")
    store.add_message(conv_id, "user", "What are the power domains in the i.MX 93?")
    store.add_message(conv_id, "assistant", 
        "The i.MX 93 has multiple power domains including the A55 core domain, "
        "M33 core domain, peripheral domains, and always-on domain. This allows "
        "fine-grained power management where unused domains can be powered down.")
    
    # Ingest it
    print(f"Ingesting conversation {conv_id}...")
    chunks = memory.ingest_conversation(conv_id)
    print(f"Added {chunks} chunks to memory")
    
    # Search memory
    print("\nSearching memory for 'power management'...")
    results = memory.search_memory("power management")
    for r in results:
        print(f"  - {r.get('content', '')[:100]}...")
    
    # Get context
    print("\nGetting memory context...")
    contexts, citations = memory.get_memory_context("i.MX power features")
    print(f"Found {len(contexts)} memory contexts")
    for c in citations:
        print(f"  - {c['source_file']}")
