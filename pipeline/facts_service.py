"""
Facts Service - Lightweight learned facts layer
Stores facts from web searches and user corrections in a separate ChromaDB collection.
Facts are injected into prompts to override stale model training data.
"""
import hashlib
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import chromadb

from user_paths import facts_collection_name


# Per-user cache
_facts_services: dict = {}


def get_facts_service(username: str = None):
    """Get or create the FactsService instance for this user."""
    if username is None:
        from auth_ctx import get_current_username
        username = get_current_username()
    if username not in _facts_services:
        try:
            _facts_services[username] = FactsService(username)
        except Exception as e:
            print(f"⚠️ Facts service not available for {username}: {e}")
            return None
    return _facts_services[username]


class FactsService:
    """Manage learned facts in a per-user ChromaDB collection."""

    def __init__(self, username: str):
        host = os.environ.get("CHROMA_HOST", "vectordb")
        port = int(os.environ.get("CHROMA_PORT", 8000))
        self.username = username
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name=facts_collection_name(username),
            metadata={"hnsw:space": "cosine"}
        )
        count = self.collection.count()
        print(f"✅ Facts service connected for {username} — {count} learned facts")

    def add_fact(self, fact: str, source: str = "unknown", category: str = "general") -> str:
        """Store a learned fact. Returns the fact ID."""
        fact_id = hashlib.md5(fact.lower().strip().encode()).hexdigest()
        now = datetime.now().isoformat()

        self.collection.upsert(
            ids=[fact_id],
            documents=[fact.strip()],
            metadatas=[{
                "source": source,
                "category": category,
                "created_at": now,
                "source_type": "learned_fact",
            }]
        )
        print(f"  💡 Fact learned: {fact[:80]}...")
        return fact_id

    def search_facts(self, query: str, k: int = 3, min_relevance: float = 0.35) -> List[Dict]:
        """Find facts relevant to a query. Returns list of {fact, source, created_at, score}."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        facts = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score (1 = identical, 0 = orthogonal)
            similarity = 1 - (distance / 2)
            if similarity >= min_relevance:
                meta = results["metadatas"][0][i]
                facts.append({
                    "fact": doc,
                    "source": meta.get("source", "unknown"),
                    "category": meta.get("category", "general"),
                    "created_at": meta.get("created_at", ""),
                    "score": round(similarity, 3),
                    "id": results["ids"][0][i],
                })
        return facts

    def get_all_facts(self) -> List[Dict]:
        """Return all stored facts."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(
            include=["documents", "metadatas"]
        )

        facts = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            facts.append({
                "id": results["ids"][i],
                "fact": doc,
                "source": meta.get("source", "unknown"),
                "category": meta.get("category", "general"),
                "created_at": meta.get("created_at", ""),
            })
        # Sort newest first
        facts.sort(key=lambda f: f["created_at"], reverse=True)
        return facts

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID."""
        try:
            self.collection.delete(ids=[fact_id])
            return True
        except Exception:
            return False

    def clear_all(self) -> int:
        """Delete all facts. Returns count deleted."""
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
        return count

    def get_fact_context(self, query: str, k: int = 3) -> Tuple[List[str], List[Dict]]:
        """Get fact context for prompt injection. Returns (fact_strings, citations)."""
        facts = self.search_facts(query, k=k)
        if not facts:
            return [], []

        fact_strings = [f["fact"] for f in facts]
        citations = [{"source": f"Learned fact ({f['source']})", "score": f["score"]} for f in facts]
        return fact_strings, citations
