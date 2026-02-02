"""
RAG Service for Personal AI Framework
Handles document ingestion, embedding, and retrieval via ChromaDB
"""
import os
import hashlib
from typing import List, Optional
import chromadb
from pydantic import BaseModel


class Document(BaseModel):
    content: str
    metadata: Optional[dict] = {}
    doc_id: Optional[str] = None


class RAGService:
    def __init__(
        self,
        chroma_host: str = "vectordb",
        chroma_port: int = 8000,
        collection_name: str = "personal_knowledge"
    ):
        """Initialize RAG service with ChromaDB."""
        
        # Connect to ChromaDB
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ RAG service connected to ChromaDB at {chroma_host}:{chroma_port}")
        
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a document based on content hash."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
                
        return chunks if chunks else [text]
    
    def add_document(self, content: str, metadata: dict = None, chunk: bool = True) -> int:
        """Add a document to the knowledge base."""
        metadata = metadata or {}
        
        if chunk:
            chunks = self._chunk_text(content)
        else:
            chunks = [content]
        
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk_text in enumerate(chunks):
            doc_id = self._generate_id(chunk_text)
            ids.append(doc_id)
            documents.append(chunk_text)
            
            chunk_metadata = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            metadatas.append(chunk_metadata)
        
        # Upsert to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add multiple documents to the knowledge base."""
        total_chunks = 0
        for doc in documents:
            chunks_added = self.add_document(
                content=doc.content,
                metadata=doc.metadata
            )
            total_chunks += chunks_added
        return total_chunks
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """Search for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        
        return formatted
    
    def get_context(self, query: str, k: int = 3) -> List[str]:
        """Get context strings for RAG prompt building."""
        results = self.search(query, k=k)
        return [r["content"] for r in results]
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count()
        }
    
    def clear(self) -> bool:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        return True

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file."""
        results = self.collection.get(
            where={"source_file": source_file},
            include=["metadatas"]
        )
        
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def get_sources(self) -> List[str]:
        """Get list of all source files in the collection."""
        results = self.collection.get(include=["metadatas"])
        sources = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "source_file" in meta:
                    sources.add(meta["source_file"])
        return list(sources)


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        chroma_host = os.getenv("CHROMA_HOST", "vectordb")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        _rag_service = RAGService(chroma_host=chroma_host, chroma_port=chroma_port)
    return _rag_service
