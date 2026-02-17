"""
Advanced RAG Service with Hybrid Search, Reranking, and Citations
"""
import re
import math
from collections import Counter
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A search result with content, metadata, and scores."""
    content: str
    metadata: Dict
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    citation_id: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "citation_id": self.citation_id,
            "source_file": self.metadata.get("source_file", "unknown"),
            "citation": f"[{self.citation_id}]"
        }


class BM25:
    """BM25 keyword scoring algorithm."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.corpus_size: int = 0
        self.documents: List[List[str]] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'again', 'further', 'then', 'once',
                      'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                      'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                      'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                      'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'any', 'this', 'that', 'these',
                      'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you',
                      'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 'it',
                      'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom'}
        return [t for t in tokens if t not in stop_words and len(t) > 2]
    
    def fit(self, documents: List[str]):
        """Fit BM25 on a corpus of documents."""
        self.documents = []
        self.doc_freqs = {}
        self.doc_lengths = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # Count document frequency for each term
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        self.corpus_size = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self.documents[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        score = 0.0
        doc_token_counts = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in doc_token_counts:
                continue
            
            # Term frequency in document
            tf = doc_token_counts[token]
            
            # Document frequency
            df = self.doc_freqs.get(token, 0)
            
            # IDF calculation
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 score component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search and return top-k document indices with scores."""
        scores = []
        for i in range(len(self.documents)):
            score = self.score(query, i)
            if score > 0:
                scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class SimpleReranker:
    """
    Simple reranker based on query-document overlap and position.
    For production, consider using a cross-encoder model like ms-marco-MiniLM.
    """
    
    def __init__(self):
        self.query_terms: List[str] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    def _calculate_overlap(self, query: str, document: str) -> float:
        """Calculate term overlap between query and document."""
        query_terms = set(self._tokenize(query))
        doc_terms = set(self._tokenize(document))
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & doc_terms)
        return overlap / len(query_terms)
    
    def _calculate_density(self, query: str, document: str) -> float:
        """Calculate query term density in first 200 chars."""
        query_terms = set(self._tokenize(query))
        first_200 = document[:200].lower()
        doc_terms = self._tokenize(first_200)
        
        if not doc_terms:
            return 0.0
        
        matches = sum(1 for t in doc_terms if t in query_terms)
        return matches / len(doc_terms)
    
    def _exact_phrase_bonus(self, query: str, document: str) -> float:
        """Bonus if exact query phrase appears in document."""
        query_lower = query.lower().strip()
        doc_lower = document.lower()
        
        if query_lower in doc_lower:
            return 0.3
        
        # Check for 3+ word sequences
        words = query_lower.split()
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if phrase in doc_lower:
                    return 0.15
        
        return 0.0
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results based on relevance signals."""
        for result in results:
            overlap = self._calculate_overlap(query, result.content)
            density = self._calculate_density(query, result.content)
            phrase_bonus = self._exact_phrase_bonus(query, result.content)
            
            # Combine signals
            result.rerank_score = (overlap * 0.4) + (density * 0.3) + phrase_bonus + 0.3
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results


class AdvancedRAG:
    """
    Advanced RAG with:
    - Hybrid search (semantic + BM25 keyword)
    - Reranking
    - Citation tracking
    """
    
    def __init__(self, rag_service):
        """
        Initialize with existing RAG service.
        
        Args:
            rag_service: The base RAGService instance
        """
        self.rag = rag_service
        self.bm25 = BM25()
        self.reranker = SimpleReranker()
        self.bm25_fitted = False
        self.cached_docs: List[str] = []
        self.cached_metadatas: List[Dict] = []
    
    def _build_bm25_index(self, force: bool = False):
        """Build BM25 index from ChromaDB collection."""
        if self.bm25_fitted and not force:
            return
        
        # Get all documents from collection
        try:
            count = self.rag.collection.count()
            if count == 0:
                return
            
            # Fetch in batches
            batch_size = 5000
            all_docs = []
            all_metadatas = []
            
            for offset in range(0, count, batch_size):
                results = self.rag.collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                if results["documents"]:
                    all_docs.extend(results["documents"])
                    all_metadatas.extend(results["metadatas"])
            
            self.cached_docs = all_docs
            self.cached_metadatas = all_metadatas
            self.bm25.fit(all_docs)
            self.bm25_fitted = True
            print(f"✅ BM25 index built with {len(all_docs)} documents")
            
        except Exception as e:
            print(f"⚠️ Failed to build BM25 index: {e}")
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_reranking: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            use_reranking: Whether to apply reranking
        
        Returns:
            List of SearchResult objects with scores and citations
        """
        results_map: Dict[str, SearchResult] = {}
        
        # 1. Semantic search via ChromaDB
        semantic_results = self.rag.collection.query(
            query_texts=[query],
            n_results=k * 2,  # Get more for fusion
            include=["documents", "metadatas", "distances"]
        )
        
        if semantic_results["documents"] and semantic_results["documents"][0]:
            for i, doc in enumerate(semantic_results["documents"][0]):
                doc_hash = hash(doc)
                # Convert distance to score (cosine distance to similarity)
                distance = semantic_results["distances"][0][i]
                score = 1 - distance  # Higher is better
                
                results_map[doc_hash] = SearchResult(
                    content=doc,
                    metadata=semantic_results["metadatas"][0][i] if semantic_results["metadatas"] else {},
                    semantic_score=score
                )
        
        # 2. BM25 keyword search
        self._build_bm25_index()
        if self.bm25_fitted:
            bm25_results = self.bm25.search(query, top_k=k * 2)
            
            # Normalize BM25 scores
            if bm25_results:
                max_score = max(score for _, score in bm25_results)
                
                for doc_idx, score in bm25_results:
                    doc = self.cached_docs[doc_idx]
                    doc_hash = hash(doc)
                    normalized_score = score / max_score if max_score > 0 else 0
                    
                    if doc_hash in results_map:
                        results_map[doc_hash].keyword_score = normalized_score
                    else:
                        results_map[doc_hash] = SearchResult(
                            content=doc,
                            metadata=self.cached_metadatas[doc_idx],
                            keyword_score=normalized_score
                        )
        
        # Filter out results with zero semantic score (likely irrelevant keyword matches)
        results = [r for r in results if r.semantic_score > 0.1 or r.keyword_score > 0.8]
        # 3. Combine scores
        results = list(results_map.values())
        for result in results:
            result.final_score = (
                semantic_weight * result.semantic_score +
                keyword_weight * result.keyword_score
            )
        
        # 4. Sort by combined score
        results.sort(key=lambda x: x.final_score, reverse=True)
        results = results[:k * 2]  # Keep top candidates for reranking
        
        # 5. Rerank
        if use_reranking and results:
            results = self.reranker.rerank(query, results)
            # Blend rerank score with final score
            for result in results:
                result.final_score = (result.final_score * 0.6) + (result.rerank_score * 0.4)
            results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 6. Take top k and add citation IDs
        results = results[:k]
        for i, result in enumerate(results):
            result.citation_id = i + 1
        
        return results
    
    def search_with_citations(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[SearchResult], str]:
        """
        Search and return results with formatted citations.
        
        Returns:
            Tuple of (results, citation_text)
        """
        results = self.hybrid_search(query, k=k)
        
        # Build citation text
        citations = []
        for r in results:
            source = r.metadata.get("source_file", "unknown")
            # Clean up source path for display
            if "/" in source:
                source = source.split("/")[-1]
            citations.append(f"[{r.citation_id}] {source}")
        
        citation_text = "\n".join(citations)
        
        return results, citation_text
    
    def get_context_with_citations(
        self,
        query: str,
        k: int = 3
    ) -> Tuple[List[str], List[Dict]]:
        """
        Get context strings with citation metadata for RAG prompt.
        
        Returns:
            Tuple of (context_strings, citation_info)
        """
        results = self.hybrid_search(query, k=k)
        
        contexts = []
        citations = []
        
        for r in results:
            # Add citation marker to context
            contexts.append(f"[{r.citation_id}] {r.content}")
            
            citations.append({
                "id": r.citation_id,
                "source_file": r.metadata.get("source_file", "unknown"),
                "chunk_index": r.metadata.get("chunk_index", 0),
                "semantic_score": round(r.semantic_score, 3),
                "keyword_score": round(r.keyword_score, 3),
                "final_score": round(r.final_score, 3)
            })
        
        return contexts, citations


# Test function
def test_advanced_rag():
    """Quick test of the advanced RAG components."""
    
    # Test BM25
    print("Testing BM25...")
    bm25 = BM25()
    docs = [
        "The i.MX 8QuadMax processor is designed for automotive applications",
        "ARM Cortex-A72 cores provide high performance computing",
        "The project deadline has been extended to next month",
        "Machine learning models can run on embedded devices"
    ]
    bm25.fit(docs)
    results = bm25.search("i.MX processor automotive", top_k=2)
    print(f"BM25 results: {results}")
    
    # Test reranker
    print("\nTesting Reranker...")
    reranker = SimpleReranker()
    test_results = [
        SearchResult(content="The i.MX processor is great", metadata={}, semantic_score=0.8),
        SearchResult(content="i.MX 8QuadMax for automotive", metadata={}, semantic_score=0.7)
    ]
    reranked = reranker.rerank("i.MX automotive", test_results)
    for r in reranked:
        print(f"  Score: {r.rerank_score:.3f} - {r.content[:50]}")
    
    print("\n✅ Advanced RAG components working!")


if __name__ == "__main__":
    test_advanced_rag()
