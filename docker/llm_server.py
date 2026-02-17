"""
LLM Inference Server for Mixtral 8x7B
Uses llama-cpp-python for efficient GGUF model serving
Integrates with RAG pipeline for context-aware generation
"""
import os
import yaml
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio
from pydantic import BaseModel
from llama_cpp import Llama
from rag_service import get_rag_service, Document
from advanced_rag import AdvancedRAG

# Global advanced RAG instance
_advanced_rag = None

def get_advanced_rag():
    global _advanced_rag
    if _advanced_rag is None:
        rag = get_rag_service()
        if rag:
            _advanced_rag = AdvancedRAG(rag)
    return _advanced_rag

# Load configuration
with open('/app/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    context: Optional[List[str]] = None  # Manual context override
    use_rag: Optional[bool] = True  # Auto-retrieve context from RAG
    rag_k: Optional[int] = 3  # Number of documents to retrieve
    conversation_history: Optional[List[ConversationMessage]] = None  # Previous messages

class GenerationResponse(BaseModel):
    text: str
    tokens_used: int
    model: str
    context_used: Optional[List[str]] = None
    citations: Optional[List[dict]] = None  # Source citations with scores
class StreamingRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    context: Optional[List[str]] = None
    use_rag: Optional[bool] = True
    rag_k: Optional[int] = 3
    conversation_history: Optional[List[ConversationMessage]] = None


class IngestRequest(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class IngestBatchRequest(BaseModel):
    documents: List[IngestRequest]

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5

# Initialize FastAPI
app = FastAPI(title="Personal AI LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance
llm = None

def load_model():
    """Load Mixtral model with GPU acceleration"""
    global llm
    
    model_path = "/app/models/mixtral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading Mixtral 8x7B from {model_path}")
    print(f"Using GPU: NVIDIA RTX 5090")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,  # Context window
        n_threads=8,   # CPU threads
        n_gpu_layers=-1,  # All layers on GPU
        verbose=True
    )
    
    print("✅ Model loaded successfully!")
    return llm

@app.on_event("startup")
async def startup_event():
    """Load model on server start"""
    load_model()
    
    # Initialize RAG service (will connect to ChromaDB)
    try:
        rag = get_rag_service()
        stats = rag.get_stats()
        print(f"✅ RAG service connected - {stats['document_count']} documents in knowledge base")
    except Exception as e:
        print(f"⚠️ RAG service not available: {e}")
        print("   (You can still use the LLM without RAG)")

@app.get("/")
def read_root():
    return {"message": "Personal AI LLM Server", "model": "Mixtral-8x7B", "rag_enabled": True}

@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    """Generate text based on prompt and context"""
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    context_docs = request.context
    
    # Auto-retrieve context from RAG if enabled and no manual context provided
    citations = None
    if request.use_rag and not context_docs:
        try:
            advanced_rag = get_advanced_rag()
            if advanced_rag:
                # Use hybrid search with reranking
                context_docs, citations = advanced_rag.get_context_with_citations(
                    request.prompt, k=request.rag_k
                )
            else:
                # Fallback to basic RAG
                rag = get_rag_service()
                context_docs = rag.get_context(request.prompt, k=request.rag_k)
        except Exception as e:
            print(f"RAG retrieval failed: {e}")
            context_docs = []
    
    # Build prompt with RAG context if available
    full_prompt = ""
    
    # Add RAG context first
    if context_docs:
        full_prompt = "Use the following context to help answer the query:\n\n"
        for i, doc in enumerate(context_docs[:3], 1):
            full_prompt += f"[{i}] {doc}\n\n"
        full_prompt += "---\n\n"
    
    # Add conversation history
    if request.conversation_history:
        full_prompt += "Previous conversation:\n"
        for msg in request.conversation_history[-6:]:  # Last 6 messages (3 turns)
            role = "User" if msg.role == "user" else "Assistant"
            full_prompt += f"{role}: {msg.content}\n\n"
        full_prompt += "---\n\n"
    
    full_prompt += f"User: {request.prompt}\n\nAssistant:"
    
    # Generate
    response = llm(
        full_prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=["</s>", "\n\n\n"],
        echo=False
    )
    
    return GenerationResponse(
        text=response['choices'][0]['text'].strip(),
        tokens_used=response['usage']['total_tokens'],
        model="mixtral-8x7b",
        context_used=context_docs if context_docs else None,
        citations=citations
    )


@app.post("/generate/stream")
def generate_stream(request: StreamingRequest):
    """Generate text with streaming response (Server-Sent Events)"""
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    context_docs = request.context
    citations = None
    
    # Auto-retrieve context from RAG if enabled
    if request.use_rag and not context_docs:
        try:
            advanced_rag = get_advanced_rag()
            if advanced_rag:
                context_docs, citations = advanced_rag.get_context_with_citations(
                    request.prompt, k=request.rag_k
                )
            else:
                rag = get_rag_service()
                context_docs = rag.get_context(request.prompt, k=request.rag_k)
        except Exception as e:
            print(f"RAG retrieval failed: {e}")
            context_docs = []
    
    # Build prompt with RAG context
    full_prompt = ""
    
    if context_docs:
        full_prompt = "Use the following context to help answer the query:\n\n"
        for i, doc in enumerate(context_docs[:3], 1):
            full_prompt += f"[{i}] {doc}\n\n"
        full_prompt += "---\n\n"
    
    if request.conversation_history:
        full_prompt += "Previous conversation:\n"
        for msg in request.conversation_history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            full_prompt += f"{role}: {msg.content}\n\n"
        full_prompt += "---\n\n"
    
    full_prompt += f"User: {request.prompt}\n\nAssistant:"
    
    def generate_tokens():
        # Send initial metadata (context, citations)
        meta = {
            "type": "meta",
            "context_used": context_docs if context_docs else None,
            "citations": citations
        }
        yield f"data: {json.dumps(meta)}\n\n"
        
        # Stream tokens
        full_response = ""
        for output in llm(
            full_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["</s>", "\n\n\n"],
            echo=False,
            stream=True
        ):
            token = output["choices"][0]["text"]
            full_response += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        
        # Send completion signal with stats
        yield f"data: {json.dumps({'type': 'done', 'text': full_response.strip()})}\n\n"
    
    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/ingest")
def ingest_document(request: IngestRequest):
    """Add a document to the knowledge base"""
    try:
        rag = get_rag_service()
        chunks_added = rag.add_document(request.content, request.metadata)
        return {
            "status": "success",
            "chunks_added": chunks_added,
            "total_documents": rag.get_stats()["document_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/ingest/batch")
def ingest_batch(request: IngestBatchRequest):
    """Add multiple documents to the knowledge base"""
    try:
        rag = get_rag_service()
        documents = [Document(content=d.content, metadata=d.metadata) for d in request.documents]
        chunks_added = rag.add_documents(documents)
        return {
            "status": "success",
            "documents_processed": len(documents),
            "chunks_added": chunks_added,
            "total_documents": rag.get_stats()["document_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")

@app.post("/search")
def search_documents(request: SearchRequest):
    """Search the knowledge base"""
    try:
        rag = get_rag_service()
        results = rag.search(request.query, k=request.k)
        return {
            "query": request.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/knowledge/stats")
def knowledge_stats():
    """Get knowledge base statistics"""
    try:
        rag = get_rag_service()
        return rag.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not get stats: {str(e)}")

@app.delete("/knowledge/clear")
def clear_knowledge():
    """Clear all documents from the knowledge base"""
    try:
        rag = get_rag_service()
        rag.clear()
        return {"status": "success", "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@app.get("/health")
def health_check():
    rag_status = "unknown"
    doc_count = 0
    try:
        rag = get_rag_service()
        stats = rag.get_stats()
        rag_status = "connected"
        doc_count = stats["document_count"]
    except:
        rag_status = "disconnected"
    
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "rag_status": rag_status,
        "knowledge_base_documents": doc_count
    }


# Sync endpoints
class SyncDeleteRequest(BaseModel):
    source_file: str

@app.post("/sync/delete")
def sync_delete(request: SyncDeleteRequest):
    """Delete all chunks for a source file."""
    try:
        rag = get_rag_service()
        deleted = rag.delete_by_source(request.source_file)
        return {"status": "ok", "deleted_count": deleted, "source_file": request.source_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sync/sources")
def sync_sources():
    """List all source files in the knowledge base."""
    try:
        rag = get_rag_service()
        sources = rag.get_sources()
        return {"sources": sources, "count": len(sources)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync/now")
def sync_now():
    """Trigger immediate sync (called from web UI)."""
    import subprocess
    try:
        result = subprocess.run(
            ['python3', '/app/sync_service.py', 'full-sync'],
            capture_output=True,
            text=True,
            timeout=300
        )
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "output": result.stdout,
            "error": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Sync timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
