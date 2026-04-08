"""
LLM Inference Server for Mixtral 8x7B
Uses llama-cpp-python for efficient GGUF model serving
Integrates with RAG pipeline for context-aware generation
"""
import gc
import os
import yaml
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import json
from pydantic import BaseModel
from llama_cpp import Llama

# Lock to prevent concurrent inference calls (llama-cpp-python is not thread-safe)
_inference_lock = threading.Lock()
from rag_service import get_rag_service, Document
from advanced_rag import AdvancedRAG
from settings_manager import get_all_settings, apply_settings, load_settings, save_settings
from conversation_store import get_conversation_store
from memory_service import get_memory_service
from facts_service import get_facts_service

# Global advanced RAG instance
_advanced_rag = None

# Email config directory
EMAIL_CONFIG_DIR = Path.home() / ".personal-ai"

def get_advanced_rag():
    global _advanced_rag
    if _advanced_rag is None:
        rag = get_rag_service()
        if rag:
            _advanced_rag = AdvancedRAG(rag)
    return _advanced_rag

import re


def _wrap_instruct(prompt: str) -> str:
    """Wrap prompt in Mixtral/Mistral [INST] template for single-shot calls.
    Note: llama-cpp-python adds <s> (BOS) automatically, so we omit it here."""
    return f"[INST] {prompt.strip()} [/INST]"


def _build_multiturn_prompt(
    system_context: str,
    conversation_history: Optional[List] = None,
    current_message: str = "",
) -> str:
    """Build a proper Mixtral multi-turn [INST] prompt.

    Format:
      [INST] {system context}

      {user_1} [/INST] {assistant_1}</s>[INST] {user_2} [/INST] {assistant_2}</s>[INST] {current} [/INST]

    System context (personality, facts, memory, RAG) goes in the first [INST] block only.
    BOS (<s>) is added automatically by llama-cpp-python.
    """
    history = conversation_history or []
    history = history[-6:]  # Last 6 messages (3 turns)

    # Pair history into complete (user, assistant) turns
    turns = []
    i = 0
    while i < len(history) - 1:
        if history[i].role == "user" and history[i + 1].role == "assistant":
            turns.append((history[i].content.strip(), history[i + 1].content.strip()))
            i += 2
        else:
            i += 1  # Skip malformed entries

    ctx = system_context.strip()

    if not turns:
        # No history — single [INST] block with system context + user message
        if ctx:
            return f"[INST] {ctx}\n\n{current_message.strip()} [/INST]"
        return f"[INST] {current_message.strip()} [/INST]"

    # First [INST]: system context + first user message
    first_user, first_assistant = turns[0]
    if ctx:
        parts = [f"[INST] {ctx}\n\n{first_user} [/INST] {first_assistant}</s>"]
    else:
        parts = [f"[INST] {first_user} [/INST] {first_assistant}</s>"]

    # Subsequent complete turns
    for user_msg, assistant_msg in turns[1:]:
        parts.append(f"[INST] {user_msg} [/INST] {assistant_msg}</s>")

    # Final: current user message
    parts.append(f"[INST] {current_message.strip()} [/INST]")

    return "".join(parts)


def _is_conversational(prompt: str) -> bool:
    """Check if a prompt is conversational/greeting rather than a knowledge query."""
    p = prompt.strip().lower()
    # Short greetings and conversational phrases
    if len(p.split()) <= 6 and any(w in p for w in [
        'hi', 'hello', 'hey', 'howdy', 'sup', 'yo',
        'good morning', 'good afternoon', 'good evening',
        'how are you', 'what\'s up', 'whats up',
        'tell me about yourself', 'who are you', 'what are you',
        'what is your name', 'what\'s your name', 'whats your name',
        'thanks', 'thank you', 'bye', 'goodbye', 'good night',
    ]):
        return True
    # Very short messages (1-3 words) are likely conversational
    if len(p.split()) <= 3 and not any(c in p for c in ['?', 'how to', 'what is', 'explain', 'find']):
        return True
    return False

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

class SettingsUpdate(BaseModel):
    digest: Optional[dict] = None
    sync: Optional[dict] = None
    model: Optional[dict] = None
    personality: Optional[dict] = None
    web_search: Optional[dict] = None
    training: Optional[dict] = None

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
_maintenance_mode = False
_current_model_path = None

def load_model(model_path_override=None):
    """Load model with GPU acceleration"""
    global llm, _current_model_path

    # Unload any existing model first to free GPU memory
    unload_model()

    model_path = model_path_override or "/app/models/mixtral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model from {model_path}")
    print(f"Using GPU: NVIDIA RTX 5090")

    llm = Llama(
        model_path=model_path,
        n_ctx=config.get('model', {}).get('context_length', 16384),
        n_threads=8,
        n_gpu_layers=-1,
        verbose=True
    )

    _current_model_path = model_path
    print("✅ Model loaded successfully!")
    return llm

def unload_model():
    """Unload model to free GPU memory for training."""
    global llm
    if llm is not None:
        del llm
        llm = None
        gc.collect()
        print("✅ Model unloaded, GPU memory freed")

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

    if _maintenance_mode:
        raise HTTPException(status_code=503, detail="Model is retraining. Please wait — this usually takes ~2.5 hours.")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    context_docs = request.context

    citations = None
    memory_context = []
    fact_context = []

    # Always search learned facts (independent of RAG setting)
    try:
        facts = get_facts_service()
        if facts:
            fact_docs, fact_citations = facts.get_fact_context(request.prompt, k=3)
            if fact_docs:
                fact_context = fact_docs
                print(f"Found {len(fact_docs)} relevant facts")
    except Exception as e:
        print(f"Facts search failed (ok if empty): {e}")

    # Auto-retrieve context from RAG if enabled and no manual context provided
    if request.use_rag and not context_docs:
        try:
            # Search conversation memory
            try:
                memory = get_memory_service()
                memory_docs, memory_citations = memory.get_memory_context(request.prompt, k=2)
                if memory_docs:
                    memory_context = memory_docs
                    print(f"Found {len(memory_docs)} relevant past conversations")
            except Exception as e:
                print(f"Memory search failed (ok if empty): {e}")

            # Search knowledge base
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

    # Build system context (personality, facts, memory, RAG)
    system_context = ""

    # Add personality/system prompt
    try:
        settings = load_settings()
        personality = settings.get("personality", {})
        ai_name = personality.get("name", "Assistant")
        system_prompt = personality.get("prompt", "You are a helpful AI assistant.")

        system_context = f"Your name is {ai_name}. {system_prompt}\n\n"
    except:
        system_context = ""

    # Add learned facts (these override stale training data)
    if fact_context:
        system_context += "SYSTEM FACTS (mandatory — override your training data, do not contradict these):\n\n"
        for fact in fact_context:
            system_context += f"- {fact}\n"
        system_context += "\n---\n\n"

    # Add memory context (past conversations)
    if memory_context:
        system_context += "From our previous conversations:\n\n"
        for doc in memory_context:
            system_context += f"{doc}\n\n"
        system_context += "---\n\n"

    # Add RAG context (knowledge base) - skip for conversational queries
    if context_docs and not _is_conversational(request.prompt):
        system_context += "Relevant information from your knowledge base:\n\n"
        for i, doc in enumerate(context_docs[:3], 1):
            system_context += f"[{i}] {doc}\n\n"
        system_context += "---\n\n"

    # Build multi-turn prompt with proper [INST] format
    full_prompt = _build_multiturn_prompt(system_context, request.conversation_history, request.prompt)

    # Re-check model availability (may have been unloaded during context retrieval)
    if _maintenance_mode or llm is None:
        raise HTTPException(status_code=503, detail="Model is retraining. Please wait — this usually takes ~2.5 hours.")

    # Generate (lock prevents concurrent CUDA access)
    with _inference_lock:
        response = llm(
            full_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["</s>", "\n\n\n", "[INST]"],
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

    if _maintenance_mode:
        raise HTTPException(status_code=503, detail="Model is retraining. Please wait — this usually takes ~2.5 hours.")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    context_docs = request.context
    citations = None
    memory_context = []
    fact_context = []

    # Always search learned facts (independent of RAG setting)
    try:
        facts = get_facts_service()
        if facts:
            fact_docs, fact_citations = facts.get_fact_context(request.prompt, k=3)
            if fact_docs:
                fact_context = fact_docs
                print(f"Found {len(fact_docs)} relevant facts")
    except Exception as e:
        print(f"Facts search failed (ok if empty): {e}")

    # Auto-retrieve context from RAG if enabled
    if request.use_rag and not context_docs:
        try:
            # Search conversation memory
            try:
                memory = get_memory_service()
                memory_docs, memory_citations = memory.get_memory_context(request.prompt, k=2)
                if memory_docs:
                    memory_context = memory_docs
                    print(f"Found {len(memory_docs)} relevant past conversations")
            except Exception as e:
                print(f"Memory search failed (ok if empty): {e}")

            # Search knowledge base
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

    # Build system context (personality, facts, memory, RAG)
    system_context = ""

    # Add personality/system prompt
    try:
        settings = load_settings()
        personality = settings.get("personality", {})
        ai_name = personality.get("name", "Assistant")
        system_prompt = personality.get("prompt", "You are a helpful AI assistant.")

        system_context = f"Your name is {ai_name}. {system_prompt}\n\n"
    except:
        system_context = ""

    # Add learned facts (these override stale training data)
    if fact_context:
        system_context += "SYSTEM FACTS (mandatory — override your training data, do not contradict these):\n\n"
        for fact in fact_context:
            system_context += f"- {fact}\n"
        system_context += "\n---\n\n"

    # Add memory context (past conversations)
    if memory_context:
        system_context += "From our previous conversations:\n\n"
        for doc in memory_context:
            system_context += f"{doc}\n\n"
        system_context += "---\n\n"

    # Add RAG context (knowledge base) - skip for conversational queries
    if context_docs and not _is_conversational(request.prompt):
        system_context += "Relevant information from your knowledge base:\n\n"
        for i, doc in enumerate(context_docs[:3], 1):
            system_context += f"[{i}] {doc}\n\n"
        system_context += "---\n\n"

    # Build multi-turn prompt with proper [INST] format
    full_prompt = _build_multiturn_prompt(system_context, request.conversation_history, request.prompt)

    # Re-check model availability (may have been unloaded during context retrieval)
    if _maintenance_mode or llm is None:
        raise HTTPException(status_code=503, detail="Model is retraining. Please wait — this usually takes ~2.5 hours.")

    def generate_tokens():
        import time

        # Send initial metadata (context, citations)
        meta = {
            "type": "meta",
            "context_used": context_docs if context_docs else None,
            "citations": citations
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Stream tokens with timing (lock prevents concurrent CUDA access)
        full_response = ""
        token_count = 0
        start_time = time.time()
        first_token_time = None

        with _inference_lock:
            for output in llm(
                full_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=["</s>", "\n\n\n", "[INST]"],
                echo=False,
                stream=True
            ):
                token = output["choices"][0]["text"]
                full_response += token
                token_count += 1

                # Record time to first token
                if first_token_time is None:
                    first_token_time = time.time()

                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        # Calculate metrics
        end_time = time.time()
        total_time = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else 0
        throughput = token_count / total_time if total_time > 0 else 0

        # Send completion signal with stats and metrics
        yield f"data: {json.dumps({'type': 'done', 'text': full_response.strip(), 'metrics': {'ttft': round(ttft, 2), 'tokens': token_count, 'total_time': round(total_time, 2), 'throughput': round(throughput, 1)}})}\n\n"
    
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
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/knowledge/clear")
def clear_knowledge():
    """Clear the knowledge base"""
    try:
        rag = get_rag_service()
        rag.clear()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    rag_status = "disconnected"
    doc_count = 0
    try:
        rag = get_rag_service()
        stats = rag.get_stats()
        rag_status = "connected"
        doc_count = stats.get('document_count', 0)
    except:
        pass
    
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "rag_status": rag_status,
        "knowledge_base_documents": doc_count
    }


@app.post("/sync/delete")
def delete_synced_content(source_file: str):
    """Delete documents from a specific source file"""
    try:
        rag = get_rag_service()
        # This would need implementation in rag_service
        return {"status": "deleted", "source": source_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sync/sources")
def list_synced_sources():
    """List all synced source files"""
    try:
        rag = get_rag_service()
        # This would need implementation to get unique sources
        return {"sources": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sync/now")
def trigger_sync_legacy():
    """Trigger immediate sync (legacy endpoint)"""
    try:
        result = subprocess.run(
            ['python3', '/app/sync_service.py', '--once'],
            capture_output=True,
            text=True,
            timeout=300
        )
        return {"status": "ok", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Settings endpoints
@app.get("/settings")
def get_settings():
    """Get all settings"""
    return get_all_settings()


@app.post("/settings")
def update_settings(settings: SettingsUpdate):
    """Update settings"""
    current = load_settings()
    
    if settings.digest:
        current["digest"].update(settings.digest)
    if settings.sync:
        current["sync"].update(settings.sync)
    if settings.model:
        current["model"].update(settings.model)
    if settings.personality:
        current["personality"] = settings.personality
    if settings.web_search:
        current["web_search"] = settings.web_search
    if settings.training:
        current.setdefault("training", {}).update(settings.training)

    result = apply_settings(current)
    return {
        "status": "ok" if result["saved"] else "error",
        "result": result,
        "settings": get_all_settings()
    }


# ---- Training Coordination Endpoints ----

TRAINING_STATE_FILE = Path.home() / ".personal-ai" / "training_state.json"
TRAINING_TRIGGER_FILE = Path.home() / ".personal-ai" / "training_trigger.json"

@app.get("/training/status")
def training_status():
    """Get current training pipeline status."""
    try:
        if TRAINING_STATE_FILE.exists():
            with open(TRAINING_STATE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return {"status": "idle", "progress_pct": 0}

@app.post("/training/trigger")
def training_trigger():
    """Trigger a training run by writing a trigger file for the host-side watcher."""
    if _maintenance_mode:
        return {"status": "error", "error": "Training already in progress"}
    TRAINING_TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_TRIGGER_FILE, "w") as f:
        json.dump({"action": "start", "timestamp": str(datetime.now())}, f)
    return {"status": "ok", "message": "Training triggered"}

@app.post("/training/prepare")
def training_prepare():
    """Unload the model and enter maintenance mode to free GPU for training."""
    global _maintenance_mode
    unload_model()
    _maintenance_mode = True
    return {"status": "ready_for_training", "previous_model": _current_model_path}

@app.post("/training/complete")
def training_complete(model_path: str = None):
    """Load a new (or the previous) model and exit maintenance mode."""
    global _maintenance_mode
    path = model_path or _current_model_path or "/app/models/mixtral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    load_model(path)
    _maintenance_mode = False
    return {"status": "ok", "model_loaded": path}

@app.post("/training/interrupt")
def training_interrupt():
    """Signal the training orchestrator to stop."""
    TRAINING_TRIGGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_TRIGGER_FILE, "w") as f:
        json.dump({"action": "interrupt", "timestamp": str(datetime.now())}, f)
    return {"status": "ok", "message": "Interrupt signal sent"}

@app.get("/training/maintenance")
def training_maintenance_status():
    """Check if server is in maintenance mode."""
    return {"maintenance_mode": _maintenance_mode}


@app.post("/settings/sync-now")
def trigger_sync():
    """Trigger immediate sync"""
    try:
        result = subprocess.run(
            ['python3', '/app/sync_service.py', '--once'],
            capture_output=True,
            text=True,
            timeout=300
        )
        return {"status": "ok", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Email OAuth Setup endpoints
@app.post("/email/upload-credentials/{provider}")
async def upload_email_credentials(provider: str, file: UploadFile = File(...)):
    """Upload OAuth credentials file for Gmail or Outlook"""
    if provider not in ["gmail", "outlook"]:
        raise HTTPException(status_code=400, detail="Provider must be 'gmail' or 'outlook'")
    
    EMAIL_CONFIG_DIR.mkdir(exist_ok=True)
    
    if provider == "gmail":
        dest = EMAIL_CONFIG_DIR / "gmail_credentials.json"
    else:
        dest = EMAIL_CONFIG_DIR / "outlook_credentials.json"
    
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"status": "ok", "message": f"{provider} credentials uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/email/auth-url/{provider}")
def get_auth_url(provider: str):
    """Get OAuth authorization URL"""
    if provider == "gmail":
        creds_file = EMAIL_CONFIG_DIR / "gmail_credentials.json"
        if not creds_file.exists():
            raise HTTPException(status_code=400, detail="Upload credentials first")
        
        try:
            from google_auth_oauthlib.flow import Flow
            
            with open(creds_file) as f:
                creds_data = json.load(f)
            
            # Create flow with localhost redirect
            flow = Flow.from_client_config(
                creds_data,
                scopes=['https://www.googleapis.com/auth/gmail.send',
                        'https://www.googleapis.com/auth/gmail.compose'],
                redirect_uri='http://localhost:8080/email/oauth-callback/gmail'
            )
            
            auth_url, state = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                prompt='consent'
            )
            
            # Store state and code_verifier as JSON (pickle doesn't work with Flow)
            oauth_state = {
                "state": state,
                "code_verifier": flow.code_verifier
            }
            state_file = EMAIL_CONFIG_DIR / "gmail_oauth_state.json"
            with open(state_file, 'w') as f:
                json.dump(oauth_state, f)
            
            return {"auth_url": auth_url, "state": state}
        except ImportError:
            raise HTTPException(status_code=500, detail="google-auth-oauthlib not installed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    elif provider == "outlook":
        creds_file = EMAIL_CONFIG_DIR / "outlook_credentials.json"
        if not creds_file.exists():
            raise HTTPException(status_code=400, detail="Upload credentials first")
        
        try:
            with open(creds_file) as f:
                config = json.load(f)
            
            client_id = config.get('client_id')
            redirect_uri = 'http://localhost:8080/email/oauth-callback/outlook'
            scope = 'https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/Mail.ReadWrite offline_access'
            
            auth_url = (
                f"https://login.microsoftonline.com/consumers/oauth2/v2.0/authorize"
                f"?client_id={client_id}"
                f"&response_type=code"
                f"&redirect_uri={redirect_uri}"
                f"&scope={scope}"
                f"&response_mode=query"
            )
            
            return {"auth_url": auth_url}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=400, detail="Provider must be 'gmail' or 'outlook'")


@app.get("/email/oauth-callback/{provider}")
def oauth_callback(provider: str, code: str = None, state: str = None, error: str = None):
    """Handle OAuth callback"""
    
    if error:
        return HTMLResponse(f"""
            <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h2>❌ Authentication Failed</h2>
            <p>{error}</p>
            <p>You can close this window.</p>
            <script>setTimeout(() => window.close(), 3000);</script>
            </body></html>
        """)
    
    if not code:
        return HTMLResponse("""
            <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h2>❌ No authorization code received</h2>
            <p>You can close this window.</p>
            </body></html>
        """)
    
    try:
        if provider == "gmail":
            from google_auth_oauthlib.flow import Flow
            import pickle
            
            # Load stored state and code_verifier
            state_file = EMAIL_CONFIG_DIR / "gmail_oauth_state.json"
            if not state_file.exists():
                raise Exception("OAuth flow expired. Please try again.")
            
            with open(state_file) as f:
                oauth_state = json.load(f)
            
            # Load credentials config
            creds_file = EMAIL_CONFIG_DIR / "gmail_credentials.json"
            with open(creds_file) as f:
                creds_data = json.load(f)
            
            # Recreate flow with stored code_verifier
            flow = Flow.from_client_config(
                creds_data,
                scopes=['https://www.googleapis.com/auth/gmail.send',
                        'https://www.googleapis.com/auth/gmail.compose'],
                redirect_uri='http://localhost:8080/email/oauth-callback/gmail'
            )
            flow.code_verifier = oauth_state.get("code_verifier")
            
            flow.fetch_token(code=code)
            creds = flow.credentials
            
            # Clean up state file
            state_file.unlink()
            
            # Save token
            token_file = EMAIL_CONFIG_DIR / "gmail_token.pickle"
            with open(token_file, 'wb') as f:
                pickle.dump(creds, f)
            
            return HTMLResponse("""
                <html><body style="font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: #eee;">
                <h2 style="color: #4ade80;">✅ Gmail Connected Successfully!</h2>
                <p>You can close this window and return to the app.</p>
                <script>
                    window.opener && window.opener.postMessage({type: 'oauth-success', provider: 'gmail'}, '*');
                    setTimeout(() => window.close(), 2000);
                </script>
                </body></html>
            """)
        
        elif provider == "outlook":
            import requests
            
            creds_file = EMAIL_CONFIG_DIR / "outlook_credentials.json"
            with open(creds_file) as f:
                config = json.load(f)
            
            # Exchange code for token
            token_response = requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    'client_id': config['client_id'],
                    'client_secret': config.get('client_secret', ''),
                    'code': code,
                    'redirect_uri': 'http://localhost:8080/email/oauth-callback/outlook',
                    'grant_type': 'authorization_code',
                    'scope': 'https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/Mail.ReadWrite offline_access'
                }
            )
            
            if token_response.status_code == 200:
                token_data = token_response.json()
                token_file = EMAIL_CONFIG_DIR / "outlook_token.json"
                with open(token_file, 'w') as f:
                    json.dump(token_data, f)
                
                return HTMLResponse("""
                    <html><body style="font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: #eee;">
                    <h2 style="color: #4ade80;">✅ Outlook Connected Successfully!</h2>
                    <p>You can close this window and return to the app.</p>
                    <script>
                        window.opener && window.opener.postMessage({type: 'oauth-success', provider: 'outlook'}, '*');
                        setTimeout(() => window.close(), 2000);
                    </script>
                    </body></html>
                """)
            else:
                raise Exception(token_response.text)
    
    except Exception as e:
        return HTMLResponse(f"""
            <html><body style="font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: #eee;">
            <h2 style="color: #f87171;">❌ Authentication Error</h2>
            <p>{str(e)}</p>
            <script>setTimeout(() => window.close(), 5000);</script>
            </body></html>
        """)
    
    return HTMLResponse("""
        <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
        <h2>❌ Unknown provider</h2>
        </body></html>
    """)


@app.delete("/email/disconnect/{provider}")
def disconnect_email(provider: str):
    """Remove email provider credentials and tokens"""
    if provider == "gmail":
        files = ["gmail_credentials.json", "gmail_token.pickle"]
    elif provider == "outlook":
        files = ["outlook_credentials.json", "outlook_token.json"]
    else:
        raise HTTPException(status_code=400, detail="Provider must be 'gmail' or 'outlook'")
    
    for filename in files:
        filepath = EMAIL_CONFIG_DIR / filename
        if filepath.exists():
            filepath.unlink()
    
    return {"status": "ok", "message": f"{provider} disconnected"}


# This must be at the very end

from pydantic import BaseModel as PydanticBaseModel

class EmailRequest(PydanticBaseModel):
    to: str
    subject: str
    body: str


@app.post("/email/send")
def send_email(request: EmailRequest):
    """Send email directly via Gmail API"""
    import pickle
    import base64
    from email.mime.text import MIMEText
    from googleapiclient.discovery import build
    
    token_file = EMAIL_CONFIG_DIR / "gmail_token.pickle"
    if not token_file.exists():
        raise HTTPException(status_code=400, detail="Gmail not connected")
    
    with open(token_file, 'rb') as f:
        creds = pickle.load(f)
    
    service = build('gmail', 'v1', credentials=creds)
    
    message = MIMEText(request.body)
    message['to'] = request.to
    message['subject'] = request.subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    try:
        service.users().messages().send(userId='me', body={'raw': raw}).execute()
        return {"status": "sent", "to": request.to}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email/draft")
def create_draft(request: EmailRequest):
    """Create Gmail draft and return URL to open it"""
    import pickle
    import base64
    from email.mime.text import MIMEText
    from googleapiclient.discovery import build
    
    token_file = EMAIL_CONFIG_DIR / "gmail_token.pickle"
    if not token_file.exists():
        raise HTTPException(status_code=400, detail="Gmail not connected")
    
    with open(token_file, 'rb') as f:
        creds = pickle.load(f)
    
    service = build('gmail', 'v1', credentials=creds)
    
    message = MIMEText(request.body)
    message['to'] = request.to
    message['subject'] = request.subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    try:
        draft = service.users().drafts().create(
            userId='me', 
            body={'message': {'raw': raw}}
        ).execute()
        
        draft_id = draft['id']
        message_id = draft.get('message', {}).get('id', '')
        
        # Gmail URL to open the draft for editing
        gmail_url = f"https://mail.google.com/mail/u/0/#drafts?compose={message_id}"
        
        return {"status": "draft_created", "draft_id": draft_id, "url": gmail_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Conversation History Endpoints
@app.post("/conversations")
def create_conversation(title: str = None):
    """Create a new conversation"""
    store = get_conversation_store()
    conv_id = store.create_conversation(title)
    return {"id": conv_id, "title": title}


@app.get("/conversations")
def list_conversations(limit: int = 50, offset: int = 0):
    """List recent conversations"""
    store = get_conversation_store()
    return {"conversations": store.list_conversations(limit, offset)}


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    """Get full conversation with messages"""
    store = get_conversation_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/conversations/{conv_id}/messages")
def add_message(conv_id: str, role: str, content: str, tokens_used: int = None):
    """Add message to conversation"""
    store = get_conversation_store()
    store.add_message(conv_id, role, content, tokens_used)
    return {"status": "ok"}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    """Delete a conversation"""
    store = get_conversation_store()
    store.delete_conversation(conv_id)
    return {"status": "deleted"}


@app.post("/conversations/{conv_id}/ingest")
def ingest_conversation(conv_id: str):
    """Ingest conversation to memory (RAG)"""
    memory = get_memory_service()
    chunks = memory.ingest_conversation(conv_id)
    return {"status": "ingested", "chunks_added": chunks}


@app.post("/memory/ingest-all")
def ingest_all_conversations():
    """Ingest all pending conversations to memory"""
    memory = get_memory_service()
    result = memory.ingest_all_pending()
    return result


@app.get("/memory/search")
def search_memory(query: str, k: int = 3):
    """Search conversation memory"""
    memory = get_memory_service()
    results = memory.search_memory(query, k)
    return {"results": results}




@app.delete("/memory/clear-all")
def clear_all_memory():
    """Delete all conversations and memory - nuclear option"""
    store = get_conversation_store()
    
    # Get all conversations
    convs = store.list_conversations(limit=1000)
    deleted_count = len(convs)
    
    # Delete each one
    for conv in convs:
        store.delete_conversation(conv["id"])
    
    return {
        "status": "cleared",
        "conversations_deleted": deleted_count,
        "message": "All conversations and memory have been cleared"
    }


@app.post("/memory/auto-ingest/{conv_id}")
def auto_ingest_conversation(conv_id: str):
    """Auto-ingest a conversation (called after each message if enabled)"""
    try:
        memory = get_memory_service()
        
        # Mark as not ingested first so we can re-ingest
        from conversation_store import get_db
        conn = get_db()
        conn.execute("UPDATE conversations SET ingested_to_rag = FALSE WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()
        
        chunks = memory.ingest_conversation(conv_id)
        return {"status": "ok", "chunks": chunks}
    except Exception as e:
        return {"status": "error", "error": str(e)}




@app.get("/memory/list")
def list_memories(limit: int = 100, offset: int = 0):
    """List all memories stored in ChromaDB"""
    try:
        from rag_service import get_rag_service
        rag = get_rag_service()
        
        # Query ChromaDB for conversation memories
        results = rag.collection.get(
            where={"source_type": "conversation_memory"},
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"]
        )
        
        memories = []
        if results and results['ids']:
            for i, id in enumerate(results['ids']):
                memories.append({
                    "id": id,
                    "content": results['documents'][i][:500] if results['documents'] else "",
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                })
        
        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        return {"error": str(e), "memories": [], "count": 0}


@app.delete("/memory/item/{memory_id}")
def delete_memory_item(memory_id: str):
    """Delete a specific memory entry from ChromaDB"""
    try:
        from rag_service import get_rag_service
        rag = get_rag_service()
        
        # Delete from ChromaDB
        rag.collection.delete(ids=[memory_id])
        
        return {"status": "deleted", "id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/memory/stats")
def memory_stats():
    """Get memory statistics"""
    try:
        from rag_service import get_rag_service
        rag = get_rag_service()
        
        # Count conversation memories
        results = rag.collection.get(
            where={"source_type": "conversation_memory"},
            include=[]
        )
        memory_count = len(results['ids']) if results and results['ids'] else 0
        
        # Get conversation count from SQLite
        store = get_conversation_store()
        convs = store.list_conversations(limit=1000)
        
        return {
            "memory_chunks": memory_count,
            "conversations": len(convs),
            "total_knowledge_docs": rag.collection.count()
        }
    except Exception as e:
        return {"error": str(e)}




@app.get("/export/conversations")
def export_conversations():
    """Export all conversations as JSON"""
    import json
    from datetime import datetime
    
    store = get_conversation_store()
    convs = store.list_conversations(limit=1000)
    
    export_data = {
        "export_date": datetime.now().isoformat(),
        "version": "3.3.1",
        "conversations": []
    }
    
    for conv in convs:
        full_conv = store.get_conversation(conv["id"])
        export_data["conversations"].append(full_conv)
    
    return export_data


@app.get("/export/full")
def export_full_backup():
    """Export conversations + settings as full backup"""
    import json
    from datetime import datetime
    
    store = get_conversation_store()
    convs = store.list_conversations(limit=1000)
    
    export_data = {
        "export_date": datetime.now().isoformat(),
        "version": "3.3.1",
        "type": "full_backup",
        "conversations": [],
        "settings": load_settings()
    }
    
    for conv in convs:
        full_conv = store.get_conversation(conv["id"])
        export_data["conversations"].append(full_conv)
    
    return export_data


class ImportData(BaseModel):
    conversations: List[dict] = []
    settings: dict = None


@app.post("/import/conversations")
def import_conversations(data: ImportData):
    """Import conversations from JSON backup"""
    store = get_conversation_store()
    imported = 0
    skipped = 0
    
    for conv in data.conversations:
        try:
            # Check if conversation already exists
            existing = store.get_conversation(conv.get("id", ""))
            if existing:
                skipped += 1
                continue
            
            # Create new conversation
            conv_id = store.create_conversation(conv.get("title", "Imported conversation"))
            
            # Add messages
            for msg in conv.get("messages", []):
                store.add_message(
                    conv_id,
                    msg.get("role", "user"),
                    msg.get("content", ""),
                    msg.get("tokens_used")
                )
            
            imported += 1
        except Exception as e:
            print(f"Failed to import conversation: {e}")
            skipped += 1
    
    # Import settings if provided
    settings_imported = False
    if data.settings:
        try:
            save_settings(data.settings)
            settings_imported = True
        except:
            pass
    
    return {
        "status": "ok",
        "imported": imported,
        "skipped": skipped,
        "settings_imported": settings_imported
    }




@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...), doc_type: str = "document"):
    """Upload a document and ingest it to the knowledge base"""
    import os
    from pathlib import Path
    
    # Determine destination directory
    base_path = Path.home() / "knowledge"
    if doc_type == "datasheet":
        dest_dir = base_path / "datasheets"
    elif doc_type == "email":
        dest_dir = base_path / "emails"
    elif doc_type == "transcript":
        dest_dir = base_path / "transcripts"
    else:
        dest_dir = base_path / "documents"
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file (create subdirs if filename includes a path from folder upload)
    file_path = dest_dir / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Ingest to RAG
    try:
        from rag_service import get_rag_service
        rag = get_rag_service()
        
        # Read and ingest based on file type
        chunks_added = 0
        if file.filename.lower().endswith('.pdf'):
            # Use PyPDF2 or similar for PDF
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(file_path))
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                # Chunk the text
                chunk_size = 1000
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    if chunk.strip():
                        rag.add_document(
                            content=chunk,
                            metadata={
                                "source_type": "uploaded_document",
                                "source_file": file.filename,
                                "doc_type": doc_type,
                                "chunk_index": i // chunk_size
                            }
                        )
                        chunks_added += 1
            except ImportError:
                return {"status": "error", "error": "PDF support not installed (PyMuPDF)"}
        else:
            # Plain text files
            text = file_path.read_text(errors='ignore')
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                if chunk.strip():
                    rag.add_document(
                        content=chunk,
                        metadata={
                            "source_type": "uploaded_document",
                            "source_file": file.filename,
                            "doc_type": doc_type,
                            "chunk_index": i // chunk_size
                        }
                    )
                    chunks_added += 1
        
        return {
            "status": "ok",
            "filename": file.filename,
            "path": str(file_path),
            "chunks_added": chunks_added,
            "doc_type": doc_type
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "filename": file.filename}





def summarize_transcript_direct(transcript: str, title: str = "Meeting") -> dict:
    """Summarize transcript using the already-loaded LLM directly"""
    prompt = f"""Analyze this meeting transcript and provide a JSON response with:
1. A brief 2-3 sentence summary
2. Key points discussed (list)
3. Action items with owners if mentioned (list)  
4. Decisions made (list)

Meeting: {title}

Transcript:
{transcript[:6000]}

Respond ONLY with valid JSON in this exact format, no other text:
{{"summary": "...", "key_points": ["..."], "action_items": ["..."], "decisions": ["..."]}}"""

    prompt = _wrap_instruct(prompt)
    try:
        with _inference_lock:
            response = llm(prompt, max_tokens=1000, temperature=0.3)
        text = response["choices"][0]["text"].strip()
        
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"summary": text, "key_points": [], "action_items": [], "decisions": []}
    except Exception as e:
        return {"error": str(e)}

def summarize_ocr_direct(text: str, title: str = "Image") -> dict:
    """Summarize OCR-extracted text using the already-loaded LLM directly"""
    prompt = f"""Analyze this text extracted from an image and provide a JSON response with:
1. A brief 2-3 sentence summary of what this text contains
2. Key points or information found (list)

Image: {title}

Extracted text:
{text[:6000]}

Respond ONLY with valid JSON in this exact format, no other text:
{{"summary": "...", "key_points": ["..."]}}"""

    prompt = _wrap_instruct(prompt)
    try:
        with _inference_lock:
            response = llm(prompt, max_tokens=1000, temperature=0.3)
        text_resp = response["choices"][0]["text"].strip()

        import re
        json_match = re.search(r'\{.*\}', text_resp, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"summary": text_resp, "key_points": []}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload/transcribe")
async def upload_and_transcribe(file: UploadFile = File(...), title: str = "Untitled Recording", summarize: bool = False):
    """Upload audio/video and transcribe with Whisper"""
    import tempfile
    from pathlib import Path
    
    # Check file type
    suffix = Path(file.filename).suffix.lower()
    audio_formats = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.wma', '.amr', '.aac'}
    video_formats = {'.mp4', '.webm', '.mkv', '.mov', '.avi'}
    supported = audio_formats | video_formats
    
    if suffix not in supported:
        return {"status": "error", "error": f"Unsupported format: {suffix}. Supported: {', '.join(sorted(supported))}"}
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name
    
    try:
        # Import and use meeting summarizer
        from meeting_summarizer import MeetingSummarizer
        
        summarizer = MeetingSummarizer(whisper_model="base", device="cpu")
        
        # Transcribe
        transcript_text, segments = summarizer.transcribe(tmp_path)
        
        # Save transcript to knowledge base
        transcript_dir = Path.home() / "knowledge" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as text file
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        transcript_file = transcript_dir / f"{safe_title}.txt"
        transcript_file.write_text(transcript_text)
        
        # Ingest to RAG
        from rag_service import get_rag_service
        rag = get_rag_service()
        
        chunk_size = 1000
        chunks_added = 0
        for i in range(0, len(transcript_text), chunk_size):
            chunk = transcript_text[i:i+chunk_size]
            if chunk.strip():
                rag.add_document(
                    content=chunk,
                    metadata={
                        "source_type": "transcript",
                        "source_file": file.filename,
                        "title": title,
                        "chunk_index": i // chunk_size
                    }
                )
                chunks_added += 1
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Summarize if requested
        summary_data = None
        if summarize:
            try:
                summary_data = summarize_transcript_direct(transcript_text, title)
            except Exception as e:
                summary_data = {"error": str(e)}
        
        return {
            "status": "ok",
            "filename": file.filename,
            "title": title,
            "transcript_length": len(transcript_text),
            "chunks_added": chunks_added,
            "transcript_preview": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
            "full_transcript": transcript_text,
            "summary": summary_data
        }
        
    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"status": "error", "error": str(e)}




@app.post("/upload/ocr")
async def upload_and_ocr(file: UploadFile = File(...), title: str = "Screenshot", summarize: bool = False):
    """Upload image and extract text with OCR"""
    import tempfile
    from pathlib import Path
    
    # Check file type
    suffix = Path(file.filename).suffix.lower()
    image_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    
    if suffix not in image_formats:
        return {"status": "error", "error": f"Unsupported format: {suffix}. Supported: {', '.join(sorted(image_formats))}"}
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name
    
    try:
        # OCR with pytesseract
        import pytesseract
        from PIL import Image
        
        img = Image.open(tmp_path)
        extracted_text = pytesseract.image_to_string(img)
        
        if not extracted_text.strip():
            os.unlink(tmp_path)
            return {"status": "error", "error": "No text found in image"}
        
        # Save to knowledge base
        docs_dir = Path.home() / "knowledge" / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as text file
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        text_file = docs_dir / f"{safe_title}.txt"
        text_file.write_text(extracted_text)
        
        # Also save original image
        img_dir = Path.home() / "knowledge" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(tmp_path, img_dir / f"{safe_title}{suffix}")
        
        # Ingest to RAG
        from rag_service import get_rag_service
        rag = get_rag_service()
        
        chunk_size = 1000
        chunks_added = 0
        for i in range(0, len(extracted_text), chunk_size):
            chunk = extracted_text[i:i+chunk_size]
            if chunk.strip():
                rag.add_document(
                    content=chunk,
                    metadata={
                        "source_type": "ocr",
                        "source_file": file.filename,
                        "title": title,
                        "chunk_index": i // chunk_size
                    }
                )
                chunks_added += 1
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Optional summarization
        summary_data = None
        if summarize:
            try:
                summary_data = summarize_ocr_direct(extracted_text, title)
            except Exception as e:
                summary_data = {"error": str(e)}

        return {
            "status": "ok",
            "filename": file.filename,
            "title": title,
            "text_length": len(extracted_text),
            "chunks_added": chunks_added,
            "text_preview": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "full_text": extracted_text,
            "summary": summary_data
        }
        
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return {"status": "error", "error": str(e)}




def _ddg_search(query: str, max_results: int = 5, retries: int = 3):
    """Search DuckDuckGo with retry/backoff for rate limiting."""
    import time
    from duckduckgo_search import DDGS

    for attempt in range(retries):
        try:
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
            if results:
                return results
            # Empty results may be rate limiting — retry
            if attempt < retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  ⏳ DDG returned 0 results, retrying in {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  ⏳ DDG error: {e}, retrying in {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
            else:
                raise
    return []


@app.get("/search/web")
def web_search(query: str, max_results: int = 5):
    """Search the web using DuckDuckGo"""
    from settings_manager import load_settings

    settings = load_settings()
    if not settings.get("web_search", {}).get("enabled", False):
        return {"status": "error", "error": "Web search is disabled. Enable it in Settings."}

    try:
        raw_results = _ddg_search(query, max_results=max_results)
        results = [{
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", "")
        } for r in raw_results]

        return {"status": "ok", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/chat/search")
def chat_with_search(query: str):
    """Answer a question using web search + LLM"""
    from settings_manager import load_settings
    
    settings = load_settings()
    if not settings.get("web_search", {}).get("enabled", False):
        return {"status": "error", "error": "Web search is disabled. Enable it in Settings."}
    
    try:
        # Search the web with retry/backoff
        raw_results = _ddg_search(query, max_results=5)
        search_results = [f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}" for r in raw_results]

        if not search_results:
            return {"status": "ok", "query": query, "answer": "Web search returned no results. DuckDuckGo may be rate-limiting this query — try again in a few minutes or rephrase.", "sources": []}
        
        context = "\n\n".join(search_results)
        print(f"  🌐 Web search returned {len(search_results)} results")

        # Get personality
        personality = settings.get("personality", {})
        ai_name = personality.get("name", "Assistant")
        system_prompt = personality.get("prompt", "You are a helpful AI assistant.")
        
        prompt = f"""Your name is {ai_name}. {system_prompt}

SYSTEM CONTEXT (mandatory — this information is current and overrides your training data):

Web Search Results:
{context}

Answer the following question using ONLY the search results above. If the results contradict your training data, trust the results.

User Question: {query}"""

        prompt = _wrap_instruct(prompt)

        with _inference_lock:
            response = llm(prompt, max_tokens=1000, temperature=0.3, stop=["</s>", "\n\n\n", "User:", "user:", "[INST]"])
        answer = response["choices"][0]["text"].strip()

        # Auto-learn facts from web search results (use raw search snippets, not model answer)
        try:
            facts = get_facts_service()
            if facts and search_results:
                fact_prompt = _wrap_instruct(f"""Extract one concise factual statement from these web search results. Return ONLY the fact, nothing else.
If no clear fact can be extracted, respond with "NONE".

Question: {query}
Search Results: {search_results[0]}""")
                with _inference_lock:
                    fact_response = llm(fact_prompt, max_tokens=100, temperature=0.1, stop=["</s>", "\n\n", "[INST]"])
                extracted = fact_response["choices"][0]["text"].strip()
                if extracted and extracted.upper() != "NONE" and len(extracted) > 10:
                    source_titles = [r.split("\n")[0].replace("Title: ", "") for r in search_results[:2]]
                    facts.add_fact(extracted, source=f"web search: {'; '.join(source_titles)}", category="web")
        except Exception as e:
            print(f"Fact extraction failed (non-critical): {e}")

        return {
            "status": "ok",
            "query": query,
            "answer": answer,
            "sources": search_results[:3]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ===== Facts API Endpoints =====

@app.get("/facts")
def list_facts():
    """List all learned facts."""
    facts = get_facts_service()
    if not facts:
        return {"facts": [], "count": 0}
    all_facts = facts.get_all_facts()
    return {"facts": all_facts, "count": len(all_facts)}


@app.post("/facts")
def add_fact(fact: str, source: str = "manual", category: str = "general"):
    """Manually add a fact."""
    facts = get_facts_service()
    if not facts:
        raise HTTPException(status_code=503, detail="Facts service not available")
    fact_id = facts.add_fact(fact, source=source, category=category)
    return {"status": "ok", "id": fact_id}


@app.delete("/facts/{fact_id}")
def delete_fact(fact_id: str):
    """Delete a specific fact."""
    facts = get_facts_service()
    if not facts:
        raise HTTPException(status_code=503, detail="Facts service not available")
    success = facts.delete_fact(fact_id)
    return {"status": "ok" if success else "error"}


@app.delete("/facts")
def clear_facts():
    """Delete all learned facts."""
    facts = get_facts_service()
    if not facts:
        raise HTTPException(status_code=503, detail="Facts service not available")
    count = facts.clear_all()
    return {"status": "ok", "deleted": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
