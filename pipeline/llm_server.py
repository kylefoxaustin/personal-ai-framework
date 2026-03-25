"""
LLM Inference Server for Mixtral 8x7B
Uses llama-cpp-python for efficient GGUF model serving
Integrates with RAG pipeline for context-aware generation
"""
import os
import yaml
import shutil
import subprocess
from typing import Optional, List
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
import json
from pydantic import BaseModel
from llama_cpp import Llama
from rag_service import get_rag_service, Document
from advanced_rag import AdvancedRAG
from settings_manager import get_all_settings, apply_settings, load_settings, save_settings
from conversation_store import get_conversation_store
from memory_service import get_memory_service

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
        n_ctx=config.get('model', {}).get('context_length', 16384),
        n_threads=8,
        n_gpu_layers=-1,
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
    memory_context = []
    
    if request.use_rag and not context_docs:
        try:
            # Search conversation memory first
            try:
                memory = get_memory_service()
                memory_docs, memory_citations = memory.get_memory_context(request.prompt, k=2)
                if memory_docs:
                    memory_context = memory_docs
                    print(f"Found {len(memory_docs)} relevant past conversations")
            except Exception as e:
                print(f"Memory search failed (ok if empty): {e}")
            
            # Then search knowledge base
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
    
    # Build prompt with memory and RAG context
    full_prompt = ""
    
    # Add personality/system prompt
    try:
        settings = load_settings()
        personality = settings.get("personality", {})
        ai_name = personality.get("name", "Assistant")
        system_prompt = personality.get("prompt", "You are a helpful AI assistant.")
        
        full_prompt = f"Your name is {ai_name}. {system_prompt}\n\n"
    except:
        full_prompt = ""
    
    # Add memory context (past conversations) first
    if memory_context:
        full_prompt = "From our previous conversations:\n\n"
        for doc in memory_context:
            full_prompt += f"{doc}\n\n"
        full_prompt += "---\n\n"
    
    # Add RAG context (knowledge base)
    if context_docs:
        full_prompt += "Relevant information from your knowledge base:\n\n"
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
    memory_context = []
    
    if request.use_rag and not context_docs:
        try:
            # Search conversation memory first
            try:
                memory = get_memory_service()
                memory_docs, memory_citations = memory.get_memory_context(request.prompt, k=2)
                if memory_docs:
                    memory_context = memory_docs
                    print(f"Found {len(memory_docs)} relevant past conversations")
            except Exception as e:
                print(f"Memory search failed (ok if empty): {e}")
            
            # Then search knowledge base
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
    
    # Build prompt with memory and RAG context
    full_prompt = ""
    
    # Add personality/system prompt
    try:
        settings = load_settings()
        personality = settings.get("personality", {})
        ai_name = personality.get("name", "Assistant")
        system_prompt = personality.get("prompt", "You are a helpful AI assistant.")
        
        full_prompt = f"Your name is {ai_name}. {system_prompt}\n\n"
    except:
        full_prompt = ""
    
    # Add memory context (past conversations) first
    if memory_context:
        full_prompt = "From our previous conversations:\n\n"
        for doc in memory_context:
            full_prompt += f"{doc}\n\n"
        full_prompt += "---\n\n"
    
    # Add RAG context (knowledge base)
    if context_docs:
        full_prompt += "Relevant information from your knowledge base:\n\n"
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
        import time
        
        # Send initial metadata (context, citations)
        meta = {
            "type": "meta",
            "context_used": context_docs if context_docs else None,
            "citations": citations
        }
        yield f"data: {json.dumps(meta)}\n\n"
        
        # Stream tokens with timing
        full_response = ""
        token_count = 0
        start_time = time.time()
        first_token_time = None
        
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
    
    result = apply_settings(current)
    return {
        "status": "ok" if result["saved"] else "error",
        "result": result,
        "settings": get_all_settings()
    }


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
