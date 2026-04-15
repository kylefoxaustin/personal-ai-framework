"""
LLM Inference Server for Mixtral 8x7B
Uses llama-cpp-python for efficient GGUF model serving
Integrates with RAG pipeline for context-aware generation
"""
import gc
import os
# Google OAuth: accept scope supersets (happens when user has already granted
# related scopes, e.g. Gmail is connected and Calendar is then added).
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
import yaml
import shutil
import subprocess
import threading
from datetime import datetime
from typing import Optional, List, Dict
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
from agent_tools import execute_tool, get_tool_descriptions, set_web_search_fn, is_confirm_tool
import metrics

# Global advanced RAG instance
_advanced_rag = None

# Email config directory (per-user — reads current user from auth context)
def _email_dir() -> Path:
    from user_paths import user_dir
    from auth_ctx import get_current_username
    return user_dir(get_current_username())

def get_advanced_rag():
    global _advanced_rag
    if _advanced_rag is None:
        rag = get_rag_service()
        if rag:
            _advanced_rag = AdvancedRAG(rag)
    return _advanced_rag

import re


def _wrap_instruct(prompt: str) -> str:
    """Wrap prompt in ChatML template for single-shot calls (Qwen 2.5)."""
    return f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n<|im_start|>assistant\n"


def _rewrite_query_for_rag(query: str, conversation_history: Optional[List] = None) -> str:
    """Rewrite a user query into a better search query for RAG retrieval.

    Uses the LLM to expand vague references, resolve pronouns from conversation
    history, and produce a clear search-optimized query.
    Falls back to the original query on any error.
    """
    if llm is None or _maintenance_mode:
        return query

    # Include last 6 messages for context — more turns lets us resolve references
    # ("that", "it", "tell me more") across deeper conversational history.
    # Truncate older messages more aggressively to stay within budget.
    context_lines = ""
    if conversation_history:
        recent = conversation_history[-6:]
        for i, msg in enumerate(recent):
            role = "User" if msg.role == "user" else "Assistant"
            # More truncation for older turns, less for recent
            max_len = 100 if i < len(recent) - 2 else 250
            content = msg.content[:max_len]
            if len(msg.content) > max_len:
                content += "..."
            context_lines += f"{role}: {content}\n"

    if context_lines:
        prompt = f"""Rewrite this query to be more specific and search-friendly for a document retrieval system.
Use the conversation context to:
- Resolve pronouns and references ("that", "it", "the thing we discussed", "tell me more")
- Expand abbreviated subjects ("the processor" → full name if mentioned earlier)
- Maintain topic continuity from the conversation
Return ONLY the rewritten search query, nothing else.

Recent conversation:
{context_lines}
Current query: {query}

Rewritten query:"""
    else:
        prompt = f"""Rewrite this query to be more specific and search-friendly for a document retrieval system.
Return ONLY the rewritten query, nothing else. If the query is already clear, return it unchanged.

Query: {query}

Rewritten query:"""

    try:
        wrapped = _wrap_instruct(prompt)
        with _inference_lock:
            response = llm(
                wrapped,
                max_tokens=60,
                temperature=0.1,
                stop=["</s>", "\n\n", "<|im_end|>"],
                echo=False,
            )
        rewritten = response["choices"][0]["text"].strip().strip('"')
        if rewritten and len(rewritten) > 5:
            print(f"  🔄 Query rewrite: \"{query}\" → \"{rewritten}\"")
            return rewritten
    except Exception as e:
        print(f"  Query rewrite failed (using original): {e}")

    return query


def _extract_conversation_topics(conversation_history: Optional[List], max_terms: int = 8) -> List[str]:
    """Pull key terms from recent user turns to bias retrieval toward ongoing topics.

    Returns a short list of salient words (nouns, technical terms) that can be
    appended to a retrieval query when the current query alone is too vague.
    """
    if not conversation_history:
        return []
    _stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "this", "that", "these", "those", "what", "which",
        "who", "how", "when", "where", "why", "about", "with", "from", "for",
        "and", "but", "or", "not", "no", "so", "if", "then", "than", "too",
        "very", "just", "also", "more", "some", "any", "all", "each", "of",
        "in", "on", "at", "to", "by", "up", "out", "into", "over", "after",
        "tell", "know", "think", "want", "like", "said", "get", "got", "make",
        "going", "go", "see", "look", "give", "take", "come", "thing", "things",
    }
    import re
    from collections import Counter
    counts = Counter()
    # Only look at user turns from last 6 messages
    for msg in conversation_history[-6:]:
        if msg.role != "user":
            continue
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]{2,}", msg.content.lower())
        for w in words:
            if w not in _stop and len(w) > 2:
                counts[w] += 1
    return [w for w, _ in counts.most_common(max_terms)]


def _check_for_tool_use(
    prompt: str,
    conversation_history: Optional[List] = None,
    tool_history: Optional[List[dict]] = None,
):
    """Pass 1: Ask the LLM if it needs a tool to answer this question.

    When `tool_history` is provided (list of {name, params, result} from prior
    iterations of the agent loop), previous tool results are included in the
    detection prompt so the model can decide whether another tool is needed.

    Returns {"name": str, "params": dict, "result": str} if a tool was used,
    or None if no tool is needed. Uses a short, low-temperature inference.
    """
    if llm is None or _maintenance_mode:
        return None
    if _is_conversational(prompt) and not tool_history:
        return None

    tool_desc = get_tool_descriptions()

    # Include last user message for context on follow-ups
    context_hint = ""
    if conversation_history:
        for msg in reversed(conversation_history[-4:]):
            if msg.role == "user":
                context_hint = f"\nPrevious question: {msg.content[:150]}"
                break

    # Include prior tool calls + results in this chain, so the model knows
    # what it has already gathered and can decide what's next.
    history_hint = ""
    if tool_history:
        parts = ["\nTools already run in this chain (use their results; do not repeat them):"]
        for step in tool_history:
            preview = step.get("result", "")
            if len(preview) > 500:
                preview = preview[:500] + " [... truncated ...]"
            parts.append(f"- {step['name']}({step.get('params', {})}) → {preview}")
        history_hint = "\n".join(parts) + "\n"

    now_local = datetime.now().astimezone()
    now_str = now_local.strftime("%Y-%m-%dT%H:%M:%S%z")
    now_human = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")

    detection_prompt = f"""You have access to these tools:
{tool_desc}

Current date/time: {now_human} (ISO: {now_str})

Decide if you need a tool. Use a tool when the user asks to read a file, list a directory, search the web, check git status, write a file, run a script, schedule a reminder, send an email, create a calendar event, or list calendar events.
Write/schedule/send tools (write_file, run_script, schedule_reminder, send_email, create_calendar_event) will NOT execute automatically — they propose an action the user must approve.

Output format — `params` MUST be a flat JSON object whose keys are the parameter names themselves, NOT literal "key"/"value" fields.

Example 1 — user asks: Read /app/config.yaml
<tool>read_file</tool>
<params>{{"path": "/app/config.yaml"}}</params>

Example 2 — user asks: Write a file hello.md in the workspace with text: hi there
<tool>write_file</tool>
<params>{{"path": "hello.md", "content": "hi there"}}</params>

Example 2b — user asks: Write austin_facts.md with a bulleted list of facts
<tool>write_file</tool>
<params>{{"path": "austin_facts.md", "content": "# Austin Facts\\n\\n- Capital of Texas\\n- Known as the Live Music Capital of the World\\n- Home to the University of Texas\\n"}}</params>
(Note: write_file content is raw file text — no 'Subject:' line, no greetings. Match the file extension's format.)

Example 3 — user asks: Remind me tomorrow at 9am to call the dentist
<tool>schedule_reminder</tool>
<params>{{"text": "Call the dentist", "due_at": "<ISO 8601 for tomorrow 9am in user's timezone>"}}</params>

Example 4 — user asks: Email alice@example.com with subject "Meeting" and body "Let's sync Friday"
<tool>send_email</tool>
<params>{{"to": "alice@example.com", "subject": "Meeting", "body": "Hi Alice,\\n\\nLet's sync Friday — does the morning work for you?\\n\\nThanks,\\nKyle"}}</params>

Example 4b — after listing calendar, user wanted summary email. Compose a real message, do NOT paste the raw tool output:
<tool>send_email</tool>
<params>{{"to": "alice@example.com", "subject": "Your week ahead", "body": "Hi Alice,\\n\\nHere is a quick summary of your week:\\n- Tue 2pm: Test event\\n\\nLet me know if anything needs changing.\\n\\nBest,\\nKyle"}}</params>

Example 5 — user asks: Add a calendar event "Dentist" tomorrow 2pm-3pm
<tool>create_calendar_event</tool>
<params>{{"summary": "Dentist", "start": "<tomorrow 2pm ISO>", "end": "<tomorrow 3pm ISO>"}}</params>

Example 6 — user asks: What's on my calendar this week?
<tool>list_calendar_events</tool>
<params>{{"days": "7"}}</params>

Example 7 — user asks: What is the capital of France?
NO_TOOLS

Now for the real question below, output either a tool call in the exact format shown, or exactly NO_TOOLS.
If you have already gathered enough information from the tools already run, output NO_TOOLS so the assistant can respond to the user.
{history_hint}{context_hint}
User question: {prompt}"""

    try:
        wrapped = _wrap_instruct(detection_prompt)
        with _inference_lock:
            response = llm(
                wrapped,
                max_tokens=2048,  # Allow long bodies for write_file/send_email
                temperature=0.05,
                # Note: removed "\n\n\n" — multi-paragraph email/file bodies
                # legitimately contain blank-line gaps and would be truncated.
                # Parser slices the FIRST <params>...</params> pair, so trailing
                # output from a chatty model is harmless.
                stop=["</s>", "<|im_end|>"],
                echo=False,
            )

        text = response["choices"][0]["text"].strip()
        print(f"  [agent] Tool detection: {text[:120]}")

        if "NO_TOOLS" in text:
            return None

        # Parse the FIRST tool call. Slice-based so we only consume the first
        # <params>...</params> pair (model may emit multiple chained calls).
        tool_match = re.search(r"<tool>([\w_]+)</tool>", text)
        if not tool_match:
            return None
        tool_name = tool_match.group(1)

        params: dict = {}
        after_tool = text[tool_match.end():]
        ps = after_tool.find("<params>")
        pe = after_tool.find("</params>", ps + 1) if ps != -1 else -1
        if ps != -1 and pe != -1:
            raw = after_tool[ps + len("<params>"):pe].strip()
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  [agent] Failed to parse params: {raw[:200]}")
                return None

        # Scrub Qwen's email-header reflex for write_file content so the UI
        # preview matches what will actually be written.
        if tool_name == "write_file" and isinstance(params.get("content"), str):
            params["content"] = re.sub(r"^\s*Subject:[^\n]*\n+", "", params["content"], count=1)

        # Write-capable tools require explicit user confirmation — don't execute.
        if is_confirm_tool(tool_name):
            print(f"  ⏸  Pending confirmation: {tool_name}({params})")
            metrics.tool_calls_total.labels(tool=tool_name, outcome="pending").inc()
            return {"name": tool_name, "params": params, "pending": True}

        print(f"  🔧 Executing tool: {tool_name}({params})")
        result = execute_tool(tool_name, params)
        print(f"  [agent] Tool result: {len(result)} chars")

        outcome = "error" if result.startswith("Error:") else "executed"
        metrics.tool_calls_total.labels(tool=tool_name, outcome=outcome).inc()

        return {"name": tool_name, "params": params, "result": result}

    except Exception as e:
        print(f"  [agent] Tool detection error: {e}")
        return None


MAX_AGENT_STEPS = 5


def _pending_action_metadata(name: str, params: dict) -> dict:
    """
    Extra context to show alongside a pending confirm-tool action so the user
    can decide with full information. For write_file, flags whether the target
    file already exists and its size/mtime.
    """
    meta: dict = {}
    if name == "write_file":
        try:
            import agent_tools as _at
            from datetime import datetime as _dt
            _at.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            target = (_at.WORKSPACE_DIR / params.get("path", "")).resolve()
            if _at.WORKSPACE_DIR.resolve() in target.parents and target.exists() and target.is_file():
                st = target.stat()
                meta["existing_file"] = {
                    "size_bytes": st.st_size,
                    "mtime_iso": _dt.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                }
        except Exception:
            pass
    return meta


def _run_agent_loop(prompt: str, conversation_history: Optional[List] = None, max_steps: int = MAX_AGENT_STEPS):
    """
    Run up to `max_steps` iterations of tool detection + execution.

    Safe tools execute and feed back into the next detection pass. A confirm-tool
    breaks the loop and is surfaced as a pending action for user approval.

    Returns:
        {
            "steps": [ {name, params, result} ... ],  # completed safe-tool steps
            "pending": {name, params} or None,        # if chain stopped for approval
        }
    """
    steps: list = []
    for i in range(max_steps):
        result = _check_for_tool_use(prompt, conversation_history, tool_history=steps)
        if result is None:
            break
        if result.get("pending"):
            return {"steps": steps, "pending": {"name": result["name"], "params": result["params"]}}
        # Safe tool — record and continue
        steps.append({
            "name": result["name"],
            "params": result["params"],
            "result": result.get("result", ""),
        })
        print(f"  [agent] Loop step {i+1}/{max_steps} done: {result['name']}")
    return {"steps": steps, "pending": None}


def _build_tool_context(tool_result: dict) -> str:
    """Format a tool result for injection into the system context."""
    name = tool_result["name"]
    result = tool_result["result"]

    if len(result) > 8000:
        result = result[:8000] + "\n[... truncated ...]"

    # Wrap the result so the model uses it silently. Avoid leaking the
    # "TOOL RESULT" / "---" / "[planning]" scaffolding into the reply.
    return (
        "You just ran a tool on the user's behalf. The tool has finished — do NOT plan, "
        "do NOT describe what you are about to do, do NOT echo the tool name or a divider. "
        "Use the data below to answer the user's question directly in natural language.\n\n"
        f"<{name}_output>\n{result}\n</{name}_output>\n\n"
    )


def _build_tool_chain_context(steps: list) -> str:
    """Format an agent-loop chain of tool results for pass-2 system context."""
    if not steps:
        return ""
    parts = [
        "You just ran a chain of tools on the user's behalf. All results are below. "
        "Answer the user's question directly in natural language — do NOT echo the tool names, "
        "do NOT emit '[planning]' or 'TOOL RESULT' or '---' dividers, do NOT describe what you did.\n"
    ]
    for step in steps:
        name = step["name"]
        result = step.get("result", "")
        if len(result) > 6000:
            result = result[:6000] + "\n[... truncated ...]"
        parts.append(f"<{name}_output>\n{result}\n</{name}_output>")
    return "\n".join(parts) + "\n\n"


def _build_multiturn_prompt(
    system_context: str,
    conversation_history: Optional[List] = None,
    current_message: str = "",
) -> str:
    """Build a proper ChatML multi-turn prompt (Qwen 2.5).

    Format:
      <|im_start|>system
      {system context}<|im_end|>
      <|im_start|>user
      {user_1}<|im_end|>
      <|im_start|>assistant
      {assistant_1}<|im_end|>
      ...
      <|im_start|>user
      {current}<|im_end|>
      <|im_start|>assistant

    System context (personality, facts, memory, RAG) goes in the system block.
    """
    history = conversation_history or []
    history = history[-6:]  # Last 6 messages (3 turns)

    parts = []

    # System block with all context
    ctx = system_context.strip()
    if ctx:
        parts.append(f"<|im_start|>system\n{ctx}<|im_end|>")

    # Conversation history
    for msg in history:
        role = "user" if msg.role == "user" else "assistant"
        parts.append(f"<|im_start|>{role}\n{msg.content.strip()}<|im_end|>")

    # Current user message + assistant prompt
    parts.append(f"<|im_start|>user\n{current_message.strip()}<|im_end|>")
    parts.append("<|im_start|>assistant")

    return "\n".join(parts)


def _is_conversational(prompt: str) -> bool:
    """Check if a prompt is conversational/greeting rather than a knowledge query."""
    import re as _re
    p = prompt.strip().lower()
    tokens = p.split()
    # Match whole words/phrases, not substrings (fixes "this" matching "hi")
    single_words = {
        'hi', 'hello', 'hey', 'howdy', 'sup', 'yo',
        'thanks', 'bye', 'goodbye',
    }
    phrases = [
        'good morning', 'good afternoon', 'good evening', 'good night',
        'how are you', "what's up", 'whats up',
        'tell me about yourself', 'who are you', 'what are you',
        'what is your name', "what's your name", 'whats your name',
        'thank you',
    ]
    if len(tokens) <= 6:
        word_set = set(_re.findall(r"[a-z']+", p))
        if single_words & word_set:
            return True
        if any(phrase in p for phrase in phrases):
            return True
    # Very short messages (1-3 words) are likely conversational
    if len(tokens) <= 3 and not any(c in p for c in ['?', 'how to', 'what is', 'explain', 'find']):
        return True
    return False


_synthesis_patterns = [
    "summarize everything", "summarize all", "compile all", "compile everything",
    "what do i know about", "what do we know about", "everything i know about",
    "everything we know about", "all info on", "all information on",
    "give me an overview", "give an overview", "broad overview",
    "what does my knowledge base", "what's in my knowledge base",
    "across all documents", "across my documents", "across all my",
    "comprehensive summary", "full summary",
]


def _is_synthesis_query(prompt: str) -> bool:
    """Detect broad synthesis requests that should pull from many documents."""
    p = prompt.strip().lower()
    return any(pat in p for pat in _synthesis_patterns)


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
    pending_action: Optional[dict] = None  # {name, params} — write tool awaiting user approval

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
    # LAN-only app — echo any origin back so session cookies work across
    # localhost/127.0.0.1/LAN-IP. Can't use allow_origins=["*"] with
    # allow_credentials=True per the CORS spec.
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run one-time legacy → multi-user migration before any service touches disk.
from migrate_to_multiuser import migrate as _migrate_multiuser
_migrate_multiuser()

# ── Auth ──
import user_service
import auth_ctx
from fastapi import Request, Response, HTTPException, Cookie, Depends

SESSION_COOKIE = "skippy_session"

# Paths that skip auth (so the login page + OAuth callbacks can work).
_PUBLIC_EXACT = {
    "/", "/docs", "/openapi.json", "/redoc", "/favicon.ico",
    "/auth/login", "/auth/logout", "/auth/bootstrap", "/auth/needs-bootstrap",
    "/metrics", "/calendar/oauth-callback",
}
_PUBLIC_PREFIXES = ("/email/oauth-callback/",)  # /email/oauth-callback/gmail etc.


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


def _extract_token(request: Request) -> Optional[str]:
    """Accept session token via Authorization: Bearer or session cookie."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return request.cookies.get(SESSION_COOKIE)


@app.middleware("http")
async def attach_user_ctx(request: Request, call_next):
    """
    Set auth_ctx.current_user from the bearer token or session cookie; reject
    unauthenticated requests to non-public paths. This is the blanket auth
    gate — individual endpoints can add Depends(require_admin) for stricter
    checks.
    """
    token = _extract_token(request)
    user = user_service.session_user(token) if token else None
    path = request.url.path
    if not user and not _is_public(path):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    token_ctx = auth_ctx.current_user.set(user["username"] if user else None)
    try:
        response = await call_next(request)
    finally:
        auth_ctx.current_user.reset(token_ctx)
    return response


def require_auth(request: Request) -> dict:
    """Dependency: resolve the current user or 401."""
    token = _extract_token(request)
    user = user_service.session_user(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(user: dict = Depends(require_auth)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin required")
    return user


# Global model instance
llm = None
_maintenance_mode = False
_current_model_path = None

def load_model(model_path_override=None):
    """Load model with GPU acceleration"""
    global llm, _current_model_path

    # Unload any existing model first to free GPU memory
    unload_model()

    model_path = model_path_override or "/app/models/qwen2.5-14b/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"

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

    # Persist active model path so it survives container restarts
    try:
        settings = load_settings()
        settings.setdefault("model", {})["active_model_path"] = model_path
        save_settings(settings)
    except Exception as e:
        print(f"  Could not persist model path: {e}")

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
    """Load model on server start, restoring last-used model if available."""
    model_path = None
    try:
        settings = load_settings()
        saved_path = settings.get("model", {}).get("active_model_path")
        if saved_path and os.path.exists(saved_path):
            print(f"Restoring last-used model: {saved_path}")
            model_path = saved_path
    except Exception:
        pass
    load_model(model_path)
    
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
    return {"message": "Personal AI LLM Server", "model": "Qwen2.5-14B", "rag_enabled": True}

@app.post("/generate", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    """Generate text based on prompt and context"""

    if _maintenance_mode:
        raise HTTPException(status_code=503, detail="Model is retraining. Please wait — this usually takes ~2.5 hours.")
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import time as _time
    _gen_start = _time.time()

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

    # Rewrite query for better RAG retrieval (skipped for conversational queries)
    retrieval_query = request.prompt
    if request.use_rag and not context_docs and not _is_conversational(request.prompt):
        retrieval_query = _rewrite_query_for_rag(request.prompt, request.conversation_history)
        # For short/vague queries, append conversation topic terms to improve recall
        if len(retrieval_query.split()) <= 6 and request.conversation_history:
            topics = _extract_conversation_topics(request.conversation_history)
            if topics:
                retrieval_query = f"{retrieval_query} {' '.join(topics[:5])}"
                print(f"  📎 Topic-boosted query: \"{retrieval_query}\"")

    # Auto-retrieve context from RAG if enabled and no manual context provided
    if request.use_rag and not context_docs:
        try:
            # Search conversation memory
            try:
                memory = get_memory_service()
                memory_docs, memory_citations = memory.get_memory_context(retrieval_query, k=2)
                if memory_docs:
                    memory_context = memory_docs
                    print(f"Found {len(memory_docs)} relevant past conversations")
            except Exception as e:
                print(f"Memory search failed (ok if empty): {e}")

            # Search knowledge base — use more results for synthesis queries
            synthesis_mode = _is_synthesis_query(request.prompt)
            rag_k = max(8, request.rag_k * 3) if synthesis_mode else request.rag_k
            if synthesis_mode:
                print(f"  🔬 Synthesis mode: expanding k from {request.rag_k} to {rag_k}")
            advanced_rag = get_advanced_rag()
            if advanced_rag:
                context_docs, citations = advanced_rag.get_context_with_citations(
                    retrieval_query, k=rag_k
                )
            else:
                rag = get_rag_service()
                context_docs = rag.get_context(retrieval_query, k=rag_k)
            metrics.rag_queries_total.labels(mode="synthesis" if synthesis_mode else "standard").inc()
            metrics.rag_docs_returned.observe(len(context_docs) if context_docs else 0)
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
        if _is_synthesis_query(request.prompt) and citations:
            # Group chunks by source for cross-document synthesis
            from collections import OrderedDict
            groups = OrderedDict()
            for i, doc in enumerate(context_docs):
                src = citations[i]["source_file"] if i < len(citations) else "unknown"
                src_name = src.split("/")[-1] if "/" in src else src
                groups.setdefault(src_name, []).append(doc)
            system_context += (
                "Information from MULTIPLE sources in the knowledge base. "
                "Synthesize across these sources — note where sources agree, "
                "disagree, or complement each other:\n\n"
            )
            for src, docs in groups.items():
                system_context += f"── {src} ──\n"
                for doc in docs:
                    system_context += f"{doc}\n\n"
            system_context += "---\n\n"
        else:
            system_context += "Relevant information from your knowledge base:\n\n"
            for doc in context_docs:
                system_context += f"{doc}\n\n"
            system_context += "---\n\n"

    # Agent loop — up to MAX_AGENT_STEPS safe-tool iterations, then either
    # continue or stop for confirm-tool approval.
    agent = _run_agent_loop(request.prompt, request.conversation_history)
    if agent["pending"]:
        name = agent["pending"]["name"]
        params = agent["pending"]["params"]
        preview_parts = [f"{k}={repr(v)[:60]}" for k, v in params.items() if k != "content"]
        if "content" in params:
            preview_parts.append(f"content=[{len(str(params['content']))} chars]")
        prefix = f"After running {len(agent['steps'])} prior step(s), " if agent["steps"] else ""
        msg = f"{prefix}I'd like to run `{name}({', '.join(preview_parts)})`. Please approve or deny below."
        return GenerationResponse(
            text=msg,
            tokens_used=0,
            model="qwen2.5-14b",
            pending_action={"name": name, "params": params, "meta": _pending_action_metadata(name, params)},
        )
    if agent["steps"]:
        system_context += _build_tool_chain_context(agent["steps"])

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
            stop=["</s>", "\n\n\n", "<|im_end|>"],
            echo=False
        )

    _elapsed = _time.time() - _gen_start
    metrics.generations_total.labels(endpoint="generate").inc()
    metrics.generation_latency_seconds.labels(endpoint="generate").observe(_elapsed)
    _tokens = response['usage']['total_tokens']
    if _elapsed > 0 and _tokens > 0:
        metrics.generation_tokens_per_second.observe(_tokens / _elapsed)

    return GenerationResponse(
        text=response['choices'][0]['text'].strip(),
        tokens_used=_tokens,
        model="qwen2.5-14b",
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

    # Rewrite query for better RAG retrieval (skipped for conversational queries)
    retrieval_query = request.prompt
    if request.use_rag and not context_docs and not _is_conversational(request.prompt):
        retrieval_query = _rewrite_query_for_rag(request.prompt, request.conversation_history)
        # For short/vague queries, append conversation topic terms to improve recall
        if len(retrieval_query.split()) <= 6 and request.conversation_history:
            topics = _extract_conversation_topics(request.conversation_history)
            if topics:
                retrieval_query = f"{retrieval_query} {' '.join(topics[:5])}"
                print(f"  📎 Topic-boosted query: \"{retrieval_query}\"")

    # Auto-retrieve context from RAG if enabled
    if request.use_rag and not context_docs:
        try:
            # Search conversation memory
            try:
                memory = get_memory_service()
                memory_docs, memory_citations = memory.get_memory_context(retrieval_query, k=2)
                if memory_docs:
                    memory_context = memory_docs
                    print(f"Found {len(memory_docs)} relevant past conversations")
            except Exception as e:
                print(f"Memory search failed (ok if empty): {e}")

            # Search knowledge base — use more results for synthesis queries
            synthesis_mode = _is_synthesis_query(request.prompt)
            rag_k = max(8, request.rag_k * 3) if synthesis_mode else request.rag_k
            if synthesis_mode:
                print(f"  🔬 Synthesis mode: expanding k from {request.rag_k} to {rag_k}")
            advanced_rag = get_advanced_rag()
            if advanced_rag:
                context_docs, citations = advanced_rag.get_context_with_citations(
                    retrieval_query, k=rag_k
                )
            else:
                rag = get_rag_service()
                context_docs = rag.get_context(retrieval_query, k=rag_k)
            metrics.rag_queries_total.labels(mode="synthesis" if synthesis_mode else "standard").inc()
            metrics.rag_docs_returned.observe(len(context_docs) if context_docs else 0)
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
        if _is_synthesis_query(request.prompt) and citations:
            # Group chunks by source for cross-document synthesis
            from collections import OrderedDict
            groups = OrderedDict()
            for i, doc in enumerate(context_docs):
                src = citations[i]["source_file"] if i < len(citations) else "unknown"
                src_name = src.split("/")[-1] if "/" in src else src
                groups.setdefault(src_name, []).append(doc)
            system_context += (
                "Information from MULTIPLE sources in the knowledge base. "
                "Synthesize across these sources — note where sources agree, "
                "disagree, or complement each other:\n\n"
            )
            for src, docs in groups.items():
                system_context += f"── {src} ──\n"
                for doc in docs:
                    system_context += f"{doc}\n\n"
            system_context += "---\n\n"
        else:
            system_context += "Relevant information from your knowledge base:\n\n"
            for doc in context_docs:
                system_context += f"{doc}\n\n"
            system_context += "---\n\n"

    # Agent loop — run up to MAX_AGENT_STEPS iterations of safe tools, then
    # either continue to the final generation or stop for a confirm-tool approval.
    agent = _run_agent_loop(request.prompt, request.conversation_history)
    completed_steps = agent["steps"]
    pending = agent["pending"]

    if pending:
        name = pending["name"]
        params = pending["params"]
        preview_parts = [f"{k}={repr(v)[:60]}" for k, v in params.items() if k != "content"]
        if "content" in params:
            preview_parts.append(f"content=[{len(str(params['content']))} chars]")
        prefix = f"After running {len(completed_steps)} prior step(s), " if completed_steps else ""
        msg = f"{prefix}I'd like to run `{name}({', '.join(preview_parts)})`. Please approve or deny below."

        def pending_stream():
            # Emit any prior safe-tool steps first so the UI can show them
            for step in completed_steps:
                yield f"data: {json.dumps({'type': 'tool_use', 'tool': step['name'], 'params': step['params'], 'result': step['result']})}\n\n"
            yield f"data: {json.dumps({'type': 'pending_action', 'name': name, 'params': params, 'meta': _pending_action_metadata(name, params)})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'text': msg, 'metrics': {}})}\n\n"

        return StreamingResponse(pending_stream(), media_type="text/event-stream")

    if completed_steps:
        system_context += _build_tool_chain_context(completed_steps)

    # Build multi-turn prompt with proper [INST] format
    full_prompt = _build_multiturn_prompt(system_context, request.conversation_history, request.prompt)

    # Re-check model availability (may have been unloaded during context retrieval)
    if _maintenance_mode or llm is None:
        raise HTTPException(status_code=503, detail="Model is retraining. Please wait — this usually takes ~2.5 hours.")

    def generate_tokens():
        import time

        # Send one tool-use event per completed step in the agent loop
        for step in completed_steps:
            trace = {
                "type": "tool_use",
                "tool": step["name"],
                "params": step["params"],
                "result": step.get("result", ""),
            }
            yield f"data: {json.dumps(trace)}\n\n"

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
                stop=["</s>", "\n\n\n", "<|im_end|>"],
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

        metrics.generations_total.labels(endpoint="stream").inc()
        metrics.generation_latency_seconds.labels(endpoint="stream").observe(total_time)
        if first_token_time is not None:
            metrics.generation_ttft_seconds.observe(ttft)
        if throughput > 0:
            metrics.generation_tokens_per_second.observe(throughput)

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
    path = model_path or _current_model_path or "/app/models/qwen2.5-14b/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
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
async def upload_email_credentials(provider: str, file: UploadFile = File(...), user: dict = Depends(require_auth)):
    """Upload OAuth credentials file for Gmail or Outlook"""
    if provider not in ["gmail", "outlook"]:
        raise HTTPException(status_code=400, detail="Provider must be 'gmail' or 'outlook'")

    _email_dir().mkdir(exist_ok=True)
    
    if provider == "gmail":
        dest = _email_dir() / "gmail_credentials.json"
    else:
        dest = _email_dir() / "outlook_credentials.json"
    
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"status": "ok", "message": f"{provider} credentials uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/email/auth-url/{provider}")
def get_auth_url(provider: str, user: dict = Depends(require_auth)):
    """Get OAuth authorization URL"""
    if provider == "gmail":
        creds_file = _email_dir() / "gmail_credentials.json"
        if not creds_file.exists():
            raise HTTPException(status_code=400, detail="Upload credentials first")

        try:
            import secrets as _secrets
            from google_auth_oauthlib.flow import Flow

            with open(creds_file) as f:
                creds_data = json.load(f)

            flow = Flow.from_client_config(
                creds_data,
                scopes=['https://www.googleapis.com/auth/gmail.send',
                        'https://www.googleapis.com/auth/gmail.compose'],
                redirect_uri='http://localhost:8080/email/oauth-callback/gmail',
                state=f"{user['username']}:{_secrets.token_urlsafe(16)}",
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
            state_file = _email_dir() / "gmail_oauth_state.json"
            with open(state_file, 'w') as f:
                json.dump(oauth_state, f)
            
            return {"auth_url": auth_url, "state": state}
        except ImportError:
            raise HTTPException(status_code=500, detail="google-auth-oauthlib not installed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    elif provider == "outlook":
        creds_file = _email_dir() / "outlook_credentials.json"
        if not creds_file.exists():
            raise HTTPException(status_code=400, detail="Upload credentials first")

        try:
            import secrets as _secrets
            with open(creds_file) as f:
                config = json.load(f)

            client_id = config.get('client_id')
            redirect_uri = 'http://localhost:8080/email/oauth-callback/outlook'
            scope = 'https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/Mail.ReadWrite offline_access'
            state = f"{user['username']}:{_secrets.token_urlsafe(16)}"

            auth_url = (
                f"https://login.microsoftonline.com/consumers/oauth2/v2.0/authorize"
                f"?client_id={client_id}"
                f"&response_type=code"
                f"&redirect_uri={redirect_uri}"
                f"&scope={scope}"
                f"&state={state}"
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
    
    # Public callback: identify user from the state prefix (set in get_auth_url).
    if not state or ":" not in state:
        return HTMLResponse("<html><body><h2>❌ Missing or malformed state</h2></body></html>")
    u_from_state = state.split(":", 1)[0]
    from user_paths import user_dir as _ud
    _user_dir = _ud(u_from_state)

    try:
        if provider == "gmail":
            from google_auth_oauthlib.flow import Flow
            import pickle

            state_file = _user_dir / "gmail_oauth_state.json"
            if not state_file.exists():
                raise Exception("OAuth flow expired. Please try again.")

            with open(state_file) as f:
                oauth_state = json.load(f)

            creds_file = _user_dir / "gmail_credentials.json"
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
            token_file = _user_dir / "gmail_token.pickle"
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

            creds_file = _user_dir / "outlook_credentials.json"
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
                token_file = _user_dir / "outlook_token.json"
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
        filepath = _email_dir() / filename
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
    from google.auth.transport.requests import Request

    token_file = _email_dir() / "gmail_token.pickle"
    if not token_file.exists():
        raise HTTPException(status_code=400, detail="Gmail not connected")

    with open(token_file, 'rb') as f:
        creds = pickle.load(f)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, 'wb') as f:
            pickle.dump(creds, f)

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
    from google.auth.transport.requests import Request

    token_file = _email_dir() / "gmail_token.pickle"
    if not token_file.exists():
        raise HTTPException(status_code=400, detail="Gmail not connected")

    with open(token_file, 'rb') as f:
        creds = pickle.load(f)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file, 'wb') as f:
            pickle.dump(creds, f)

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


# ─────────────── Metrics ───────────────

from fastapi import Response


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus text-format metrics scrape endpoint."""
    # Refresh gauges that are cheap to compute on-demand.
    metrics.model_loaded.set(1 if llm is not None else 0)
    metrics.maintenance_mode.set(1 if _maintenance_mode else 0)
    try:
        stats = get_rag_service().get_stats()
        metrics.knowledge_base_docs.set(stats.get("document_count", 0))
    except Exception:
        pass
    return Response(content=metrics.render_text(), media_type="text/plain; version=0.0.4")


# ─────────────── System / Phone Access helpers ───────────────

@app.get("/system/lan-ips")
def system_lan_ips():
    """
    Return LAN IPv4 addresses of the host, written at container start by
    `run.sh` to ~/.personal-ai/lan_ips.txt. Filters out docker-bridge and
    loopback addresses so the UI gets the phone-reachable ones.
    """
    ip_file = Path("/root/.personal-ai/lan_ips.txt")
    if not ip_file.exists():
        return {"ips": [], "note": "Run ./run.sh start to refresh LAN IP list"}
    try:
        raw = [ln.strip() for ln in ip_file.read_text().splitlines() if ln.strip()]
    except Exception:
        return {"ips": [], "note": "Could not read LAN IP file"}

    def is_real_lan(ip: str) -> bool:
        if ip.startswith("127.") or ip.startswith("172.17.") or ip.startswith("172.18."):
            return False
        if ip.startswith("169.254."):
            return False
        return True

    return {"ips": [ip for ip in raw if is_real_lan(ip)]}


@app.get("/system/qr")
def system_qr(data: str):
    """Return an SVG QR code for `data`. Used by the phone-access panel."""
    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        raise HTTPException(status_code=500, detail="qrcode library not installed")
    factory = qrcode.image.svg.SvgImage
    img = qrcode.make(data, image_factory=factory, box_size=10, border=2)
    import io
    buf = io.BytesIO()
    img.save(buf)
    svg = buf.getvalue().decode()
    return HTMLResponse(content=svg, media_type="image/svg+xml")


# ─────────────── Training Dashboard ───────────────

TRAINING_OUTPUT_DIR = Path("/app/../training/output")


@app.get("/training/runs")
def training_runs():
    """
    Enumerate LoRA training checkpoints in training/output/ and summarize each.
    """
    import glob
    # Path inside the container is /app; host training/ is not mounted by default.
    # Try a few likely locations so this works in both dev and container.
    candidates = [
        Path("/app/training/output"),
        Path("/app/../training/output"),
        Path("/training/output"),
    ]
    out_dir = next((p for p in candidates if p.exists()), None)
    if out_dir is None:
        return {"runs": [], "output_dir": None, "note": "training/output not mounted in container"}

    runs = []
    for sub in sorted(out_dir.iterdir()):
        if not sub.is_dir():
            continue
        state = sub / "trainer_state.json"
        if not state.exists():
            continue
        try:
            with open(state) as f:
                data = json.load(f)
        except Exception:
            continue
        log = data.get("log_history", [])
        train_losses = [x.get("loss") for x in log if "loss" in x and "eval_loss" not in x]
        eval_losses = [x.get("eval_loss") for x in log if "eval_loss" in x]
        runs.append({
            "name": sub.name,
            "global_step": data.get("global_step"),
            "epochs": data.get("num_train_epochs"),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_eval_loss": eval_losses[-1] if eval_losses else None,
            "log_entries": len(log),
            "mtime_iso": datetime.fromtimestamp(state.stat().st_mtime).isoformat(timespec="seconds"),
        })
    return {"runs": runs, "output_dir": str(out_dir)}


@app.get("/training/run/{name}")
def training_run_detail(name: str):
    """Return trainer_state.log_history for plotting."""
    # Same candidate search as above; and sanitize `name` (no '..').
    if ".." in name or "/" in name:
        raise HTTPException(status_code=400, detail="Invalid checkpoint name")
    candidates = [
        Path("/app/training/output"),
        Path("/app/../training/output"),
        Path("/training/output"),
    ]
    out_dir = next((p for p in candidates if p.exists()), None)
    if out_dir is None:
        raise HTTPException(status_code=404, detail="training/output not available")
    state = out_dir / name / "trainer_state.json"
    if not state.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    with open(state) as f:
        data = json.load(f)
    log = data.get("log_history", [])
    return {
        "name": name,
        "global_step": data.get("global_step"),
        "epochs": data.get("num_train_epochs"),
        "series": [
            {"step": x.get("step"), "loss": x.get("loss"), "eval_loss": x.get("eval_loss"), "lr": x.get("learning_rate")}
            for x in log
        ],
    }


# ─────────────── Feedback (👍/👎) ───────────────

# feedback helpers imported below with get_feedback_store


class FeedbackRequest(PydanticBaseModel):
    message_id: int
    rating: str  # 'up' or 'down'
    comment: Optional[str] = None


from feedback_service import get_feedback_store


@app.post("/feedback")
def feedback_rate(req: FeedbackRequest):
    try:
        result = get_feedback_store().rate(req.message_id, req.rating, req.comment)
        metrics.feedback_total.labels(rating=req.rating).inc()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/feedback/{message_id}")
def feedback_clear(message_id: int):
    ok = get_feedback_store().clear(message_id)
    return {"cleared": ok}


@app.get("/feedback/stats")
def feedback_stats():
    return get_feedback_store().stats()


@app.get("/feedback/history")
def feedback_history(days: int = 30):
    return {"days": days, "buckets": get_feedback_store().history(days)}


@app.get("/feedback/{message_id}")
def feedback_get(message_id: int):
    row = get_feedback_store().get(message_id)
    return row or {"message_id": message_id, "rating": None}


# ─────────────── Reminders ───────────────

from reminder_service import get_reminder_store


class ReminderCreate(PydanticBaseModel):
    text: str
    due_at: str


@app.post("/reminders")
def reminders_create(req: ReminderCreate):
    try:
        result = get_reminder_store().schedule(req.text, req.due_at)
        metrics.reminders_created_total.inc()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid due_at: {e}")


@app.get("/reminders/upcoming")
def reminders_upcoming(limit: int = 50):
    return {"reminders": get_reminder_store().list_upcoming(limit=limit)}


@app.get("/reminders/due")
def reminders_due():
    return {"reminders": get_reminder_store().get_due_unacked()}


@app.post("/reminders/{reminder_id}/ack")
def reminders_ack(reminder_id: int):
    ok = get_reminder_store().ack(reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found or already acked")
    metrics.reminders_fired_total.inc()
    return {"status": "ok"}


@app.delete("/reminders/{reminder_id}")
def reminders_cancel(reminder_id: int):
    ok = get_reminder_store().cancel(reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found or already closed")
    return {"status": "ok"}


# ─────────────── Agent Tool Execution ───────────────

class AgentExecuteRequest(PydanticBaseModel):
    name: str
    params: Dict = {}


@app.get("/agent/next-free-path")
def agent_next_free_path(path: str):
    """
    Return the first non-colliding variant of `path` inside the workspace.
    If 'austin_facts.md' exists, returns 'austin_facts_01.md'; if that
    exists too, tries _02, _03, ... Safe against path traversal.
    """
    import agent_tools as _at
    _at.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        target = (_at.WORKSPACE_DIR / path).resolve()
        if _at.WORKSPACE_DIR.resolve() not in target.parents and _at.WORKSPACE_DIR.resolve() != target.parent:
            raise HTTPException(status_code=400, detail="Path escapes workspace")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not target.exists():
        return {"path": path}

    stem = target.stem
    suffix = target.suffix
    parent_rel = target.parent.relative_to(_at.WORKSPACE_DIR.resolve())
    for i in range(1, 100):
        candidate_name = f"{stem}_{i:02d}{suffix}"
        candidate = target.parent / candidate_name
        if not candidate.exists():
            rel = (parent_rel / candidate_name).as_posix()
            if rel.startswith("./"):
                rel = rel[2:]
            return {"path": rel}
    raise HTTPException(status_code=409, detail="Too many variants exist")


@app.post("/agent/execute")
def agent_execute(req: AgentExecuteRequest):
    """
    Execute a tool on behalf of the user after they approved it in the UI.
    Used for write-capable tools that require confirmation (write_file, run_script).
    Also works for safe tools if the UI wants to invoke them directly.
    """
    try:
        result = execute_tool(req.name, req.params, require_safe=False)
        outcome = "error" if result.startswith("Error:") else "executed"
        metrics.tool_calls_total.labels(tool=req.name, outcome=outcome).inc()
        return {"name": req.name, "params": req.params, "result": result}
    except Exception as e:
        metrics.tool_calls_total.labels(tool=req.name, outcome="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────── Google Calendar Endpoints ───────────────

import calendar_service


class CalendarEventRequest(PydanticBaseModel):
    summary: str
    start: str
    end: str
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: Optional[List[str]] = None


@app.get("/calendar/status")
def calendar_status(user: dict = Depends(require_auth)):
    u = user["username"]
    return {
        "configured": calendar_service.is_configured(u),
        "connected": calendar_service.is_connected(u),
    }


@app.get("/calendar/auth-url")
def calendar_auth_url(user: dict = Depends(require_auth)):
    """Start Google Calendar OAuth. Requires gmail_credentials.json present."""
    u = user["username"]
    if not calendar_service.is_configured(u):
        raise HTTPException(
            status_code=400,
            detail="Upload Google OAuth credentials first (same file used for Gmail)",
        )

    try:
        import secrets as _secrets
        from google_auth_oauthlib.flow import Flow

        with open(calendar_service.creds_path(u)) as f:
            creds_data = json.load(f)

        flow = Flow.from_client_config(
            creds_data,
            scopes=calendar_service.SCOPES,
            redirect_uri="http://localhost:8080/calendar/oauth-callback",
            state=f"{u}:{_secrets.token_urlsafe(16)}",
        )

        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        with open(calendar_service.oauth_state_path(u), "w") as f:
            json.dump({"state": state, "code_verifier": flow.code_verifier}, f)

        return {"auth_url": auth_url, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calendar/oauth-callback")
def calendar_oauth_callback(code: str = None, state: str = None, error: str = None):
    """Public endpoint — Google redirects here. User identified via `state` prefix."""
    if error:
        return HTMLResponse(
            f"""<html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h2>❌ Authentication Failed</h2><p>{error}</p>
            <script>setTimeout(() => window.close(), 3000);</script></body></html>"""
        )
    if not code or not state or ":" not in state:
        return HTMLResponse(
            """<html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h2>❌ Missing code or state</h2></body></html>"""
        )

    u = state.split(":", 1)[0]
    try:
        import pickle
        from google_auth_oauthlib.flow import Flow

        state_path = calendar_service.oauth_state_path(u)
        if not state_path.exists():
            raise Exception("OAuth flow expired. Please try again.")

        with open(state_path) as f:
            oauth_state = json.load(f)

        with open(calendar_service.creds_path(u)) as f:
            creds_data = json.load(f)

        flow = Flow.from_client_config(
            creds_data,
            scopes=calendar_service.SCOPES,
            redirect_uri="http://localhost:8080/calendar/oauth-callback",
        )
        flow.code_verifier = oauth_state.get("code_verifier")
        flow.fetch_token(code=code)

        state_path.unlink()

        with open(calendar_service.token_path(u), "wb") as f:
            pickle.dump(flow.credentials, f)

        return HTMLResponse(
            """<html><body style="font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: #eee;">
            <h2 style="color: #4ade80;">✅ Calendar Connected!</h2>
            <p>You can close this window.</p>
            <script>
                window.opener && window.opener.postMessage({type: 'oauth-success', provider: 'calendar'}, '*');
                setTimeout(() => window.close(), 1500);
            </script></body></html>"""
        )
    except Exception as e:
        return HTMLResponse(
            f"""<html><body style="font-family: sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: #eee;">
            <h2 style="color: #f87171;">❌ Authentication Error</h2><p>{str(e)}</p>
            <script>setTimeout(() => window.close(), 5000);</script></body></html>"""
        )


@app.delete("/calendar/disconnect")
def calendar_disconnect(user: dict = Depends(require_auth)):
    calendar_service.disconnect(user["username"])
    return {"status": "ok", "message": "calendar disconnected"}


@app.get("/calendar/events")
def calendar_events(days: int = 7, max_results: int = 25, user: dict = Depends(require_auth)):
    u = user["username"]
    if not calendar_service.is_connected(u):
        raise HTTPException(status_code=400, detail="Calendar not connected")
    try:
        return {"events": calendar_service.list_events(u, days=days, max_results=max_results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calendar/create")
def calendar_create(req: CalendarEventRequest, user: dict = Depends(require_auth)):
    u = user["username"]
    if not calendar_service.is_connected(u):
        raise HTTPException(status_code=400, detail="Calendar not connected")
    try:
        return calendar_service.create_event(
            u,
            summary=req.summary,
            start=req.start,
            end=req.end,
            description=req.description,
            location=req.location,
            attendees=req.attendees,
        )
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


@app.get("/conversations/search")
def search_conversations(q: str, limit: int = 20):
    """Search conversations by message content"""
    store = get_conversation_store()
    return {"conversations": store.search_conversations(q, limit)}


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    """Get full conversation with messages"""
    store = get_conversation_store()
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/conversations/{conv_id}/exclude-from-training")
def exclude_conversation(conv_id: str):
    get_conversation_store().set_excluded(conv_id, True)
    return {"status": "ok", "excluded": True}


@app.delete("/conversations/{conv_id}/exclude-from-training")
def include_conversation(conv_id: str):
    get_conversation_store().set_excluded(conv_id, False)
    return {"status": "ok", "excluded": False}


@app.post("/conversations/{conv_id}/messages")
def add_message(conv_id: str, role: str, content: str, tokens_used: int = None):
    """Add message to conversation. Returns the new message's id."""
    store = get_conversation_store()
    message_id = store.add_message(conv_id, role, content, tokens_used)
    return {"status": "ok", "message_id": message_id}


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
        store = get_conversation_store()
        conn = store._conn()
        conn.execute("UPDATE conversations SET ingested_to_rag = FALSE WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()
        
        chunks = memory.ingest_conversation(conv_id)
        return {"status": "ok", "chunks": chunks}
    except Exception as e:
        return {"status": "error", "error": str(e)}




@app.get("/memory/list")
def list_memories(limit: int = 100, offset: int = 0):
    """List all memories stored in ChromaDB (per-user collection)"""
    try:
        rag = get_memory_service().rag

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
    """Delete a specific memory entry from ChromaDB (per-user collection)"""
    try:
        rag = get_memory_service().rag
        rag.collection.delete(ids=[memory_id])
        
        return {"status": "deleted", "id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@app.get("/memory/stats")
def memory_stats():
    """Get memory statistics (this user's conversation memory + shared knowledge base)"""
    try:
        mem_rag = get_memory_service().rag
        results = mem_rag.collection.get(
            where={"source_type": "conversation_memory"},
            include=[]
        )
        memory_count = len(results['ids']) if results and results['ids'] else 0

        store = get_conversation_store()
        convs = store.list_conversations(limit=1000)

        from rag_service import get_rag_service
        kb = get_rag_service()

        return {
            "memory_chunks": memory_count,
            "conversations": len(convs),
            "total_knowledge_docs": kb.collection.count()
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

@app.post("/voice/transcribe")
async def voice_transcribe(file: UploadFile = File(...)):
    """Lightweight voice transcription — mic audio to text, no ingestion."""
    import tempfile

    suffix = Path(file.filename or "audio.webm").suffix.lower() or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name

    try:
        from meeting_summarizer import MeetingSummarizer
        summarizer = MeetingSummarizer(whisper_model="base", device="cpu")
        transcript_text, _ = summarizer.transcribe(tmp_path)
        return {"status": "ok", "text": transcript_text.strip()}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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


# Wire up web_search tool to use existing _ddg_search
set_web_search_fn(_ddg_search)


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
            response = llm(prompt, max_tokens=1000, temperature=0.3, stop=["</s>", "\n\n\n", "<|im_end|>"])
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
                    fact_response = llm(fact_prompt, max_tokens=100, temperature=0.1, stop=["</s>", "\n\n", "<|im_end|>"])
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


# ── Auth + user-admin endpoints ──

class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class BootstrapRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False


@app.get("/auth/needs-bootstrap")
def auth_needs_bootstrap():
    """If there are no users yet, the UI should show a first-run admin create form."""
    return {"needs_bootstrap": user_service.user_count() == 0}


@app.post("/auth/bootstrap")
def auth_bootstrap(req: BootstrapRequest):
    """First-run endpoint: creates the initial admin account. Disabled once any user exists."""
    if user_service.user_count() > 0:
        raise HTTPException(status_code=409, detail="Users already exist; bootstrap disabled")
    try:
        user = user_service.create_user(req.username, req.password, is_admin=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = user_service.create_session(req.username)
    return {"user": user, "token": token}


@app.post("/auth/login")
def auth_login(req: LoginRequest):
    if not user_service.verify_password(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = user_service.create_session(req.username)
    user = user_service.session_user(token)
    return {"user": user, "token": token}


@app.post("/auth/logout")
def auth_logout(request: Request):
    token = _extract_token(request)
    if token:
        user_service.revoke_session(token)
    return {"ok": True}


@app.get("/auth/me")
def auth_me(user: dict = Depends(require_auth)):
    return {"user": user}


@app.post("/auth/change-password")
def auth_change_password(req: ChangePasswordRequest, user: dict = Depends(require_auth)):
    if not user_service.verify_password(user["username"], req.current_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    try:
        user_service.set_password(user["username"], req.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.get("/users")
def admin_list_users(user: dict = Depends(require_admin)):
    return {"users": user_service.list_users()}


@app.post("/users")
def admin_create_user(req: CreateUserRequest, user: dict = Depends(require_admin)):
    try:
        created = user_service.create_user(req.username, req.password, is_admin=req.is_admin, must_change=True)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Ensure the new user's directory + default settings exist.
    from user_paths import user_dir
    user_dir(req.username)
    return {"user": created}


@app.delete("/users/{username}")
def admin_delete_user(username: str, user: dict = Depends(require_admin)):
    if username == user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    user_service.revoke_all_sessions(username)
    user_service.delete_user(username)
    return {"ok": True}


@app.post("/users/{username}/reset-password")
def admin_reset_password(username: str, user: dict = Depends(require_admin)):
    temp = user_service.reset_password(username)
    user_service.revoke_all_sessions(username)
    return {"temp_password": temp}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
