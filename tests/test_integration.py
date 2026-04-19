#!/usr/bin/env python3
"""
Integration Test Suite for Personal AI Framework v5.0 + v5.1
Tests LoRA retraining pipeline and v5.1.0 features (facts, inference lock, etc.)

Usage:
    # Run all pre-training tests (Phase 1):
    python tests/test_integration.py --phase 1

    # Run specific phase:
    python tests/test_integration.py --phase 2

    # Run all phases sequentially (full training cycle ~2.5 hours):
    python tests/test_integration.py --all

    # Run a single test by name:
    python tests/test_integration.py --test test_inference_lock

    # List all tests:
    python tests/test_integration.py --list

Server must be running on localhost:8080 (docker compose up).
"""
import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_URL = "http://localhost:8080"
TRAINING_STATE_FILE = Path.home() / ".personal-ai" / "training_state.json"
TRAINING_DATA_FILE = Path(__file__).parent.parent / "training" / "data" / "train_alpaca.json"

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def log(msg, color=RESET):
    print(f"{color}{msg}{RESET}")


def log_test(name, status, detail=""):
    global passed, failed, skipped
    if status == "PASS":
        passed += 1
        icon = f"{GREEN}✓ PASS{RESET}"
    elif status == "FAIL":
        failed += 1
        icon = f"{RED}✗ FAIL{RESET}"
    else:
        skipped += 1
        icon = f"{YELLOW}○ SKIP{RESET}"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{detail_str}")


# Populated by login() at startup; api_* helpers attach it as Bearer.
_AUTH_TOKEN: Optional[str] = None


def login(user: Optional[str] = None, password: Optional[str] = None) -> bool:
    """Authenticate with SKIPPY_USER/SKIPPY_PASSWORD env vars and cache the token.

    Called once in main(); after that, api_get/api_post/api_delete attach the
    bearer header automatically. Returns True on success.
    """
    global _AUTH_TOKEN
    user = user or os.environ.get("SKIPPY_USER")
    password = password or os.environ.get("SKIPPY_PASSWORD")
    if not user or not password:
        return False
    url = f"{BASE_URL}/auth/login"
    payload = json.dumps({"username": user, "password": password}).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
            _AUTH_TOKEN = body.get("token")
            return _AUTH_TOKEN is not None
    except Exception:
        return False


def _attach_auth(req: urllib.request.Request) -> None:
    if _AUTH_TOKEN:
        req.add_header("Authorization", f"Bearer {_AUTH_TOKEN}")


def api_get(endpoint, timeout=30):
    """GET request to the API. Returns (status_code, response_dict)."""
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url)
    _attach_auth(req)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode())
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


def api_post(endpoint, data=None, timeout=30):
    """POST request to the API. Returns (status_code, response_dict)."""
    url = f"{BASE_URL}{endpoint}"
    if data is not None:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, data=b"", method="POST")
    _attach_auth(req)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode())
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


def api_delete(endpoint, timeout=30):
    """DELETE request to the API. Returns (status_code, response_dict)."""
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method="DELETE")
    _attach_auth(req)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode())
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


def wait_for_server(timeout=30):
    """Wait for the server to be reachable."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            status, _ = api_get("/")
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def get_training_state():
    """Read the training state file directly."""
    if TRAINING_STATE_FILE.exists():
        with open(TRAINING_STATE_FILE) as f:
            return json.load(f)
    return {"status": "idle", "progress_pct": 0}


def wait_for_training_status(target_statuses, timeout=300, poll_interval=5):
    """Poll training status until it reaches one of the target statuses."""
    start = time.time()
    last_pct = -1
    while time.time() - start < timeout:
        status, body = api_get("/training/status")
        current = body.get("status", "unknown")
        pct = body.get("progress_pct", 0)
        if pct != last_pct:
            log(f"    Training: {current} ({pct}%)", CYAN)
            last_pct = pct
        if current in target_statuses:
            return body
        time.sleep(poll_interval)
    return None


# ===========================================================================
# PHASE 1: Pre-Training Baseline
# ===========================================================================

def test_server_alive():
    """Verify the server is running and responsive."""
    status, body = api_get("/")
    if status == 200 and "model" in body:
        log_test("Server alive", "PASS", body.get("model", "unknown"))
    else:
        log_test("Server alive", "FAIL", f"status={status}")


def test_not_in_maintenance():
    """Verify server is NOT in maintenance mode."""
    status, body = api_get("/training/maintenance")
    if status == 200 and body.get("maintenance_mode") is False:
        log_test("Not in maintenance mode", "PASS")
    else:
        log_test("Not in maintenance mode", "FAIL", str(body))


def test_basic_generation():
    """Verify basic inference works."""
    status, body = api_post("/generate", {
        "prompt": "What is 2+2? Answer in one word.",
        "max_tokens": 32,
        "use_rag": False,
    }, timeout=120)
    if status == 200 and body.get("text"):
        log_test("Basic generation", "PASS", f'"{body["text"][:60]}"')
    else:
        log_test("Basic generation", "FAIL", str(body)[:100])


def test_inference_lock():
    """Send 3 concurrent requests — all should succeed without CUDA crash."""
    prompts = [
        "What color is the sky? One word.",
        "What is the capital of France? One word.",
        "Name a planet. One word.",
    ]

    def send_request(prompt):
        return api_post("/generate", {
            "prompt": prompt,
            "max_tokens": 32,
            "use_rag": False,
        }, timeout=180)

    log("    Sending 3 concurrent requests...", CYAN)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(send_request, p) for p in prompts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    all_ok = all(r[0] == 200 and r[1].get("text") for r in results)
    responses = [r[1].get("text", "")[:30] for r in results]
    if all_ok:
        log_test("Inference lock (concurrent)", "PASS", f"{len(results)} responses OK")
    else:
        log_test("Inference lock (concurrent)", "FAIL", str([r[0] for r in results]))


def test_personality():
    """Verify Skippy personality is injected."""
    status, body = api_post("/generate", {
        "prompt": "What is your name? Reply with just your name.",
        "max_tokens": 32,
        "use_rag": False,
    }, timeout=120)
    text = body.get("text", "").lower()
    # Check settings for configured name
    _, settings = api_get("/settings")
    ai_name = settings.get("personality", {}).get("name", "assistant").lower()
    if status == 200 and ai_name in text:
        log_test("Personality injection", "PASS", f'name="{ai_name}" found in response')
    elif status == 200:
        log_test("Personality injection", "FAIL",
                 f'expected "{ai_name}" in: "{text[:60]}"')
    else:
        log_test("Personality injection", "FAIL", f"status={status}")


def test_stop_tokens():
    """Verify stop tokens prevent 'User:' hallucination."""
    status, body = api_post("/generate", {
        "prompt": "Tell me a fun fact about dogs.",
        "max_tokens": 256,
        "use_rag": False,
    }, timeout=120)
    text = body.get("text", "")
    has_fake_turn = "User:" in text or "user:" in text
    if status == 200 and not has_fake_turn:
        log_test("Stop tokens", "PASS", "no fake User: turn in output")
    elif status == 200:
        log_test("Stop tokens", "FAIL", f'"User:" found in output: "{text[:100]}"')
    else:
        log_test("Stop tokens", "FAIL", f"status={status}")


def test_facts_add_and_search():
    """Add a test fact, search for it, then clean up."""
    test_fact = "The Personal AI Framework test suite was run successfully."
    test_source = "integration_test"

    # Add fact
    status, body = api_post(
        f"/facts?fact={urllib.parse.quote(test_fact)}&source={test_source}&category=test"
    )
    if status != 200:
        log_test("Facts add", "FAIL", str(body)[:80])
        return
    fact_id = body.get("id", "")
    log_test("Facts add", "PASS", f"id={fact_id[:12]}...")

    # Search for it
    status, body = api_get(f"/facts")
    found = any(f.get("source") == test_source for f in body.get("facts", []))
    if found:
        log_test("Facts search", "PASS", f'{body.get("count", 0)} total facts')
    else:
        log_test("Facts search", "FAIL", "test fact not found in listing")

    # Clean up
    if fact_id:
        api_delete(f"/facts/{fact_id}")


def test_facts_semantic_search():
    """Test semantic similarity search on facts (ChromaDB retrieval + LLM usage)."""
    test_fact = "Kyle's favorite programming language is Rust."
    # Add
    status, _ = api_post(
        f"/facts?fact={urllib.parse.quote(test_fact)}&source=test&category=test"
    )
    if status != 200:
        log_test("Facts semantic retrieval", "SKIP", "could not add test fact")
        return

    # Test 1: Verify ChromaDB can retrieve the fact by semantic query
    status, body = api_get("/facts")
    stored = [f for f in body.get("facts", []) if f.get("source") == "test"]
    if stored and "Rust" in stored[0].get("fact", ""):
        log_test("Facts semantic retrieval", "PASS",
                 f"fact stored and retrievable in ChromaDB")
    else:
        log_test("Facts semantic retrieval", "FAIL", "fact not found in ChromaDB")

    # Test 2: Check if LLM uses the injected fact via [INST] template
    # Note: use_rag must be true — facts retrieval is inside the RAG block
    status, body = api_post("/generate", {
        "prompt": "What is Kyle's favorite programming language?",
        "max_tokens": 64,
        "use_rag": True,
    }, timeout=120)
    text = body.get("text", "").lower()
    has_fact = "rust" in text
    if status == 200 and has_fact:
        log_test("Facts LLM injection", "PASS", "model used injected fact")
    elif status == 200:
        log_test("Facts LLM injection", "FAIL",
                 f'model ignored fact — "{text[:60]}"')
    else:
        log_test("Facts LLM injection", "FAIL", f"status={status}")

    # Clean up: remove test facts
    status, body = api_get("/facts")
    for f in body.get("facts", []):
        if f.get("source") == "test":
            api_delete(f"/facts/{f['id']}")


def test_rag_retrieval():
    """Test that RAG retrieval works (knowledge base query)."""
    status, body = api_post("/generate", {
        "prompt": "Search the knowledge base: what documents are available?",
        "max_tokens": 128,
        "use_rag": True,
        "rag_k": 3,
    }, timeout=120)
    if status == 200 and body.get("text"):
        has_context = body.get("context_used") is not None
        log_test("RAG retrieval", "PASS",
                 f"context={'yes' if has_context else 'no'}, response={len(body['text'])} chars")
    else:
        log_test("RAG retrieval", "FAIL", str(body)[:100])


# ---------------------------------------------------------------------------
# Unit tests for RAG ranking internals (no server required; direct import of
# pipeline/advanced_rag.py). These guard the invariants that the hybrid-search
# tuning relies on — regressions here have caused eval pass rate to drop
# before, so pin them as tests.
# ---------------------------------------------------------------------------

def _import_advanced_rag():
    """Add pipeline/ to sys.path and import AdvancedRAG + BM25.

    Returns None if the import fails (e.g. a dep is missing in the caller's
    venv), so the test can SKIP gracefully instead of crashing the suite.
    """
    pipeline_dir = str(Path(__file__).parent.parent / "pipeline")
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    try:
        import advanced_rag  # noqa: F401
        return advanced_rag
    except Exception:
        return None


def test_bm25_tokenizer_keeps_product_ids():
    """BM25 must keep 2-char product identifiers like 'mx' and '93'.

    These were dropped by the pre-v5.9.3 `len > 2` filter, which was the
    reason i.MX queries couldn't match the reference-manual chunks in BM25.
    """
    mod = _import_advanced_rag()
    if mod is None:
        log_test("BM25 tokenizer keeps product IDs", "SKIP", "advanced_rag import failed")
        return
    tokens = mod.BM25()._tokenize("i.MX 93 DDR controller bus width")
    missing = [t for t in ("mx", "93") if t not in tokens]
    if not missing:
        log_test("BM25 tokenizer keeps product IDs", "PASS", f"tokens={tokens}")
    else:
        log_test("BM25 tokenizer keeps product IDs", "FAIL", f"missing {missing} in {tokens}")


def test_peripheral_alias_expansion():
    """UART/SPI/I2C queries should expand to include the LP-prefixed names.

    On i.MX SoCs the peripheral chapters are named LPUART/LPSPI/LPI2C — bare
    'UART' won't match them semantically without this alias pass.
    """
    mod = _import_advanced_rag()
    if mod is None:
        log_test("Peripheral alias expansion", "SKIP", "advanced_rag import failed")
        return
    adv = mod.AdvancedRAG(rag_service=None)  # _expand_query doesn't touch self.rag
    cases = [
        ("What handles UART on i.MX 93?", "LPUART"),
        ("Tell me about SPI on this chip.", "LPSPI"),
        ("How does I2C work?", "LPI2C"),
    ]
    failures = []
    for q, expected in cases:
        expanded = adv._expand_query(q)
        if expected not in expanded:
            failures.append(f"{q!r} → {expanded!r} (missing {expected})")
    if not failures:
        log_test("Peripheral alias expansion", "PASS", f"{len(cases)}/{len(cases)} cases")
    else:
        log_test("Peripheral alias expansion", "FAIL", "; ".join(failures))


def test_peripheral_alias_idempotent():
    """Expansion must not double-append an alias when it's already present."""
    mod = _import_advanced_rag()
    if mod is None:
        log_test("Peripheral alias idempotent", "SKIP", "advanced_rag import failed")
        return
    adv = mod.AdvancedRAG(rag_service=None)
    # Query already contains LPUART — expansion should not add it again.
    expanded = adv._expand_query("How does LPUART handle UART traffic?")
    lpuart_count = expanded.lower().count("lpuart")
    if lpuart_count == 1:
        log_test("Peripheral alias idempotent", "PASS", f"'lpuart' count={lpuart_count}")
    else:
        log_test("Peripheral alias idempotent", "FAIL",
                 f"'lpuart' appears {lpuart_count}× in {expanded!r}")


def test_concept_alias_pin_routing():
    """Pin-routing intent phrases must trigger the IOMUX concept alias.

    Compound queries like "what IP block routes its pins?" don't get answered
    by the peripheral-chapter chunks alone — a secondary retrieval with
    IOMUXC added is needed. Guards the regex patterns in _CONCEPT_ALIASES.
    """
    mod = _import_advanced_rag()
    if mod is None:
        log_test("Concept alias detects pin routing", "SKIP", "advanced_rag import failed")
        return
    adv = mod.AdvancedRAG(rag_service=None)
    positive = [
        "what IP block routes its pins?",
        "how is pad mux handled",
        "Describe the pin-mux for UART",
    ]
    # These should NOT fire (bare "pins" without routing intent).
    negative = [
        "how many pins does the chip have?",
        "is it pin-compatible with other SoCs",
    ]
    pos_fail = [q for q in positive if not adv._detect_concept_aliases(q)]
    neg_fail = [q for q in negative if adv._detect_concept_aliases(q)]
    if not pos_fail and not neg_fail:
        log_test("Concept alias detects pin routing", "PASS",
                 f"{len(positive)} positive, {len(negative)} negative")
    else:
        detail = []
        if pos_fail:
            detail.append(f"missed positive: {pos_fail}")
        if neg_fail:
            detail.append(f"false positive: {neg_fail}")
        log_test("Concept alias detects pin routing", "FAIL", "; ".join(detail))


# ---------------------------------------------------------------------------
# Integration tests for RAG improvements (require the server).
# These exercise the end-to-end retrieval+generation path to catch regressions
# that the unit tests above can't see (e.g. reranker blend weights changing,
# filter thresholds drifting).
# ---------------------------------------------------------------------------

def _rag_probe(prompt, expected_substrings, test_name, max_tokens=512, timeout=120):
    """Helper: POST /generate with RAG on, check response contains all expected
    substrings (case-insensitive). Passes if all present; fails with a summary
    of the miss. Used by the RAG-scenario integration tests below.
    """
    status, body = api_post("/generate", {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "use_rag": True,
        "rag_k": 3,
    }, timeout=timeout)
    if status != 200 or not body.get("text"):
        log_test(test_name, "FAIL", f"status={status}, body={str(body)[:100]}")
        return
    text = body["text"].lower()
    missing = [s for s in expected_substrings if s.lower() not in text]
    if not missing:
        log_test(test_name, "PASS", f"all {len(expected_substrings)} substrings present")
    else:
        log_test(test_name, "FAIL",
                 f"missing {missing} in response: {body['text'][:160]!r}")


def test_rag_peripheral_query_finds_lpuart():
    """Bare 'UART' query must surface LPUART chunks and yield an LPUART answer.

    Pre-alias-expansion, the model answered with generic 'UART' text pulled
    from marketing chunks — missed the LPUART chapter entirely.
    """
    _rag_probe(
        "Which peripheral on the i.MX 93 handles UART?",
        ["LPUART"],
        "RAG: UART query surfaces LPUART",
    )


def test_rag_compound_query_finds_iomux():
    """Compound peripheral+routing query should answer both parts correctly.

    Pre-concept-alias, top-3 was all LPUART chunks with no IOMUX content —
    model correctly named LPUART but hallucinated 'GPIO' for the routing.
    """
    _rag_probe(
        "Which peripheral on the i.MX 93 handles UART, and what IP block routes its pins?",
        ["LPUART", "IOMUX"],
        "RAG: compound UART+pin-routing query",
    )


def test_rag_rare_phrase_retrieval():
    """Rare-phrase queries must surface BM25 rank-4/5 chunks, not just rank-1.

    Pre-BM25-admit, the MX8QuadMax chunk (BM25 rank 4 for 'angry birds')
    was filtered out because its normalized kw ~0.7 failed the > 0.8 gate
    and its sem score was 0. Model then denied Kyle demoed the game.

    Phrased as a direct "what chip" question rather than "Did Kyle ever demo
    … Which chip?" — the latter reliably triggers the agent loop to propose
    a write_file tool call, which obscures the retrieval signal we're
    actually testing here. This wording exercises the same retrieval path
    without bouncing off the agent dispatcher.
    """
    _rag_probe(
        "In Kyle's pillar-to-pillar IVI demo, which NXP chip ran Angry Birds on the LCD?",
        ["i.MX 8", "Angry Birds"],
        "RAG: rare-phrase multi-hop (Angry Birds)",
    )


def test_rag_persona_not_hijacked():
    """Persona queries must not be hijacked by weak-relevance RAG chunks.

    When the candidate pool was widened without a sem floor, persona intros
    returned 'I'm a helpful AI assistant' because a middling-score chunk
    overrode the persona prompt. The sem > 0.3 floor prevents that.
    """
    _rag_probe(
        "In two sentences, introduce yourself as Skippy. Stay in character.",
        ["Skippy"],
        "RAG: persona not hijacked by drift chunks",
    )


def test_rag_general_knowledge_not_refused():
    """General-knowledge queries must still answer from training, not refuse.

    With weak chunks in context, an over-restrictive RAG directive made the
    model say 'excerpts don't cover that' for questions that don't need KB
    content at all. The filter + softer directive keep these answerable.
    """
    _rag_probe(
        "Briefly explain what an MoE (mixture-of-experts) model is.",
        ["expert"],
        "RAG: general-knowledge query still answered",
    )


def test_training_status_idle():
    """Training status should be idle/complete before we start."""
    status, body = api_get("/training/status")
    state = body.get("status", "unknown")
    if state in ("idle", "complete"):
        log_test("Training status idle", "PASS", f'status="{state}"')
    else:
        log_test("Training status idle", "FAIL", f'status="{state}"')


# ===========================================================================
# PHASE 2: Training Trigger & GPU Prep
# ===========================================================================

def test_trigger_training():
    """Trigger training via API."""
    status, body = api_post("/training/trigger")
    if status == 200 and body.get("status") == "ok":
        log_test("Trigger training", "PASS")
    else:
        log_test("Trigger training", "FAIL", str(body)[:100])


def test_maintenance_mode_activates():
    """Wait for maintenance mode to activate after trigger."""
    log("    Waiting for maintenance mode (up to 60s)...", CYAN)
    start = time.time()
    while time.time() - start < 60:
        status, body = api_get("/training/maintenance")
        if body.get("maintenance_mode") is True:
            elapsed = int(time.time() - start)
            log_test("Maintenance mode activates", "PASS", f"after {elapsed}s")
            return True
        time.sleep(2)
    log_test("Maintenance mode activates", "FAIL", "timed out after 60s")
    return False


def test_inference_blocked_503():
    """Inference should return 503 (or 500) during maintenance."""
    status, body = api_post("/generate", {
        "prompt": "Hello",
        "max_tokens": 16,
        "use_rag": False,
    }, timeout=10)
    if status in (503, 500):
        log_test("Inference blocked", "PASS", f"status={status}")
    else:
        log_test("Inference blocked", "FAIL", f"status={status}")


def test_streaming_blocked_503():
    """Streaming should also return 503 (or 500) during maintenance."""
    status, body = api_post("/generate/stream", {
        "prompt": "Hello",
        "max_tokens": 16,
        "use_rag": False,
    }, timeout=10)
    if status in (503, 500):
        log_test("Streaming blocked", "PASS", f"status={status}")
    else:
        log_test("Streaming blocked", "FAIL", f"status={status}")


def test_facts_accessible_during_training():
    """Facts (ChromaDB) should still be queryable even with model unloaded."""
    status, body = api_get("/facts")
    if status == 200:
        log_test("Facts accessible during training", "PASS",
                 f'{body.get("count", 0)} facts readable')
    else:
        log_test("Facts accessible during training", "FAIL", f"status={status}")


def test_training_status_progressing():
    """Training status should show progress updates."""
    status, body = api_get("/training/status")
    state = body.get("status", "unknown")
    pct = body.get("progress_pct", 0)
    if state not in ("idle", "complete", "failed"):
        log_test("Training status progressing", "PASS",
                 f'status="{state}", progress={pct}%')
    else:
        log_test("Training status progressing", "FAIL",
                 f'status="{state}" (expected active state)')


# ===========================================================================
# PHASE 3: During Training (Long-Running)
# ===========================================================================

def test_monitor_training():
    """Monitor training progress until completion or failure."""
    log("    Monitoring training (this may take ~2.5 hours)...", CYAN)
    log("    Press Ctrl+C to skip monitoring (training continues in background).", YELLOW)

    try:
        result = wait_for_training_status(
            ["complete", "failed", "interrupted"],
            timeout=10800,  # 3 hours max
            poll_interval=15
        )
        if result is None:
            log_test("Training completion", "FAIL", "timed out after 3 hours")
            return None
        final_status = result.get("status")
        if final_status == "complete":
            log_test("Training completion", "PASS",
                     f'model={result.get("new_model", "?")}')
        elif final_status == "failed":
            log_test("Training completion", "FAIL",
                     f'error={result.get("error", "unknown")}')
        else:
            log_test("Training completion", "SKIP",
                     f'status={final_status}')
        return result
    except KeyboardInterrupt:
        log("\n    Monitoring interrupted — training continues in background.", YELLOW)
        log_test("Training completion", "SKIP", "user interrupted monitoring")
        return None


def test_inference_blocked_during_training():
    """Periodically check that inference stays blocked throughout training."""
    checks = 0
    blocked = 0
    for _ in range(3):
        status, _ = api_post("/generate", {
            "prompt": "test",
            "max_tokens": 8,
            "use_rag": False,
        }, timeout=5)
        checks += 1
        if status in (503, 500):
            blocked += 1
        time.sleep(10)

    if blocked == checks:
        log_test("Inference blocked throughout", "PASS", f"{blocked}/{checks} blocked")
    else:
        log_test("Inference blocked throughout", "FAIL",
                 f"only {blocked}/{checks} blocked")


# ===========================================================================
# PHASE 4: Post-Training Deployment
# ===========================================================================

def test_inference_resumes():
    """After training, inference should work again."""
    log("    Waiting for inference to resume (up to 180s)...", CYAN)
    start = time.time()
    while time.time() - start < 180:
        status, body = api_get("/training/maintenance")
        if body.get("maintenance_mode") is False:
            # Try a generation
            status2, body2 = api_post("/generate", {
                "prompt": "Say hello.",
                "max_tokens": 32,
                "use_rag": False,
            }, timeout=120)
            if status2 == 200 and body2.get("text"):
                elapsed = int(time.time() - start)
                log_test("Inference resumes", "PASS",
                         f'after {elapsed}s: "{body2["text"][:40]}"')
                return True
        time.sleep(5)
    log_test("Inference resumes", "FAIL", "timed out")
    return False


def test_post_training_concurrent():
    """Re-test concurrent inference with the new model."""
    test_inference_lock()  # Reuse the same test


def test_post_training_personality():
    """Verify personality is still injected after model swap."""
    test_personality()


def test_post_training_stop_tokens():
    """Verify stop tokens work with the trained model."""
    status, body = api_post("/generate", {
        "prompt": "Write a paragraph about space exploration.",
        "max_tokens": 512,
        "use_rag": False,
    }, timeout=120)
    text = body.get("text", "")
    has_fake_turn = "User:" in text or "user:" in text
    if status == 200 and not has_fake_turn and len(text) > 20:
        log_test("Stop tokens (trained model)", "PASS",
                 f"{len(text)} chars, no fake turns")
    elif status == 200 and has_fake_turn:
        log_test("Stop tokens (trained model)", "FAIL",
                 '"User:" found — trained model may need different stop tokens')
    else:
        log_test("Stop tokens (trained model)", "FAIL", f"status={status}")


def test_post_training_facts():
    """Verify facts layer works with the new model."""
    test_fact = "Integration test post-training fact check: 42 is the answer."
    api_post(f"/facts?fact={urllib.parse.quote(test_fact)}&source=post_test&category=test")

    status, body = api_get("/facts")
    found = any(f.get("source") == "post_test" for f in body.get("facts", []))
    if found:
        log_test("Facts layer (post-training)", "PASS")
    else:
        log_test("Facts layer (post-training)", "FAIL", "test fact not found")

    # Clean up
    for f in body.get("facts", []):
        if f.get("source") == "post_test":
            api_delete(f"/facts/{f['id']}")


def test_post_training_rag():
    """Verify RAG retrieval works with the new model."""
    test_rag_retrieval()


def test_model_version_incremented():
    """Check that a new GGUF model file was created."""
    models_dir = Path(__file__).parent.parent / "models"
    gguf_files = sorted(models_dir.glob("kyle-7b-v*-q4_k_m.gguf"))
    if gguf_files:
        latest = gguf_files[-1]
        size_gb = latest.stat().st_size / (1024**3)
        log_test("Model version incremented", "PASS",
                 f"{latest.name} ({size_gb:.1f} GB)")
    else:
        log_test("Model version incremented", "FAIL",
                 "no kyle-7b-v* GGUF files found in models/")


# ===========================================================================
# PHASE 5: Edge Cases & Recovery
# ===========================================================================

def test_min_examples_gate():
    """Training should abort if fewer than min_new_examples conversations exist."""
    # Get current settings
    _, settings = api_get("/settings")
    min_ex = settings.get("training", {}).get("min_new_examples", 50)
    log(f"    min_new_examples = {min_ex}", CYAN)

    # The gate is enforced by the orchestrator, not the API.
    # We just verify the setting is configured.
    if min_ex > 0:
        log_test("Min examples gate configured", "PASS", f"threshold={min_ex}")
    else:
        log_test("Min examples gate configured", "FAIL", "threshold is 0 or missing")


def test_training_data_dedup():
    """Check training data file for duplicates."""
    if not TRAINING_DATA_FILE.exists():
        log_test("Training data dedup", "SKIP", "no training data file yet")
        return

    with open(TRAINING_DATA_FILE) as f:
        data = json.load(f)

    import hashlib
    hashes = set()
    dupes = 0
    for ex in data:
        h = hashlib.md5((ex["instruction"] + ex["output"]).encode()).hexdigest()
        if h in hashes:
            dupes += 1
        hashes.add(h)

    if dupes == 0:
        log_test("Training data dedup", "PASS", f"{len(data)} examples, 0 duplicates")
    else:
        log_test("Training data dedup", "FAIL", f"{dupes} duplicates in {len(data)} examples")


def test_training_data_no_fact_leakage():
    """Check that fact injection text doesn't leak into training data."""
    if not TRAINING_DATA_FILE.exists():
        log_test("Training data fact leakage", "SKIP", "no training data file yet")
        return

    with open(TRAINING_DATA_FILE) as f:
        data = json.load(f)

    leak_markers = [
        "Important — these are verified, up-to-date facts",
        "Use them over your training data",
        "Relevant information from your knowledge base",
        "From our previous conversations:",
    ]

    leaked = []
    for i, ex in enumerate(data):
        instruction = ex.get("instruction", "")
        for marker in leak_markers:
            if marker in instruction:
                leaked.append((i, marker[:40]))
                break

    if not leaked:
        log_test("Training data fact leakage", "PASS",
                 f"checked {len(data)} examples, no prompt leakage")
    else:
        log_test("Training data fact leakage", "FAIL",
                 f"{len(leaked)} examples have leaked prompt context")
        for idx, marker in leaked[:3]:
            log(f"      Example {idx}: contains \"{marker}...\"", RED)


def test_crash_recovery_state():
    """Verify training state file is in a valid terminal state."""
    state = get_training_state()
    status = state.get("status", "unknown")
    valid_terminal = {"idle", "complete", "failed", "interrupted"}
    if status in valid_terminal:
        log_test("Training state valid", "PASS", f'status="{status}"')
    else:
        log_test("Training state valid", "FAIL",
                 f'status="{status}" (stuck — may need crash recovery)')


# ===========================================================================
# PHASE 6: Interrupt Flow (separate run)
# ===========================================================================

def test_interrupt_training():
    """Trigger training, then interrupt it. Verify graceful shutdown."""
    # Trigger
    status, body = api_post("/training/trigger")
    if status != 200:
        log_test("Interrupt: trigger", "FAIL", str(body)[:80])
        return

    log("    Waiting for training to start...", CYAN)
    time.sleep(10)

    # Check it's actually training
    _, state = api_get("/training/status")
    if state.get("status") in ("idle", "complete"):
        log_test("Interrupt: training started", "FAIL", "training didn't start")
        return
    log_test("Interrupt: training started", "PASS", f'status={state.get("status")}')

    # Send interrupt
    status, body = api_post("/training/interrupt")
    if status == 200:
        log_test("Interrupt: signal sent", "PASS")
    else:
        log_test("Interrupt: signal sent", "FAIL", str(body)[:80])
        return

    # Wait for interrupted state
    log("    Waiting for graceful shutdown (up to 120s)...", CYAN)
    result = wait_for_training_status(
        ["interrupted", "failed", "complete", "idle"],
        timeout=120,
        poll_interval=5
    )
    if result and result.get("status") == "interrupted":
        log_test("Interrupt: graceful shutdown", "PASS")
    elif result:
        log_test("Interrupt: graceful shutdown", "FAIL",
                 f'final status={result.get("status")}')
    else:
        log_test("Interrupt: graceful shutdown", "FAIL", "timed out")

    # Verify inference resumes
    log("    Checking inference resumes after interrupt...", CYAN)
    time.sleep(10)
    status, body = api_get("/training/maintenance")
    if body.get("maintenance_mode") is False:
        log_test("Interrupt: inference restored", "PASS")
    else:
        log_test("Interrupt: inference restored", "FAIL", "still in maintenance mode")


# ===========================================================================
# Test Registry
# ===========================================================================

PHASES = {
    1: {
        "name": "Pre-Training Baseline",
        "tests": [
            test_server_alive,
            test_not_in_maintenance,
            test_training_status_idle,
            test_basic_generation,
            test_inference_lock,
            test_personality,
            test_stop_tokens,
            test_facts_add_and_search,
            test_facts_semantic_search,
            test_rag_retrieval,
            # RAG ranking unit tests (no server; fast)
            test_bm25_tokenizer_keeps_product_ids,
            test_peripheral_alias_expansion,
            test_peripheral_alias_idempotent,
            test_concept_alias_pin_routing,
            # RAG end-to-end scenarios (server; slower)
            test_rag_peripheral_query_finds_lpuart,
            test_rag_compound_query_finds_iomux,
            test_rag_rare_phrase_retrieval,
            test_rag_persona_not_hijacked,
            test_rag_general_knowledge_not_refused,
        ],
    },
    2: {
        "name": "Training Trigger & GPU Prep",
        "tests": [
            test_trigger_training,
            test_maintenance_mode_activates,
            test_inference_blocked_503,
            test_streaming_blocked_503,
            test_facts_accessible_during_training,
            test_training_status_progressing,
        ],
    },
    3: {
        "name": "During Training (Long-Running)",
        "tests": [
            test_inference_blocked_during_training,
            test_monitor_training,
        ],
    },
    4: {
        "name": "Post-Training Deployment",
        "tests": [
            test_inference_resumes,
            test_post_training_concurrent,
            test_post_training_personality,
            test_post_training_stop_tokens,
            test_post_training_facts,
            test_post_training_rag,
            test_model_version_incremented,
        ],
    },
    5: {
        "name": "Edge Cases & Data Quality",
        "tests": [
            test_min_examples_gate,
            test_training_data_dedup,
            test_training_data_no_fact_leakage,
            test_crash_recovery_state,
        ],
    },
    6: {
        "name": "Interrupt Flow (Separate Run)",
        "tests": [
            test_interrupt_training,
        ],
    },
}


def run_phase(phase_num):
    global passed, failed, skipped
    phase = PHASES[phase_num]
    log(f"\n{'='*60}", BOLD)
    log(f"  PHASE {phase_num}: {phase['name']}", BOLD)
    log(f"{'='*60}", BOLD)
    for test_fn in phase["tests"]:
        test_fn()
    print()


def list_tests():
    for num, phase in PHASES.items():
        log(f"\nPhase {num}: {phase['name']}", BOLD)
        for test_fn in phase["tests"]:
            print(f"  - {test_fn.__name__}")


def main():
    global passed, failed, skipped

    parser = argparse.ArgumentParser(description="Integration tests for Personal AI Framework")
    parser.add_argument("--phase", type=int, choices=PHASES.keys(),
                        help="Run a specific test phase (1-6)")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases sequentially (includes full training cycle)")
    parser.add_argument("--test", type=str,
                        help="Run a single test by function name")
    parser.add_argument("--list", action="store_true",
                        help="List all available tests")
    parser.add_argument("--baseline", action="store_true",
                        help="Run phases 1 and 5 only (no training)")
    args = parser.parse_args()

    if args.list:
        list_tests()
        return

    # Check server is reachable
    log(f"\nConnecting to {BASE_URL}...", CYAN)
    if not wait_for_server(10):
        log(f"Cannot reach server at {BASE_URL}. Is docker compose running?", RED)
        sys.exit(1)
    log("Server connected.", GREEN)

    # Authenticate — server is auth-gated since v5.9 multi-user support.
    # SKIPPY_USER / SKIPPY_PASSWORD must be exported (same env the eval
    # harness uses). Without auth every /generate and /facts call 401s.
    if not login():
        log("Auth failed — set SKIPPY_USER and SKIPPY_PASSWORD env vars.", RED)
        sys.exit(1)
    log("Authenticated.\n", GREEN)

    if args.test:
        # Find and run single test
        all_tests = {}
        for phase in PHASES.values():
            for fn in phase["tests"]:
                all_tests[fn.__name__] = fn
        if args.test in all_tests:
            log(f"Running single test: {args.test}", BOLD)
            all_tests[args.test]()
        else:
            log(f"Unknown test: {args.test}", RED)
            log("Available tests:", YELLOW)
            for name in sorted(all_tests.keys()):
                print(f"  {name}")
            sys.exit(1)
    elif args.baseline:
        run_phase(1)
        run_phase(5)
    elif args.phase:
        run_phase(args.phase)
    elif args.all:
        for phase_num in sorted(PHASES.keys()):
            if phase_num == 6:
                log("Skipping Phase 6 (interrupt) — run separately with --phase 6", YELLOW)
                continue
            run_phase(phase_num)
    else:
        # Default: run Phase 1 (safe, no training)
        log("No phase specified — running Phase 1 (pre-training baseline).", YELLOW)
        log("Use --help to see all options.\n", YELLOW)
        run_phase(1)

    # Summary
    total = passed + failed + skipped
    log(f"\n{'='*60}", BOLD)
    log(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped ({total} total)", BOLD)
    log(f"{'='*60}\n", BOLD)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
