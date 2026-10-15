#!/usr/bin/env python3
"""Build Skippy's email-free fine-tuning set: grounded RAG answers + abstentions + refusals.

The old fine-tunes were ~97.5% Kyle's personal email (archived 2026-10-13). This set contains NO
personal data. It teaches the behaviour the evals say is missing: answer datasheet questions from the
retrieved excerpts, quote values verbatim, and say so when the excerpts don't cover the question
(the old v4 was trained without any system prompt or context, and overrode good context with
memorised i.MX 93 specs).

Every example uses Skippy's REAL inference prompt (persona + RAG directive + excerpts, read from
pipeline/llm_server.py so training and serving can't drift) and REAL retrieval (Skippy /generate).

Stages (resumable; state in training/data/clean_v5/):
  holdout   retrieve k=20 for every eval prompt; those chunks are never used for training
  sample    pick datasheet chunks spread across source documents
  questions Claude writes an answerable + an unanswerable-but-related question per chunk
  retrieve  Skippy retrieval for each question (answerable: target chunk spliced in if missed)
  answers   Claude writes the ideal grounded response
  tidy      (v5.1) rewrite abstentions: extra facts only if they partly answer the question
  general   general-knowledge questions with tangential excerpts (guards against over-refusal)
  build     HHEM consistency filter + PII scan + refusals -> train/val messages JSON

    python3 training/build_grounded_dataset.py all      # needs Skippy up, a GPU lease, API key
"""
import concurrent.futures as cf, hashlib, json, os, random, re, sys, time
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "eval")); sys.path.insert(0, os.path.join(REPO, "training"))
from scrub_signatures import PHONE, EMAIL

OUT = os.path.join(REPO, "training/data/clean_v5")
BASE, CHROMA = "http://localhost:8080", "http://localhost:8000/api/v1"
USER = os.environ.get("SKIPPY_USER", "kyle")
TEACHER = "claude-sonnet-5-5"
N_CHUNKS, N_ANSWERABLE, N_UNANSWERABLE, N_GENERAL, N_REFUSAL = 1000, 700, 300, 150, 150
RAG_K, SEED = 4, 1013
PERSONA = ("Your name is Skippy. You are Skippy, a helpful AI assistant focused on embedded systems engineering.\n"
           "You help your user with hardware engineering, datasheets, and code. You're friendly and concise.\n"
           "Always answer as Skippy. Never claim to be Qwen or a generic AI assistant.\n\n")

def rag_directive():
    """The exact directive llm_server.py injects, parsed from its source so the two can't drift."""
    src = open(os.path.join(REPO, "pipeline/llm_server.py")).read()
    block = src[src.index("rag_directive = (") + len("rag_directive = ("):]
    block = block[:block.index("\n        )")]
    return "".join(eval(s) for s in re.findall(r'^\s*("(?:[^"\\]|\\.)*")\s*$', block, re.M))

def system_prompt(excerpts):
    s = PERSONA + rag_directive() + "Excerpts from your knowledge base:\n\n"
    return s + "".join(f"{d}\n\n" for d in excerpts) + "---\n\n"

def path(name): return os.path.join(OUT, name)
def load(name, default):
    return json.load(open(path(name))) if os.path.exists(path(name)) else default
def save(name, obj):
    os.makedirs(OUT, exist_ok=True); tmp = path(name) + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, ensure_ascii=False); os.replace(tmp, path(name))
def h(text): return hashlib.sha1(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()

_TOKEN = None
def skippy_retrieve(q, k=RAG_K):
    global _TOKEN
    if _TOKEN is None:
        pw = os.environ.get("SKIPPY_PASSWORD")  # env or ~/.personal-ai/keys.env; never hard-coded (public repo)
        if not pw:
            sys.exit("set SKIPPY_PASSWORD (local Skippy login) in the environment or ~/.personal-ai/keys.env")
        _TOKEN = requests.post(f"{BASE}/auth/login", json={"username": USER, "password": pw}, timeout=15).json()["token"]
    while True:  # a large k can overflow the 32k context (long chunks); retry smaller rather than lose the item
        r = requests.post(f"{BASE}/generate", headers={"Authorization": f"Bearer {_TOKEN}"}, timeout=1800, json={
            "prompt": q, "use_rag": True, "rag_k": k, "max_tokens": 1, "temperature": 0, "skip_agent_loop": True})
        if r.ok or k <= 1:
            break
        k = max(1, k // 2)
    d = r.json() if r.ok else {}
    return d.get("context_used") or [], [c.get("source_file", "?") for c in (d.get("citations") or [])]

def claude(system, user, max_tokens=900):
    from anthropic import Anthropic
    for attempt in range(5):
        try:
            r = Anthropic().messages.create(model=TEACHER, max_tokens=max_tokens, system=system,  # 5.5 rejects temperature
                                            messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")  # skip thinking blocks
        except Exception as e:
            print(f"   claude retry {attempt}: {str(e)[:120]}", flush=True); time.sleep(10 * (attempt + 1))
    raise RuntimeError("teacher unavailable")

def claude_json(system, user, max_tokens=900):
    """Parse the teacher's JSON; one retry on malformed output, then skip the item (None) rather than crash."""
    for attempt in range(2):
        t = claude(system, user if attempt == 0 else user + "\n\nReturn ONLY valid JSON with double-quoted keys.", max_tokens)
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None

# ---------------------------------------------------------------- stages
def holdout():
    """Chunks any eval prompt retrieves (k=20) are excluded from training: no eval contamination."""
    hold = set(load("holdout.json", []))
    prompts = []
    for f in ("eval/prompts.json", "eval/prompts_v2.json", "eval/prompts_v2_1.json"):
        d = json.load(open(os.path.join(REPO, f)))
        prompts += [p["prompt"] for p in (d["prompts"] if isinstance(d, dict) else d)]
    import faithfulness_sweep
    prompts += [p for p, _, _ in faithfulness_sweep.AGENTIC.values()]
    for i, q in enumerate(dict.fromkeys(prompts)):
        ctx, _ = skippy_retrieve(q, k=20)
        hold |= {h(c) for c in ctx}
        if i % 20 == 0:
            print(f"  holdout {i}/{len(set(prompts))}: {len(hold)} chunks", flush=True); save("holdout.json", sorted(hold))
    save("holdout.json", sorted(hold)); print(f"holdout: {len(hold)} chunks from {len(set(prompts))} eval prompts")

def sample():
    hold = set(load("holdout.json", []))
    cid = {c["name"]: c["id"] for c in requests.get(f"{CHROMA}/collections", timeout=30).json()}["personal_knowledge"]
    docs, off = [], 0
    while True:
        r = requests.post(f"{CHROMA}/collections/{cid}/get", timeout=300, json={
            "where": {"category": "datasheets"}, "include": ["documents", "metadatas"], "limit": 5000, "offset": off}).json()
        if not r["ids"]:
            break
        docs += [(d, m.get("source_file", "?")) for d, m in zip(r["documents"], r["metadatas"])]
        off += 5000
    def usable(t):
        letters = sum(ch.isalpha() for ch in t)
        return 500 <= len(t) <= 3000 and letters / len(t) > 0.55 and h(t) not in hold
    pool = [(d, s) for d, s in docs if usable(d)]
    by_src = {}
    for d, s in pool:
        by_src.setdefault(s, []).append(d)
    rnd = random.Random(SEED); picked = []
    srcs = sorted(by_src); [rnd.shuffle(by_src[s]) for s in srcs]
    while len(picked) < N_CHUNKS and any(by_src.values()):  # round-robin across documents
        for s in srcs:
            if by_src[s] and len(picked) < N_CHUNKS:
                picked.append({"chunk": by_src[s].pop(), "source": s})
    save("chunks.json", picked)
    print(f"sample: {len(docs)} datasheet chunks, {len(pool)} usable after holdout, {len(picked)} picked from {len(srcs)} documents")

Q_SYSTEM = ("You write realistic user questions for training a hardware-engineering assistant that answers from "
            "retrieved datasheet excerpts. Return STRICT JSON only.")
Q_USER = """Source document: {src}
Excerpt:
\"\"\"{chunk}\"\"\"

Write two questions an embedded engineer might type to the assistant:
1. "answerable": a specific question whose answer IS stated in this excerpt (a value, setting, register, feature, limit, procedure). Name the part (infer it from the document name if the excerpt doesn't say), so the question stands on its own.
2. "unanswerable": a closely related, plausible question about the same part/topic whose answer is NOT in this excerpt (a different parameter, a neighbouring spec, or the same spec for a sibling part). It must not be answerable from the excerpt.
Return {{"answerable": "...", "unanswerable": "..."}}. If the excerpt has no usable technical content, return {{"skip": true}}."""

def questions():
    chunks = load("chunks.json", []); qs = load("questions.json", {})
    todo = [i for i in range(len(chunks)) if str(i) not in qs]
    def work(i):
        c = chunks[i]
        return i, claude_json(Q_SYSTEM, Q_USER.format(src=c["source"], chunk=c["chunk"]), 2000)
    with cf.ThreadPoolExecutor(8) as ex:
        for n, (i, j) in enumerate(ex.map(work, todo)):
            qs[str(i)] = j or {"skip": True}
            if n % 25 == 0:
                save("questions.json", qs); print(f"  questions {len(qs)}/{len(chunks)}", flush=True)
    save("questions.json", qs); print(f"questions: {sum(1 for v in qs.values() if not v.get('skip'))} usable")

_LABEL = re.compile(r"^\[\d+\]\s*")
def _unlabel(c): return _LABEL.sub("", c)
def _relabel(docs): return [f"[{i + 1}] {d}" for i, d in enumerate(docs)]  # Skippy's own context format
def _norm(s): return re.sub(r"\s+", " ", _unlabel(s)).strip()
def _same(ctx_chunk, target):
    """Skippy prefixes '[n] ' and may merge neighbouring text, so match on a 300-char overlap."""
    a, b = _norm(ctx_chunk), _norm(target)
    return b[:300] in a or a[:300] in b

def retrieve():
    chunks = load("chunks.json", []); qs = load("questions.json", {}); items = load("items.json", [])
    done = {(it["chunk_i"], it["kind"]) for it in items}
    rnd = random.Random(SEED)
    keys = [k for k in sorted(qs, key=int) if not qs[k].get("skip")]
    for k in keys:
        for kind in ("answerable", "unanswerable"):
            if (int(k), kind) in done or not qs[k].get(kind):
                continue
            q = qs[k][kind]; ctx, srcs = skippy_retrieve(q)
            target = chunks[int(k)]["chunk"]; spliced = False
            if kind == "answerable" and not any(_same(c, target) for c in ctx):
                body = [_unlabel(c) for c in ctx][:RAG_K - 1]
                body.insert(rnd.randrange(len(body) + 1), target)
                ctx = _relabel(body); spliced = True
            if kind == "unanswerable":  # drop the chunk it was written against, however it was formatted
                kept = [_unlabel(c) for c in ctx if not _same(c, target)]
                ctx = _relabel(kept) if kept else ctx
            items.append({"chunk_i": int(k), "kind": kind, "question": q, "context": ctx, "sources": srcs, "spliced": spliced})
        if len(items) % 50 < 2:
            save("items.json", items); print(f"  retrieve {len(items)}", flush=True)
    save("items.json", items); print(f"retrieve: {len(items)} items ({sum(i['spliced'] for i in items)} answerable needed the target spliced in)")

A_SYSTEM = ("You write the ideal reply of 'Skippy', a concise, friendly embedded-engineering assistant, for a "
            "fine-tuning dataset. Skippy answers datasheet questions ONLY from the retrieved excerpts it was given.")
A_USER = """Retrieved excerpts Skippy was given:
\"\"\"
{ctx}
\"\"\"

User question: {q}

Write Skippy's reply. Rules:
- Every specific (number, unit, register, pin, part name, feature) must be stated in the excerpts. Quote values verbatim with units. Use no outside knowledge for specifics.
- If the excerpts answer it: answer directly in 1-5 sentences (or a short list), no preamble.
- If the excerpts do NOT answer it: say so plainly in the first sentence (e.g. "The retrieved excerpts don't state ..."). Add more ONLY if it partly answers the actual question (a value, limit or condition the user could use for what they asked); otherwise stop there. Never add loosely related facts, never use a "Related:" label, never guess.
- If an excerpt describes a different part than the one asked about, say which part it describes; don't transfer specs between parts.
- No greetings, sign-offs, apologies, or "As an AI". Don't mention excerpt numbers; say "the datasheet" / "the reference manual" if needed.
Return STRICT JSON: {{"reply": "...", "answered": true|false}}"""

def answers():
    items = load("items.json", [])
    todo = [i for i, it in enumerate(items) if "reply" not in it]
    def work(i):
        it = items[i]
        return i, claude_json(A_SYSTEM, A_USER.format(ctx="\n\n".join(it["context"]), q=it["question"]), 3000)
    with cf.ThreadPoolExecutor(8) as ex:
        for n, (i, j) in enumerate(ex.map(work, todo)):
            if j and j.get("reply"):
                items[i]["reply"], items[i]["answered"] = j["reply"].strip(), bool(j.get("answered"))
            else:
                items[i]["reply"] = None
            if n % 25 == 0:
                save("items.json", items); print(f"  answers {sum('reply' in x for x in items)}/{len(items)}", flush=True)
    save("items.json", items); print(f"answers: {sum(bool(x.get('reply')) for x in items)} written")

G_USER = """Write {n} diverse questions a hardware/embedded engineer might ask an assistant that are GENERAL knowledge, not lookups in a specific datasheet: C/C++/Python/Rust code, RTOS and Linux concepts, digital design, signal integrity, power, communication protocols (I2C/SPI/CAN/USB/PCIe) in general, math/unit conversions, debugging strategy, career/tooling questions. Mix short and detailed asks. No questions about specific part numbers' specs.
Return STRICT JSON: {{"questions": ["...", ...]}}"""
GA_USER = """Retrieved excerpts (tangential; the question is general knowledge):
\"\"\"
{ctx}
\"\"\"

User question: {q}

Write Skippy's reply: answer from general engineering knowledge, correct and concise (code in fenced blocks if asked). Don't invent datasheet specifics for any particular part; don't refer to the excerpts unless they genuinely help. No greetings or sign-offs.
Return STRICT JSON: {{"reply": "..."}}"""

T_USER = """Retrieved excerpts Skippy was given:
\"\"\"
{ctx}
\"\"\"

User question: {q}

Skippy's current reply (the excerpts don't answer the question):
\"\"\"
{reply}
\"\"\"

Rewrite the reply. Rules:
- First sentence: say plainly what the excerpts don't state (keep the current wording if it's fine).
- Keep any further information ONLY if it partly answers the actual question: a value, limit or condition the user could directly use for what they asked. Drop loosely related facts (neighbouring parameters, other peripherals, register trivia) entirely.
- Never use a "Related:" label or a stock lead-in. If you keep something, work it into plain prose and vary the phrasing.
- Copy any kept value verbatim from the current reply; add nothing new.
- No greetings, sign-offs, apologies or suggestions to consult other documents.
Return STRICT JSON: {{"reply": "..."}}"""

def tidy():
    """v5.1: v5 abstentions carried a 'Related:' fact ~89% of the time (the A_USER rule asked for one), and
    v5 learned it as a template. Rewrite them so extra facts appear only when they partly answer the
    question. Originals are kept in reply_v1."""
    items = load("items.json", [])
    todo = [i for i, it in enumerate(items) if it.get("kind") == "unanswerable" and it.get("reply")
            and not it.get("answered") and "reply_v1" not in it]
    def work(i):
        it = items[i]
        return i, claude_json(A_SYSTEM, T_USER.format(ctx="\n\n".join(it["context"]), q=it["question"], reply=it["reply"]), 3000)
    with cf.ThreadPoolExecutor(8) as ex:
        for n, (i, j) in enumerate(ex.map(work, todo)):
            if j and j.get("reply"):
                items[i]["reply_v1"], items[i]["reply"] = items[i]["reply"], j["reply"].strip()
            if n % 50 == 0:
                save("items.json", items); print(f"  tidy {n}/{len(todo)}", flush=True)
    save("items.json", items)
    una = [it for it in items if it.get("kind") == "unanswerable" and it.get("reply") and not it.get("answered")]
    rel = sum(bool(re.search(r"\bRelated\b[^.:]{0,40}:", it["reply"])) for it in una)
    print(f"tidy: {sum('reply_v1' in it for it in items)} rewritten; 'Related:' now in {rel}/{len(una)}")

def general():
    g = load("general.json", [])
    if not g:
        qs = []
        while len(qs) < N_GENERAL:
            j = claude_json("Return STRICT JSON only.", G_USER.format(n=50), 8000) or {}
            qs += [q for q in j.get("questions", []) if q not in qs]
        g = [{"question": q} for q in qs[:N_GENERAL]]
    for it in g:
        if "context" not in it:
            it["context"], it["sources"] = skippy_retrieve(it["question"])
    save("general.json", g)
    def work(i):
        return i, claude_json(A_SYSTEM, GA_USER.format(ctx="\n\n".join(g[i]["context"]), q=g[i]["question"]), 4000)
    with cf.ThreadPoolExecutor(8) as ex:
        for i, j in ex.map(work, [i for i, it in enumerate(g) if "reply" not in it]):
            g[i]["reply"] = (j or {}).get("reply")
    save("general.json", g); print(f"general: {sum(bool(x.get('reply')) for x in g)} written")

def build():
    import faithfulness_gate as fg
    items = [it for it in load("items.json", []) if it.get("reply")]
    rnd = random.Random(SEED)
    ans = [it for it in items if it["kind"] == "answerable" and it.get("answered")]
    una = [it for it in items if it["kind"] == "unanswerable" and not it.get("answered")]
    rnd.shuffle(ans); rnd.shuffle(una)
    kept, dropped = [], {"hhem": 0, "pii": 0}
    for pool, n in ((ans, N_ANSWERABLE), (una, N_UNANSWERABLE)):
        took = 0
        for it in pool:
            if took >= n:
                break
            if not fg.gate(it["reply"], "\n\n".join(it["context"]))["faithful"]:
                dropped["hhem"] += 1; continue
            kept.append(it); took += 1
    general = [g for g in load("general.json", []) if g.get("reply")]
    refusals = json.load(open(os.path.join(REPO, "training/data/refusal_alpaca.json")))
    rnd.shuffle(refusals)
    rows = [{"messages": [{"role": "system", "content": system_prompt(it["context"])},
                          {"role": "user", "content": it["question"]},
                          {"role": "assistant", "content": it["reply"]}], "kind": it["kind"]} for it in kept]
    rows += [{"messages": [{"role": "system", "content": system_prompt(g["context"])},
                           {"role": "user", "content": g["question"]},
                           {"role": "assistant", "content": g["reply"]}], "kind": "general"} for g in general]
    for r in refusals[:N_REFUSAL]:  # existing synthetic refusals, served through real retrieval context
        ctx, _ = skippy_retrieve(r["instruction"])
        rows.append({"messages": [{"role": "system", "content": system_prompt(ctx)},
                                  {"role": "user", "content": r["instruction"]},
                                  {"role": "assistant", "content": r["output"]}], "kind": "refusal"})
    clean = []
    for r in rows:  # assistant turns must never carry a phone number or email address
        if PHONE.search(r["messages"][2]["content"]) or EMAIL.search(r["messages"][2]["content"]):
            dropped["pii"] += 1; continue
        clean.append(r)
    rnd.shuffle(clean)
    n_val = max(40, len(clean) // 20)
    save("train_messages.json", clean[n_val:]); save("val_messages.json", clean[:n_val])
    from collections import Counter
    report = {"train": len(clean) - n_val, "val": n_val, "by_kind": Counter(r["kind"] for r in clean), "dropped": dropped,
              "teacher": TEACHER, "rag_k": RAG_K, "holdout_chunks": len(load("holdout.json", []))}
    save("build_report.json", report); print(json.dumps(report, indent=1, default=dict))

if __name__ == "__main__":
    stages = {"holdout": holdout, "sample": sample, "questions": questions, "retrieve": retrieve,
              "answers": answers, "tidy": tidy, "general": general, "build": build}
    from _keys import ensure_keys
    ensure_keys(("ANTHROPIC_API_KEY",))
    for s in (list(stages) if sys.argv[1:] == ["all"] else sys.argv[1:]):
        print(f"=== {s} ===", flush=True); stages[s]()
