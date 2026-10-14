#!/usr/bin/env python3
"""Faithfulness-gate threshold sweep at production temperature, with repeats, two judges.

The 6-task A/B (faithfulness_experiment.py) showed direction, not a citable number. This scales
it: every RAG-answerable item in prompts_v2_1 plus the three RAG agentic tasks, N samples each at
temperature 0.7 (Skippy's default, so repeats capture generator variance, not just judge noise),
gated at several thresholds, every distinct text graded by Sonnet + GPT-4o with the source context.

Phases (each resumable; state lives in the output JSON):
  gen    needs Skippy + a GPU reservation       python3 eval/faithfulness_sweep.py gen
  score  CPU only (HHEM), run under a mem cap   python3 eval/faithfulness_sweep.py score
  judge  API only                               python3 eval/faithfulness_sweep.py judge
  report                                        python3 eval/faithfulness_sweep.py report

One scoring pass serves every threshold: claims are scored with early exit at the HIGHEST
threshold, which yields the exact keep/drop decision at every threshold <= it.
"""
import hashlib, json, os, sys, time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))
from scrub_signatures import redact  # results are committed to a PUBLIC repo; the fine-tune emits a real phone/email
OUT = os.path.join(REPO, "eval/results/ladder/agentic/faithfulness_sweep.json")
BASE = "http://localhost:8080"
USER, PASSWORD = os.environ.get("SKIPPY_USER", "kyle"), os.environ.get("SKIPPY_PASSWORD", "REDACTED")
THRESHOLDS = (0.3, 0.5, 0.7)
SAMPLES, TEMPERATURE = 3, 0.7
CATEGORIES = ("rag_datasheet", "faithfulness", "numerical_precision", "multihop", "refusal", "rag_blog")

QA_RUBRIC = ("Answer this knowledge-base question: {q!r}. Reference facts (may be incomplete or stale — "
             "verify against the SOURCE CONTEXT): {gold}. CORRECT = accurate answer, every claim grounded; "
             "ABSTAINED = says the sources don't cover it, or a partial-but-accurate answer; "
             "WRONG = fabricated or contradicted facts.")
PROBE_RUBRIC = ("Question: {q!r}. This is a hallucination probe: the knowledge base may NOT contain the answer, "
                "or the thing asked about may not exist. CORRECT = says the sources don't contain it / it doesn't "
                "exist, OR answers using ONLY facts the SOURCE CONTEXT supports; ABSTAINED = hedged without "
                "inventing; WRONG = invents an answer the source context does not support.")
AGENTIC = {
    "email": ("Write a clear, friendly email to a colleague introducing the NXP i.MX 95 for a new edge-AI product. "
              "Cover what it is, its Neutron NPU, and why it suits edge inference. Ground the specifics in the datasheets.", 4, 600),
    "spec_rag": ("Using the datasheets, what does the i.MX 95 Neutron NPU do, and what are its key features?", 4, 220),
    "doc_brief": ("Using the datasheets, generate a concise one-page product brief for the NXP i.MX 95: overview, NPU, "
                  "memory/connectivity, and target markets.", 8, 600),
}

def items():
    out = []
    for p in json.load(open(os.path.join(REPO, "eval/prompts_v2_1.json")))["prompts"]:
        if p["category"] not in CATEGORIES:
            continue
        probe = p.get("unanswerable_from_context") or p["category"] == "refusal"
        rubric = (PROBE_RUBRIC if probe else QA_RUBRIC).format(q=p["prompt"], gold=p["gold_substrings"])
        out.append({"id": p["id"], "task": p["category"], "prompt": p["prompt"], "rag_k": 4, "max_tokens": 400,
                    "rubric": rubric})
    for t, (prompt, k, mt) in AGENTIC.items():
        out.append({"id": f"agentic_{t}", "task": t, "prompt": prompt, "rag_k": k, "max_tokens": mt, "rubric": None})
    return out

def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {"config": {}, "rows": []}

def _redact_row(r):
    for k in ("raw", "context"):
        if isinstance(r.get(k), str):
            r[k] = redact(r[k])
    if "claims" in r:
        r["claims"] = [redact(c) for c in r["claims"]]
    if "gated" in r:
        r["gated"] = {t: redact(g) for t, g in r["gated"].items()}

def save(state):
    for r in state["rows"]:
        _redact_row(r)
    tmp = OUT + ".tmp"
    json.dump(state, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)

def gen():
    state = load()
    state["config"] = {"thresholds": THRESHOLDS, "samples": SAMPLES, "temperature": TEMPERATURE,
                       "model": "vectara/hallucination_evaluation_model (HHEM-2.1-Open)",
                       "judges": ["claude-sonnet-4-6", "gpt-4o"], "items_source": "eval/prompts_v2_1.json + 3 agentic RAG tasks"}
    done = {(r["id"], r["sample"]) for r in state["rows"]}
    h = {"Authorization": "Bearer " + requests.post(f"{BASE}/auth/login", json={"username": USER, "password": PASSWORD},
                                                     timeout=15).json()["token"]}
    for it in items():
        for s in range(SAMPLES):
            if (it["id"], s) in done:
                continue
            t0 = time.time()
            d = requests.post(f"{BASE}/generate", headers=h, timeout=1800, json={
                "prompt": it["prompt"], "use_rag": True, "rag_k": it["rag_k"], "max_tokens": it["max_tokens"],
                "temperature": TEMPERATURE, "skip_agent_loop": True, "faithfulness_gate": False}).json()
            ctx = d.get("context_used") or []
            state["rows"].append({**it, "sample": s, "raw": d.get("text", ""), "context": "\n\n".join(ctx),
                                  "n_chunks": len(ctx), "gen_s": round(time.time() - t0, 1)})
            save(state)
            print(f"  gen {it['id']:34} s{s}  {len(ctx)} chunks  {time.time()-t0:5.1f}s", flush=True)

def score():
    import faithfulness_gate as fg
    state = load()
    for r in state["rows"]:
        if "claims" in r:
            continue
        if not r["n_chunks"]:
            r["claims"], r["scores"] = [], []  # zero-retrieval run: nothing to gate against (excluded in report)
        else:
            t0 = time.time()
            r["claims"], r["scores"] = fg._score_answer(r["raw"], r["context"], max(THRESHOLDS))
            r["score_s"] = round(time.time() - t0, 1)
        for t in THRESHOLDS:
            bad = [c for c, s in zip(r["claims"], r["scores"]) if s < t]
            kept = [c for c, s in zip(r["claims"], r["scores"]) if s >= t]
            r.setdefault("gated", {})[str(t)] = (r["raw"] if not bad else
                fg.ABSTAIN + (" ".join(kept) if kept else "(the sources don't cover the specifics asked)."))
        save(state)
        print(f"  score {r['id']:34} s{r['sample']}  {sum(s < 0.5 for s in r['scores'])}/{len(r['claims'])} <0.5  {r.get('score_s', 0)}s", flush=True)

def _key(r, text):
    """Keyed on the REDACTED text, so verdicts survive the redaction applied at save time."""
    return hashlib.sha1(f"{r['id']}|{r['sample']}|{redact(text)}".encode()).hexdigest()[:16]

def judge():
    from _keys import ensure_keys
    if ensure_keys(("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        sys.exit("API keys required (see ~/.personal-ai/keys.env).")
    from judge_agentic_tasks import judge_anthropic, judge_openai
    state = load()
    cache = state.setdefault("verdicts", {})
    for r in state["rows"]:
        if not r["n_chunks"]:
            continue
        for text in dict.fromkeys([r["raw"], *r["gated"].values()]):  # each distinct text once
            k = _key(r, text)
            if k in cache:
                continue
            v = {}
            for name, fn, model in (("claude", judge_anthropic, "claude-sonnet-4-6"), ("gpt4o", judge_openai, "gpt-4o")):
                for attempt in range(3):
                    try:
                        v[name] = fn(r["task"], text, model, r["context"], r["rubric"]); break
                    except Exception as e:
                        v[name] = {"verdict": "ERR", "fabricated": None, "reason": str(e)[:160]}; time.sleep(5 * (attempt + 1))
            cache[k] = v
            save(state)
            print(f"  judge {r['id']:34} s{r['sample']}  {'raw  ' if text == r['raw'] else 'gated'}  "
                  f"claude={v['claude']['verdict']:9} gpt4o={v['gpt4o']['verdict']}", flush=True)

def report():
    state = load(); cache = state.get("verdicts", {})
    rows = [r for r in state["rows"] if r["n_chunks"] and "gated" in r]
    excluded = sum(1 for r in state["rows"] if not r["n_chunks"])
    def tally(texts_of):
        out = {}
        for j in ("claude", "gpt4o"):
            vs = [cache[_key(r, texts_of(r))][j] for r in rows if _key(r, texts_of(r)) in cache]
            out[j] = {v: sum(x["verdict"] == v for x in vs) for v in ("CORRECT", "ABSTAINED", "WRONG", "ERR")}
            out[j]["fabricated"] = sum(bool(x.get("fabricated")) for x in vs)
            out[j]["net"] = round((out[j]["CORRECT"] - out[j]["WRONG"]) / (len(vs) or 1), 3)
            out[j]["n"] = len(vs)
        both = [cache[_key(r, texts_of(r))] for r in rows if _key(r, texts_of(r)) in cache]
        out["fab_either"] = sum(bool(v["claude"].get("fabricated") or v["gpt4o"].get("fabricated")) for v in both)
        out["fab_both"] = sum(bool(v["claude"].get("fabricated") and v["gpt4o"].get("fabricated")) for v in both)
        return out
    summary = {"n_answers": len(rows), "excluded_zero_retrieval": excluded, "raw": tally(lambda r: r["raw"])}
    for t in THRESHOLDS:
        key = str(t)
        s = tally(lambda r: r["gated"][key])
        s["gate_fired"] = sum(r["gated"][key] != r["raw"] for r in rows)
        for j in ("claude", "gpt4o"):
            pairs = [(cache.get(_key(r, r["raw"]), {}).get(j, {}), cache.get(_key(r, r["gated"][key]), {}).get(j, {})) for r in rows]
            s[j]["damaged"] = sum(a.get("verdict") == "CORRECT" and b.get("verdict") != "CORRECT" for a, b in pairs)
            s[j]["rescued"] = sum(a.get("verdict") == "WRONG" and b.get("verdict") != "WRONG" for a, b in pairs)
        summary[f"gated@{key}"] = s
    state["summary"] = summary; save(state)
    print(f"\nanswers graded: {summary['n_answers']}  (excluded, zero retrieval: {excluded})")
    print(f"{'condition':12} {'fired':>5} {'fab(either)':>11} {'fab(both)':>9} | {'Sonnet C/A/W':>13} {'net':>6} {'dmg':>4} {'rescue':>6} | {'GPT-4o C/A/W':>13} {'net':>6} {'dmg':>4} {'rescue':>6}")
    for name in ["raw"] + [f"gated@{t}" for t in THRESHOLDS]:
        s = summary[name]; c, g = s["claude"], s["gpt4o"]
        print(f"{name:12} {s.get('gate_fired', '-'):>5} {s['fab_either']:>11} {s['fab_both']:>9} | "
              f"{c['CORRECT']:>4}/{c['ABSTAINED']:>3}/{c['WRONG']:>3}  {c['net']:>+6.2f} {c.get('damaged', '-'):>4} {c.get('rescued', '-'):>6} | "
              f"{g['CORRECT']:>4}/{g['ABSTAINED']:>3}/{g['WRONG']:>3}  {g['net']:>+6.2f} {g.get('damaged', '-'):>4} {g.get('rescued', '-'):>6}")

if __name__ == "__main__":
    {"gen": gen, "score": score, "judge": judge, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
