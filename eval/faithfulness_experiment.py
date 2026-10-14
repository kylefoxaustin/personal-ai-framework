#!/usr/bin/env python3
"""A/B the HHEM faithfulness gate on the fabricating tasks: grade RAW vs GATED outputs.

For each generation task that fabricated, this gets (raw_output, context), runs the gate to
produce a safe gated_output (grounded sentences kept, unsupported ones dropped → abstention),
then grades BOTH with the two-judge correctness grader. Shows the fabrication count drop — the
point: an edge model that fabricates 5-6/10 can be made SAFE with a 110M gate, no bigger model.

    python3 eval/faithfulness_experiment.py            # needs Skippy up + API keys
"""
import json, os, sys, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faithfulness_gate as fg
from _keys import skippy_password
from judge_agentic_tasks import judge_anthropic, judge_openai, GROUND_TRUTH, RUBRIC

BASE = "http://localhost:8080"; USER = "kyle"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(REPO, "knowledge/recordings/resources_sample1.wav")

def tok():
    return requests.post(f"{BASE}/auth/login", json={"username": USER, "password": skippy_password()}, timeout=15).json()["token"]

H = {}

def gen(prompt, rag=False, rag_k=4, mt=400):
    d = requests.post(f"{BASE}/generate", json={"prompt": prompt, "use_rag": rag, "rag_k": rag_k,
        "max_tokens": mt, "temperature": 0, "skip_agent_loop": True}, headers=H, timeout=300).json()
    ctx = d.get("context_used") or []
    return d.get("text", ""), "\n".join(ctx if isinstance(ctx, list) else [str(ctx)])

def collect():
    """(task, raw_output, context) for the fabricating generation tasks."""
    items = {}
    t, c = gen("Write a clear, friendly email to a colleague introducing the NXP i.MX 95 for a new edge-AI product. Cover what it is, its Neutron NPU, and why it suits edge inference. Ground the specifics in the datasheets.", rag=True, rag_k=4, mt=600)
    items["email"] = (t, c)
    t, c = gen("Using the datasheets, what does the i.MX 95 Neutron NPU do, and what are its key features?", rag=True, rag_k=4, mt=220)
    items["spec_rag"] = (t, c)
    t, c = gen("Using the datasheets, generate a concise one-page product brief for the NXP i.MX 95: overview, NPU, memory/connectivity, and target markets.", rag=True, rag_k=8, mt=600)
    items["doc_brief"] = (t, c)
    # web_summarize: context = the web snippet
    r = requests.get(f"{BASE}/search/web", params={"query": "NXP i.MX 95 applications processor edge AI"}, headers=H, timeout=60).json()
    snip = json.dumps(r.get("results") or r)[:2000]
    t, _ = gen(f"Summarize these web search results about the NXP i.MX 95 in 4-5 sentences:\n{snip}", rag=False, mt=220)
    items["web_summarize"] = (t, snip)
    # meeting_summarize: context = the transcript
    with open(AUDIO, "rb") as fh:
        d = requests.post(f"{BASE}/upload/transcribe", files={"file": (os.path.basename(AUDIO), fh)},
                          data={"title": "Sample", "summarize": "true"}, headers=H, timeout=600).json()
    summ = d.get("summary"); summ = summ if isinstance(summ, str) else json.dumps(summ)
    items["meeting_summarize"] = (summ or "", d.get("full_transcript", ""))
    # file_ops: context = the notes (same text trace_bench writes to imx95_notes.txt)
    notes = ("i.MX 95: 6x Cortex-A55, eIQ Neutron NPU 2.0 TOPS, Mali GPU, ISP, LPDDR5/LPDDR4X, "
             "PCIe Gen3, GbE-TSN. Targets automotive/industrial/consumer edge AI.")
    t, _ = gen(f"Summarize these notes in 3 bullet points:\n{notes}", rag=False, mt=180)
    items["file_ops"] = (t, notes)
    return items

def grade(task, output, context=""):
    sv = judge_anthropic(task, output, "claude-sonnet-4-6", context)
    gv = judge_openai(task, output, "gpt-4o", context)
    return sv, gv

def main():
    global H
    from _keys import ensure_keys
    if ensure_keys(("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
        sys.exit("API keys required (see ~/.personal-ai/keys.env).")
    H = {"Authorization": f"Bearer {tok()}"}
    print("collecting raw outputs + context…")
    items = collect()
    print("priming HHEM gate…"); fg._model()
    rows, agg = [], {"raw_fab": 0, "gated_fab": 0, "raw_wrong": 0, "gated_wrong": 0,
                     "raw_correct": 0, "damaged": 0}  # damaged = a judge's CORRECT that the gate un-did
    for task, (raw, ctx) in items.items():
        gated, verdict = fg.gated_answer(raw, ctx, threshold=0.5)
        rs, rg = grade(task, raw, ctx)
        gs, gg = grade(task, gated, ctx)
        raw_fab = bool(rs.get("fabricated") or rg.get("fabricated"))
        gated_fab = bool(gs.get("fabricated") or gg.get("fabricated"))
        agg["raw_fab"] += raw_fab; agg["gated_fab"] += gated_fab
        agg["raw_wrong"] += (rs["verdict"] == "WRONG"); agg["gated_wrong"] += (gs["verdict"] == "WRONG")
        for r_, g_ in ((rs, gs), (rg, gg)):
            agg["raw_correct"] += (r_["verdict"] == "CORRECT")
            agg["damaged"] += (r_["verdict"] == "CORRECT" and g_["verdict"] != "CORRECT")
        rows.append({"task": task, "gate": verdict, "raw_output": raw, "gated_output": gated,
                     "context": ctx,
                     "raw": {"claude": rs["verdict"], "gpt4o": rg["verdict"], "fabricated": raw_fab},
                     "gated": {"claude": gs["verdict"], "gpt4o": gg["verdict"], "fabricated": gated_fab}})
        print(f"\n  [{task}]  gate: {verdict['n_unsupported']}/{verdict['n_claims']} claims unsupported "
              f"(min {verdict['min_score']})")
        print(f"     RAW   claude={rs['verdict']:9s} gpt4o={rg['verdict']:9s} fab={raw_fab}")
        print(f"     GATED claude={gs['verdict']:9s} gpt4o={gg['verdict']:9s} fab={gated_fab}")
    n = len(rows)
    print(f"\n  ===== HHEM gate A/B ({n} fabricating tasks) =====")
    print(f"  fabrications (either judge):  RAW {agg['raw_fab']}/{n}  ->  GATED {agg['gated_fab']}/{n}")
    print(f"  WRONG (claude):               RAW {agg['raw_wrong']}/{n}  ->  GATED {agg['gated_wrong']}/{n}")
    print(f"  CORRECT verdicts the gate un-did (both judges): {agg['damaged']}/{agg['raw_correct']}")
    out = os.path.join(REPO, "eval/results/ladder/agentic/faithfulness_gate_ab.json")
    sys.path.insert(0, os.path.join(REPO, "training"))
    from scrub_signatures import redact  # committed to a PUBLIC repo; the fine-tune emits a real phone/email
    for r in rows:
        for k in ("raw_output", "gated_output", "context"):
            r[k] = redact(r[k])
    json.dump({"threshold": 0.5, "model": "vectara/hallucination_evaluation_model (HHEM-2.1-Open)",
               "summary": agg, "rows": rows}, open(out, "w"), indent=2)
    print(f"  wrote {os.path.relpath(out, REPO)}")

if __name__ == "__main__":
    main()
