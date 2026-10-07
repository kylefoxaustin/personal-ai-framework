#!/usr/bin/env python3
"""10-task agentic-edge benchmark on the 5090 — diverse resource profiles, grounded in
OpenClaw's capability set (A/B), measured on Skippy with a GPU sampler around each task.

Each task: fork an nvidia-smi sampler -> run (dispatched by kind) -> stop -> record
wall_s + telemetry (generate only) + GPU signature + FULL output (for the coherence gate:
a task's resource numbers only count if its output is coherent). One JSON per task.

Run:   python3 scripts/agentic_bench/trace_bench.py [task ...]   (default: all 10)
"""
import json, os, subprocess, sys, time, statistics, urllib.request
from datetime import datetime
import requests

BASE = "http://localhost:8080"; USER, PASSWORD = "kyle", "123456"
REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
OUT = os.path.join(REPO, "eval/results/ladder/agentic")
AUDIO = os.path.join(REPO, "knowledge/recordings/resources_sample1.wav")
IMAGE = None  # picked at runtime (newest screenshot)

def tok():
    return requests.post(f"{BASE}/auth/login", json={"username": USER, "password": PASSWORD}, timeout=15).json()["token"]

# ---- GPU sampler (sm%=compute, mem%=bandwidth proxy, power) ----
def sampler_start(path):
    f = open(path, "w")
    p = subprocess.Popen(["nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
        "--format=csv,noheader,nounits", "-lms", "200"], stdout=f, stderr=subprocess.DEVNULL)
    return p, f
def sampler_summ(path):
    rows = []
    for ln in open(path):
        a = [x.strip() for x in ln.split(",")]
        if len(a) >= 4:
            try: rows.append((float(a[0]), float(a[1]), float(a[2]), float(a[3])))
            except ValueError: pass
    if not rows: return {}
    return {"samples": len(rows),
            "sm_pct_mean": round(statistics.mean(r[0] for r in rows), 1),
            "mem_pct_mean": round(statistics.mean(r[1] for r in rows), 1),
            "power_w_mean": round(statistics.mean(r[2] for r in rows), 1),
            "sm_pct_peak": max(r[0] for r in rows), "vram_mib_peak": max(r[3] for r in rows)}

# ---- task runners (return dict: text, telemetry, extra) ----
H = {}
def gen(prompt, rag=False, rag_k=3, mt=256, skip_loop=False):
    body = {"prompt": prompt, "use_rag": rag, "rag_k": rag_k, "max_tokens": mt,
            "include_telemetry": True, "skip_agent_loop": skip_loop}
    d = requests.post(f"{BASE}/generate", json=body, headers=H, timeout=600).json()
    return {"text": d.get("text", ""), "telemetry": d.get("telemetry") or {},
            "rag_docs": (d.get("telemetry") or {}).get("rag_docs_used")}

def run_email():  return gen("Write a clear, friendly email to a colleague introducing the NXP i.MX 95 for a new edge-AI product. Cover what it is, its Neutron NPU, and why it suits edge inference. Ground the specifics in the datasheets.", rag=True, rag_k=4, mt=600)
def run_spec_rag(): return gen("Using the datasheets, what does the i.MX 95 Neutron NPU do, and what are its key features?", rag=True, rag_k=4, mt=220)
def run_short_chat(): return gen("In two or three sentences, what is the NXP i.MX 95?", rag=True, rag_k=3, mt=90)
def run_doc_brief(): return gen("Using the datasheets, generate a concise one-page product brief for the NXP i.MX 95: overview, NPU, memory/connectivity, and target markets.", rag=True, rag_k=8, mt=768)

def run_web_summarize():
    t0 = time.time()
    r = requests.get(f"{BASE}/search/web", params={"query": "NXP i.MX 95 applications processor edge AI"}, headers=H, timeout=60)
    web = r.json(); web_s = time.time() - t0
    results = web.get("results") or web
    snippet = json.dumps(results)[:2000]
    g = gen(f"Summarize these web search results about the NXP i.MX 95 in 4-5 sentences:\n{snippet}", rag=False, mt=220)
    g["extra"] = {"web_search_s": round(web_s, 3), "web_status": web.get("status", "?"),
                  "n_results": len(results) if isinstance(results, list) else None}
    return g

def _agent(name, params):
    return requests.post(f"{BASE}/agent/execute", json={"name": name, "params": params}, headers=H, timeout=120).json()
def run_file_ops():
    t0 = time.time()
    WS = "/root/.personal-ai/users/kyle/skippy-workspace"
    # ensure the file exists (write_file resolves the workspace correctly)
    _agent("write_file", {"path": "imx95_notes.txt", "content":
        "i.MX 95: 6x Cortex-A55, eIQ Neutron NPU 2.0 TOPS, Mali GPU, ISP, LPDDR5X, "
        "PCIe Gen3, GbE-TSN. Targets automotive/industrial/consumer edge AI."})
    # read_file needs the FULL path (known bug: read_file doesn't resolve the workspace)
    rd = _agent("read_file", {"path": f"{WS}/imx95_notes.txt"})
    content = rd.get("result", "")
    g = gen(f"Summarize these notes in 3 bullet points:\n{content}", rag=False, mt=180)
    wr = _agent("write_file", {"path": "imx95_summary.txt", "content": g["text"]})
    g["extra"] = {"read_ok": not str(rd.get("result","")).startswith("Error"),
                  "write_result": str(wr.get("result",""))[:80], "file_ops_total_s": round(time.time()-t0,3)}
    return g
def run_run_script():
    t0 = time.time(); d = _agent("run_script", {"path": "specsum.py"})
    return {"text": str(d.get("result", "")), "telemetry": {}, "extra": {"run_script_s": round(time.time()-t0,3)}}

def _upload(path, endpoint, extra_data=None):
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh)}
        r = requests.post(f"{BASE}{endpoint}", files=files, data=extra_data or {}, headers=H, timeout=600)
    return r.json()
def run_transcribe():
    d = _upload(AUDIO, "/upload/transcribe", {"title": "Sample", "summarize": "false"})
    txt = d.get("full_transcript") or d.get("transcript_preview") or d.get("text") or json.dumps(d)[:600]
    return {"text": txt[:1600], "telemetry": {}, "extra": {"engine": "whisper-base"}}
def run_meeting_summarize():
    d = _upload(AUDIO, "/upload/transcribe", {"title": "Sample", "summarize": "true"})
    summ = d.get("summary") or d.get("text") or json.dumps(d)[:800]
    return {"text": (summ if isinstance(summ,str) else json.dumps(summ))[:1600], "telemetry": {}, "extra": {"engine": "whisper->llm"}}
def run_ocr():
    d = _upload(IMAGE, "/upload/ocr", {})
    txt = d.get("extracted_text") or d.get("full_text") or d.get("text_preview") or d.get("text") or json.dumps(d)[:600]
    return {"text": txt[:1600], "telemetry": {}, "extra": {"engine": "tesseract", "image": os.path.basename(IMAGE)}}
def run_multi_tool_chain():
    return gen("Do these steps: (1) note the key specs of the i.MX 95 from the datasheets, "
               "(2) write them to a file called chain_notes.txt, (3) draft a short email to a "
               "colleague summarizing them.", rag=True, rag_k=4, mt=600)

TASKS = {
    "email": run_email, "web_summarize": run_web_summarize, "spec_rag": run_spec_rag,
    "file_ops": run_file_ops, "transcribe": run_transcribe, "ocr": run_ocr,
    "meeting_summarize": run_meeting_summarize, "run_script": run_run_script,
    "multi_tool_chain": run_multi_tool_chain, "doc_brief": run_doc_brief,
}

def main():
    global H, IMAGE
    import glob
    ds = sorted(glob.glob(os.path.join(REPO, "knowledge/images/ocr_imx_datasheet*.png")))
    imgs = ds or sorted(glob.glob(os.path.join(REPO, "knowledge/images/*.png")), key=os.path.getsize, reverse=True)
    IMAGE = imgs[0] if imgs else None
    H = {"Authorization": f"Bearer {tok()}"}
    which = sys.argv[1:] or list(TASKS)
    os.makedirs(OUT, exist_ok=True)
    for name in which:
        fn = TASKS.get(name)
        if not fn: print(f"  ?? unknown task {name}"); continue
        tp = os.path.join(OUT, f"_gputrace_bench_{name}.csv")
        samp, f = sampler_start(tp); time.sleep(0.5)
        t0 = time.time()
        try: r = fn()
        except Exception as e: r = {"text": f"EXCEPTION: {e}", "telemetry": {}, "extra": {"error": str(e)}}
        wall = time.time() - t0
        time.sleep(0.3); samp.terminate(); f.close()
        te = r.get("telemetry", {})
        rec = {"task": name, "ts": datetime.now().isoformat(timespec="seconds"), "board": "rtx5090",
               "wall_s": round(wall, 3),
               "telemetry": {k: te.get(k) for k in ("prefill_ms","decode_ms","prefill_tok_per_s","decode_tok_per_s","rag_docs_used") if k in te},
               "gpu": sampler_summ(tp), "extra": r.get("extra", {}), "output": (r.get("text") or "")[:1600]}
        json.dump(rec, open(os.path.join(OUT, f"bench_{name}_{datetime.now():%Y%m%d-%H%M%S}.json"), "w"), indent=2)
        pf, dc = te.get("prefill_ms") or 0, te.get("decode_ms") or 0
        print(f"[{name:18s}] wall={wall:6.2f}s  pf={pf:7.1f} dc={dc:7.1f}  sm%={rec['gpu'].get('sm_pct_mean')} mem%={rec['gpu'].get('mem_pct_mean')} pwr={rec['gpu'].get('power_w_mean')}  :: {rec['output'][:70].replace(chr(10),' ')}")

if __name__ == "__main__":
    main()
