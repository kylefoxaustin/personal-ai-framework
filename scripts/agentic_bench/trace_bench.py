#!/usr/bin/env python3
"""10-task agentic-edge benchmark on the 5090 — diverse resource profiles, grounded in
OpenClaw's capability set (A/B), measured on Skippy with a GPU sampler around each task.

Each task: fork an nvidia-smi sampler -> run (dispatched by kind) -> stop -> record
wall_s + telemetry (generate only) + GPU signature + FULL output (for the coherence gate:
a task's resource numbers only count if its output is coherent). One JSON per task.

Run:   python3 scripts/agentic_bench/trace_bench.py [task ...]   (default: all 10)
"""
import json, os, subprocess, sys, time, statistics, threading, urllib.request
from datetime import datetime
import requests

BASE = "http://localhost:8080"; USER, PASSWORD = "kyle", "REDACTED"
REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
OUT = os.environ.get("AGENTIC_OUT") or os.path.join(REPO, "eval/results/ladder/agentic")
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

# ---- CPU sampler (system-wide, /proc/stat) ----
# Why system-wide: the LLM runs in the llm-server CONTAINER, a separate process; sampling only
# this harness process would miss it. System-wide %busy (all cores share the host kernel) captures
# the container's decode/sample work PLUS host-side retrieval + tool glue. The headline is
# cores_busy (busy_fraction x ncores) — "how many CPU cores this task needs", the edge SoC budget
# even when an accelerator does the matmuls (decode still burns CPU on tokenize/sample/detokenize;
# OCR/ASR are almost entirely CPU work).
class CpuSampler:
    def __init__(self, interval=0.2):
        self.interval = interval; self.samples = []; self._stop = threading.Event()
        self.ncores = os.cpu_count() or 1
    @staticmethod
    def _read():
        with open("/proc/stat") as fh:
            v = list(map(int, fh.readline().split()[1:]))
        idle = v[3] + (v[4] if len(v) > 4 else 0)   # idle + iowait
        return idle, sum(v)
    def _run(self):
        pi, pt = self._read()
        while not self._stop.wait(self.interval):
            i, t = self._read(); di, dt = i - pi, t - pt; pi, pt = i, t
            if dt > 0: self.samples.append(max(0.0, 1 - di / dt))
    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self
    def stop(self):
        self._stop.set(); self._t.join(timeout=1)
        if not self.samples: return {}
        m, pk = statistics.mean(self.samples), max(self.samples)
        return {"samples": len(self.samples), "ncores": self.ncores,
                "cpu_pct_mean": round(m * 100, 1), "cpu_pct_peak": round(pk * 100, 1),
                "cores_busy_mean": round(m * self.ncores, 2), "cores_busy_peak": round(pk * self.ncores, 2)}

# ---- task runners (return dict: text, telemetry, extra) ----
H = {}
MODEL = "?"
def gen(prompt, rag=False, rag_k=3, mt=256, skip_loop=False):
    body = {"prompt": prompt, "use_rag": rag, "rag_k": rag_k, "max_tokens": mt,
            "include_telemetry": True, "skip_agent_loop": skip_loop, "temperature": 0}
    d = requests.post(f"{BASE}/generate", json=body, headers=H, timeout=600).json()
    return {"text": d.get("text", ""), "telemetry": d.get("telemetry") or {},
            "rag_docs": (d.get("telemetry") or {}).get("rag_docs_used"),
            # the retrieved chunks the model was given — the judges need them to tell a grounded
            # datasheet detail from an invented one (the 5-line GROUND_TRUTH alone can't)
            "context": "\n\n".join(d.get("context_used") or [])}

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
                  "n_results": len(results) if isinstance(results, list) else None,
                  "work_ms": round(web_s * 1000, 1)}  # the web fetch is the "work"
    g["context"] = snippet
    return g

def _agent(name, params):
    return requests.post(f"{BASE}/agent/execute", json={"name": name, "params": params}, headers=H, timeout=120).json()
def run_file_ops():
    t0 = time.time()
    WS = "/root/.personal-ai/users/kyle/skippy-workspace"
    # ensure the file exists (write_file resolves the workspace correctly). Time the file-tool
    # calls separately from the LLM summary — the file I/O is the "work".
    tt = time.time()
    _agent("write_file", {"path": "imx95_notes.txt", "content":
        "i.MX 95: 6x Cortex-A55, eIQ Neutron NPU 2.0 TOPS, Mali GPU, ISP, LPDDR5/LPDDR4X, "
        "PCIe Gen3, GbE-TSN. Targets automotive/industrial/consumer edge AI."})
    # read_file needs the FULL path (known bug: read_file doesn't resolve the workspace)
    rd = _agent("read_file", {"path": f"{WS}/imx95_notes.txt"})
    tool_pre = time.time() - tt
    content = rd.get("result", "")
    g = gen(f"Summarize these notes in 3 bullet points:\n{content}", rag=False, mt=180)
    tt = time.time()
    wr = _agent("write_file", {"path": "imx95_summary.txt", "content": g["text"]})
    tool_ms = round((tool_pre + (time.time() - tt)) * 1000, 1)
    g["extra"] = {"read_ok": not str(rd.get("result","")).startswith("Error"),
                  "write_result": str(wr.get("result",""))[:80], "file_ops_total_s": round(time.time()-t0,3),
                  "work_ms": tool_ms}  # file read+write = the work
    g["context"] = content
    return g
def run_run_script():
    t0 = time.time(); d = _agent("run_script", {"path": "specsum.py"}); s = time.time()-t0
    return {"text": str(d.get("result", "")), "telemetry": {}, "extra": {"run_script_s": round(s,3), "work_ms": round(s*1000,1)}}

def _upload(path, endpoint, extra_data=None):
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh)}
        r = requests.post(f"{BASE}{endpoint}", files=files, data=extra_data or {}, headers=H, timeout=600)
    return r.json()
def run_transcribe():
    d = _upload(AUDIO, "/upload/transcribe", {"title": "Sample", "summarize": "false"})
    txt = d.get("full_transcript") or d.get("transcript_preview") or d.get("text") or json.dumps(d)[:600]
    tm = d.get("timing_ms") or {}
    return {"text": txt[:1600], "telemetry": {}, "extra": {"engine": "whisper-base", "work_ms": tm.get("transcribe")}}
def run_meeting_summarize():
    d = _upload(AUDIO, "/upload/transcribe", {"title": "Sample", "summarize": "true"})
    summ = d.get("summary") or d.get("text") or json.dumps(d)[:800]
    tm = d.get("timing_ms") or {}
    return {"text": (summ if isinstance(summ,str) else json.dumps(summ))[:1600], "telemetry": {},
            "context": d.get("full_transcript", ""),
            "extra": {"engine": "whisper->llm", "work_ms": tm.get("transcribe"), "compose_ms": tm.get("summarize")}}
def run_ocr():
    d = _upload(IMAGE, "/upload/ocr", {})
    txt = d.get("extracted_text") or d.get("full_text") or d.get("text_preview") or d.get("text") or json.dumps(d)[:600]
    tm = d.get("timing_ms") or {}
    return {"text": txt[:1600], "telemetry": {}, "extra": {"engine": "tesseract", "image": os.path.basename(IMAGE), "work_ms": tm.get("ocr")}}
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
    global MODEL
    try: MODEL = requests.get(f"{BASE}/health", timeout=10).json().get("model") or "?"
    except Exception: MODEL = "?"
    which = sys.argv[1:] or list(TASKS)
    os.makedirs(OUT, exist_ok=True)
    # Warm the retrieval index so per-task "work" reflects steady-state, not a one-time ~6s cold
    # load landing on whichever RAG task runs first. (Cold-start is a real, separate phenomenon.)
    try: gen("warm up the retrieval index", rag=True, rag_k=4, mt=1)
    except Exception: pass
    for name in which:
        fn = TASKS.get(name)
        if not fn: print(f"  ?? unknown task {name}"); continue
        tp = os.path.join(OUT, f"_gputrace_bench_{name}.csv")
        samp, f = sampler_start(tp); cpu = CpuSampler().start(); time.sleep(0.5)
        t0 = time.time()
        try: r = fn()
        except Exception as e: r = {"text": f"EXCEPTION: {e}", "telemetry": {}, "extra": {"error": str(e)}}
        wall = time.time() - t0
        time.sleep(0.3); samp.terminate(); f.close(); cpu_summ = cpu.stop()
        te = r.get("telemetry", {}); ex = r.get("extra", {})
        # work_ms = the engine "work": explicit tool time (uploads/file/web/script) or, for RAG
        # /generate tasks, the retrieval time from telemetry. Everything else is overhead.
        work_ms = ex.get("work_ms")
        if work_ms is None: work_ms = (te.get("retrieval_ms") or 0) + (te.get("tool_ms") or 0)
        rec = {"task": name, "ts": datetime.now().isoformat(timespec="seconds"), "board": "rtx5090", "model": MODEL,
               "wall_s": round(wall, 3), "work_ms": work_ms,
               "telemetry": {k: te.get(k) for k in ("prefill_ms","decode_ms","host_ms","retrieval_ms","tool_ms","prefill_tok_per_s","decode_tok_per_s","rag_docs_used") if k in te},
               "gpu": sampler_summ(tp), "cpu": cpu_summ, "extra": ex, "output": (r.get("text") or "")[:1600],
               "context": r.get("context") or ""}
        json.dump(rec, open(os.path.join(OUT, f"bench_{name}_{datetime.now():%Y%m%d-%H%M%S}.json"), "w"), indent=2)
        if name in ("email", "spec_rag", "doc_brief", "multi_tool_chain") and te.get("rag_docs_used") == 0:
            print(f"  ⚠ {name}: RAG returned ZERO chunks — retrieval failed; the judge will skip this run")
        pf, dc = te.get("prefill_ms") or 0, te.get("decode_ms") or 0
        print(f"[{name:18s}] wall={wall:6.2f}s  pf={pf:7.1f} dc={dc:7.1f}  cpu={cpu_summ.get('cores_busy_mean')}cores sm%={rec['gpu'].get('sm_pct_mean')} mem%={rec['gpu'].get('mem_pct_mean')} pwr={rec['gpu'].get('power_w_mean')}  :: {rec['output'][:60].replace(chr(10),' ')}")

if __name__ == "__main__":
    main()
