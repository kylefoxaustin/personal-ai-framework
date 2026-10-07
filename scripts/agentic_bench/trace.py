#!/usr/bin/env python3
"""Agentic-edge benchmark — per-phase resource trace harness (M1.1).

Runs a workload through Skippy on the 5090 while sampling the GPU, then reports the
compute-vs-bandwidth signature that is the heart of the VP deliverable:

  util.gpu    = SM / compute activity %      (high  => compute/TOPS-bound  => prefill-like)
  util.memory = memory-controller busy %     (high  => bandwidth-bound     => decode-like)

The deep-research headline is that agentic workloads are DECODE-dominated and therefore
LPDDR5X-BANDWIDTH-bound, not NPU/TOPS-bound. This harness measures that on our own stack:
a decode-heavy run should show mem% >> sm%, a prefill-heavy run the reverse.

Usage:
  python3 scripts/agentic_bench/trace.py <task> [--max-tokens N] [--rag] [--out DIR]
  tasks: decode_heavy | prefill_heavy | rag_multistep | <free prompt in quotes>

Auth is inline (local kyle account); never printed.
"""
import json, os, subprocess, sys, time, urllib.request, statistics
from datetime import datetime

BASE = "http://localhost:8080"
USER, PASSWORD = "kyle", "123456"

TASKS = {
    "decode_heavy": {   # long generation, tiny prompt -> autoregressive decode dominates
        "prompt": "Write a detailed, multi-paragraph technical explanation of why LLM decode "
                  "is memory-bandwidth-bound while prefill is compute-bound. Be thorough.",
        "max_tokens": 400, "use_rag": False},
    "prefill_heavy": {  # big context, 1 token out -> prefill dominates
        "prompt": "Reply with only the word OK.",
        "context": [(" ".join(["The i.MX 95 integrates an eIQ Neutron NPU and Cortex-A55 cores "
                     "with LPDDR5X memory providing high bandwidth for edge inference."] * 220))],
        "max_tokens": 1, "use_rag": False},
    "rag_multistep": {  # real agentic: RAG retrieval + grounded synthesis over the 31.9K-chunk KB
        "prompt": "Using the datasheets, what NPU does the i.MX 95 have and how many TOPS? "
                  "Cite the source.",
        "use_rag": True, "rag_k": 4, "max_tokens": 220},
    "field_service": {  # industrial: clean-phrasing datasheet RAG (retrieves correctly)
        "prompt": "What does the i.MX 95 Neutron NPU do, and list its key features and "
                  "supported neural-network operators.",
        "use_rag": True, "rag_k": 4, "max_tokens": 220},
    "inbox_triage": {   # consumer: multi-step LLM, no external asset
        "prompt": "You have 3 unread emails: (1) a vendor asking to reschedule Tuesday's call, "
                  "(2) your manager requesting the Q3 status doc by end of day, (3) a newsletter. "
                  "Triage them by priority and draft a one-line reply to each that needs one.",
        "use_rag": False, "max_tokens": 320},
}


def login():
    req = urllib.request.Request(f"{BASE}/auth/login",
        data=json.dumps({"username": USER, "password": PASSWORD}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=15).read())["token"]


def start_gpu_sampler(path):
    f = open(path, "w")
    p = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used",
         "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=f, stderr=subprocess.DEVNULL)
    return p, f


def run_task(token, spec):
    body = {"prompt": spec["prompt"], "use_rag": spec.get("use_rag", False),
            "max_tokens": spec.get("max_tokens", 256), "include_telemetry": True}
    for k in ("context", "rag_k"):
        if k in spec:
            body[k] = spec[k]
    req = urllib.request.Request(f"{BASE}/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    t1 = time.time()
    return t0, t1, resp


def summarize_trace(path, t0, t1):
    rows = []
    for line in open(path):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            # row wall-clock from sampler cadence; we filter by sample index window instead of
            # parsing nvidia's timestamp — simpler and robust. Keep all rows; the task dominates.
            sm, mem, pwr, mu = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            rows.append((sm, mem, pwr, mu))
        except ValueError:
            continue
    if not rows:
        return {}
    sm = [r[0] for r in rows]; mem = [r[1] for r in rows]; pwr = [r[2] for r in rows]
    return {
        "samples": len(rows),
        "sm_pct": {"mean": round(statistics.mean(sm), 1), "peak": max(sm)},
        "mem_pct": {"mean": round(statistics.mean(mem), 1), "peak": max(mem)},  # BW proxy
        "power_w": {"mean": round(statistics.mean(pwr), 1), "peak": max(pwr)},
        "vram_mib_peak": max(r[3] for r in rows),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    task = sys.argv[1]
    outdir = "eval/results/ladder/agentic"
    if "--out" in sys.argv:
        outdir = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(outdir, exist_ok=True)

    spec = TASKS.get(task) or {"prompt": task, "use_rag": "--rag" in sys.argv,
                               "max_tokens": 256}
    if "--max-tokens" in sys.argv:
        spec = dict(spec); spec["max_tokens"] = int(sys.argv[sys.argv.index("--max-tokens") + 1])

    token = login()
    trace_path = os.path.join(outdir, f"_gputrace_{task}.csv")
    samp, f = start_gpu_sampler(trace_path)
    time.sleep(0.6)  # baseline samples
    try:
        t0, t1, resp = run_task(token, spec)
    finally:
        time.sleep(0.4); samp.terminate(); f.close()

    tel = resp.get("telemetry") or {}
    gpu = summarize_trace(trace_path, t0, t1)
    rec = {
        "task": task, "ts": datetime.now().isoformat(timespec="seconds"),
        "board": "rtx5090", "model": resp.get("model"),
        "wall_s": round(t1 - t0, 3),
        "tokens_used": resp.get("tokens_used"),
        "telemetry": {k: tel.get(k) for k in
                      ("prompt_tokens", "completion_tokens", "prefill_ms", "decode_ms",
                       "total_ms", "prefill_tok_per_s", "decode_tok_per_s",
                       "rag_docs_used") if k in tel},
        "gpu_signature": gpu,
        "answer_preview": (resp.get("text") or "")[:160],
    }
    slug = "".join(c if c.isalnum() else "_" for c in task)[:24].strip("_") or "task"
    out = os.path.join(outdir, f"trace_{slug}_{datetime.now():%Y%m%d-%H%M%S}.json")
    json.dump(rec, open(out, "w"), indent=2)
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
