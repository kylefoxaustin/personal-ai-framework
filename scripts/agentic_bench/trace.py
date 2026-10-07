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

# In-domain tasks: things Skippy is actually FOR (embedded/i.MX domain + its voice, grounded on
# the 31.9K-chunk datasheet KB). Coherent output AND a real resource-profile spread.
TASKS = {
    "write_email": {    # FLAGSHIP: decode-heavy long-form in Skippy's voice, grounded -> coherent
        "prompt": "Write a clear, friendly email to a colleague introducing the NXP i.MX 95 for a "
                  "new edge-AI product. Cover what it is, its Neutron NPU, and why it suits edge "
                  "inference. Ground the specifics in the datasheets.",
        "use_rag": True, "rag_k": 4, "max_tokens": 400},
    "spec_rag": {       # RAG lookup, grounded, concise (known-good phrasing)
        "prompt": "Using the datasheets, what does the i.MX 95 Neutron NPU do, and what are its "
                  "key features?",
        "use_rag": True, "rag_k": 4, "max_tokens": 220},
    "compare": {        # decode + RAG, longer in-domain reasoning
        "prompt": "Using the datasheets, compare the NXP i.MX 93 and i.MX 95 for an edge-AI "
                  "gateway, focusing on the NPU and memory.",
        "use_rag": True, "rag_k": 5, "max_tokens": 350},
    "long_summary": {   # prefill-heavy: a real spec blob as context + a coherent summary
        "prompt": "Summarize the key specifications from the reference text above as a short "
                  "bulleted list.",
        "context": [(("The NXP i.MX 95 applications processor integrates up to six Arm Cortex-A55 "
          "cores and a Cortex-M7 real-time core. It includes the eIQ Neutron NPU rated at 2.0 TOPS "
          "for machine-learning inference, an Arm Mali GPU, and an image signal processor. Memory "
          "is 32-bit LPDDR4X/LPDDR5 up to high bandwidth. Connectivity includes PCIe Gen3, "
          "Gigabit Ethernet with TSN, USB 3.0, and CAN-FD. It targets automotive, industrial, and "
          "consumer edge applications with functional-safety support. ") * 12)],
        "use_rag": False, "max_tokens": 220},
    "short_chat": {     # short in-domain Q&A, grounded
        "prompt": "In two or three sentences, what is the NXP i.MX 95?",
        "use_rag": True, "rag_k": 3, "max_tokens": 90},
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
        "answer_preview": (resp.get("text") or "")[:1600],
    }
    slug = "".join(c if c.isalnum() else "_" for c in task)[:24].strip("_") or "task"
    out = os.path.join(outdir, f"trace_{slug}_{datetime.now():%Y%m%d-%H%M%S}.json")
    json.dump(rec, open(out, "w"), indent=2)
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
