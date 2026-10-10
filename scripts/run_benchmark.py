#!/usr/bin/env python3
"""Agentic-edge benchmark — ONE command to run the whole system.

    python3 scripts/run_benchmark.py

Sequences the four layers into a single run and one unified result:
  1. WORKLOAD   — the 10-task agentic harness on the 5090 (resource profile per task)   [needs Skippy up]
  2. LADDER     — decode/prefill across boards (5090 from harness telemetry; Thor/Orin via llama-bench)
  3. GRADE      — two-judge (Sonnet+GPT-4o) task-success                                  [needs API keys]
  4. AGGREGATE  — unify into eval/results/system/run_<ts>.json + a printed rollup

Graceful: if Skippy is down / a board is unreachable / keys are missing, that phase is SKIPPED with a
clear note — the rest still runs. Scope today: 5090, Thor, Orin (iq9 + i.MX95 come in the port phase).

Flags:  --boards 5090,thor,orin   --skip-grade   --skip-ladder   --only harness|ladder|grade|aggregate
"""
import argparse, glob, json, os, subprocess, sys, time
from datetime import datetime

REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
AG   = os.path.join(REPO, "eval/results/ladder/agentic")
BR   = os.path.join(REPO, "eval/results/ladder/board_runs")
OUT  = os.path.join(REPO, "eval/results/system")
BASE = "http://localhost:8080"
LADDER_BOARDS = ["thor", "orin"]          # 5090 decode comes from the harness telemetry
def ts(): return datetime.now().strftime("%Y%m%d-%H%M%S")
def run(cmd, timeout=1200): return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=timeout)

CONFIG = os.path.join(REPO, "pipeline/config.yaml")
MODELS = {  # friendly aliases -> container path (the LLM server sees /app/models)
  "base-7b":    "/app/models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf",
  "prod-7b-v4": "/app/models/qwen2.5-7b-kyle/kyle-qwen25-7b-v4-q4_k_m.gguf",
  "v4":         "/app/models/qwen2.5-7b-kyle/kyle-qwen25-7b-v4-q4_k_m.gguf",
  "14b":        "/app/models/qwen2.5-14b/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf",
}
def resolve_model(spec): return MODELS.get(spec, spec)   # alias or raw /app path
def current_model():
    import re
    m = re.search(r'^\s*path:\s*"([^"]+)"', open(CONFIG).read(), re.M)
    return os.path.basename(m.group(1)) if m else "?"
def _gpu_free_mib():
    r = run("nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits")
    try:
        free, tot = [int(x.strip()) for x in r.stdout.strip().split(",")[:2]]
        return free, tot
    except Exception:
        return None, None

def swap_model(spec, log):
    """Point config.yaml at `spec`, restart the server, wait for load — but NEVER tear down a
    running Skippy to load a model that won't fit the free GPU (that orphaned production once)."""
    path = resolve_model(spec)
    need_mib = {"14b": 9500, "qwen2.5-14b": 9500}.get(spec, 5500)   # rough weights+KV footprint
    free, tot = _gpu_free_mib()
    up_before, _ = skippy_up()
    if free is not None and free < need_mib and free < 0.4 * tot:
        # the card is heavily used by someone else; a restart would evict Skippy and still not fit
        log(f"  ⚠ GPU too full for {os.path.basename(path)}: free {free} MiB < need ~{need_mib} MiB "
            f"({tot} total). NOT restarting — leaving the current Skippy up. Free the GPU and retry.")
        return False
    import re
    s = open(CONFIG).read()
    s2 = re.sub(r'(^\s*path:\s*")[^"]+(")', lambda m: m.group(1)+path+m.group(2), s, count=1, flags=re.M)
    open(CONFIG, "w").write(s2)
    log(f"  ↻ swapping model -> {os.path.basename(path)} (restarting llm-server; GPU free {free} MiB)")
    run("docker compose restart llm-server", timeout=180)
    for _ in range(100):            # up to ~300s — big GGUFs load slowly
        up, _m = skippy_up()
        if up: log("    model loaded"); return True
        time.sleep(3)
    log("  ⚠ model did not report loaded within 300s"); return False

def skippy_up():
    try:
        import requests
        h = requests.get(f"{BASE}/health", timeout=5).json()
        return bool(h.get("model_loaded")), h.get("model", "?")
    except Exception:
        return False, None

def phase_harness(log):
    up, model = skippy_up()
    if not up:
        log("  ⏭  WORKLOAD skipped — Skippy not up on :8080 (start it: ./run.sh start)"); return None
    log(f"  ▶ WORKLOAD — 10-task harness on the 5090 (model: {model})")
    r = run(f"python3 scripts/agentic_bench/trace_bench.py", timeout=1800)
    if r.returncode != 0:
        log(f"  ⚠ harness exited {r.returncode}: {r.stderr[-200:]}");
    tasks = {}
    for f in glob.glob(f"{AG}/bench_*.json"):
        d = json.load(open(f)); t = d["task"]
        # keep the most recent per task
        if t not in tasks or d["ts"] > tasks[t]["ts"]:
            te = d.get("telemetry", {})
            tasks[t] = {"ts": d["ts"], "wall_s": d.get("wall_s"),
                        "decode_tok_s": te.get("decode_tok_per_s"), "prefill_tok_s": te.get("prefill_tok_per_s"),
                        "gpu": d.get("gpu", {}), "output_head": (d.get("output") or "")[:120]}
    log(f"    captured {len(tasks)} tasks")
    return {"model": current_model(), "tasks": tasks}

# run_benchmark --model alias -> provision registry alias, so the LADDER benches the same model as the harness
RLADDER = {"14b": "qwen2.5-14b", "base-7b": "qwen2.5-7b", "qwen2.5-14b": "qwen2.5-14b",
           "qwen2.5-7b": "qwen2.5-7b", "3b": "qwen2.5-3b", "qwen2.5-3b": "qwen2.5-3b"}

def phase_ladder(boards, log, provision_alias=None):
    out = {}
    for board in boards:
        if board in ("5090", "rtx5090"):
            continue
        prov = f" --provision {provision_alias}" if provision_alias else ""
        log(f"  ▶ LADDER — {board}{(' @ ' + provision_alias) if provision_alias else ''}")
        r = run(f"python3 scripts/bench_board.py {board}{prov}", timeout=3600)
        p = f"{BR}/{board}.json"
        if os.path.exists(p):
            d = json.load(open(p)); res = d.get("result", {})
            ml = provision_alias or (res.get("live") or {}).get("model", "7B")
            model = "14B" if "14b" in str(ml).lower() else ("3B" if "3b" in str(ml).lower() else "7B")
            prov = res.get("decode_prov", "MEASURED" if str(res.get("backend","")).startswith(("cuda","cpu")) else "SOURCED")
            out[board] = {"decode_tok_s": res.get("decode_tok_s"), "prefill_tok_s": res.get("prefill_tok_s"),
                          "backend": res.get("backend"), "model": model, "prov": prov, "census": d.get("census", {})}
            log(f"    {board}: decode {out[board]['decode_tok_s']} t/s ({model}, {res.get('backend')})")
        else:
            log(f"    ⚠ {board}: no result ({r.stderr[-160:] if r.returncode else 'missing json'})")
    return out

def phase_grade(log):
    sys.path.insert(0, os.path.join(REPO, "eval"))
    try:
        from _keys import ensure_keys
        if ensure_keys(("ANTHROPIC_API_KEY", "OPENAI_API_KEY")):
            log("  ⏭  GRADE skipped — API keys not in ~/.personal-ai/keys.env (see docs/api-keys-setup.md)"); return None
    except Exception as e:
        log(f"  ⏭  GRADE skipped — {e}"); return None
    log("  ▶ GRADE — two-judge (Sonnet + GPT-4o)")
    r = run("python3 eval/judge_agentic_tasks.py", timeout=900)
    p = f"{AG}/task_success_twojudge.json"
    if os.path.exists(p):
        d = json.load(open(p)); log(f"    agreement: {d.get('agreement')}"); return d
    log(f"  ⚠ grade produced no json: {r.stderr[-160:]}"); return None

def rollup(man):
    L = []
    L.append(f"\n{'='*64}\n AGENTIC-EDGE BENCHMARK — run {man['run']['ts']}\n{'='*64}")
    res = man.get("resource") or {}
    if res.get("tasks"):
        L.append(f" model: {res.get('model')}   tasks: {len(res['tasks'])}")
        # 5090 decode from spec_rag telemetry
        sr = res["tasks"].get("spec_rag", {})
        L.append(f" 5090 decode (spec_rag): {sr.get('decode_tok_s')} t/s  [MEASURED, harness telemetry]")
    lad = man.get("ladder") or {}
    if lad:
        L.append(" decode ladder:")
        for b, v in lad.items():
            L.append(f"   {b:11s} {str(v.get('decode_tok_s')):>8s} t/s  {v.get('model','7B'):>3s}  {str(v.get('backend','')):14s} {v.get('prov','')[:22]}")
    acc = man.get("accuracy")
    if acc:
        L.append(f" task-success (two-judge): agree {acc.get('agreement',{}).get('agree')} / split {acc.get('agreement',{}).get('split')}")
    else:
        L.append(" task-success: (not graded this run)")
    L.append(f" written: {man['run']['path']}\n{'='*64}")
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default="5090,thor,orin")
    ap.add_argument("--skip-grade", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--only", choices=["harness", "ladder", "grade", "aggregate"])
    ap.add_argument("--model", help="swap+restart before the run: alias (base-7b|prod-7b-v4|14b) or an /app/... GGUF path")
    ap.add_argument("--full", action="store_true", help="all boards incl the small ones: 5090/thor/orin/iq9/imx95-cpu/imx95-ara")
    a = ap.parse_args()
    boards = ["5090","thor","orin","iq9","imx95-cpu","imx95-ara"] if a.full else [b.strip() for b in a.boards.split(",") if b.strip()]
    os.makedirs(OUT, exist_ok=True)
    logs = []
    def log(m): print(m); logs.append(m)
    log(f"\n▶ agentic-edge benchmark · boards={boards} · {ts()}")
    if a.model and a.only in (None, "harness"):
        swap_model(a.model, log)

    man = {"run": {"ts": ts(), "boards": boards, "scope": "5090/thor/orin (iq9+i.MX95 deferred to port phase)"},
           "resource": None, "ladder": None, "accuracy": None,
           "provenance": {"harness": "MEASURED (5090 full-stack telemetry)", "ladder": "MEASURED (llama-bench)",
                          "accuracy": "MEASURED (two-judge, Sonnet+GPT-4o)"}}
    only = a.only
    if only in (None, "harness"):     man["resource"] = phase_harness(log)
    if (only in (None, "ladder")) and not a.skip_ladder: man["ladder"] = phase_ladder(boards, log, RLADDER.get(a.model))
    if (only in (None, "grade")) and not a.skip_grade:   man["accuracy"] = phase_grade(log)

    man["run"]["path"] = os.path.join(OUT, f"run_{man['run']['ts']}.json")
    man["log"] = logs
    json.dump(man, open(man["run"]["path"], "w"), indent=2)
    json.dump(man, open(os.path.join(OUT, "latest.json"), "w"), indent=2)
    try:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from report import build_report
        build_report(man["run"]["path"])                                   # report beside the stamped run
        rp = build_report(os.path.join(OUT, "latest.json"), os.path.join(OUT, "report.html"))
        log(f"  report (open in a browser): {rp}")
    except Exception as e:
        log(f"  ⚠ HTML report generation failed: {e}")
    print(rollup(man))

if __name__ == "__main__":
    main()
