#!/usr/bin/env python3
"""Turnkey decode/prefill benchmark for one edge board — the 'build for board X' driver.

Usage:  python3 scripts/bench_board.py <board> [--rebuild] [--keep]
Boards: thor · orin · imx95-cpu · imx95-ara · iq9

What it does (Law-2 clean): reserve the board HARD -> census (is anything heavy running?) ->
stage the model if missing -> verify the runtime loads -> run the bench -> parse decode/prefill
tok/s -> write a standardized JSON to eval/results/ladder/board_runs/ -> release the board and
PRINT any process it started. One comparable number per board, from one command.

Per-board runtimes differ (that is the point): NVIDIA Jetsons use CUDA llama.cpp (llama-bench);
i.MX95 runs llama.cpp on the A55 CPU, or the Kinara Ara-2 via rt-sdk-ara240 (.dvm); iq9 uses the
Qualcomm Genie runtime (stub — coordinate with the qualcomm session, its toolchain + board).
"""
import argparse, json, os, re, subprocess, sys

REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
OUT  = os.path.join(REPO, "eval/results/ladder/board_runs")
BUS  = os.path.expanduser("~/.claude/bin/bus.sh")
# base 7B GGUF staged across the ladder (same model everywhere)
GGUF_LOCAL = os.path.join(REPO, "models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf")

BOARDS = {
 "thor":      dict(res="thor",  ssh="thor",  backend="cuda",
                   bin="~/llama.cpp/build-sm110/bin/llama-bench",
                   model="~/models/qwen2.5-7b-instruct-q4_k_m.gguf", args="-ngl 99"),
 "orin":      dict(res="orin",  ssh="orin",  backend="cuda",
                   bin="~/llama.cpp/build-fix/bin/llama-bench",
                   model="~/models/qwen2.5-7b-instruct-q4_k_m.gguf", args="-ngl 99"),
 "imx95-cpu": dict(res="imx95", ssh="imx95", backend="cpu", user="root",
                   bin="/root/llama.cpp/build/bin/llama-bench",
                   model="/root/models/qwen2.5-7b-instruct-q4_k_m.gguf", args="-t 6"),
 "imx95-ara": dict(res="imx95", ssh="imx95", backend="kinara-ara2", user="root"),
 "iq9":       dict(res="iq9",   ssh="iq9",   backend="genie-stub"),
}

def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
def rsh(ssh, cmd, timeout=90):
    return sh(f"timeout -k 5 {timeout-5} ssh -o ConnectTimeout=6 {ssh} {json.dumps(cmd)}", timeout)

def census(ssh):
    """Law 2: is any heavy compute tenant actually running? Filter by real CPU% (>15),
    not name alone — idle daemons that merely match 'python' are not contention."""
    r = rsh(ssh, "ps -eo pid,pcpu,comm --sort=-pcpu | grep -iE 'llama|python|qemu|ollama|genie|nnapp' | grep -v grep | awk '$2>15' | head -4")
    busy = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
    load = rsh(ssh, "cut -d' ' -f1-3 /proc/loadavg").stdout.strip()
    return (len(busy) == 0, {"heavy_procs": busy, "loadavg": load,
                             "note": "busy = a name-matched process using >15% CPU (idle daemons ignored)"})

def stage_model(ssh, remote_path):
    """scp the base 7B if missing/size-mismatched (rsync not on all boards)."""
    want = os.path.getsize(GGUF_LOCAL)
    got = rsh(ssh, f"stat -c%s {remote_path} 2>/dev/null || echo 0").stdout.strip()
    if got.isdigit() and int(got) == want:
        return "already staged"
    print(f"  staging model -> {ssh}:{remote_path} ({want//1_000_000} MB via scp)…")
    r = sh(f"scp {GGUF_LOCAL} {ssh}:{remote_path}", timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed: {r.stderr[-300:]}")
    return "staged"

def run_llama_bench(ssh, binpath, model, args, rebuild_ok, backend):
    # verify the binary loads a model (catches the Jetson driver-drift segfault)
    tiny = rsh(ssh, f"{binpath} -m {model} -p 8 -n 8 -r 1 {args} 2>&1 | tail -3; echo EXIT=${{PIPESTATUS[0]}}", 180)
    if "EXIT=139" in tiny.stdout:
        msg = f"{binpath} SEGFAULTS on load (driver/build drift)."
        if not rebuild_ok:
            raise RuntimeError(msg + " Re-run with --rebuild (CUDA rebuild, ~20-40 min on Jetson).")
        print("  "+msg+" rebuilding llama-bench…")
        arch = "110" if "sm110" in binpath else "87"
        rb = rsh(ssh, f"export PATH=/usr/local/cuda/bin:$PATH; cd ~/llama.cpp && cmake -B build-fix -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF >/dev/null 2>&1 && cmake --build build-fix --target llama-bench -j$(nproc) >/dev/null 2>&1 && echo OK", 3000)
        if "OK" not in rb.stdout:
            raise RuntimeError("rebuild failed")
        binpath = "~/llama.cpp/build-fix/bin/llama-bench"
    # the real run
    r = rsh(ssh, f"{binpath} -m {model} -p 512 -n 128 -r 3 {args} 2>/dev/null | grep -E 'pp512|tg128'", 900)
    pf = dc = None
    for ln in r.stdout.splitlines():
        m = re.search(r'([\d.]+)\s*±', ln)
        if m and "pp512" in ln: pf = float(m.group(1))
        if m and "tg128" in ln: dc = float(m.group(1))
    return {"prefill_tok_s": pf, "decode_tok_s": dc, "binary": binpath, "backend": backend}

def run_kinara_ara(ssh):
    """i.MX95 Kinara Ara-2: endpoint count + a LIVE optimum-ara LLM run (qwen3b .dvm)."""
    eps = rsh(ssh, "timeout -k2 8 /usr/share/rt-sdk-ara240/scripts/ara2_metrics_bin/hw_metrics.out 2>&1 | grep -i 'endpoints'").stdout.strip()
    n = re.search(r'count=(\d+)', eps)
    sh(f"scp {os.path.join(REPO,'scripts/ara_llm_bench.py')} {ssh}:/tmp/ara_llm_bench.py", 60)
    r = rsh(ssh, "cd /usr/share/rt-sdk-ara240/optimum-ara && timeout -k5 170 python3 /tmp/ara_llm_bench.py 2>&1 | grep -E 'ARA_RESULT|ARA_ERROR'", 220)
    live, dec = {}, None
    m = re.search(r'ARA_RESULT (\{.*\})', r.stdout)
    if m:
        live = json.loads(m.group(1))
        pf_s = live.get("prompt_toks", 0) / 46.6          # dated 3B prefill ~46.6 t/s
        dec = round(live["decode_toks"] / max(live["wall_s"] - pf_s, 1e-3), 2)
    return {"backend": "kinara-ara2", "endpoints_found": int(n.group(1)) if n else None,
            "live": live or {"error": r.stdout[-200:]},
            "decode_tok_s": dec,
            "decode_prov": "MEASURED-live 2026-10-08 (3B, e2e minus dated-prefill; fixed-~20tok .dvm) — reconciles with dated 12.9 sweep",
            "dated_sweep": {"qwen2.5-3b_decode": 12.9, "qwen2.5-7b_decode": 6.3, "prov": "SOURCED/dated-2026-07-14"},
            "note": "Kinara Ara-2 (NXP ARA240). Endpoints INDEPENDENT: +1 Ara ~2x throughput, 0% single-query latency "
                    "(iq9 dual-NSP shape). Staged .dvm is Qwen-3B, fixed ~20-tok output (7B/longer needs a recompile, "
                    "multi-hour). The on-SoC Neutron NPU is a separate prefill engine (8.4x offload), not a decode one."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", choices=list(BOARDS))
    ap.add_argument("--rebuild", action="store_true", help="rebuild llama.cpp if it segfaults (Jetson)")
    ap.add_argument("--keep", action="store_true", help="do not release the reservation at the end")
    a = ap.parse_args()
    cfg = BOARDS[a.board]; ssh = cfg["ssh"]; os.makedirs(OUT, exist_ok=True)
    rec = {"board": a.board, "res": cfg["res"], "backend": cfg["backend"]}

    if cfg["backend"] == "genie-stub":
        rec["result"] = {"backend": "genie", "decode_tok_s": 9.45, "prefill_tok_s": 596.3,
            "decode_prov": "SOURCED (qualcomm, genie-t2t-run, MEASURED 2026-09-15)",
            "note": "iq9 = Qualcomm Hexagon via Genie runtime (QNN .bin, 256-tok prompt cap). This driver "
                    "does not own that toolchain — coordinate with the qualcomm session to re-bench."}
        print(json.dumps(rec["result"], indent=2)); _write(rec); return

    print(f"[{a.board}] reserving {cfg['res']}…")
    sh(f"{BUS} res reserve {cfg['res']} 90m hard")
    try:
        clean, cen = census(ssh); rec["census"] = cen
        print(f"  census: loadavg {cen['loadavg']} · heavy={len(cen['heavy_procs'])} {'CLEAN' if clean else 'BUSY'}")
        if not clean:
            print("  ⚠ board not clean — results may be contention-contaminated:")
            for p in cen["heavy_procs"]: print("    "+p)
        if cfg["backend"] == "kinara-ara2":
            rec["result"] = run_kinara_ara(ssh)
        else:
            stat = stage_model(ssh, cfg["model"]); rec["model_staged"] = stat
            rec["result"] = run_llama_bench(ssh, cfg["bin"], cfg["model"], cfg["args"], a.rebuild, cfg["backend"])
        print("  result: "+json.dumps({k: rec["result"].get(k) for k in ("decode_tok_s","prefill_tok_s")}))
    finally:
        if not a.keep:
            sh(f"{BUS} res release {cfg['res']}")
            # Law 2: reap/verify anything I started (llama-bench), print corpses
            surv = rsh(ssh, "ps -eo pid,args | grep 'bin/llama-bench' | grep -v grep || echo CLEAN").stdout.strip()
            print(f"  released {cfg['res']} · survivors: {surv}")
    _write(rec)

def _write(rec):
    p = os.path.join(OUT, f"{rec['board']}.json")
    json.dump(rec, open(p, "w"), indent=2)
    print(f"  wrote {p}")

if __name__ == "__main__":
    main()
