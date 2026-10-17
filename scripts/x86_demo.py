#!/usr/bin/env python3
"""x86/AMD pure-CPU decode-ladder demo — "pull the repo and run it on your laptop".

The rest of the ladder (bench_board.py) drives *fleet* boards over SSH with reservations.
This one is the opposite on purpose: it runs on the machine you're sitting at, CPU-only, no
GPU, no Docker, no Skippy server, no SSH, no fleet bus. A stranger clones the repo, installs
one pip wheel, and reproduces the headline finding on their own silicon:

    DECODE IS MEMORY-BANDWIDTH-BOUND, NOT TOPS-BOUND.

It self-downloads a small GGUF (via llama-cpp-python's from_pretrained), runs a prefill+decode
bench on the CPU, and — the pedagogical payload — derives the *effective memory bandwidth* the
decode loop sustains (decode_tok/s x weights_bytes) and, if the DIMM speed is knowable, the % of
peak DDR bandwidth that represents. On a laptop that lands near the memory ceiling, which is the
whole point: the generation loop is fed by DDR, not by any accelerator's TOPS.

Cross-platform: Linux (native) and Windows (native or WSL2). Pure CPU.

    python3 scripts/x86_demo.py                 # default: qwen2.5-3b (2 GB, laptop-friendly)
    python3 scripts/x86_demo.py --model 7b       # the ladder's reference size (needs ~6 GB RAM)
    python3 scripts/x86_demo.py --threads 8       # override thread count
    python3 scripts/x86_demo.py --ddr-gbps 89.6   # supply peak DDR BW if auto-detect can't

Provenance (Fleet Law 1): decode/prefill tok/s are MEASURED (ran here, this box). The effective
bandwidth is DERIVED (labelled). Peak DDR BW is SOURCED (DIMM speed) — never compared bare against
a measured number from another board; the within-box util% is the honest headline.
"""
import argparse, ctypes, json, os, platform, re, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(REPO, "eval/results/ladder/board_runs")
REGISTRY = os.path.join(REPO, "eval/model_registry.json")
KV_HEADROOM_GB = 1.5  # weights + a modest KV/context budget for the feasibility gate


# ---------- hardware census (cross-platform, stdlib only) ----------
def total_ram_gb():
    try:
        if platform.system() == "Windows":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / 1e9, 1)
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
        return round(kb * 1024 / 1e9, 1)
    except Exception:
        return None

def simd_flags():
    """What the installed llama.cpp build can actually use — prefill (compute-bound) lives or
    dies on this. A generic wheel with AVX2 off understates prefill badly."""
    try:
        import llama_cpp
        info = llama_cpp.llama_print_system_info().decode()
        on = [f for f in ("AVX512", "AVX2", "AVX", "FMA", "F16C", "NEON") if re.search(rf"{f}\s*=\s*1", info)]
        return ", ".join(on) or "none detected"
    except Exception:
        return "unknown"

def cpu_model():
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                m = re.search(r"model name\s*:\s*(.+)", f.read())
                if m: return m.group(1).strip()
        if platform.system() == "Windows":
            return platform.processor() or "x86 CPU"
        return platform.processor() or platform.machine()
    except Exception:
        return platform.machine()

def peak_ddr_gbps():
    """SOURCED peak DDR bandwidth = channels x MT/s x 8 bytes. Best-effort, never fatal.
    Returns (gbps, how) or (None, why). Needs dmidecode (sudo) on Linux or wmic on Windows."""
    sysname = platform.system()
    try:
        if sysname == "Linux":
            r = subprocess.run(["dmidecode", "-t", "memory"], capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return None, "dmidecode needs sudo (run with --ddr-gbps to supply it)"
            speeds = [int(x) for x in re.findall(r"Configured Memory Speed:\s*(\d+)\s*MT/s", r.stdout)]
            if not speeds:
                speeds = [int(x) for x in re.findall(r"Speed:\s*(\d+)\s*MT/s", r.stdout)]
            n = len(speeds)
            if n:
                return round(n * speeds[0] * 8 / 1000, 1), f"{n} channel(s) x {speeds[0]} MT/s x 8B (dmidecode)"
        if sysname == "Windows":
            # PowerShell CIM first: wmic is deprecated and is gone from Windows 11 24H2+.
            ps = ("Get-CimInstance Win32_PhysicalMemory | ForEach-Object "
                  "{ if ($_.ConfiguredClockSpeed) { $_.ConfiguredClockSpeed } else { $_.Speed } }")
            for cmd, tool in ((["powershell", "-NoProfile", "-Command", ps], "PowerShell CIM"),
                              (["wmic", "memorychip", "get", "Speed"], "wmic")):
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                except (OSError, subprocess.SubprocessError):
                    continue
                speeds = [int(x) for x in re.findall(r"(\d+)", r.stdout) if int(x) > 0]
                if speeds:
                    return (round(len(speeds) * speeds[0] * 8 / 1000, 1),
                            f"{len(speeds)} DIMM(s) x {speeds[0]} MT/s x 8B ({tool}; assumes 1 DIMM/channel)")
    except Exception as e:
        return None, f"auto-detect failed ({type(e).__name__})"
    return None, "DIMM speed not detectable — pass --ddr-gbps for the util%"

def default_threads():
    """A safe cross-platform thread default. Two traps this avoids, both measured:
    (1) oversubscribing logical threads (hyperthreads) hurts; (2) on Intel hybrid chips
    (P+E cores) throwing work at the slow E-cores *tanks* the compute-bound prefill (we saw
    40 vs 215 tok/s at 24 vs 16 threads on a 14900KF). So: half the logical count, capped at
    16 — lands on the fast cores for prefill and is plenty for bandwidth-bound decode. Tune
    with --threads if you know your topology (P-core count is usually the sweet spot)."""
    lg = os.cpu_count() or 4
    return max(1, min(lg // 2 if lg > 8 else lg, 16))


# ---------- model + gate ----------
def load_model_spec(alias):
    reg = json.load(open(REGISTRY, encoding="utf-8"))
    key = {"3b": "qwen2.5-3b", "7b": "qwen2.5-7b", "14b": "qwen2.5-14b"}.get(alias, alias)
    if key not in reg["models"]:
        sys.exit(f"unknown model '{alias}' — choose one of: 3b, 7b, 14b")
    return key, reg["models"][key]

def feasible(spec, ram_gb):
    need = spec["weights_gb"] + KV_HEADROOM_GB
    if ram_gb is None:
        return True, f"RAM unknown — proceeding (need ~{need:.1f} GB)"
    if ram_gb < need:
        return False, f"won't fit: needs ~{need:.1f} GB (weights {spec['weights_gb']} + KV {KV_HEADROOM_GB}), box has {ram_gb} GB"
    return True, f"fits: needs ~{need:.1f} GB, box has {ram_gb} GB"


# ---------- the "needs only the model" quick tasks (no server, DB, network, or extra deps) ----------
FACTS = ("NXP i.MX 95: 6x Arm Cortex-A55 + a Cortex-M33 + a Cortex-M7; eIQ Neutron NPU (N3-1024S) "
         "rated 2.0 TOPS (1024 MACs x 1.0 GHz x 2 OPS/MAC); Arm Mali GPU; on-chip ISP; LPDDR5 or LPDDR4X memory; "
         "PCIe Gen3; Gigabit Ethernet with TSN; 'Energy Flex' power architecture. For edge AI across "
         "automotive/industrial/consumer.")

def _notes():
    try:
        return open(os.path.join(REPO, "scripts/agentic_bench/sample_notes.txt"), encoding="utf-8").read()
    except Exception:
        return FACTS

QUICK_TASKS = [
    {"id": "email", "name": "Email drafting", "max_tokens": 256,
     "prompt": lambda: ("Draft a short, friendly email introducing the NXP i.MX 95 for an edge-AI "
                        f"product. Cover what it is, the Neutron NPU, and why it suits edge.\n\nFacts:\n{FACTS}")},
    {"id": "file_ops", "name": "File read + summarize", "max_tokens": 160,
     "prompt": lambda: ("Summarize the following notes in exactly 3 concise bullet points. Use only "
                        f"what the notes say.\n\nNOTES:\n{_notes()}")},
    {"id": "spec_inline", "name": "Grounded spec Q&A (inline context)", "max_tokens": 200,
     "prompt": lambda: (f"Context (datasheet facts):\n{FACTS}\n\nQuestion: What does the i.MX 95 Neutron "
                        "NPU do, and what are its key rated specs? Answer accurately, grounded ONLY in the context.")},
]

# ---------- bench (two-run subtraction: isolates decode from prefill) ----------
def load_llm(repo_id, filename, n_threads):
    try:
        from llama_cpp import Llama
        import huggingface_hub  # noqa: F401  (Llama.from_pretrained needs it; not a llama-cpp dependency)
    except ImportError as e:
        sys.exit(f"missing package: {e.name}  ({e})\n"
                 f"  this interpreter: {sys.executable}  (install into THIS one: <it> -m pip install ...)\n"
                 "  Linux/WSL2:  pip install llama-cpp-python huggingface-hub\n"
                 "  Windows:     pip install --prefer-binary llama-cpp-python huggingface-hub "
                 "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
                 "  See docs/RUN_X86_DEMO.md.")
    print(f"  loading {repo_id} :: {filename}  (CPU, {n_threads} threads)…")
    t0 = time.time()
    llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=2048,
                                n_threads=n_threads, n_threads_batch=n_threads,
                                n_gpu_layers=0, verbose=False)
    return llm, round(time.time() - t0, 1)

def bench_decode(llm, reps):
    # a deterministic ~512-token prompt (repeat a sentence, then trim by tokenizing)
    seed = ("The i.MX 95 integrates an eIQ Neutron NPU for edge AI inference. ") * 40
    toks = llm.tokenize(seed.encode())[:512]
    prompt = llm.detokenize(toks).decode(errors="ignore")
    n_prompt = len(toks)

    def once(max_tokens):
        llm.reset()
        t = time.time()
        llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0, top_k=1)
        return time.time() - t

    # warmup (fills caches / mmaps weights) then timed reps; keep the FASTEST (least OS jitter)
    once(1)
    pf_times, tot_times = [], []
    for i in range(reps):
        pf_times.append(once(1))          # prompt-eval + 1 token ~= prefill time
        tot_times.append(once(128))       # prefill + 128 decode tokens
        print(f"    rep {i+1}/{reps}: prefill {n_prompt/pf_times[-1]:6.1f} tok/s · "
              f"decode {127/max(tot_times[-1]-pf_times[-1],1e-3):6.2f} tok/s")
    t_pf, t_tot = min(pf_times), min(tot_times)
    prefill_tok_s = round(n_prompt / t_pf, 1)
    decode_tok_s  = round(127 / max(t_tot - t_pf, 1e-3), 2)
    return {"prefill_tok_s": prefill_tok_s, "decode_tok_s": decode_tok_s,
            "n_prompt_tokens": n_prompt, "reps": reps}

def run_quick_tasks(llm, rates):
    """Run the three model-only agentic tasks on the already-loaded model. Each produces real
    output (read it — the coherence gate) and a DERIVED decode-share: prefill/decode time split
    estimated from the measured rates, showing even real agentic prompts are decode-dominated."""
    pf_r, dc_r = rates["prefill_tok_s"], rates["decode_tok_s"]
    print("\n  --- quick tasks (model-only; no server / DB / network) ---")
    recs = []
    for t in QUICK_TASKS:
        prompt = t["prompt"]()
        llm.reset()
        t0 = time.time()
        # greedy (temp 0, deterministic) + a mild repeat penalty — what a real deployment runs;
        # without it a small base model greedily loops (not a resource property, a sampling one).
        r = llm.create_completion(prompt, max_tokens=t["max_tokens"], temperature=0.0, top_k=1,
                                  repeat_penalty=1.1)
        wall = round(time.time() - t0, 2)
        out = r["choices"][0]["text"].strip()
        u = r.get("usage", {})
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        est_pf, est_dc = pt / pf_r, ct / dc_r
        dshare = round(100 * est_dc / max(est_pf + est_dc, 1e-9), 1)
        print(f"\n  [{t['name']}]  wall {wall}s · {pt} prompt -> {ct} gen tok · "
              f"~{dshare}% decode (bandwidth-bound) [DERIVED]")
        snippet = " ".join(out.split())[:280]
        print(f"      {snippet}{'…' if len(out) > 280 else ''}")
        recs.append({"id": t["id"], "name": t["name"], "wall_s": wall, "prompt_tokens": pt,
                     "gen_tokens": ct, "decode_share_pct": dshare,
                     "decode_share_prov": "DERIVED from measured prefill/decode rates", "output": out})
    print("\n  (read the outputs above — the coherence gate. Optional: grade with "
          "eval/judge_agentic_tasks.py if you have API keys.)")
    return recs


def main():
    # Windows consoles/pipes can be cp1252: never crash printing model output.
    # Piped/redirected output is written as UTF-8 so logs and files don't mangle the dashes.
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(errors="replace")
            if os.name == "nt" and not st.isatty():
                st.reconfigure(encoding="utf-8")
        except Exception: pass
    ap = argparse.ArgumentParser(description="x86/AMD pure-CPU decode-ladder demo")
    ap.add_argument("--model", default="3b", help="3b (default) | 7b | 14b")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: physical-core-ish)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ddr-gbps", type=float, default=None, help="peak DDR bandwidth (GB/s) if auto-detect can't")
    ap.add_argument("--tasks", action="store_true",
                    help="also run the 3 model-only agentic tasks (email, file-summarize, grounded Q&A)")
    a = ap.parse_args()

    key, spec = load_model_spec(a.model)
    ram = total_ram_gb(); cpu = cpu_model(); threads = a.threads or default_threads()
    ddr, ddr_how = (a.ddr_gbps, "user-supplied") if a.ddr_gbps else peak_ddr_gbps()

    print("\n=== x86/AMD pure-CPU decode-ladder demo ===")
    print(f"  OS       : {platform.platform()}")
    print(f"  CPU      : {cpu}  ({os.cpu_count()} logical cores, using {threads} threads)")
    print(f"  SIMD     : {simd_flags()}  (llama.cpp build — prefill speed depends on this)")
    print(f"  RAM      : {ram} GB")
    print(f"  DDR peak : {str(ddr)+' GB/s ('+ddr_how+')' if ddr else 'unknown — '+ddr_how}  [SOURCED]")
    print(f"  model    : {key}  (Q4_K_M, {spec['weights_gb']} GB weights, {spec['params_b']}B params)")

    ok, why = feasible(spec, ram)
    print(f"  gate     : {'RUNNABLE' if ok else 'UNRUNNABLE'} — {why}")
    if not ok:
        rec = {"board": "localhost-x86", "backend": "cpu-llamacpp", "model": key,
               "gate": {"runnable": False, "reason": why},
               "host": {"os": platform.platform(), "cpu": cpu, "ram_gb": ram, "threads": threads}}
        _write(rec); sys.exit(2)

    llm, load_s = load_llm(spec["hf_repo"], spec["glob"], threads)
    res = bench_decode(llm, a.reps)
    res["load_s"] = load_s

    # DERIVED: effective memory bandwidth the decode loop sustains.
    # A dense transformer reads ~all its weights from DRAM once per generated token, so
    #   effective_read_BW ~= decode_tok/s x weights_bytes.
    weights_bytes = spec["weights_gb"] * 1e9
    eff_gbps = round(res["decode_tok_s"] * weights_bytes / 1e9, 1)
    util = round(100 * eff_gbps / ddr, 1) if ddr else None
    res.update({"eff_bandwidth_gbps": eff_gbps, "eff_bandwidth_prov": "DERIVED = decode_tok/s x weights_bytes",
                "ddr_peak_gbps": ddr, "ddr_peak_prov": f"SOURCED ({ddr_how})",
                "ddr_bandwidth_util_pct": util, "backend": "cpu-llamacpp", "model": key})

    print(f"\n  --- result (model load {res['load_s']}s) ---")
    print(f"  prefill        : {res['prefill_tok_s']:>8} tok/s   [MEASURED]  (compute-bound; scales with --threads)")
    print(f"  decode         : {res['decode_tok_s']:>8} tok/s   [MEASURED]  (bandwidth-bound; ~flat vs threads)")
    print(f"  eff. mem BW    : {eff_gbps:>8} GB/s    [DERIVED: decode x weights]")
    if util is not None:
        print(f"  DDR util       : {util:>8} %       [DERIVED/SOURCED] <- decode runs at {util}% of peak DDR BW")
        print(f"\n  ==> Decode is memory-bandwidth-bound: the generation loop sustains {util}% of this box's")
        print(f"      peak DDR bandwidth with ZERO GPU/NPU. More TOPS would not move this number; more")
        print(f"      memory bandwidth (faster/more DIMM channels) would. That is the agentic-edge thesis.")
    else:
        print(f"\n  ==> Decode sustains ~{eff_gbps} GB/s of effective read bandwidth on CPU alone (GPU/NPU idle).")
        print(f"      Pass --ddr-gbps <peak> to see what % of your DIMMs' peak that is (the headline).")

    tasks = run_quick_tasks(llm, res) if a.tasks else None

    rec = {"board": "localhost-x86", "backend": "cpu-llamacpp", "model": key,
           "gate": {"runnable": True, "reason": why},
           "host": {"os": platform.platform(), "cpu": cpu, "ram_gb": ram,
                    "logical_cores": os.cpu_count(), "threads": threads, "simd": simd_flags()},
           "result": res}
    if tasks is not None:
        rec["quick_tasks"] = tasks
    _write(rec)
    print("\n  Next: python3 scripts/aggregate_boards.py   # this rung joins the cross-board ladder")


def _write(rec):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "localhost-x86.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"  wrote {os.path.relpath(p, REPO)}")


if __name__ == "__main__":
    main()
