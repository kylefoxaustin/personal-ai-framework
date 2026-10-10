#!/usr/bin/env python3
"""Phase-A model provisioning + feasibility gate for the benchmark.

The question that gates everything: CAN THIS BOARD RUN THIS MODEL? Answered before any download
or run. Then: make the artifact exist on the board (prebuilt → download → [Phase B: build]).

    python3 scripts/provision.py <board> <model> [--check-only]
    python3 scripts/provision.py orin qwen2.5-14b          # gate -> ensure staged -> print path
    python3 scripts/provision.py --list                    # models + boards

Scope: NVIDIA Jetsons (llama.cpp GGUF) — download from HF if missing, scp to the board. iq9
(Genie .bin) + i.MX95 (Kinara .dvm) are the Phase-B BUILD targets; the gate already knows to
refuse models that don't fit / aren't supported there once those boards are added.
"""
import argparse, glob, json, os, subprocess, sys

REPO = "/home/kyle/Documents/GitHub/personal-ai-framework"
REG = json.load(open(os.path.join(REPO, "eval/model_registry.json")))
MODELS, BOARDS = REG["models"], REG["boards"]

def sh(cmd, timeout=1800): return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=timeout)

def feasible(board, model):
    """The gate. Returns (ok: bool, reason: str)."""
    if model not in MODELS: return False, f"unknown model '{model}' (see --list)"
    if board not in BOARDS: return False, f"unknown board '{board}' (see --list)"
    m, b = MODELS[model], BOARDS[board]
    if m["arch"] not in b["supports_arch"]:
        return False, f"{b['runtime']} does not support arch '{m['arch']}'"
    need = m["weights_gb"] + b.get("kv_headroom_gb", 2)
    if need > b["ram_gb"]:
        return False, f"won't fit: ~{need:.0f} GB (weights+KV) > {b['ram_gb']} GB on {board}"
    return True, f"fits: ~{need:.0f} GB of {b['ram_gb']} GB; {b['runtime']} supports {m['arch']}"

def ensure_on_host(model):
    """Download the GGUF from HF into its local_dir if not already present. Returns host path."""
    m = MODELS[model]; d = os.path.join(REPO, m["local_dir"])
    present = glob.glob(os.path.join(d, m["glob"]))
    if not present:
        os.makedirs(d, exist_ok=True)
        print(f"  downloading {model} from {m['hf_repo']} -> {m['local_dir']} …")
        r = sh(f"hf download {m['hf_repo']} --include {json.dumps(m['glob'])} --local-dir {json.dumps(d)}", timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"hf download failed: {r.stderr[-300:]}")
        present = glob.glob(os.path.join(d, m["glob"]))
        if not present: raise RuntimeError("download reported ok but no gguf found")
    return os.path.join(d, m["gguf"])

def ensure_on_board(board, model):
    """Gate -> ensure on host -> scp any missing shard to the board. Returns remote main-gguf path."""
    ok, why = feasible(board, model)
    if not ok: raise RuntimeError(f"INFEASIBLE: {why}")
    m, b = MODELS[model], BOARDS[board]
    host_main = ensure_on_host(model)
    host_shards = sorted(glob.glob(os.path.join(REPO, m["local_dir"], m["glob"])))
    for hs in host_shards:
        fn = os.path.basename(hs); remote = f"{b['model_dir']}/{fn}"
        got = sh(f"ssh -o ConnectTimeout=6 {b['ssh']} stat -c%s {json.dumps(remote)} 2>/dev/null || echo 0").stdout.strip()
        if got.isdigit() and int(got) == os.path.getsize(hs): continue
        print(f"  staging {fn} -> {board}:{b['model_dir']} …")
        r = sh(f"scp {json.dumps(hs)} {b['ssh']}:{json.dumps(remote)}", timeout=1200)
        if r.returncode != 0: raise RuntimeError(f"scp {fn} failed: {r.stderr[-200:]}")
    return f"{b['model_dir']}/{m['gguf']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("board", nargs="?"); ap.add_argument("model", nargs="?")
    ap.add_argument("--check-only", action="store_true", help="feasibility gate only; no download/stage")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.board and a.model):
        print("models:", ", ".join(MODELS)); print("boards:", ", ".join(BOARDS)); return
    ok, why = feasible(a.board, a.model)
    print(f"[gate] {a.board} × {a.model}: {'RUNNABLE' if ok else 'UNRUNNABLE'} — {why}")
    if not ok or a.check_only: sys.exit(0 if ok else 2)
    path = ensure_on_board(a.board, a.model)
    print(f"[ready] {a.board}:{path}")

if __name__ == "__main__":
    main()
