#!/usr/bin/env python3
"""Drive llama-server on the Jetson AGX Orin: accuracy + decode-perf + power.

Runs ON the Orin. Stdlib only (no requests, no torch) — Jetson pip is a swamp.

For each GGUF it:
  1. starts `llama-server` with full GPU offload,
  2. replays the Skippy prompt pack through /v1/chat/completions
     (llama-server applies the GGUF's OWN chat template — which is exactly the
     step `skippy_eval.py build` deliberately leaves to the target runtime),
  3. records per-sample TTFT and decode tok/s from the server's timings,
  4. samples `tegrastats` for board power across the run,
  5. writes completions.jsonl (scored later by skippy_eval.py) and perf.json.

THE EXPERIMENT THIS EXISTS FOR
------------------------------
qualcomm's IQ-9075 dossier §4.4/§4.5 asserts, from a bandwidth model, that LLM
decode is weight-streaming: per-token latency ≈ weight-bytes ÷ memory-bandwidth,
and for a sparse MoE the streamed bytes track **active** params, not total.

That predicts something falsifiable and counterintuitive on this board:

    Qwen3-30B-A3B  Q4   17.3 GB on disk,  ~3.3B active  ──┐
    Qwen2.5-14B    Q4    8.4 GB on disk,  14.7B active  ──┴─► MoE should be FASTER

i.e. the model that is 2× larger on disk should decode faster, because only a
fraction of its experts stream per token. If decode instead tracks file size,
the MoE is slower and the "streams only active experts" assumption is wrong.

The Orin's bus is 204.8 GB/s; the IQ-9075's is 76.8 GB/s. Confirming the same
law on two silicon platforms with a 2.7× bandwidth ratio is worth far more than
confirming it twice on one.

Secondary: the dense Q4-vs-Q8 pair (8.4 GB vs 14.6 GB, same architecture, same
weights) is a clean bandwidth probe. If decode is bus-bound, tok/s should fall by
roughly the weight-byte ratio (~0.57×). Deviation quantifies how much of decode
is NOT weight streaming on this platform.

USAGE
    python3 run_orin.py --models-dir ~/skippy_corpus/models \
                        --bundle-dir ~/skippy_corpus/bundle \
                        --out-dir    ~/skippy_corpus/out
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER_PORT = 8099
BASE = f"http://127.0.0.1:{SERVER_PORT}"

# (label, gguf relative path, note)
MODELS = [
    ("qwen2.5-7b-instruct-q4",   "qwen2.5-7b-instruct-q4_k_m.gguf",        "dense 7.6B, Q4"),
    ("skippy-7b-v4-q4",          "kyle-7b-v4-q4_k_m.gguf",                 "PRODUCTION fine-tune, dense 7.6B, Q4"),
    ("qwen2.5-14b-instruct-q4",  "qwen2.5-14b-instruct-stock-q4_k_m.gguf", "dense 14.7B, Q4 — bandwidth probe A"),
    ("qwen2.5-14b-instruct-q8",  "Qwen2.5-14B-Instruct-Q8_0.gguf",         "dense 14.7B, Q8 — bandwidth probe B"),
    ("qwen3-30b-a3b-q4",         "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf","MoE 30.5B total / 3.3B active — THE experiment"),
]


def wait_for_server(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def post_json(path, payload, timeout=600):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Tegrastats:
    """Sample board power. VDD_GPU_SOC + VDD_CPU_CV are the two rails that move."""
    PAT = re.compile(r"(VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0)\s+(\d+)mW")

    def __init__(self, logpath):
        self.logpath = Path(logpath)
        self.proc = None

    def __enter__(self):
        try:
            self.fh = open(self.logpath, "w")
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", "1000"],
                stdout=self.fh, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("  ! tegrastats not found — skipping power", file=sys.stderr)
            self.proc = None
        return self

    def __exit__(self, *a):
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.proc.kill()
            self.fh.close()

    def summarize(self):
        if not self.logpath.exists():
            return None
        rails = {}
        for line in self.logpath.read_text().splitlines():
            for rail, mw in self.PAT.findall(line):
                rails.setdefault(rail, []).append(int(mw))
        if not rails:
            return None
        out = {r: {"mean_mW": round(sum(v)/len(v)), "peak_mW": max(v), "n": len(v)}
               for r, v in rails.items()}
        out["total_mean_mW"] = sum(o["mean_mW"] for r, o in out.items()
                                   if r in ("VDD_GPU_SOC", "VDD_CPU_CV"))
        return out


def run_model(label, gguf, note, bundle_dir, out_dir, prompts, limit=None):
    print(f"\n{'='*72}\n▶ {label}   ({note})\n  {gguf.name}  {gguf.stat().st_size/2**30:.2f} GB\n{'='*72}")
    srv = subprocess.Popen(
        ["llama-server", "-m", str(gguf), "--port", str(SERVER_PORT),
         "-ngl", "999", "-c", "8192", "--host", "127.0.0.1",
         "-t", "8", "--no-warmup"],
        stdout=open(out_dir / f"server_{label}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_for_server():
            print(f"  ✗ server never came up for {label}", file=sys.stderr)
            return None

        power_log = out_dir / f"tegrastats_{label}.log"
        completions, perf = [], []
        work = prompts[:limit] if limit else prompts

        with Tegrastats(power_log) as tg:
            t_start = time.time()
            for i, e in enumerate(work, 1):
                payload = {
                    "messages": [
                        {"role": "system", "content": e["system"]},
                        {"role": "user", "content": e["user"]},
                    ],
                    "temperature": 0.0,
                    "max_tokens": e["gen"]["max_tokens"],
                    "cache_prompt": False,   # never let cache contaminate TTFT
                }
                t0 = time.time()
                try:
                    r = post_json("/v1/chat/completions", payload)
                except Exception as ex:
                    print(f"  ! {e['uid']}: {ex}", file=sys.stderr)
                    continue
                wall = time.time() - t0
                text = r["choices"][0]["message"]["content"]
                ntok = (r.get("usage") or {}).get("completion_tokens")
                rec = {
                    "uid": e["uid"], "prompt_id": e["prompt_id"],
                    "sample_index": e["sample_index"], "text": text,
                    "tokens": ntok,
                    "elapsed_s": round(wall, 3),
                }
                # llama-server usually attaches `timings` to the OpenAI-compat
                # response, but not on every build/flag combination. Without a
                # fallback the entire perf half of this run silently yields nulls
                # — which is the whole reason we booked the board. So derive it.
                tim = r.get("timings") or {}
                if tim.get("predicted_per_second"):
                    rec["ttft_ms"] = round(tim.get("prompt_ms", 0), 2)
                    rec["decode_tok_s"] = round(tim["predicted_per_second"], 2)
                    rec["timing_source"] = "server"
                elif ntok and wall > 0:
                    rec["decode_tok_s"] = round(ntok / wall, 2)
                    rec["timing_source"] = "derived_wall_clock"  # includes prefill; a LOWER bound
                else:
                    rec["timing_source"] = "none"
                completions.append(rec)
                perf.append({k: rec.get(k) for k in ("ttft_ms", "decode_tok_s", "tokens")})
                if i % 20 == 0 or i == len(work):
                    print(f"  [{i}/{len(work)}] {rec.get('decode_tok_s','?')} tok/s")
            wall_total = time.time() - t_start

        (out_dir / f"completions_{label}.jsonl").write_text(
            "\n".join(json.dumps(c) for c in completions) + "\n")

        dts = [p["decode_tok_s"] for p in perf if p.get("decode_tok_s")]
        ttfts = [p["ttft_ms"] for p in perf if p.get("ttft_ms")]
        dts.sort(); ttfts.sort()
        med = lambda v: v[len(v)//2] if v else None
        srcs = {}
        for c in completions:
            srcs[c.get("timing_source","none")] = srcs.get(c.get("timing_source","none"),0)+1
        summary = {
            "label": label, "note": note,
            "timing_source": srcs,   # {"server": n} is trustworthy; derived_wall_clock is a LOWER bound
            "gguf": gguf.name,
            "weight_bytes": gguf.stat().st_size,
            "weight_GiB": round(gguf.stat().st_size / 2**30, 3),
            "n_samples": len(completions),
            "wall_s": round(wall_total, 1),
            "decode_tok_s_median": med(dts),
            "decode_tok_s_mean": round(sum(dts)/len(dts), 2) if dts else None,
            "ttft_ms_median": med(ttfts),
            "power": Tegrastats(power_log).summarize(),
        }
        (out_dir / f"perf_{label}.json").write_text(json.dumps(summary, indent=2))
        print(f"  ✓ {summary['decode_tok_s_median']} tok/s median, "
              f"TTFT {summary['ttft_ms_median']} ms, "
              f"{(summary['power'] or {}).get('total_mean_mW','?')} mW")
        return summary
    finally:
        srv.send_signal(signal.SIGINT)
        try: srv.wait(timeout=30)
        except subprocess.TimeoutExpired: srv.kill()
        time.sleep(3)   # let VRAM actually free before the next model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="only first N prompts (smoke test)")
    ap.add_argument("--only", default=None, help="comma-separated labels")
    a = ap.parse_args()

    models_dir = Path(a.models_dir); bundle = Path(a.bundle_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    pack = out / "prompt_pack.jsonl"
    if not pack.exists():
        subprocess.run([sys.executable, str(bundle / "skippy_eval.py"), "build",
                        "--rag", "--samples", "3", "-o", str(pack)], check=True,
                       cwd=bundle)
    prompts = [json.loads(l) for l in pack.open()]
    print(f"loaded {len(prompts)} prompts")

    only = set(a.only.split(",")) if a.only else None
    results = []
    for label, fname, note in MODELS:
        if only and label not in only:
            continue
        g = models_dir / fname
        if not g.exists():
            print(f"  ! missing {g}, skipping", file=sys.stderr); continue
        s = run_model(label, g, note, bundle, out, prompts, a.limit)
        if s: results.append(s)

    (out / "perf_all.json").write_text(json.dumps(results, indent=2))

    # The headline: does decode track ACTIVE params or FILE SIZE?
    print(f"\n{'='*72}\nDECODE-LAW TEST\n{'='*72}")
    print(f"{'model':<28} {'GiB':>7} {'tok/s':>9} {'TTFT ms':>9} {'mW':>7}")
    for r in results:
        p = (r.get("power") or {}).get("total_mean_mW")
        print(f"{r['label']:<28} {r['weight_GiB']:>7.2f} {str(r['decode_tok_s_median']):>9} "
              f"{str(r['ttft_ms_median']):>9} {str(p):>7}")
    d14 = next((r for r in results if r["label"] == "qwen2.5-14b-instruct-q4"), None)
    moe = next((r for r in results if r["label"] == "qwen3-30b-a3b-q4"), None)
    if d14 and moe and d14["decode_tok_s_median"] and moe["decode_tok_s_median"]:
        ratio = moe["decode_tok_s_median"] / d14["decode_tok_s_median"]
        print(f"\nMoE (17.3 GiB, 3.3B active) vs dense-14B (8.4 GiB, 14.7B active)")
        print(f"  file-size ratio  : {moe['weight_GiB']/d14['weight_GiB']:.2f}x  (MoE is BIGGER)")
        print(f"  decode tok/s ratio: {ratio:.2f}x")
        if ratio > 1.0:
            print("  ⇒ MoE decodes FASTER despite being larger on disk.")
            print("    Decode tracks ACTIVE params, not total. qualcomm §4.5 CONFIRMED on Orin.")
        else:
            print("  ⇒ MoE decodes SLOWER. Decode does NOT track active params here.")
            print("    qualcomm §4.5 REFUTED on Orin — the interesting outcome. Check whether")
            print("    llama.cpp gathers experts per-token or materialises the full weight set.")
    q4 = d14
    q8 = next((r for r in results if r["label"] == "qwen2.5-14b-instruct-q8"), None)
    if q4 and q8 and q4["decode_tok_s_median"] and q8["decode_tok_s_median"]:
        # Pure weight-streaming ⇒ tok/s ∝ 1/weight_bytes, so
        # predicted (q8 tok/s)/(q4 tok/s) = q4_bytes/q8_bytes < 1.
        predicted = q4["weight_bytes"] / q8["weight_bytes"]
        measured = q8["decode_tok_s_median"] / q4["decode_tok_s_median"]
        print(f"\nBandwidth probe — same model/arch, Q8 vs Q4 "
              f"({q8['weight_GiB']:.1f} vs {q4['weight_GiB']:.1f} GiB):")
        print(f"  predicted tok/s ratio if PURELY bus-bound : {predicted:.3f}x")
        print(f"  measured  tok/s ratio                     : {measured:.3f}x")
        print(f"  ratio of ratios (1.0 = perfectly bus-bound): {measured/predicted:.3f}")
        if measured > predicted * 1.05:
            print("  ⇒ Q8 is FASTER than weight-streaming predicts: decode is not purely")
            print("    bus-bound here (dequant cost, compute, or cache effects matter).")
        elif measured < predicted * 0.95:
            print("  ⇒ Q8 is SLOWER than predicted: Q8_0 dequant overhead exceeds its")
            print("    bandwidth disadvantage. Weight-byte math under-predicts the cost.")
        else:
            print("  ⇒ within 5% of pure weight-streaming. Decode is bus-bound on Orin,")
            print("    which is the premise qualcomm's §4.4 latency model rests on.")


if __name__ == "__main__":
    main()
