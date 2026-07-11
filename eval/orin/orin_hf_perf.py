#!/usr/bin/env python3
"""Decode-rate + TTFT + board-power measurement for HF fp16 models on Jetson AGX Orin.

Runs ON the Orin. Uses only what is already installed there (torch + transformers);
no llama.cpp, no vLLM, no new packages.

WHAT THIS MEASURES AND WHY
--------------------------
qualcomm's IQ-9075 dossier §4.4 asserts, from a bandwidth model, that LLM decode
is weight-streaming: every decoded token drags the full weight set across the
memory bus, so

    per-token latency ≈ weight_bytes / memory_bandwidth

That is *derived* there, validated against a single OpenVLA datapoint. It is the
premise under the entire §4.4/§4.6 latency table. It has never been tested on a
second memory system.

Here it is testable directly. Two models, same architecture family
(Qwen2ForCausalLM), different weight footprints:

    skippy-7b-v4    14.19 GiB fp16   (production Skippy fine-tune)
    qwen2.5-14b     27.51 GiB fp16

weight ratio = 1.94x. If decode is purely bus-bound, then

    tok/s(14B) / tok/s(7B)  ≈  1 / 1.94  =  0.516

Deviation from 0.516 quantifies how much of decode is NOT weight streaming on
Orin's 204.8 GB/s LPDDR5. Combined with the same test on the IQ-9075's 76.8 GB/s
bus, it says whether the law is a property of transformers or of one board.

METHOD — separating prefill from decode without a streamer
----------------------------------------------------------
Naive `generate()` timing conflates prefill (compute-bound, scales with prompt
length) with decode (bus-bound, scales with weights). We separate them by timing
the same prompt twice:

    t1   = time to generate  1 new token   = prefill + 1 decode step
    tN   = time to generate  N new tokens  = prefill + N decode steps

    ⇒  decode_tok_s = (N - 1) / (tN - t1)      # prefill cancels exactly
    ⇒  ttft_ms      = t1 * 1000                # prefill + one step

This is exact under greedy decoding with a fixed prompt and no cache reuse
between the two calls, because both calls do identical prefill work.

Also derives an effective achieved bandwidth:

    achieved_GB_s = weight_bytes * decode_tok_s / 1e9

which, compared against the board's 204.8 GB/s spec, is the honest "how close to
the bus are we" number. (Reading real weights, not a synthetic memcpy.)
"""
import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _patch_sdpa_enable_gqa():
    """JetPack's torch is 2.5.0a0 (an NVIDIA pre-release) whose
    scaled_dot_product_attention has no `enable_gqa` kwarg, but transformers
    5.13 passes it unconditionally → TypeError on every forward.

    We cannot simply drop the kwarg: both Qwen2.5 models use grouped-query
    attention (7B: 28 q heads / 4 kv heads; 14B: 40 / 8). `enable_gqa=True`
    tells SDPA to broadcast the kv heads across query-head groups. Dropping it
    would silently compute attention against mismatched head counts — either a
    shape error or, worse, wrong numbers.

    So we replicate it explicitly: expand k/v along the head dim before calling
    the original kernel. Mathematically identical; costs a little extra memory
    traffic for the expanded KV. With 42-168 char contexts and 128 new tokens
    the KV cache is a few MB against 14-27 GiB of weights, so it does not
    perturb the weight-streaming measurement.
    """
    import torch.nn.functional as F
    orig = F.scaled_dot_product_attention
    # SDPA is a C builtin: inspect.signature() raises "no signature found".
    # Probe by calling it. Only shim when the failure is specifically the
    # missing kwarg — any other TypeError means the kwarg WAS accepted.
    try:
        _q = torch.zeros(1, 2, 2, 4)
        _k = torch.zeros(1, 1, 2, 4)
        orig(_q, _k, _k, enable_gqa=True)
        return "native"
    except TypeError as e:
        if "enable_gqa" not in str(e):
            return "native"
    except Exception:
        return "native"

    def shim(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
             scale=None, enable_gqa=False, **kw):
        if enable_gqa and k.shape[-3] != q.shape[-3]:
            rep = q.shape[-3] // k.shape[-3]
            k = k.repeat_interleave(rep, dim=-3)
            v = v.repeat_interleave(rep, dim=-3)
        return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale, **kw)

    F.scaled_dot_product_attention = shim
    return "shimmed"


SDPA_MODE = _patch_sdpa_enable_gqa()

SYSTEM_PROMPT = (
    "Your name is Skippy. You are a helpful AI assistant. Be direct, "
    "concise, and specific."
)


class Tegrastats:
    PAT = re.compile(r"(VDD_GPU_SOC|VDD_CPU_CV)\s+(\d+)mW")

    def __init__(self, logpath):
        self.logpath = Path(logpath)
        self.proc = None
        self.fh = None

    def __enter__(self):
        try:
            self.fh = open(self.logpath, "w")
            self.proc = subprocess.Popen(["tegrastats", "--interval", "500"],
                                         stdout=self.fh, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("  ! tegrastats missing — no power data", file=sys.stderr)
        return self

    def __exit__(self, *a):
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.proc.kill()
        if self.fh: self.fh.close()

    def summarize(self):
        if not self.logpath.exists(): return None
        rails = {}
        for line in self.logpath.read_text().splitlines():
            for rail, mw in self.PAT.findall(line):
                rails.setdefault(rail, []).append(int(mw))
        if not rails: return None
        out = {r: {"mean_mW": round(sum(v)/len(v)), "peak_mW": max(v)} for r, v in rails.items()}
        out["total_mean_mW"] = sum(o["mean_mW"] for o in out.values() if isinstance(o, dict))
        return out


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    # transformers v5 renamed torch_dtype → dtype; support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    return tok, model


def weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


@torch.inference_mode()
def timed_generate(model, inputs, n_new):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=n_new, min_new_tokens=n_new,
                         do_sample=False, temperature=None, top_p=None, top_k=None,
                         use_cache=True, pad_token_id=model.config.eos_token_id
                         if isinstance(model.config.eos_token_id, int) else None)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def measure(label, path, prompts, n_new, out_dir):
    print(f"\n{'='*70}\n▶ {label}\n{'='*70}", flush=True)
    tok, model = load(path)
    wb = weight_bytes(model)
    print(f"  weights: {wb/2**30:.2f} GiB ({wb/1e9:.2f} GB)  params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)

    # warmup — first CUDA kernels + autotune must not pollute the measurement
    warm = tok(tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True),
        return_tensors="pt").to("cuda")
    timed_generate(model, warm, 8)

    rows = []
    power_log = out_dir / f"tegra_{label}.log"
    tg = Tegrastats(power_log)
    with tg:
        for i, p in enumerate(prompts, 1):
            text = tok.apply_chat_template(
                [{"role": "system", "content": p["system"]},
                 {"role": "user", "content": p["user"]}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to("cuda")
            n_prompt = enc["input_ids"].shape[1]

            t1, _ = timed_generate(model, enc, 1)
            tN, _ = timed_generate(model, enc, n_new)
            decode_s = (tN - t1) / (n_new - 1)
            rows.append({
                "prompt_id": p["prompt_id"], "prompt_tokens": n_prompt,
                "ttft_ms": round(t1 * 1000, 2),
                "decode_tok_s": round(1.0 / decode_s, 3),
                "t_1tok_s": round(t1, 4), "t_Ntok_s": round(tN, 4),
            })
            print(f"  [{i}/{len(prompts)}] ctx={n_prompt:<5} TTFT={rows[-1]['ttft_ms']:>8.1f} ms  "
                  f"decode={rows[-1]['decode_tok_s']:>6.2f} tok/s", flush=True)

    dts = sorted(r["decode_tok_s"] for r in rows)
    tts = sorted(r["ttft_ms"] for r in rows)
    med = lambda v: v[len(v)//2]
    summary = {
        "label": label,
        "weight_bytes": wb,
        "weight_GiB": round(wb / 2**30, 3),
        "params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
        "dtype": "float16",
        "n_prompts": len(rows),
        "n_new_tokens": n_new,
        "decode_tok_s_median": med(dts),
        "decode_tok_s_min": dts[0], "decode_tok_s_max": dts[-1],
        "ttft_ms_median": med(tts),
        "achieved_GB_s": round(wb * med(dts) / 1e9, 1),
        "bus_spec_GB_s": 204.8,
        "bus_utilization": round(wb * med(dts) / 1e9 / 204.8, 3),
        "power": tg.summarize(),
        "rows": rows,
    }
    (out_dir / f"perf_{label}.json").write_text(json.dumps(summary, indent=2))
    print(f"  ✓ decode {summary['decode_tok_s_median']} tok/s | TTFT {summary['ttft_ms_median']} ms "
          f"| {summary['achieved_GB_s']} GB/s = {summary['bus_utilization']:.0%} of bus "
          f"| {(summary['power'] or {}).get('total_mean_mW','?')} mW", flush=True)

    del model
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--n-new", type=int, default=64)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pack = [json.loads(l) for l in open(a.pack)]
    # one sample per prompt id, spread across context lengths
    seen, prompts = set(), []
    for e in pack:
        if e["prompt_id"] in seen: continue
        seen.add(e["prompt_id"]); prompts.append(e)
    prompts.sort(key=lambda e: len(e["user"]))
    step = max(1, len(prompts) // a.n_prompts)
    prompts = prompts[::step][:a.n_prompts]
    print(f"selected {len(prompts)} prompts, context spread "
          f"{len(prompts[0]['user'])}..{len(prompts[-1]['user'])} chars")

    results = []
    for spec in a.models:
        label, path = spec.split("=", 1)
        results.append(measure(label, path, prompts, a.n_new, out))

    (out / "perf_summary.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*70}\nWEIGHT-STREAMING LAW — is decode bus-bound on Orin?\n{'='*70}")
    print(f"{'model':<22} {'GiB':>7} {'tok/s':>8} {'GB/s':>8} {'% bus':>7} {'mW':>7}")
    for r in results:
        p = (r.get("power") or {}).get("total_mean_mW", "?")
        print(f"{r['label']:<22} {r['weight_GiB']:>7.2f} {r['decode_tok_s_median']:>8.2f} "
              f"{r['achieved_GB_s']:>8.1f} {r['bus_utilization']*100:>6.1f}% {str(p):>7}")

    if len(results) == 2:
        small, big = sorted(results, key=lambda r: r["weight_bytes"])
        predicted = small["weight_bytes"] / big["weight_bytes"]
        measured = big["decode_tok_s_median"] / small["decode_tok_s_median"]
        print(f"\n  weight ratio {small['label']}:{big['label']} = "
              f"1 : {big['weight_bytes']/small['weight_bytes']:.2f}")
        print(f"  predicted tok/s ratio if PURELY weight-streaming : {predicted:.3f}")
        print(f"  measured  tok/s ratio                            : {measured:.3f}")
        print(f"  ratio-of-ratios (1.000 = perfect bus-bound)      : {measured/predicted:.3f}")
        if abs(measured / predicted - 1) <= 0.10:
            print("\n  ⇒ WITHIN 10% OF PURE WEIGHT-STREAMING.")
            print("    Decode is bus-bound on Orin. qualcomm §4.4's premise holds on a")
            print("    204.8 GB/s bus, not just the IQ-9075's 76.8 GB/s.")
        elif measured > predicted:
            print("\n  ⇒ The big model is FASTER than weight-streaming predicts.")
            print("    Decode is not purely bus-bound: fixed per-token overhead (kernel")
            print("    launch, attention, sampling) is a material share at this size.")
        else:
            print("\n  ⇒ The big model is SLOWER than weight-streaming predicts.")
            print("    Something beyond weight bytes scales with size here — check")
            print("    KV-cache traffic and attention cost at this context length.")


if __name__ == "__main__":
    main()
