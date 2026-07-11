#!/usr/bin/env python3
"""Runtime perf bench: prefill + decode tok/s for a vLLM model on the 5090.

Used to compare FP8 vs bf16 runtime throughput — completes the "quality
matches AND runs faster" side of the silicon-architecture argument.

Usage:
  ~/.venvs/quant-venv/bin/python3 eval/run_perf_vllm.py \\
      --model-path models/qwen2.5-14b-hf --label fp16 --runs 10
  ~/.venvs/quant-venv/bin/python3 eval/run_perf_vllm.py \\
      --model-path models/qwen2.5-14b-fp8 --label fp8 --runs 10

Fires N warmup + N measured completions at a standard short-chat shape
(250 prefill, 200 decode) and reports p50/p90 prefill tok/s, decode tok/s,
and total latency. Short enough to fit in constrained max-model-len.
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=768)
    parser.add_argument("--gpu-mem-utilization", type=float, default=0.92)
    args = parser.parse_args()

    print(f"▶ Loading vLLM model from {args.model_path} (label={args.label})...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_utilization,
        enforce_eager=True,
        dtype="bfloat16",
    )

    # A representative short-chat prompt that fits in max-model-len.
    # Using a real prompt (not random tokens) to hit realistic code paths.
    user_msg = (
        "Explain in three sentences what mixture-of-experts means in large "
        "language models, why it can be faster at decode than a dense model "
        "of the same parameter count, and what the main drawback is."
    )
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Your name is Skippy."},
         {"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )

    # Force fixed-length decode so we measure steady-state tok/s.
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        min_tokens=args.decode_tokens,  # prevents early-stop drift
    )

    print(f"▶ Warmup ({args.warmup} runs)...")
    for _ in range(args.warmup):
        model.generate([chat_prompt], sampling, use_tqdm=False)

    print(f"▶ Measured runs ({args.runs})...")
    results = []
    for i in range(args.runs):
        t0 = time.time()
        out = model.generate([chat_prompt], sampling, use_tqdm=False)
        elapsed = time.time() - t0
        prompt_toks = len(out[0].prompt_token_ids)
        completion_toks = len(out[0].outputs[0].token_ids)
        # vLLM's outputs object often exposes first-token timing; fall back
        # to overall timing if not available.
        ttft = None
        try:
            if out[0].metrics and out[0].metrics.first_token_time is not None:
                ttft = out[0].metrics.first_token_time - out[0].metrics.arrival_time
        except (AttributeError, TypeError):
            ttft = None
        results.append({
            "elapsed_s": elapsed,
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "ttft_s": ttft,
        })
        if ttft is not None:
            decode_s = elapsed - ttft
            prefill_ts = prompt_toks / ttft if ttft > 0 else None
            decode_ts = completion_toks / decode_s if decode_s > 0 else None
            print(f"  [{i+1}/{args.runs}] total={elapsed:.3f}s "
                  f"ttft={ttft*1000:.1f}ms "
                  f"prefill={prefill_ts:.0f} tok/s decode={decode_ts:.1f} tok/s")
        else:
            print(f"  [{i+1}/{args.runs}] total={elapsed:.3f}s "
                  f"(no ttft) overall={completion_toks/elapsed:.1f} tok/s")

    # Aggregate
    totals = [r["elapsed_s"] for r in results]
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    overall_ts = [r["completion_tokens"] / r["elapsed_s"] for r in results]
    prefill_ts = [r["prompt_tokens"] / r["ttft_s"]
                  for r in results if r["ttft_s"] and r["ttft_s"] > 0]
    decode_ts = [r["completion_tokens"] / (r["elapsed_s"] - r["ttft_s"])
                 for r in results if r["ttft_s"] and r["ttft_s"] > 0]

    summary = {
        "label": args.label,
        "model_path": args.model_path,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": args.runs,
        "prompt_tokens": results[0]["prompt_tokens"],
        "completion_tokens": results[0]["completion_tokens"],
        "total_s_p50": round(statistics.median(totals), 3),
        "total_s_p90": round(sorted(totals)[min(len(totals)-1, int(0.9*len(totals)))], 3),
    }
    if ttfts:
        summary["ttft_s_p50"] = round(statistics.median(ttfts), 4)
        summary["prefill_tok_s_p50"] = round(statistics.median(prefill_ts), 1)
        summary["decode_tok_s_p50"] = round(statistics.median(decode_ts), 1)
    else:
        summary["overall_tok_s_p50"] = round(statistics.median(overall_ts), 1)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"perf_{args.label}_{stamp}.json"
    out_path.write_text(json.dumps(
        {"summary": summary, "runs": results}, indent=2,
    ))
    print(f"\n📄 {out_path}")


if __name__ == "__main__":
    main()
