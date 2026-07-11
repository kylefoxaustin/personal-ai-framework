#!/usr/bin/env python3
"""Accuracy eval via vLLM — for compressed-tensors W8A8 models.

vLLM has native CUTLASS INT8 matmul kernels that don't dequantize to
fp16 at compute time, so the full 5090 memory budget stays usable for
KV cache + activations. transformers' CompressedLinear path would run
the matmul in fp16 (dequantizing), doubling peak memory and OOMing on
14B models.

Output schema matches `run_accuracy_eval.py` so diffs via
`compare_accuracy_runs.py` work identically.

Usage:
  /tmp/quant-venv/bin/python3 eval/run_accuracy_eval_vllm.py \\
      --model-path models/qwen2.5-14b-w8a8 \\
      --name candidate-dense-w8a8-v1 \\
      --prompts eval/prompts.json \\
      --samples 3
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path


SYSTEM_PROMPT = (
    "Your name is Skippy. You are a helpful AI assistant. Be direct, "
    "concise, and specific."
)


def build_prompt(user_message: str, tokenizer, rag_chunks: list = None) -> str:
    """Build a ChatML prompt. If rag_chunks is non-empty, prepend them to
    the user message framed as 'excerpts from knowledge base'."""
    if rag_chunks:
        chunk_block = "\n\n".join(
            f"— Excerpt {i+1}{' (source: ' + (c.get('source') or 'unknown') + ')' if isinstance(c, dict) else ''} —\n"
            f"{(c.get('text', '') if isinstance(c, dict) else str(c)).strip()}"
            for i, c in enumerate(rag_chunks)
            if (c.get('text') if isinstance(c, dict) else c)
        )
        full_user = (
            f"Excerpts retrieved from your knowledge base follow. Use them to answer.\n\n"
            f"{chunk_block}\n\n---\n\nQuestion: {user_message.strip()}"
        )
    else:
        full_user = user_message.strip()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def score(response_text: str, gold_substrings: list, match_mode: str = "all") -> str:
    if not gold_substrings:
        return "manual"
    lower = response_text.lower()
    if match_mode == "any":
        for g in gold_substrings:
            if g.lower() in lower:
                return "pass"
        return "fail (no refusal phrase matched)"
    missing = [g for g in gold_substrings if g.lower() not in lower]
    if not missing:
        return "pass"
    return f"fail (missing: {', '.join(missing)})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompts", default="eval/prompts.json")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="vLLM max context length (default 4K — small prompts)")
    parser.add_argument("--gpu-mem-utilization", type=float, default=0.85,
                        help="Fraction of GPU memory vLLM can use (default 0.85)")
    parser.add_argument("--categories", default="")
    parser.add_argument("--rag-chunks", default=None,
                        help="Path to rag_chunks_for_v2.json — when set, "
                             "each prompt gets the corresponding chunks prepended")
    parser.add_argument("--max-rag-chunks", type=int, default=8,
                        help="Cap on chunks per prompt (keeps prefix tractable for VRAM-constrained runs)")
    args = parser.parse_args()

    print(f"▶ Loading vLLM model from {args.model_path}...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # For compressed-tensors W8A8, vLLM auto-detects the quantization from
    # config.json's quantization_config block. No need to pass quantization=...
    # enforce_eager=True skips torch.inductor autotune/cudagraphs which
    # otherwise demands ~1.5 GB extra during compile. Slight per-call perf
    # hit but keeps us under the 5090's 32 GB VRAM ceiling at 14B fp16.
    model = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_utilization,
        enforce_eager=True,
        dtype="bfloat16",
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    sampling = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0.0,
        top_p=1.0 if args.temperature == 0 else 0.9,
        max_tokens=args.max_tokens,
    )

    with open(args.prompts) as f:
        prompt_set = json.load(f)

    cat_filter = set(c.strip() for c in args.categories.split(",") if c.strip())
    prompts_to_run = [p for p in prompt_set["prompts"]
                      if not cat_filter or p.get("category") in cat_filter]

    rag_chunks_by_id = {}
    if args.rag_chunks:
        with open(args.rag_chunks) as f:
            rag_data = json.load(f)
        rag_chunks_by_id = rag_data.get("chunks_by_id", {})
        print(f"▶ Loaded {len(rag_chunks_by_id)} RAG chunk sets from {args.rag_chunks}")

    out = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": f"vllm:{args.model_path}",
        "prompts_version": prompt_set.get("version"),
        "samples_per_prompt": args.samples,
        "config": {
            "use_rag": bool(args.rag_chunks),
            "rag_chunks_file": args.rag_chunks,
            "skip_agent_loop": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_model_len": args.max_model_len,
        },
        "prompts": [],
        "summary": {},
    }

    cum_pass, cum_total = 0, 0
    for p in prompts_to_run:
        print(f"▶ {p['id']} [{p['category']}]")
        chunks = rag_chunks_by_id.get(p["id"], [])[: args.max_rag_chunks]
        full_prompt = build_prompt(p["prompt"], tokenizer, rag_chunks=chunks)
        samples = []
        for s_idx in range(args.samples):
            t_start = time.time()
            outputs = model.generate([full_prompt], sampling, use_tqdm=False)
            elapsed = time.time() - t_start
            text = outputs[0].outputs[0].text.strip()
            prompt_tokens = len(outputs[0].prompt_token_ids)
            completion_tokens = len(outputs[0].outputs[0].token_ids)
            r = {
                "text": text,
                "tokens": prompt_tokens + completion_tokens,
                "elapsed_s": round(elapsed, 2),
                "sample_index": s_idx,
                "telemetry": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_ms": round(elapsed * 1000, 2),
                },
            }
            samples.append(r)
            s = score(r["text"], p.get("gold_substrings", []),
                      p.get("match_mode", "all"))
            flag = {"pass": "✅", "manual": "🖐️", "error": "💥"}.get(s, "❌")
            print(f"  [{s_idx+1}/{args.samples}] {flag} {s}")

        statuses = [score(s["text"], p.get("gold_substrings", []),
                          p.get("match_mode", "all"))
                    for s in samples]
        agg = {
            "per_sample": statuses,
            "pass_n": sum(1 for s in statuses if s == "pass"),
            "fail_n": sum(1 for s in statuses if s.startswith("fail")),
            "error_n": sum(1 for s in statuses if s == "error"),
            "manual_n": sum(1 for s in statuses if s == "manual"),
            "n": len(statuses),
        }
        cum_pass += agg["pass_n"]
        cum_total += agg["n"]
        out["prompts"].append({**p, "samples": samples, "aggregate": agg})

    out["summary"] = {
        "total_samples": cum_total,
        "passed": cum_pass,
        "pass_rate": round(cum_pass / cum_total, 3) if cum_total else 0.0,
    }
    results_dir = Path(args.prompts).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"acc_{args.name}_{stamp}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n📄 {json_path}")
    print(f"   pass: {cum_pass}/{cum_total} "
          f"({out['summary']['pass_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
