#!/usr/bin/env python3
"""Accuracy eval for a HuggingFace-format model (fp16, W8A8 compressed-tensors,
etc.) — bypasses Skippy entirely. Use for Tier 2 int8-compute experiments.

Matches the output schema of `run_accuracy_eval.py` so outputs can be diffed
by `compare_accuracy_runs.py`.

Key difference from run_accuracy_eval.py: no Skippy server dependency, no
RAG, no agent loop, no auth. Just load the HF model, run the prompts,
score. This is the "pure LLM" path — answers the question "does activation-
precision quantization damage the model's text-generation ability" directly
without the retrieval-divergence confound.

Usage:
  python3 eval/run_accuracy_eval_hf.py \\
      --model-path models/qwen2.5-14b-w8a8 \\
      --name candidate-dense-w8a8-v1 \\
      --prompts eval/prompts.json \\
      --samples 3

Outputs:
  eval/results/acc_<name>_<timestamp>.json
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch


SYSTEM_PROMPT = (
    "Your name is Skippy. You are a helpful AI assistant. Be direct, "
    "concise, and specific."
)


def build_prompt(user_message: str, tokenizer) -> str:
    """Build the same ChatML-style multi-turn prompt Skippy uses for a
    single-turn user message with the default system context. Mirrors
    `pipeline/llm_server.py::_build_multiturn_prompt` in the no-RAG path."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
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
        return f"fail (no refusal phrase matched)"
    missing = [g for g in gold_substrings if g.lower() not in lower]
    if not missing:
        return "pass"
    return f"fail (missing: {', '.join(missing)})"


def run_one_prompt(model, tokenizer, prompt_text: str, *, max_tokens: int,
                   temperature: float) -> dict:
    t0 = time.time()
    full_prompt = build_prompt(prompt_text, tokenizer)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]
    with torch.inference_mode():
        out = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    # Slice off the prompt tokens, decode only the new ones
    gen_ids = out[0][prompt_tokens:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    completion_tokens = gen_ids.shape[0]
    return {
        "text": text,
        "tokens": int(prompt_tokens + completion_tokens),
        "elapsed_s": round(elapsed, 2),
        "telemetry": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_ms": round(elapsed * 1000, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True,
                        help="Path to HF model dir (fp16 or compressed-tensors W8A8)")
    parser.add_argument("--name", required=True, help="Run label")
    parser.add_argument("--prompts", default="eval/prompts.json")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--categories", default="")
    args = parser.parse_args()

    print(f"▶ Loading model from {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print(f"  loaded: {type(model).__name__} on {next(model.parameters()).device}")

    with open(args.prompts) as f:
        prompt_set = json.load(f)

    cat_filter = set(c.strip() for c in args.categories.split(",") if c.strip())
    prompts_to_run = [p for p in prompt_set["prompts"]
                      if not cat_filter or p.get("category") in cat_filter]

    out = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": f"hf:{args.model_path}",
        "prompts_version": prompt_set.get("version"),
        "samples_per_prompt": args.samples,
        "config": {
            "use_rag": False,  # HF-only path, no RAG
            "skip_agent_loop": True,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "prompts": [],
        "summary": {},
    }

    cum_pass, cum_total = 0, 0
    for p in prompts_to_run:
        print(f"▶ {p['id']} [{p['category']}]")
        samples = []
        for s_idx in range(args.samples):
            r = run_one_prompt(model, tokenizer, p["prompt"],
                               max_tokens=args.max_tokens,
                               temperature=args.temperature)
            r["sample_index"] = s_idx
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
