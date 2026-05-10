#!/usr/bin/env python3
"""Cross-judge corroboration variant — OpenAI / DeepSeek (OpenAI-compatible).

Mirrors `eval/run_llm_judge_eval.py` (Anthropic Sonnet judge) exactly, swapping
the API client for OpenAI's. Uses the same JudgeScore Pydantic schema, the
same JUDGE_SYSTEM_PROMPT, the same select_held_out_samples() seeded sampler.

This makes outputs apples-to-apples comparable to the Anthropic-judge JSONs:
same prompt, same samples, same rubric, different judge.

Why: cross-judge corroboration with a non-Anthropic model is the highest-value
single methodology hardening per reviewer (2026-05-09). This script is the
implementation.

Usage:
    OPENAI_API_KEY=sk-... python3 eval/run_llm_judge_eval_openai.py \\
        --eval-json eval/results/acc_candidate-kyle-qwen25-7b-v4-v2-rag_20260502-175416.json \\
        --max-prompts 50 \\
        --judge-model gpt-4o-2024-08-06

For DeepSeek:
    OPENAI_API_KEY=sk-deepseek-... python3 eval/run_llm_judge_eval_openai.py \\
        --eval-json ... --judge-model deepseek-chat \\
        --base-url https://api.deepseek.com/v1
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from openai import OpenAI, APIStatusError
from pydantic import BaseModel, Field, ValidationError

# Reuse the schema + prompt + sample-selection from the Anthropic script.
# Importing here avoids drift if the rubric ever changes.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llm_judge_eval import (
    JudgeScore,
    JUDGE_SYSTEM_PROMPT,
    load_eval_data,
    select_held_out_samples,
    RESULTS_DIR,
)


def judge_one(client: OpenAI, judge_model: str, sample: dict) -> dict:
    """Score one (prompt, response) pair via the OpenAI-compatible judge."""
    user_block = f"""## User prompt
{sample['prompt_text']}

## AI assistant's response
{sample['response_text']}

## Optional metadata
- Category: {sample['category']}
- Gold substrings (substring grader expected): {sample['gold_substrings']}
- Match mode: {sample['match_mode']}
- Used RAG citations: {sample['has_citations']} ({sample['n_citations']} citations)

Score the response."""

    response = client.chat.completions.parse(
        model=judge_model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
        response_format=JudgeScore,
    )
    parsed: JudgeScore = response.choices[0].message.parsed

    usage = response.usage
    return {
        "prompt_id": sample["prompt_id"],
        "category": sample["category"],
        "correctness": parsed.correctness,
        "instruction_following": parsed.instruction_following,
        "faithfulness": parsed.faithfulness,
        "conciseness": parsed.conciseness,
        "total": parsed.correctness + parsed.instruction_following + parsed.faithfulness + parsed.conciseness,
        "notes": parsed.notes,
        "tokens_in": usage.prompt_tokens,
        "tokens_in_cached_read": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        "tokens_in_cached_write": 0,  # OpenAI doesn't expose cache-write metric
        "tokens_out": usage.completion_tokens,
    }


def aggregate(judged: list[dict]) -> dict:
    if not judged:
        return {}
    return {
        "n_judged": len(judged),
        "mean_correctness": round(mean(j["correctness"] for j in judged), 3),
        "mean_instruction_following": round(mean(j["instruction_following"] for j in judged), 3),
        "mean_faithfulness": round(mean(j["faithfulness"] for j in judged), 3),
        "mean_conciseness": round(mean(j["conciseness"] for j in judged), 3),
        "mean_total": round(mean(j["total"] for j in judged), 3),
        "max_total": 8,
        "tokens_in_total": sum(j["tokens_in"] for j in judged),
        "tokens_in_cached_read_total": sum(j["tokens_in_cached_read"] for j in judged),
        "tokens_in_cached_write_total": sum(j["tokens_in_cached_write"] for j in judged),
        "tokens_out_total": sum(j["tokens_out"] for j in judged),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True,
                        help="Path to eval/results/acc_*.json to judge")
    parser.add_argument("--judge-model", default="gpt-4o-2024-08-06",
                        help="OpenAI-compatible model ID (default: gpt-4o-2024-08-06)")
    parser.add_argument("--base-url", default=None,
                        help="Override OpenAI base URL (e.g., https://api.deepseek.com/v1 for DeepSeek)")
    parser.add_argument("--max-prompts", type=int, default=50,
                        help="Max prompts to judge (default 50, deterministic seed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection (matches Anthropic-judge default)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: eval/results/judge_<model>_<ts>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select samples + print summary without calling the API")
    parser.add_argument("--inter-call-sleep", type=float,
                        default=float(os.environ.get("JUDGE_SLEEP_SEC", "0.0")),
                        help="Sleep between API calls in seconds (default 0; OpenAI tier limits typically generous)")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ and not args.dry_run:
        print("ERROR: OPENAI_API_KEY not set. Export your key, or pass --dry-run.")
        sys.exit(1)

    eval_path = Path(args.eval_json).resolve()
    if not eval_path.exists():
        print(f"ERROR: {eval_path} does not exist")
        sys.exit(1)

    model_name, data = load_eval_data(eval_path)
    samples = select_held_out_samples(data, max_prompts=args.max_prompts, seed=args.seed)

    print(f"Eval JSON:    {eval_path.name}")
    print(f"Model judged: {model_name}")
    print(f"Judge model:  {args.judge_model}")
    if args.base_url:
        print(f"Base URL:     {args.base_url}")
    print(f"Samples:      {len(samples)}")
    print(f"Categories:   {sorted(set(s['category'] for s in samples))}")
    print()

    if args.dry_run:
        print("DRY RUN — not calling the judge.")
        return

    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    judged = []
    for i, sample in enumerate(samples, 1):
        try:
            result = judge_one(client, args.judge_model, sample)
            judged.append(result)
            cache_marker = "🟢" if result["tokens_in_cached_read"] > 0 else "  "
            print(f"  [{i:2d}/{len(samples)}] {cache_marker} {result['prompt_id']:40s} "
                  f"c={result['correctness']} i={result['instruction_following']} "
                  f"f={result['faithfulness']} sz={result['conciseness']} = {result['total']}/8")
        except APIStatusError as e:
            print(f"  [{i:2d}/{len(samples)}] ❌ {sample['prompt_id']}: {e.status_code} {e.message}")
            judged.append({"prompt_id": sample["prompt_id"], "error": f"APIStatusError {e.status_code}: {e.message}"})
        except ValidationError as e:
            print(f"  [{i:2d}/{len(samples)}] ⚠️  {sample['prompt_id']}: judge returned non-conforming output ({e})")
            judged.append({"prompt_id": sample["prompt_id"], "error": f"ValidationError: {e}"})
        except Exception as e:
            print(f"  [{i:2d}/{len(samples)}] ❌ {sample['prompt_id']}: {type(e).__name__}: {e}")
            judged.append({"prompt_id": sample["prompt_id"], "error": f"{type(e).__name__}: {e}"})
        if args.inter_call_sleep > 0 and i < len(samples):
            time.sleep(args.inter_call_sleep)

    summary = aggregate([j for j in judged if "error" not in j])

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"judge_{model_name}_{args.judge_model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    payload = {
        "evaluated_eval_json": eval_path.name,
        "model_judged": model_name,
        "judge_model": args.judge_model,
        "judge_provider": "openai" if not args.base_url else f"openai-compatible:{args.base_url}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(samples),
        "seed": args.seed,
        "summary": summary,
        "per_sample": judged,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print()
    print(f"📄 wrote {out_path}")
    print(f"   mean total: {summary.get('mean_total', '?')}/8 across {summary.get('n_judged', 0)} samples")
    print(f"   correctness {summary.get('mean_correctness')} | instruct {summary.get('mean_instruction_following')} | faithful {summary.get('mean_faithfulness')} | concise {summary.get('mean_conciseness')}")


if __name__ == "__main__":
    main()
