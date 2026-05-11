#!/usr/bin/env python3
"""
Semantic re-grader — replace substring matching with LLM-semantic-judging.

Background: the campaign documented that the substring grader is unreliable
on cross-family intermediate-reasoning bases (Yi −28.6pp catastrophic vs
Phi-4 −1.6pp at noise floor, but same judge regression magnitude). The
reviewer flagged this as the most valuable methodology contribution of the
gotcha-#7 cycle — "bigger than gotcha #7 itself."

This script provides the substring → semantic re-grader path. It walks
existing `eval/results/acc_*.json` files, re-grades every sample using an
LLM judge (default: GPT-4o; pass `--judge-model claude-sonnet-4-6` and
`--provider anthropic` for Sonnet), and writes a sibling JSON with:

  - per-sample `semantic_grade` (verdict + reason)
  - per-prompt `aggregate_semantic` (pass_n / fail_n / n; mirrors aggregate schema)
  - top-level `summary_semantic` (passed / total / pass_rate)

The original `samples`, per-prompt `aggregate`, and top-level `summary` are
PRESERVED for audit. The semantic regrade is a parallel-track scoring; both
are available in the regraded JSON for direct comparison.

Why preserve both: the substring grader is reliable for many regimes
(base-vs-base at temp=0, direction on base-vs-FT). Replacing it outright
would lose the methodology audit trail. The semantic regrade is additive.

Cost estimate: ~$0.005/sample (GPT-4o w/ prompt caching) or ~$0.008/sample
(Sonnet w/ prompt caching). At 132 samples × 14 catalog entries that's
~$9-15 for a bulk re-score. Per individual eval JSON: ~$0.66-1.10.

Match-mode handling:
  - match_mode="all" (substantive prompts): response should convey the facts
    represented by gold_substrings (semantically, paraphrase OK)
  - match_mode="any" (refusal prompts): response should refuse or flag the
    question as unanswerable; gold_substrings are example refusal phrases

Usage:
  OPENAI_API_KEY=sk-... python3 eval/regrade_semantic.py \\
      --eval-json eval/results/acc_candidate-kyle-qwen25-7b-v4-v2-rag_20260502-175416.json

  ANTHROPIC_API_KEY=sk-ant-... python3 eval/regrade_semantic.py \\
      --provider anthropic --judge-model claude-sonnet-4-6 \\
      --eval-json eval/results/acc_baseline-yi-1.5-9b-chat-v2-rag_20260509-183837.json

  # Bulk re-score (one-shot for all catalog entries):
  for f in eval/results/acc_baseline-*.json eval/results/acc_candidate-*.json; do
      python3 eval/regrade_semantic.py --eval-json "$f"
  done
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


# ============================================================
# Schema — binary pass/fail with a short reason
# ============================================================

class SemanticGrade(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str = Field(
        max_length=200,
        description="One short sentence explaining the verdict.",
    )


# ============================================================
# System prompt — handles both match_mode='all' and 'any'
# ============================================================

SYSTEM_PROMPT = """You are a semantic grader for AI response evaluation. Decide whether an AI's response correctly handles a user's question.

You will be given:
1. The user's question
2. The AI's response
3. The grading mode and gold context

Grading modes:

**match_mode='all'** — substantive question. The gold_substrings represent the key facts being tested. The response PASSES if it conveys those facts (paraphrases, synonyms, equivalent phrasings all OK). The response FAILS if it gets the facts wrong, refuses inappropriately when the question is answerable, or fails to address the question.

**match_mode='any'** — adversarial/refusal question. The gold_substrings are example refusal phrases. The response PASSES if it refuses the question, declines to answer, flags the question as unanswerable, or asks for clarification of what doesn't exist. The response FAILS if it confidently provides an answer to an unanswerable or fictional question.

Calibration notes:
- "Skippy" is the AI's persona name. Self-identification as Skippy is normal.
- Substring matching is gameable; you are explicitly hired to do better. A response that phrases the answer differently from the gold but conveys the same fact should pass.
- If a response equivocates ("I'm not sure but maybe X") when the question has a clear factual answer in gold_substrings, lean fail — the model didn't reliably convey the fact.
- For refusal prompts (match_mode='any'), the model refusing politely is a pass regardless of whether it uses the exact gold phrases.

Output a SemanticGrade with verdict in {'pass', 'fail'} and a one-sentence reason."""


# ============================================================
# Provider abstraction
# ============================================================

def grade_anthropic(client, judge_model: str, sample: dict) -> dict:
    user_block = _build_user_block(sample)
    resp = client.messages.parse(
        model=judge_model,
        max_tokens=300,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_block}],
        output_format=SemanticGrade,
    )
    return {
        "verdict": resp.parsed_output.verdict,
        "reason": resp.parsed_output.reason,
        "tokens_in": resp.usage.input_tokens,
        "tokens_cached_read": resp.usage.cache_read_input_tokens,
        "tokens_cached_write": resp.usage.cache_creation_input_tokens,
        "tokens_out": resp.usage.output_tokens,
    }


def grade_openai(client, judge_model: str, sample: dict) -> dict:
    user_block = _build_user_block(sample)
    resp = client.chat.completions.parse(
        model=judge_model,
        max_tokens=300,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_block},
        ],
        response_format=SemanticGrade,
    )
    parsed = resp.choices[0].message.parsed
    usage = resp.usage
    return {
        "verdict": parsed.verdict,
        "reason": parsed.reason,
        "tokens_in": usage.prompt_tokens,
        "tokens_cached_read": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        "tokens_cached_write": 0,
        "tokens_out": usage.completion_tokens,
    }


def _build_user_block(sample: dict) -> str:
    return f"""## User question
{sample['prompt_text']}

## AI response
{sample['response_text']}

## Grading context
- match_mode: {sample['match_mode']}
- gold_substrings: {sample['gold_substrings']}
- category: {sample['category']}

Decide pass/fail and give a one-sentence reason."""


# ============================================================
# JSON walker
# ============================================================

def regrade_eval_json(in_path: Path, out_path: Path, grader_fn, judge_model: str,
                      max_samples_per_prompt: int = None,
                      inter_call_sleep: float = 0.0,
                      dry_run: bool = False) -> dict:
    data = json.loads(in_path.read_text())
    prompts = data.get("prompts", [])
    if not prompts:
        raise ValueError(f"no prompts in {in_path}")

    print(f"Eval JSON:    {in_path.name}")
    print(f"Model judged: {data.get('name', '?')}")
    print(f"Judge model:  {judge_model}")
    print(f"Prompts:      {len(prompts)}")
    print()

    total_pass = total_n = total_err = 0
    by_cat = {}  # category → [pass_n, total_n]

    for pi, p in enumerate(prompts, 1):
        # Preserve broken-category exclusion semantics
        cat_status = p.get("category_status")
        if cat_status == "BROKEN_SUBSTRING_INCOMPATIBLE":
            # Mark but don't grade — substring-broken is by-design-excluded from headlines
            p["aggregate_semantic"] = {
                "per_sample": ["skipped"] * len(p.get("samples", [])),
                "pass_n": 0, "fail_n": 0, "error_n": 0, "skipped_n": len(p.get("samples", [])),
                "n": len(p.get("samples", [])),
                "category_status": "BROKEN_SUBSTRING_INCOMPATIBLE",
            }
            continue

        cat = p.get("category", "?")
        samples = p.get("samples", [])
        if max_samples_per_prompt:
            samples = samples[:max_samples_per_prompt]

        verdicts = []
        for si, s in enumerate(samples):
            sample = {
                "prompt_text": p.get("prompt", ""),
                "response_text": s.get("text", ""),
                "gold_substrings": p.get("gold_substrings", []),
                "match_mode": p.get("match_mode", "all"),
                "category": cat,
            }
            if dry_run:
                verdicts.append({"verdict": "pass", "reason": "[dry-run]"})
                continue
            try:
                result = grader_fn(sample, judge_model)
                verdicts.append(result)
            except ValidationError as e:
                verdicts.append({"verdict": "error", "reason": f"validation: {e}"})
            except Exception as e:
                verdicts.append({"verdict": "error", "reason": f"{type(e).__name__}: {e}"})
            if inter_call_sleep > 0 and si < len(samples) - 1:
                time.sleep(inter_call_sleep)

        # Annotate samples in-place with semantic_grade
        for s, v in zip(p.get("samples", []), verdicts):
            s["semantic_grade"] = v

        pass_n = sum(1 for v in verdicts if v["verdict"] == "pass")
        fail_n = sum(1 for v in verdicts if v["verdict"] == "fail")
        err_n  = sum(1 for v in verdicts if v["verdict"] == "error")
        p["aggregate_semantic"] = {
            "per_sample": [v["verdict"] for v in verdicts],
            "pass_n": pass_n,
            "fail_n": fail_n,
            "error_n": err_n,
            "n": len(verdicts),
        }

        # Track running totals (excluding broken categories already skipped above)
        total_pass += pass_n
        total_n += len(verdicts)
        total_err += err_n
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += pass_n
        by_cat[cat][1] += len(verdicts)

        cache_marker = ""  # could surface cached-read but the in-place hook gets noisy
        print(f"  [{pi:2d}/{len(prompts)}] {cat_status or cat:25s} {pass_n}/{len(verdicts)}")

    data["summary_semantic"] = {
        "passed": total_pass,
        "total": total_n,
        "errored": total_err,
        "pass_rate": round(total_pass / max(1, total_n), 4),
        "judge_model": judge_model,
        "grading_method": "semantic_llm_binary",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_category": {c: {"pass": pc, "total": tc, "rate": round(pc / max(1, tc), 4)}
                         for c, (pc, tc) in sorted(by_cat.items())},
    }

    if not dry_run:
        out_path.write_text(json.dumps(data, indent=2))

    return {
        "in":   in_path.name,
        "out":  out_path.name if not dry_run else None,
        "summary_semantic": data["summary_semantic"],
        "summary_substring": data.get("summary", {}),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True,
                        help="Path to eval/results/acc_*.json to regrade")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                        help="LLM provider for the grader (default: openai)")
    parser.add_argument("--judge-model", default=None,
                        help="Model ID. Defaults: gpt-4o-2024-08-06 for openai, claude-sonnet-4-6 for anthropic")
    parser.add_argument("--base-url", default=None,
                        help="Override OpenAI base URL (e.g., https://api.deepseek.com/v1)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: in-place suffix _semantic.json)")
    parser.add_argument("--max-samples-per-prompt", type=int, default=None,
                        help="Cap samples per prompt for cost control (default: all)")
    parser.add_argument("--inter-call-sleep", type=float,
                        default=float(os.environ.get("REGRADE_SLEEP_SEC", "0.0")),
                        help="Sleep between calls in seconds (default 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk the JSON without calling the API; verifies schema only")
    args = parser.parse_args()

    if not args.dry_run:
        if args.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
            print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
        if args.provider == "anthropic" and "ANTHROPIC_API_KEY" not in os.environ:
            print("ERROR: ANTHROPIC_API_KEY not set"); sys.exit(1)

    in_path = Path(args.eval_json).resolve()
    if not in_path.exists():
        print(f"ERROR: {in_path} does not exist"); sys.exit(1)

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = in_path.with_name(in_path.stem + "_semantic.json")

    judge_model = args.judge_model or ("gpt-4o-2024-08-06" if args.provider == "openai" else "claude-sonnet-4-6")

    if args.dry_run:
        grader_fn = lambda sample, m: {"verdict": "pass", "reason": "[dry-run]",
                                       "tokens_in": 0, "tokens_cached_read": 0,
                                       "tokens_cached_write": 0, "tokens_out": 0}
    elif args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        grader_fn = lambda sample, m: grade_anthropic(client, m, sample)
    else:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
        grader_fn = lambda sample, m: grade_openai(client, m, sample)

    result = regrade_eval_json(
        in_path, out_path, grader_fn, judge_model,
        max_samples_per_prompt=args.max_samples_per_prompt,
        inter_call_sleep=args.inter_call_sleep,
        dry_run=args.dry_run,
    )

    print()
    print(f"📄 wrote {out_path}" if not args.dry_run else "📄 dry-run; no file written")
    sub_summary = result["summary_substring"]
    sem_summary = result["summary_semantic"]
    sub_rate = sub_summary.get("pass_rate", "?")
    sem_rate = sem_summary.get("pass_rate", "?")
    sub_passed = sub_summary.get("passed", "?")
    sub_total = sub_summary.get("total", "?")
    print(f"   substring: {sub_passed}/{sub_total} = {sub_rate}")
    print(f"   semantic:  {sem_summary.get('passed')}/{sem_summary.get('total')} = {sem_rate}")
    if isinstance(sub_rate, float) and isinstance(sem_rate, float):
        delta = (sem_rate - sub_rate) * 100
        print(f"   Δ (semantic − substring): {delta:+.2f}pp")


if __name__ == "__main__":
    main()
