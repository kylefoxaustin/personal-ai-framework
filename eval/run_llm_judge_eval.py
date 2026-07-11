#!/usr/bin/env python3
"""
SK-P1-002 — LLM-as-judge tertiary capability gate.

Reads an existing eval JSON (eval/results/acc_*.json), picks a held-out
50-sample subset, asks Claude Sonnet 4.6 to score each response against a
faithfulness + correctness + instruction-following + conciseness rubric,
and writes the aggregated judge scores back to a sibling
`judge_<model>_<timestamp>.json`.

Why: substring grading is gameable (the team admits this). Voice + safety
gates catch their own failure modes but not "wrong-but-confidently-stated
answer that hits the gold tokens." LLM-judge is the missing third leg —
the acceptance check is "does Claude-judge agree that v4 > v3 > v1?" If
yes, substring grading is validated for this corpus. If no, the production
ship decision needs revisiting.

Cost: ~50 prompts × ~$0.008/call ≈ $0.40 per model with prompt caching
(rubric + system prompt cached across the run; ~1500 token cached prefix).
For 5 anchored models: ~$2 total.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python3 eval/run_llm_judge_eval.py \\
        --eval-json eval/results/acc_candidate-kyle-qwen25-7b-v4-v2-rag_20260502-175416.json \\
        --max-prompts 50 \\
        --judge-model claude-sonnet-4-6
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import anthropic
from pydantic import BaseModel, Field, ValidationError


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


# ============================================================
# Rubric schema (Pydantic — Claude is asked to fill this in)
# ============================================================

class JudgeScore(BaseModel):
    """One response's score across four rubric dimensions, each 0-2."""
    correctness: int = Field(
        ge=0, le=2,
        description="Factual accuracy of the response. 0 = wrong, 1 = partially correct or evasive, 2 = correct."
    )
    instruction_following: int = Field(
        ge=0, le=2,
        description="Did the response answer the actual question asked? 0 = ignored or off-topic, 1 = partial, 2 = fully addressed."
    )
    faithfulness: int = Field(
        ge=0, le=2,
        description="If RAG context was provided in the response (citations), did the response stay grounded in it? 0 = fabricated beyond context, 1 = mostly grounded with minor extrapolation, 2 = fully grounded. If no RAG context was used, score 2 (n/a, default-pass)."
    )
    conciseness: int = Field(
        ge=0, le=2,
        description="Appropriate length and no rambling. 0 = severely over- or under-length for the question, 1 = somewhat off, 2 = well-calibrated."
    )
    notes: str = Field(
        max_length=300,
        description="One short sentence justifying the lowest-scoring dimension, OR if all are 2, summarising why the response is good."
    )


# ============================================================
# System prompt + rubric (cached across the run)
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator scoring AI assistant responses against a four-dimensional rubric. Your task is to be a fair, calibrated judge — not too harsh, not too lenient.

You will be given:
1. A user prompt (asking about embedded systems hardware, NXP datasheets, code, or general questions)
2. The AI assistant's response to that prompt
3. Optional context: gold-substring expectations and / or RAG citation metadata

Score the response on four dimensions, each 0-2:

**1. Correctness (0-2)**
- 0: The response contains factually wrong claims or invents information that doesn't exist
- 1: The response is partially correct, evasive, or refuses when an answer was clearly available
- 2: The response is factually correct (or correctly refuses when the question is unanswerable / fictional)

**2. Instruction Following (0-2)**
- 0: The response ignores the question or is off-topic
- 1: The response addresses part of the question but misses important elements
- 2: The response fully addresses what was asked

**3. Faithfulness to Context (0-2)**
- This applies only when the response cited RAG sources (numbered citations like [1], [2]).
- 0: The response fabricates claims beyond what the cited context could support
- 1: Mostly grounded with minor extrapolation
- 2: All factual claims trace back to citations, OR the response had no retrieved context (n/a — default pass)

**4. Conciseness (0-2)**
- 0: Severely over-long (rambling, repetitive) or severely under-length (one-word answer to a substantive question)
- 1: Somewhat off — too verbose for the question, or terse to the point of unhelpfulness
- 2: Well-calibrated length for the complexity of the question

**Important calibration notes:**
- "Skippy" is the AI assistant's persona name. Responses identifying as Skippy are normal, not fabrication.
- Substring graders mark a response "pass" if specific gold strings appear. A response can score "pass" on the substring grader and STILL score low on this rubric (verbose-but-incidentally-correct), or vice versa (terse-but-wrong-on-substring-but-actually-correct). Use your own judgment, not the substring verdict.
- Adversarial probes about fictional products (e.g., "QuantumFlow Engine," "Flux-9000") are correctly handled by REFUSING. A confident-sounding answer to such a question scores 0 on correctness.
- If the response is well-calibrated to the question's complexity (short question → short answer; complex question → detailed answer), score conciseness 2 even if absolute length is unusual.

Score quickly and decisively. Output the scores via the structured JudgeScore schema."""


# ============================================================
# Eval JSON ingestion
# ============================================================

def load_eval_data(path: Path):
    data = json.loads(path.read_text())
    model_name = data.get("name", path.stem)
    return model_name, data


def select_held_out_samples(data: dict, max_prompts: int = 50, seed: int = 42):
    """Pick a stable subset of (prompt, sample) pairs across categories.

    Stratified-ish: includes one sample per prompt up to max_prompts, prioritising
    diverse categories. Skips the BROKEN_SUBSTRING_INCOMPATIBLE persona category.
    """
    import random
    rng = random.Random(seed)
    prompts = data.get("prompts", [])
    eligible = [p for p in prompts
                if p.get("category_status") != "BROKEN_SUBSTRING_INCOMPATIBLE"
                and p.get("samples")]

    # One sample per prompt to maximise breadth; bias toward earlier samples
    # (most representative — temperature 0 typically produces near-identical
    # samples, so the first is fine).
    pairs = []
    for p in eligible:
        s = p["samples"][0]
        pairs.append({
            "prompt_id": p["id"],
            "category": p.get("category", "unknown"),
            "prompt_text": p["prompt"],
            "response_text": s.get("text", ""),
            "gold_substrings": p.get("gold_substrings", []),
            "match_mode": p.get("match_mode", "all"),
            "has_citations": bool(s.get("citations")),
            "n_citations": len(s.get("citations", [])),
        })

    rng.shuffle(pairs)
    return pairs[:max_prompts]


# ============================================================
# Judge call (cached system prompt + Pydantic-validated output)
# ============================================================

def judge_one(client: anthropic.Anthropic, judge_model: str, sample: dict) -> dict:
    """Score one (prompt, response) pair via the Sonnet 4.6 judge."""
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

    response = client.messages.parse(
        model=judge_model,
        max_tokens=512,
        system=[{
            "type": "text",
            "text": JUDGE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user_block}],
        output_format=JudgeScore,
    )
    parsed: JudgeScore = response.parsed_output

    return {
        "prompt_id": sample["prompt_id"],
        "category": sample["category"],
        "correctness": parsed.correctness,
        "instruction_following": parsed.instruction_following,
        "faithfulness": parsed.faithfulness,
        "conciseness": parsed.conciseness,
        "total": parsed.correctness + parsed.instruction_following + parsed.faithfulness + parsed.conciseness,
        "notes": parsed.notes,
        "tokens_in": response.usage.input_tokens,
        "tokens_in_cached_read": response.usage.cache_read_input_tokens,
        "tokens_in_cached_write": response.usage.cache_creation_input_tokens,
        "tokens_out": response.usage.output_tokens,
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True,
                        help="Path to eval/results/acc_*.json to judge")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6",
                        help="Anthropic model ID for the judge (default: claude-sonnet-4-6)")
    parser.add_argument("--max-prompts", type=int, default=50,
                        help="Max prompts to judge (default 50, deterministic seed)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: eval/results/judge_<model>_<ts>.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select samples + print summary without calling the API")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set. Export your key, or pass --dry-run.")
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
    print(f"Samples:      {len(samples)}")
    print(f"Categories:   {sorted(set(s['category'] for s in samples))}")
    print()

    if args.dry_run:
        print("DRY RUN — not calling the judge.")
        return

    client = anthropic.Anthropic()
    judged = []
    inter_call_sleep = float(os.environ.get("JUDGE_SLEEP_SEC", "6.5"))  # rate limit pacing
    for i, sample in enumerate(samples, 1):
        try:
            result = judge_one(client, args.judge_model, sample)
            judged.append(result)
            cache_marker = "🟢" if result["tokens_in_cached_read"] > 0 else "🟠" if result["tokens_in_cached_write"] > 0 else "  "
            print(f"  [{i:2d}/{len(samples)}] {cache_marker} {result['prompt_id']:40s} "
                  f"c={result['correctness']} i={result['instruction_following']} "
                  f"f={result['faithfulness']} sz={result['conciseness']} = {result['total']}/8")
        except anthropic.APIStatusError as e:
            print(f"  [{i:2d}/{len(samples)}] ❌ {sample['prompt_id']}: {e.status_code} {e.message}")
            judged.append({"prompt_id": sample["prompt_id"], "error": f"APIStatusError {e.status_code}: {e.message}"})
        except ValidationError as e:
            print(f"  [{i:2d}/{len(samples)}] ⚠️  {sample['prompt_id']}: judge returned non-conforming output ({e})")
            judged.append({"prompt_id": sample["prompt_id"], "error": f"ValidationError: {e}"})
        except Exception as e:
            print(f"  [{i:2d}/{len(samples)}] ❌ {sample['prompt_id']}: {type(e).__name__}: {e}")
            judged.append({"prompt_id": sample["prompt_id"], "error": f"{type(e).__name__}: {e}"})
        # Pace inter-call to stay under Tier 1 30K input TPM (~9 calls/min ceiling)
        if i < len(samples):
            time.sleep(inter_call_sleep)

    summary = aggregate([j for j in judged if "error" not in j])

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"judge_{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    payload = {
        "evaluated_eval_json": eval_path.name,
        "model_judged": model_name,
        "judge_model": args.judge_model,
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
    cached_pct = (
        100 * summary.get("tokens_in_cached_read_total", 0)
        / max(1, summary.get("tokens_in_cached_read_total", 0)
              + summary.get("tokens_in_cached_write_total", 0)
              + summary.get("tokens_in_total", 0))
    )
    print(f"   cache hit rate: {cached_pct:.0f}% of input tokens served from cache")


if __name__ == "__main__":
    main()
