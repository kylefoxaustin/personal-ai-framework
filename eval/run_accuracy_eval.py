#!/usr/bin/env python3
"""Capture Skippy's outputs on a fixed prompt set — for later variant comparison.

Motivating use case: build a reference "fp16-compute Q4_K_M" output set today;
re-run against a future INT8 (W8A8) re-quantization; diff the two with
`eval/compare_accuracy_runs.py` to answer "does INT8 match bf16 for Skippy's
workload?"

Differs from run_eval.py in two ways tailored to the quantization-comparison
question:
  1. `--samples N` — fires each prompt N times so intra-variant variance is
     captured alongside inter-variant differences.
  2. `--skip-agent-loop` (default ON) and `--no-rag` (default ON) — removes
     external noise sources (tool-detection LLM calls, RAG retrieval
     variability) so the diff reflects only the LLM's own quantization damage.

Outputs:
    eval/results/acc_<name>_<timestamp>.json

Usage:
    SKIPPY_USER=... SKIPPY_PASSWORD=... \\
        python3 eval/run_accuracy_eval.py --name reference-q4km --samples 3

    # With RAG on (comparing end-to-end behaviour including retrieval):
    python3 eval/run_accuracy_eval.py --name reference-full --samples 3 --with-rag

    # Category filter (useful for iterating on one slice):
    python3 eval/run_accuracy_eval.py --name dbg --categories rag_datasheet,coding
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


# Prompts excluded from the headline denominator (SK-P0-001, 2026-05-08).
# Kept in sync with eval/regrade_for_broken_categories.py and
# eval/regrade_semantic.py — all three match by ID, not by a flag, because
# historical result JSONs embed their own prompt copies.
BROKEN_PROMPT_IDS = {"persona_skippy_voice", "persona_brief_role"}


def login(endpoint: str) -> dict:
    user = os.environ.get("SKIPPY_USER")
    password = os.environ.get("SKIPPY_PASSWORD")
    if not user or not password:
        print("❌ SKIPPY_USER and SKIPPY_PASSWORD must be set.")
        sys.exit(2)
    resp = requests.post(
        f"{endpoint}/auth/login",
        json={"username": user, "password": password},
        timeout=15,
    )
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def run_prompt(endpoint: str, headers: dict, prompt: str, *,
               use_rag: bool, skip_agent_loop: bool,
               max_tokens: int, temperature: float) -> dict:
    t0 = time.time()
    try:
        resp = requests.post(
            f"{endpoint}/generate",
            json={
                "prompt": prompt,
                "use_rag": use_rag,
                "skip_agent_loop": skip_agent_loop,
                "include_telemetry": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers=headers,
            timeout=600,
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"request failed: {e}", "elapsed_s": round(time.time() - t0, 2)}
    elapsed = time.time() - t0
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:300]}", "elapsed_s": round(elapsed, 2)}
    data = resp.json()
    return {
        "text": data.get("text", ""),
        "tokens": data.get("tokens_used", 0),
        "elapsed_s": round(elapsed, 2),
        "telemetry": data.get("telemetry"),
        "citations": data.get("citations"),
    }


def score(response_text: str, gold_substrings: list, match_mode: str = "all") -> str:
    """Score a response against gold substrings.

    match_mode="all" (default): every substring must appear (case-insensitive).
    match_mode="any":           at least one substring must appear. Used for
                                refusal detection — the model should refuse,
                                and there are many valid ways to phrase "I don't know".
    """
    if not gold_substrings:
        return "manual"
    lower = response_text.lower()
    if match_mode == "any":
        for g in gold_substrings:
            if g.lower() in lower:
                return "pass"
        return f"fail (no refusal phrase matched; expected any of: {', '.join(gold_substrings[:5])}...)"
    # default "all"
    missing = [g for g in gold_substrings if g.lower() not in lower]
    if not missing:
        return "pass"
    return f"fail (missing: {', '.join(missing)})"


def aggregate_pass_rate(samples: list, gold: list, match_mode: str = "all") -> dict:
    """Count how many of N samples passed — captures intra-run variance
    (e.g., 3/3 = deterministic pass, 2/3 = occasional fail, 0/3 = consistent fail)."""
    statuses = [score(s["text"], gold, match_mode) if "text" in s else "error"
                for s in samples]
    return {
        "per_sample": statuses,
        "pass_n": sum(1 for s in statuses if s == "pass"),
        "fail_n": sum(1 for s in statuses if s.startswith("fail")),
        "error_n": sum(1 for s in statuses if s == "error"),
        "manual_n": sum(1 for s in statuses if s == "manual"),
        "n": len(statuses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--prompts", default="eval/prompts.json")
    parser.add_argument("--samples", type=int, default=3,
                        help="Re-run each prompt N times (captures intra-variant variance)")
    parser.add_argument("--with-rag", action="store_true",
                        help="Enable RAG (default OFF — isolate LLM-only variance)")
    parser.add_argument("--with-agent-loop", action="store_true",
                        help="Enable tool-detection agent loop (default OFF — isolate LLM-only variance)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = deterministic greedy; slight values surface quantization noise")
    parser.add_argument("--categories", default="",
                        help="Comma-separated category filter")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompt_set = json.load(f)

    cat_filter = set(c.strip() for c in args.categories.split(",") if c.strip())
    prompts_to_run = [p for p in prompt_set["prompts"]
                      if not cat_filter or p.get("category") in cat_filter]
    if not prompts_to_run:
        print(f"❌ no prompts match categories={cat_filter}")
        sys.exit(2)

    headers = login(args.endpoint)

    out = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "prompts_version": prompt_set.get("version"),
        "samples_per_prompt": args.samples,
        "config": {
            "use_rag": args.with_rag,
            "skip_agent_loop": not args.with_agent_loop,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        },
        "prompts": [],
        "summary": {},
    }

    cum_pass, cum_total, excluded = 0, 0, 0
    for p in prompts_to_run:
        print(f"▶ {p['id']} [{p['category']}]")
        samples = []
        for s_idx in range(args.samples):
            r = run_prompt(args.endpoint, headers, p["prompt"],
                           use_rag=args.with_rag,
                           skip_agent_loop=not args.with_agent_loop,
                           max_tokens=args.max_tokens,
                           temperature=args.temperature)
            r["sample_index"] = s_idx
            samples.append(r)
            s_status = (score(r["text"], p.get("gold_substrings", []),
                             p.get("match_mode", "all"))
                        if "text" in r else "error")
            flag = {"pass": "✅", "manual": "🖐️", "error": "💥"}.get(s_status, "❌")
            print(f"  [{s_idx+1}/{args.samples}] {flag} {s_status}")

        agg = aggregate_pass_rate(samples, p.get("gold_substrings", []),
                                  p.get("match_mode", "all"))
        entry = {**p, "samples": samples, "aggregate": agg}

        # Persona is quarantined from the headline (SK-P0-001): both prompts have
        # empty gold_substrings and the system prompt injects "Skippy" into every
        # model, so the category scored 0/6 for stock and fine-tuned models alike.
        # We still RUN and RECORD it — eval/voice_metrics.py grades voice transfer
        # from these samples — we just don't let it into the denominator.
        #
        # This exclusion used to live only in the post-hoc
        # regrade_for_broken_categories.py. That fixed the data but not the
        # producer, so every run after 2026-05-08 silently re-emitted the
        # pre-quarantine denominator (132 instead of 126) and 41 of 77 result
        # JSONs drifted back. Fixed at the source 2026-07-09.
        if p["id"] in BROKEN_PROMPT_IDS:
            entry["category_status"] = "BROKEN_SUBSTRING_INCOMPATIBLE"
            excluded += agg["n"]
        else:
            cum_pass += agg["pass_n"]
            cum_total += agg["n"]

        out["prompts"].append(entry)

    out["summary"] = {
        "passed": cum_pass,
        "total": cum_total,
        "pass_rate": round(cum_pass / cum_total, 4) if cum_total else 0.0,
        "excluded_samples": excluded,
        "excluded_reason": "BROKEN_SUBSTRING_INCOMPATIBLE prompts (persona category)",
        # `total_samples` is retained ONLY because older tooling reads it. It is
        # the pre-exclusion count. Never divide `passed` by it.
        "total_samples": cum_total + excluded,
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
