#!/usr/bin/env python3
"""Run standard LLM benchmarks (TruthfulQA, MMLU, HellaSwag) against Skippy's /generate.

These are base-model / world-knowledge probes — they measure whether Qwen3-30B-A3B
reasons correctly, NOT whether Skippy's RAG layer is working. RAG is disabled by
default; pass --with-rag to include it.

Usage:
    SKIPPY_USER=... SKIPPY_PASSWORD=... \\
        python3 eval/run_standard_benchmarks.py --name qwen3-moe-2026-04-21

    # Custom sample size and benchmarks:
    python3 eval/run_standard_benchmarks.py --name probe-run \\
        --benchmarks truthfulqa,mmlu --n 30

    # Full run (all 817 TruthfulQA + all MMLU subjects):
    python3 eval/run_standard_benchmarks.py --name full --full

Outputs:
    eval/results/standard_<name>_<timestamp>.{json,md}

Scoring:
    - TruthfulQA (generation): pass if any entry in `correct_answers` appears in
      the response AND no `incorrect_answers` entry appears as a distinct substring.
    - MMLU (multiple-choice): response must start with or clearly contain the
      gold letter (A/B/C/D).
    - HellaSwag (multi-choice ending): same letter-match scoring as MMLU.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


# Curated MMLU subject mix — diverse, basic-world-knowledge oriented.
# Kyle's "is the sky blue" test maps closest to global_facts + elementary science.
DEFAULT_MMLU_SUBJECTS = [
    "global_facts",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_world_history",
    "professional_psychology",
    "miscellaneous",
]


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


def call_generate(endpoint: str, headers: dict, prompt: str, *, use_rag: bool,
                  max_tokens: int, temperature: float) -> dict:
    t0 = time.time()
    resp = requests.post(
        f"{endpoint}/generate",
        json={
            "prompt": prompt,
            "use_rag": use_rag,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers=headers,
        timeout=300,
    )
    elapsed = time.time() - t0
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:300]}", "elapsed_s": round(elapsed, 2)}
    data = resp.json()
    text = data.get("text", "")
    tokens = data.get("tokens_used", 0)
    return {
        "text": text,
        "tokens": tokens,
        "elapsed_s": round(elapsed, 2),
        "tok_per_s": round(tokens / elapsed, 1) if elapsed > 0 else None,
    }


# ---- TruthfulQA -------------------------------------------------------------

def load_truthfulqa(n: int, seed: int) -> list:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    items = list(ds)
    if n > 0 and n < len(items):
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items


def score_truthfulqa(response: str, correct: list, incorrect: list) -> str:
    if not response.strip():
        return "fail (empty)"
    rl = response.lower()
    correct_hit = any(c.lower() in rl for c in correct if len(c) >= 4)
    incorrect_hit = any(
        i.lower() in rl
        for i in incorrect
        if len(i) >= 8 and i.lower() not in [c.lower() for c in correct]
    )
    if correct_hit and not incorrect_hit:
        return "pass"
    if incorrect_hit and not correct_hit:
        return "fail (incorrect answer present)"
    if correct_hit and incorrect_hit:
        return "fail (mixed)"
    return "manual"


def run_truthfulqa(items: list, endpoint: str, headers: dict, use_rag: bool) -> list:
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        prompt = f"Answer this question truthfully in one sentence: {q}"
        print(f"  [TQA {i+1}/{len(items)}] {q[:70]}...")
        r = call_generate(endpoint, headers, prompt, use_rag=use_rag,
                          max_tokens=120, temperature=0.3)
        if "error" in r:
            s = "error"
        else:
            s = score_truthfulqa(
                r["text"],
                item.get("correct_answers") or [item.get("best_answer", "")],
                item.get("incorrect_answers") or [],
            )
        flag = {"pass": "✅", "manual": "🖐️", "error": "💥"}.get(s, "❌")
        print(f"    {flag} {s}")
        results.append({
            "id": f"truthfulqa_{i}",
            "benchmark": "truthfulqa",
            "category": item.get("category", ""),
            "prompt": prompt,
            "question": q,
            "best_answer": item.get("best_answer"),
            "correct_answers": item.get("correct_answers"),
            "incorrect_answers": item.get("incorrect_answers"),
            "response": r,
            "score": s,
        })
    return results


# ---- MMLU -------------------------------------------------------------------

MMLU_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(subjects: list, n_per_subject: int, seed: int) -> list:
    from datasets import load_dataset
    items = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception as e:
            print(f"  ⚠️  skipping MMLU subject {subj}: {e}")
            continue
        rows = list(ds)
        if n_per_subject > 0 and n_per_subject < len(rows):
            random.Random(seed + hash(subj) % 10000).shuffle(rows)
            rows = rows[:n_per_subject]
        for r in rows:
            r["_subject"] = subj
            items.append(r)
    return items


def score_mmlu(response: str, gold_index: int) -> str:
    if not response.strip():
        return "fail (empty)"
    # Look for a standalone letter at the start or after "answer" keywords.
    match = re.search(
        r"(?:answer(?:\s+is)?[:\s]*|^|\n|\s)([ABCD])\b",
        response.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return "manual"
    predicted = match.group(1).upper()
    gold = MMLU_LETTERS[gold_index]
    return "pass" if predicted == gold else f"fail (said {predicted}, gold {gold})"


def run_mmlu(items: list, endpoint: str, headers: dict, use_rag: bool) -> list:
    results = []
    for i, item in enumerate(items):
        q = item["question"]
        choices = item["choices"]
        choice_block = "\n".join(f"{MMLU_LETTERS[j]}) {c}" for j, c in enumerate(choices))
        prompt = (
            f"{q}\n\n{choice_block}\n\n"
            "Answer with just the letter (A, B, C, or D)."
        )
        subj = item["_subject"]
        print(f"  [MMLU {i+1}/{len(items)}] [{subj}] {q[:60]}...")
        r = call_generate(endpoint, headers, prompt, use_rag=use_rag,
                          max_tokens=16, temperature=0.1)
        if "error" in r:
            s = "error"
        else:
            s = score_mmlu(r["text"], item["answer"])
        flag = {"pass": "✅", "manual": "🖐️", "error": "💥"}.get(s, "❌")
        print(f"    {flag} {s}")
        results.append({
            "id": f"mmlu_{subj}_{i}",
            "benchmark": "mmlu",
            "category": subj,
            "prompt": prompt,
            "question": q,
            "choices": choices,
            "gold_letter": MMLU_LETTERS[item["answer"]],
            "response": r,
            "score": s,
        })
    return results


# ---- HellaSwag --------------------------------------------------------------

def load_hellaswag(n: int, seed: int) -> list:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    items = list(ds)
    if n > 0 and n < len(items):
        random.Random(seed).shuffle(items)
        items = items[:n]
    return items


def run_hellaswag(items: list, endpoint: str, headers: dict, use_rag: bool) -> list:
    results = []
    for i, item in enumerate(items):
        ctx = item["ctx"]
        endings = item["endings"]
        ending_block = "\n".join(
            f"{MMLU_LETTERS[j]}) {e}" for j, e in enumerate(endings)
        )
        prompt = (
            f"Which ending most naturally continues this context?\n\n"
            f"Context: {ctx}\n\n{ending_block}\n\n"
            "Answer with just the letter (A, B, C, or D)."
        )
        print(f"  [HSwag {i+1}/{len(items)}] {ctx[:60]}...")
        r = call_generate(endpoint, headers, prompt, use_rag=use_rag,
                          max_tokens=16, temperature=0.1)
        if "error" in r:
            s = "error"
        else:
            s = score_mmlu(r["text"], int(item["label"]))
        flag = {"pass": "✅", "manual": "🖐️", "error": "💥"}.get(s, "❌")
        print(f"    {flag} {s}")
        results.append({
            "id": f"hellaswag_{i}",
            "benchmark": "hellaswag",
            "category": item.get("activity_label", ""),
            "prompt": prompt,
            "context": ctx,
            "endings": endings,
            "gold_letter": MMLU_LETTERS[int(item["label"])],
            "response": r,
            "score": s,
        })
    return results


# ---- Reporting --------------------------------------------------------------

def render_markdown(out: dict) -> str:
    lines = [
        f"# Standard benchmarks: {out['name']}",
        "",
        f"- Timestamp: {out['timestamp']}",
        f"- Endpoint: {out['endpoint']}",
        f"- RAG enabled: {out['rag_enabled']}",
        f"- Benchmarks: {', '.join(out['benchmarks'])}",
        "",
        "## Summary",
        "",
    ]
    for bench in out["benchmarks"]:
        bench_items = [p for p in out["items"] if p["benchmark"] == bench]
        passes = sum(1 for p in bench_items if p["score"] == "pass")
        fails = sum(1 for p in bench_items if p["score"].startswith("fail"))
        manual = sum(1 for p in bench_items if p["score"] == "manual")
        errors = sum(1 for p in bench_items if p["score"] == "error")
        total = len(bench_items) or 1
        lines.append(
            f"- **{bench}** ({total}): ✅ {passes} · ❌ {fails} · 🖐️ {manual} · 💥 {errors} "
            f"— accuracy {passes/total*100:.1f}% (pass / total)"
        )
    lines += [
        f"- Avg decode: {out.get('avg_tok_per_s', '?')} tok/s",
        "",
        "## Details",
        "",
    ]
    for p in out["items"]:
        lines.append(f"### {p['id']} — {p['score']}")
        lines.append("")
        if p["benchmark"] == "truthfulqa":
            lines.append(f"**Question:** {p['question']}")
            lines.append(f"**Best answer:** {p.get('best_answer', '')}")
        elif p["benchmark"] == "mmlu":
            lines.append(f"**Subject:** {p['category']}")
            lines.append(f"**Question:** {p['question']}")
            lines.append(f"**Gold:** {p['gold_letter']}")
        elif p["benchmark"] == "hellaswag":
            lines.append(f"**Context:** {p['context']}")
            lines.append(f"**Gold:** {p['gold_letter']}")
        r = p["response"]
        if "error" in r:
            lines.append(f"**Error:** {r['error']}")
        else:
            lines.append(f"**Response** ({r.get('tokens', 0)} tokens, {r.get('elapsed_s')}s):")
            lines.append("")
            lines.append("```")
            lines.append(r.get("text", "").strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Label for this run")
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--benchmarks", default="truthfulqa,mmlu,hellaswag",
                        help="Comma-separated subset of truthfulqa,mmlu,hellaswag")
    parser.add_argument("--n", type=int, default=20,
                        help="Samples per benchmark (MMLU: per subject). 0 = all.")
    parser.add_argument("--full", action="store_true",
                        help="Run complete benchmarks (overrides --n to 0)")
    parser.add_argument("--with-rag", action="store_true",
                        help="Enable RAG (default off — these probe the base model)")
    parser.add_argument("--mmlu-subjects", default=",".join(DEFAULT_MMLU_SUBJECTS),
                        help="Comma-separated MMLU subject list")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for b in benchmarks:
        if b not in ("truthfulqa", "mmlu", "hellaswag"):
            print(f"❌ unknown benchmark: {b}")
            sys.exit(2)

    n = 0 if args.full else args.n

    headers = login(args.endpoint)

    all_items = []

    if "truthfulqa" in benchmarks:
        print("▶ Loading TruthfulQA...")
        tqa = load_truthfulqa(n, args.seed)
        print(f"  {len(tqa)} items")
        all_items += run_truthfulqa(tqa, args.endpoint, headers, args.with_rag)

    if "mmlu" in benchmarks:
        subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
        print(f"▶ Loading MMLU ({len(subjects)} subjects)...")
        # Split the per-benchmark budget across subjects.
        per_subj = max(1, n // len(subjects)) if n > 0 else 0
        mmlu = load_mmlu(subjects, per_subj, args.seed)
        print(f"  {len(mmlu)} items")
        all_items += run_mmlu(mmlu, args.endpoint, headers, args.with_rag)

    if "hellaswag" in benchmarks:
        print("▶ Loading HellaSwag...")
        hs = load_hellaswag(n, args.seed)
        print(f"  {len(hs)} items")
        all_items += run_hellaswag(hs, args.endpoint, headers, args.with_rag)

    tok_rates = [p["response"]["tok_per_s"] for p in all_items
                 if p["response"].get("tok_per_s")]
    out = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "rag_enabled": args.with_rag,
        "benchmarks": benchmarks,
        "n_per_benchmark": n,
        "avg_tok_per_s": round(sum(tok_rates) / len(tok_rates), 1) if tok_rates else None,
        "items": all_items,
    }

    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"standard_{args.name}_{stamp}.json"
    md_path = results_dir / f"standard_{args.name}_{stamp}.md"
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(render_markdown(out))
    print(f"\n📄 {json_path}")
    print(f"📄 {md_path}")


if __name__ == "__main__":
    main()
