#!/usr/bin/env python3
"""Run the Skippy eval set against a /generate endpoint.

Usage:
    SKIPPY_USER=... SKIPPY_PASSWORD=... \\
        python3 eval/run_eval.py --name qwen3-moe-v1 [--endpoint http://localhost:8080]

Outputs:
    eval/results/<name>_<timestamp>.json   Raw per-prompt responses and metadata
    eval/results/<name>_<timestamp>.md     Human-readable report with auto-pass flags

Auto-pass = every substring in `gold_substrings` appears (case-insensitive) in
the response. Prompts with an empty gold list are always shown as "manual".
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import requests


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


def run_prompt(endpoint: str, headers: dict, prompt: str, use_rag: bool) -> dict:
    t0 = time.time()
    resp = requests.post(
        f"{endpoint}/generate",
        json={"prompt": prompt, "use_rag": use_rag, "max_tokens": 512, "temperature": 0.3},
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
        "citations": data.get("citations"),
    }


def score(response_text: str, gold_substrings: list) -> str:
    if not gold_substrings:
        return "manual"
    lower = response_text.lower()
    missing = [g for g in gold_substrings if g.lower() not in lower]
    if not missing:
        return "pass"
    return f"fail (missing: {', '.join(missing)})"


def render_markdown(results: dict) -> str:
    lines = [
        f"# Eval run: {results['name']}",
        "",
        f"- Timestamp: {results['timestamp']}",
        f"- Endpoint: {results['endpoint']}",
        f"- Prompt set: {results['prompts_version']} ({len(results['prompts'])} prompts)",
        "",
        "## Summary",
        "",
    ]
    pass_count = sum(1 for p in results["prompts"] if p["score"] == "pass")
    fail_count = sum(1 for p in results["prompts"] if p["score"].startswith("fail"))
    manual_count = sum(1 for p in results["prompts"] if p["score"] == "manual")
    lines += [
        f"- ✅ pass: {pass_count}",
        f"- ❌ fail: {fail_count}",
        f"- 🖐️ manual: {manual_count}",
        f"- Avg decode: {results.get('avg_tok_per_s', '?')} tok/s",
        "",
        "## Details",
        "",
    ]
    for p in results["prompts"]:
        lines.append(f"### {p['id']} [{p['category']}] — {p['score']}")
        lines.append("")
        lines.append(f"**Prompt:** {p['prompt']}")
        if p.get("gold_substrings"):
            lines.append(f"**Gold:** {p['gold_substrings']}")
        r = p.get("response", {})
        if "error" in r:
            lines.append(f"**Error:** {r['error']}")
        else:
            lines.append(f"**Response** ({r.get('tokens', 0)} tokens, {r.get('elapsed_s')}s, {r.get('tok_per_s')} tok/s):")
            lines.append("")
            lines.append("```")
            lines.append(r.get("text", "").strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Label for this run (shows up in filenames)")
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--prompts", default="eval/prompts.json")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG (useful for base-model comparisons)")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompt_set = json.load(f)

    headers = login(args.endpoint)

    out = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "rag_enabled": not args.no_rag,
        "prompts_version": prompt_set.get("version"),
        "prompts": [],
    }

    tok_rates = []
    for p in prompt_set["prompts"]:
        print(f"▶ {p['id']} [{p['category']}]")
        r = run_prompt(args.endpoint, headers, p["prompt"], use_rag=not args.no_rag)
        s = "error" if "error" in r else score(r["text"], p.get("gold_substrings", []))
        if r.get("tok_per_s"):
            tok_rates.append(r["tok_per_s"])
        out["prompts"].append({**p, "response": r, "score": s})
        flag = {"pass": "✅", "manual": "🖐️", "error": "💥"}.get(s, "❌")
        print(f"  {flag} {s}")

    if tok_rates:
        out["avg_tok_per_s"] = round(sum(tok_rates) / len(tok_rates), 1)

    results_dir = Path(args.prompts).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"{args.name}_{stamp}.json"
    md_path = results_dir / f"{args.name}_{stamp}.md"
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(render_markdown(out))
    print(f"\n📄 {json_path}")
    print(f"📄 {md_path}")


if __name__ == "__main__":
    main()
