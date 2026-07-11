#!/usr/bin/env python3
"""Voice/style metrics for accuracy-eval result JSONs.

Captures stylistic dimensions that the substring grader misses:
length, bullet density, bold/heading density, emoji usage, boilerplate
openers, trailing-question rate. Useful for answering "does this fine-tune
preserve Kyle's voice?" — substring grade tells you capability, voice
metrics tell you fit.

Runs on existing result JSONs — no re-evaluation needed. Works in two
modes:

    # Single file — print metrics for one run
    python3 eval/voice_metrics.py eval/results/acc_*7b-v4*.json

    # Multi-file compare — markdown table across all runs (most useful)
    python3 eval/voice_metrics.py \\
        eval/results/acc_baseline-qwen25-7b-base-v2-rag_*.json \\
        eval/results/acc_candidate-kyle-qwen25-7b-v4-v2-rag_*.json \\
        eval/results/acc_baseline-qwen3-30b-a3b-instruct-2507-v2-rag_*.json \\
        eval/results/acc_candidate-kyle-qwen3-30b-a3b-v4-v2-rag_*.json

By design, ALL result files (including known-failed fine-tunes like MoE v4)
are valid input — failure-mode style data is itself useful for the
white-paper / customer-template story.
"""
import argparse
import json
import re
import sys
from pathlib import Path


# Patterns
_BULLET_LINE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_BOLD_SPAN = re.compile(r"\*\*[^*\n]+\*\*")
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # symbols & pictographs, supplemental, etc.
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "]"
)
# Conservative boilerplate-opener detector — match only the start of the response.
# These are the "AI assistant cadence" tells that fine-tuning typically removes.
_OPENER_PATTERNS = re.compile(
    r"^\s*(?:I'?d be happy to|I'?m here to help|I'?m happy to|Great question"
    r"|Sure[!,. ]|Of course|Absolutely|Let me|I'?ll be glad)",
    re.IGNORECASE,
)


def text_metrics(text: str) -> dict:
    if not text:
        return {
            "chars": 0, "bullets": 0, "bolds": 0, "headings": 0,
            "emojis": 0, "questions": 0, "opener_match": 0,
        }
    return {
        "chars": len(text),
        "bullets": len(_BULLET_LINE.findall(text)),
        "bolds": len(_BOLD_SPAN.findall(text)),
        "headings": len(_HEADING.findall(text)),
        "emojis": len(_EMOJI.findall(text)),
        "questions": text.count("?"),
        "opener_match": 1 if _OPENER_PATTERNS.match(text) else 0,
    }


def aggregate(result: dict) -> dict:
    """Compute mean voice metrics across all (non-error) samples in a result JSON."""
    sums = {"chars": 0, "bullets": 0, "bolds": 0, "headings": 0,
            "emojis": 0, "questions": 0, "opener_match": 0}
    n = 0
    for prompt in result.get("prompts", []):
        for sample in prompt.get("samples", []):
            text = sample.get("text", "") or ""
            if not text:  # skip errors / empties
                continue
            m = text_metrics(text)
            for k in sums:
                sums[k] += m[k]
            n += 1
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "avg_chars": sums["chars"] / n,
        "bullets_per_response": sums["bullets"] / n,
        "bolds_per_response": sums["bolds"] / n,
        "headings_per_response": sums["headings"] / n,
        "emojis_per_response": sums["emojis"] / n,
        "questions_per_response": sums["questions"] / n,
        "boilerplate_opener_rate": sums["opener_match"] / n,
    }


def short_label(name: str) -> str:
    """Trim noisy run-name prefixes for table readability."""
    s = name
    for prefix in ("baseline-", "candidate-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.replace("-v2-rag", "")
    return s


def print_single(path: Path):
    result = json.load(path.open())
    name = result.get("name", path.stem)
    headline = result.get("summary", {}).get("pass_rate")
    m = aggregate(result)
    print(f"# {name}")
    print(f"  source:   {path}")
    if headline is not None:
        print(f"  headline: {headline*100:.1f}%")
    if m["n"] == 0:
        print("  (no scored samples)")
        return
    print(f"  samples:  {m['n']}")
    print(f"  avg_chars/response:        {m['avg_chars']:8.1f}")
    print(f"  bullets/response:          {m['bullets_per_response']:8.2f}")
    print(f"  bolds/response:            {m['bolds_per_response']:8.2f}")
    print(f"  headings/response:         {m['headings_per_response']:8.2f}")
    print(f"  emojis/response:           {m['emojis_per_response']:8.2f}")
    print(f"  questions/response:        {m['questions_per_response']:8.2f}")
    print(f"  boilerplate_opener_rate:   {m['boilerplate_opener_rate']*100:7.1f}%")


def print_compare(paths: list):
    rows = []
    for p in paths:
        result = json.load(p.open())
        name = short_label(result.get("name", p.stem))
        headline = result.get("summary", {}).get("pass_rate")
        m = aggregate(result)
        rows.append((name, headline, m, p))

    # Markdown table
    print("# Voice/style comparison\n")
    print("| Run | Headline | Avg chars | Bullets | Bolds | Headings | Emojis | Q/resp | Opener-boilerplate |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, headline, m, p in rows:
        if m["n"] == 0:
            print(f"| {name} | — | (no samples) | | | | | | |")
            continue
        h = f"{headline*100:.1f}%" if headline is not None else "—"
        print(
            f"| {name} | {h} | "
            f"{m['avg_chars']:.0f} | "
            f"{m['bullets_per_response']:.2f} | "
            f"{m['bolds_per_response']:.2f} | "
            f"{m['headings_per_response']:.2f} | "
            f"{m['emojis_per_response']:.2f} | "
            f"{m['questions_per_response']:.2f} | "
            f"{m['boilerplate_opener_rate']*100:.0f}% |"
        )
    print()
    print("**Reading the table:**")
    print("- `Avg chars` — Kyle voice tends concise. Stock Qwen-instruct cadence is ~3-10× longer.")
    print("- `Bullets / Bolds / Headings` — formatting density. Stock instruct models bullet-bold-heading aggressively; Kyle voice does not.")
    print("- `Emojis` — Kyle voice has none in technical answers; stock instruct bases sprinkle them.")
    print("- `Questions/resp` — trailing 'What else can I help with?' deflectors. Kyle voice answers, doesn't deflect.")
    print("- `Opener-boilerplate` — fraction of responses starting with 'I'm here to help', 'Sure!', 'Great question', etc. Kyle voice ≈ 0%.")
    print()
    print("A fine-tune that **preserves Kyle voice** moves all of these toward zero (or toward Kyle's natural numbers) vs the same model's unmodified base.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="One or more accuracy-eval result JSONs")
    args = ap.parse_args()

    for p in args.paths:
        if not p.exists():
            print(f"❌ not found: {p}", file=sys.stderr)
            sys.exit(2)

    if len(args.paths) == 1:
        print_single(args.paths[0])
    else:
        print_compare(args.paths)


if __name__ == "__main__":
    main()
