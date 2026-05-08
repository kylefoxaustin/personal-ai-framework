#!/usr/bin/env python3
"""
Post-hoc re-grader: walk eval/results/acc_*.json and recompute summary fields
to exclude prompts marked with category_status="BROKEN_SUBSTRING_INCOMPATIBLE".

Why this exists: SK-P0-001 from REMEDIATION_PLAN.md flagged the persona category
as broken-by-design (substring grader can't capture persona; system prompt
injects identity into every model). We added category_status flags to the
persona prompts in prompts_v2.json, but every existing eval JSON has stale
summary numbers that include the persona contribution. This script walks
those JSONs, identifies prompts that match the broken set, and rewrites
summary.passed / summary.total / summary.pass_rate to exclude them.

The original `samples` and per-prompt `aggregate` data are NOT touched —
only the top-level `summary`. A new field `summary_v1_legacy` preserves
the original numbers for audit.

Usage:
    python3 eval/regrade_for_broken_categories.py
    python3 eval/regrade_for_broken_categories.py --dry-run   # preview only
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"

# IDs of prompts to exclude from headline computation. Hardcoded rather than
# read from prompts_v2.json so the script works against historical eval JSONs
# that include their own (potentially older) prompt copies.
BROKEN_PROMPT_IDS = {"persona_skippy_voice", "persona_brief_role"}


def regrade_one(path: Path, dry_run: bool = False) -> dict:
    data = json.loads(path.read_text())
    prompts = data.get("prompts", [])
    if not prompts:
        return {"path": path.name, "skipped": True, "reason": "no prompts"}

    # Recompute by summing aggregate over all NON-broken prompts.
    new_passed = 0
    new_total = 0
    excluded = 0
    for p in prompts:
        agg = p.get("aggregate", {})
        n = agg.get("total", len(p.get("samples", [])))
        if p.get("id") in BROKEN_PROMPT_IDS:
            excluded += n
            # Annotate the prompt itself so per-category logic can recognise it
            p["category_status"] = "BROKEN_SUBSTRING_INCOMPATIBLE"
            continue
        new_passed += agg.get("pass_n", 0)
        new_total += n

    new_pass_rate = (new_passed / new_total) if new_total else 0.0

    summary = data.get("summary", {})
    if "summary_v1_legacy" not in data:
        # Preserve original numbers for audit trail
        data["summary_v1_legacy"] = dict(summary)
    summary["passed"] = new_passed
    summary["total"] = new_total
    summary["pass_rate"] = round(new_pass_rate, 4)
    summary["excluded_samples"] = excluded
    summary["excluded_reason"] = "BROKEN_SUBSTRING_INCOMPATIBLE prompts (persona category)"
    summary["regraded_at"] = "2026-05-08-post-remediation"
    data["summary"] = summary

    legacy = data["summary_v1_legacy"]
    delta_pp = round((new_pass_rate - legacy.get("pass_rate", 0)) * 100, 2)

    if not dry_run:
        path.write_text(json.dumps(data, indent=2) + "\n")
    return {
        "path": path.name,
        "name": data.get("name", path.stem),
        "old": f"{legacy.get('passed')}/{legacy.get('total')} = {legacy.get('pass_rate', 0)*100:.1f}%",
        "new": f"{new_passed}/{new_total} = {new_pass_rate*100:.1f}%",
        "delta_pp": delta_pp,
        "excluded_samples": excluded,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing")
    args = parser.parse_args()

    results = []
    for path in sorted(RESULTS.glob("acc_*.json")):
        # Skip diff JSONs; they have a different schema.
        try:
            head = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "diffs" in head and "prompts" not in head:
            continue
        results.append(regrade_one(path, dry_run=args.dry_run))

    print(f"{'eval JSON':70} {'old':>14} {'new':>14} {'Δpp':>6}")
    print("-" * 110)
    for r in results:
        if r.get("skipped"):
            continue
        print(f"{r['name'][:68]:70} {r['old']:>14} {r['new']:>14} {r['delta_pp']:+6.2f}")
    print()
    print(f"{len(results)} JSONs processed (dry_run={args.dry_run}).")


if __name__ == "__main__":
    main()
