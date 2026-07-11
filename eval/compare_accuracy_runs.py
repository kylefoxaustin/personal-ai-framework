#!/usr/bin/env python3
"""Diff two accuracy-eval runs — answer 'does variant B match variant A?'

Typical use:
  1. Record a reference run of current Skippy (fp16-compute Q4_K_M):
       python3 eval/run_accuracy_eval.py --name reference-q4km
  2. Later, when an INT8 W8A8 re-quant of Skippy is available, record:
       python3 eval/run_accuracy_eval.py --name candidate-w8a8
  3. Diff:
       python3 eval/compare_accuracy_runs.py \\
           --reference eval/results/acc_reference-q4km_*.json \\
           --candidate eval/results/acc_candidate-w8a8_*.json \\
           --out eval/results/acc_diff_q4km_vs_w8a8.md

Metrics per prompt (and aggregated):
  - pass-rate delta (reference N/M vs candidate N/M)
  - exact-match rate across samples (did the text change at all?)
  - token-set Jaccard similarity (word-level overlap)
  - character-length delta (did length change dramatically?)
  - category pass-rate breakdown

The Jaccard + length ratios are crude proxies for semantic drift. For a more
rigorous comparison drop in a sentence-transformers cosine-similarity pass or
an LLM-as-judge score — the JSON outputs are structured to support that.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def diff_prompt(ref_prompt: dict, cand_prompt: dict) -> dict:
    """Per-prompt diff metrics. Both runs should have matching `id`."""
    ref_samples = ref_prompt.get("samples", [])
    cand_samples = cand_prompt.get("samples", [])
    n = min(len(ref_samples), len(cand_samples))
    if n == 0:
        return {"id": ref_prompt["id"], "error": "no samples to compare"}

    exact_hits, jaccards, length_ratios = 0, [], []
    for i in range(n):
        r_text = ref_samples[i].get("text", "") or ""
        c_text = cand_samples[i].get("text", "") or ""
        if r_text.strip() == c_text.strip():
            exact_hits += 1
        jaccards.append(jaccard(r_text, c_text))
        length_ratios.append((len(c_text) + 1) / (len(r_text) + 1))

    ref_pass = ref_prompt.get("aggregate", {}).get("pass_n", 0)
    cand_pass = cand_prompt.get("aggregate", {}).get("pass_n", 0)

    return {
        "id": ref_prompt["id"],
        "category": ref_prompt.get("category"),
        "n_samples": n,
        "ref_pass_n": ref_pass,
        "cand_pass_n": cand_pass,
        "pass_delta": cand_pass - ref_pass,
        "exact_match_rate": round(exact_hits / n, 3),
        "jaccard_p50": round(statistics.median(jaccards), 3),
        "jaccard_min": round(min(jaccards), 3),
        "length_ratio_p50": round(statistics.median(length_ratios), 3),
    }


def aggregate_by_category(rows: list) -> dict:
    by_cat = defaultdict(lambda: {"pass_delta": 0, "n_prompts": 0,
                                   "jaccards": [], "exact_matches": []})
    for r in rows:
        if "error" in r:
            continue
        c = r.get("category") or "uncategorized"
        by_cat[c]["pass_delta"] += r["pass_delta"]
        by_cat[c]["n_prompts"] += 1
        by_cat[c]["jaccards"].append(r["jaccard_p50"])
        by_cat[c]["exact_matches"].append(r["exact_match_rate"])
    out = {}
    for c, data in by_cat.items():
        out[c] = {
            "n_prompts": data["n_prompts"],
            "total_pass_delta": data["pass_delta"],
            "jaccard_p50": round(statistics.median(data["jaccards"]), 3) if data["jaccards"] else None,
            "exact_match_p50": round(statistics.median(data["exact_matches"]), 3) if data["exact_matches"] else None,
        }
    return out


def render_markdown(ref: dict, cand: dict, diffs: list) -> str:
    by_cat = aggregate_by_category(diffs)
    total_ref = ref["summary"]["passed"]
    total_cand = cand["summary"]["passed"]
    overall_jacc = statistics.median([d["jaccard_p50"] for d in diffs if "jaccard_p50" in d]) \
        if diffs else 0.0
    overall_exact = statistics.median([d["exact_match_rate"] for d in diffs if "exact_match_rate" in d]) \
        if diffs else 0.0

    lines = [
        f"# Accuracy comparison: `{ref['name']}` vs `{cand['name']}`",
        "",
        f"- Reference: {ref['name']} — {ref['timestamp']}",
        f"- Candidate: {cand['name']} — {cand['timestamp']}",
        f"- Prompt set: v{ref.get('prompts_version')} · samples/prompt: {ref.get('samples_per_prompt')}",
        "",
        "## Headline numbers",
        "",
        f"- **Pass rate:** {ref['summary']['pass_rate']*100:.1f}% → "
        f"{cand['summary']['pass_rate']*100:.1f}% "
        f"(Δ = {(cand['summary']['pass_rate'] - ref['summary']['pass_rate'])*100:+.1f} pp)",
        f"- **Total passes:** {total_ref} → {total_cand} "
        f"(Δ = {total_cand - total_ref:+d})",
        f"- **Median Jaccard overlap** (word-level): {overall_jacc:.3f} "
        f"(1.0 = identical word bags, 0.0 = no words shared)",
        f"- **Median exact-match rate** across samples: {overall_exact:.3f} "
        f"(fraction of prompts where candidate produced byte-identical text)",
        "",
        "## By category",
        "",
        "| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, d in sorted(by_cat.items()):
        lines.append(f"| {c} | {d['n_prompts']} | {d['total_pass_delta']:+d} "
                     f"| {d['jaccard_p50']} | {d['exact_match_p50']} |")

    # Flag the most-divergent prompts — these are where the variants disagree
    # and deserve manual inspection.
    divergent = sorted([d for d in diffs if "jaccard_p50" in d],
                       key=lambda x: x["jaccard_p50"])[:10]
    lines += ["", "## Most-divergent prompts (lowest Jaccard = most drift)", ""]
    lines.append("| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for d in divergent:
        lines.append(f"| `{d['id']}` | {d['category']} | {d['pass_delta']:+d} "
                     f"| {d['jaccard_p50']} | {d['exact_match_rate']} | {d['length_ratio_p50']} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True,
                        help="Path to reference run (JSON from run_accuracy_eval.py)")
    parser.add_argument("--candidate", required=True,
                        help="Path to candidate run (JSON from run_accuracy_eval.py)")
    parser.add_argument("--out", required=True,
                        help="Output markdown path")
    args = parser.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    cand = json.loads(Path(args.candidate).read_text())

    # Index by id for joining
    ref_by_id = {p["id"]: p for p in ref["prompts"]}
    cand_by_id = {p["id"]: p for p in cand["prompts"]}

    shared = sorted(set(ref_by_id) & set(cand_by_id))
    missing_from_ref = set(cand_by_id) - set(ref_by_id)
    missing_from_cand = set(ref_by_id) - set(cand_by_id)
    if missing_from_ref or missing_from_cand:
        print(f"⚠️  prompt-set mismatch — "
              f"missing from ref: {sorted(missing_from_ref)}, "
              f"missing from candidate: {sorted(missing_from_cand)}")

    diffs = [diff_prompt(ref_by_id[pid], cand_by_id[pid]) for pid in shared]

    md = render_markdown(ref, cand, diffs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    # Also emit a sibling JSON with the raw per-prompt diffs for the sizer to consume
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "reference": ref["name"], "candidate": cand["name"],
        "reference_timestamp": ref["timestamp"], "candidate_timestamp": cand["timestamp"],
        "diffs": diffs,
        "by_category": aggregate_by_category(diffs),
        "headline": {
            "ref_pass_rate": ref["summary"]["pass_rate"],
            "cand_pass_rate": cand["summary"]["pass_rate"],
            "pass_rate_delta_pp": round((cand["summary"]["pass_rate"] -
                                         ref["summary"]["pass_rate"]) * 100, 2),
        }
    }, indent=2))
    print(f"📄 {out_path}")
    print(f"📄 {json_path}")


if __name__ == "__main__":
    main()
