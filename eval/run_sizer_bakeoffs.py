#!/usr/bin/env python3
"""Sizer bake-off harness for personal-ai-assistant-sizer.

Fires a fixed set of workload profiles (eval/sizer_workload_profiles.json) against
Skippy's /generate endpoint with include_telemetry=True, captures per-call
{host_ms, prefill_ms, decode_ms, prompt_tokens, completion_tokens, ...}, and
aggregates per-profile p50/p90/p95 distributions into a vendorable JSON + MD.

The output feeds the sizer's bundle so the MODELS × WORKLOADS × TIERS grid has
Skippy-measured baselines instead of keyhole-shape-calibrated ones.

Usage:
    SKIPPY_USER=... SKIPPY_PASSWORD=... \\
        python3 eval/run_sizer_bakeoffs.py --name qwen3-moe-2026-04-22

    # Custom sample count, single profile for quick checks:
    python3 eval/run_sizer_bakeoffs.py --name probe --profiles short_chat --samples 3

Outputs:
    eval/results/sizer_bakeoff_<name>_<timestamp>.{json,md}
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


PROFILES_PATH = Path(__file__).parent / "sizer_workload_profiles.json"


# ---- Auth ------------------------------------------------------------------


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


# ---- Synthetic transcript generator ----------------------------------------

# Plausible meeting-transcript-shaped sentences. Combined + shuffled to hit
# target token counts without semantic repetition triggering any caching path.
_TRANSCRIPT_SEEDS = [
    "Alright, let's get started. Kyle, can you walk us through the latest numbers on the Blackwell bake-off?",
    "Sure. So we saw roughly a 1.4x speedup on the MoE path once we tuned the expert routing threshold.",
    "That's promising. Did that hold up across both the short-context and long-context workloads?",
    "Mostly. Long-context held at about 1.35x. Short dropped to 1.2x because the fixed overhead dominates.",
    "Hmm, that's the pattern we saw with the dense model too. Is the fixed overhead primarily the scheduler?",
    "Partly. The other piece is CPU-side token sampling — we haven't vectorized that path yet.",
    "Could we get a rough estimate on how much of the decode time that represents at NPU Mid tier?",
    "Rough estimate, maybe 8 to 12 percent. Higher on the low-BW SKUs where the math runs faster relative to sampling.",
    "Okay. Action item: let's prototype a vectorized sampler and see if it moves the needle.",
    "I can take that. Estimated a couple of days, plus integration testing. Targeting end of next sprint.",
    "Good. Next topic — the customer feasibility review for the NXP 95 tier. Who has that?",
    "That's me. I met with the customer last Thursday. They want sub-50ms TTFT for their voice interface.",
    "At what prompt length?",
    "Typical 400 token system prompt plus about 150 tokens of dialogue history. So call it 550 prefill.",
    "Is the NPU Mid tier reachable for that?",
    "Borderline. Current measurements on our Q4 MoE show 72ms TTFT at 550 prefill on the Mid SKU.",
    "What brings it down? Is it compiler-quality headroom, or do we need to profile a different quant?",
    "Probably both. The Q5_K_M variant gave us a 6% TTFT improvement in the last bake-off. Compiler tuning should claw back another 10-15%.",
    "Okay. Let's get a firm number by Wednesday so we can greenlight or flag it.",
    "Before we move on — any concerns about the measurement methodology? The customer is going to push back on synthetic workloads.",
    "Valid concern. We should run at least a few of their real prompts through the pipeline as a sanity check.",
    "Can we get anonymized samples from their staging environment?",
    "I'll ask in the sync tomorrow. If they won't share, we'll template-match to their stated shape.",
    "Moving on — roadmap review. Where are we on the sizer tool?",
    "Phase 1 is close. Bake-off harness is in, patterns are cribbed from the keyhole sizer. Should have MoE measurements by end of week.",
    "Nice. And the dense 14B numbers?",
    "Blocked on a config swap — one run per model, serially. Should be a half-day once MoE is wrapped.",
    "Blockers for Phase 2?",
    "We need model_name labels on the Prometheus histograms and the prefill-decode split exposed. Maybe a 30-line patch to the server.",
    "Priority on that patch — when we get to continuous instrumentation we want the bridge script to just work.",
    "Agreed. I'll file it as a follow-up after the Phase 1 results land.",
    "Any other open items? Okay, wrapping up. Thanks everyone.",
]


def make_synthetic_transcript(target_tokens: int, seed: int = 42) -> str:
    """Assemble a plausible meeting transcript by concatenating shuffled seed
    sentences until approximate target token count is reached. Uses ~0.75
    tokens-per-word approximation for English."""
    target_words = int(target_tokens * 0.75)
    rng = random.Random(seed)
    pool = list(_TRANSCRIPT_SEEDS)
    rng.shuffle(pool)
    out = []
    word_count = 0
    idx = 0
    speaker_turn = 0
    speakers = ["Kyle", "Priya", "Marco", "Asha", "Dev", "Jin"]
    while word_count < target_words:
        if idx >= len(pool):
            rng.shuffle(pool)
            idx = 0
        line = pool[idx]
        speaker = speakers[speaker_turn % len(speakers)]
        out.append(f"{speaker}: {line}")
        word_count += len(line.split()) + 1
        idx += 1
        speaker_turn += 1
    return "\n".join(out)


# ---- Prompt materialization ------------------------------------------------


def materialize_prompts(profile: dict) -> list:
    """Return a list of {prompt_text, source_id} for each prompt entry in a
    profile. Dict entries with transcript_synth get their templates filled in."""
    out = []
    for i, entry in enumerate(profile["prompts"]):
        if isinstance(entry, str):
            out.append({"prompt": entry, "source_id": f"{profile['id']}__{i}"})
        elif isinstance(entry, dict) and "template" in entry:
            synth = entry.get("transcript_synth", {})
            target = synth.get("target_transcript_tokens", 10000)
            transcript = make_synthetic_transcript(target, seed=42 + i)
            filled = entry["template"].format(transcript=transcript)
            out.append({"prompt": filled, "source_id": f"{profile['id']}__{i}_synth"})
        else:
            print(f"  ⚠️  skipping unrecognized prompt entry in {profile['id']}: {entry!r}")
    return out


# ---- Call harness ----------------------------------------------------------


def call_generate(endpoint: str, headers: dict, prompt: str,
                  request_params: dict, defaults: dict) -> dict:
    payload = {
        "prompt": prompt,
        "include_telemetry": True,
        "skip_agent_loop": True,
        "max_tokens": request_params.get("max_tokens", 500),
        "use_rag": request_params.get("use_rag", False),
        "rag_k": request_params.get("rag_k", 0),
        "temperature": request_params.get("temperature", defaults.get("temperature", 0.1)),
        "top_p": request_params.get("top_p", defaults.get("top_p", 0.9)),
    }
    t0 = time.time()
    try:
        resp = requests.post(
            f"{endpoint}/generate",
            json=payload,
            headers=headers,
            timeout=600,
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"request failed: {e}", "wall_clock_s": round(time.time() - t0, 2)}
    wall_clock_s = time.time() - t0
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:300]}", "wall_clock_s": round(wall_clock_s, 2)}
    data = resp.json()
    return {
        "text_preview": (data.get("text", "") or "")[:160],
        "tokens_used": data.get("tokens_used"),
        "wall_clock_s": round(wall_clock_s, 2),
        "telemetry": data.get("telemetry"),
    }


# ---- Aggregation -----------------------------------------------------------


def pct(values: list, p: float) -> float:
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def summarize_samples(samples: list) -> dict:
    """Take a list of per-call telemetry dicts and produce p50/p90/p95 across fields."""
    fields = ["host_ms", "prefill_ms", "decode_ms", "total_ms",
              "prefill_tok_per_s", "decode_tok_per_s",
              "prompt_tokens", "completion_tokens"]
    valid_telemetry = [s.get("telemetry") for s in samples
                       if s.get("telemetry") and not s.get("error")]
    summary = {"n_valid": len(valid_telemetry), "n_total": len(samples)}
    for f in fields:
        vals = [t[f] for t in valid_telemetry if t.get(f) is not None]
        if not vals:
            summary[f] = None
            continue
        summary[f] = {
            "p50": round(pct(vals, 50), 2),
            "p90": round(pct(vals, 90), 2),
            "p95": round(pct(vals, 95), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "mean": round(statistics.mean(vals), 2),
            "n": len(vals),
        }
    return summary


# ---- Main ------------------------------------------------------------------


def run_profile(profile: dict, defaults: dict, endpoint: str, headers: dict,
                samples_per_prompt: int) -> dict:
    print(f"\n▶ Profile: {profile['id']} ({profile['description'][:80]}...)")
    materialized = materialize_prompts(profile)
    all_samples = []
    for prompt_entry in materialized:
        prompt = prompt_entry["prompt"]
        for s in range(samples_per_prompt):
            print(f"  [{prompt_entry['source_id']} sample {s+1}/{samples_per_prompt}] ", end="", flush=True)
            result = call_generate(
                endpoint, headers, prompt,
                profile.get("request_params", {}), defaults,
            )
            if "error" in result:
                print(f"💥 {result['error'][:80]}")
            else:
                tm = result.get("telemetry") or {}
                print(f"✅ prefill={tm.get('prefill_ms','?')}ms "
                      f"decode={tm.get('decode_ms','?')}ms "
                      f"host={tm.get('host_ms','?')}ms "
                      f"tokens={tm.get('prompt_tokens','?')}/{tm.get('completion_tokens','?')}")
            result["source_id"] = prompt_entry["source_id"]
            result["sample_index"] = s
            all_samples.append(result)
    summary = summarize_samples(all_samples)
    return {
        "profile_id": profile["id"],
        "description": profile["description"],
        "target_prefill_tokens": profile.get("target_prefill_tokens"),
        "target_decode_tokens": profile.get("target_decode_tokens"),
        "critical_metric": profile.get("critical_metric"),
        "request_params": profile.get("request_params"),
        "summary": summary,
        "samples": all_samples,
    }


def render_markdown(out: dict) -> str:
    lines = [
        f"# Sizer bake-off: {out['name']}",
        "",
        f"- Timestamp: {out['timestamp']}",
        f"- Endpoint: {out['endpoint']}",
        f"- Samples per prompt: {out['samples_per_prompt']}",
        f"- Model reported by server: {out.get('model_reported', '?')}",
        "",
        "## Per-profile summary",
        "",
        "| Profile | n | prefill ms (p50/p90) | decode ms (p50/p90) | host ms (p50) | prefill tok/s (p50) | decode tok/s (p50) |",
        "|---|---|---|---|---|---|---|",
    ]
    for prof in out["profiles"]:
        s = prof["summary"]
        def _pair(field, keys=("p50", "p90")):
            v = s.get(field)
            if not v:
                return "-"
            return "/".join(f"{v[k]:.0f}" for k in keys)
        def _single(field, key="p50"):
            v = s.get(field)
            return f"{v[key]:.1f}" if v else "-"
        lines.append(
            f"| {prof['profile_id']} | {s.get('n_valid', 0)}/{s.get('n_total', 0)} "
            f"| {_pair('prefill_ms')} | {_pair('decode_ms')} | {_single('host_ms')} "
            f"| {_single('prefill_tok_per_s')} | {_single('decode_tok_per_s')} |"
        )
    lines += ["", "## Per-profile details", ""]
    for prof in out["profiles"]:
        lines.append(f"### {prof['profile_id']}")
        lines.append("")
        lines.append(f"_{prof['description']}_")
        lines.append("")
        lines.append(f"- Target: ~{prof.get('target_prefill_tokens', '?')} prefill / "
                     f"~{prof.get('target_decode_tokens', '?')} decode tokens")
        lines.append(f"- Critical metric: **{prof.get('critical_metric', '?')}**")
        lines.append("")
        s = prof["summary"]
        for field in ("host_ms", "prefill_ms", "decode_ms", "total_ms",
                      "prefill_tok_per_s", "decode_tok_per_s",
                      "prompt_tokens", "completion_tokens"):
            v = s.get(field)
            if not v:
                continue
            lines.append(f"- **{field}**: p50={v['p50']} p90={v['p90']} p95={v['p95']} "
                         f"(min={v['min']} max={v['max']} mean={v['mean']} n={v['n']})")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Label for this run (e.g. qwen3-moe-2026-04-22)")
    parser.add_argument("--endpoint", default="http://localhost:8080")
    parser.add_argument("--profiles", default="",
                        help="Comma-separated subset (default: all)")
    parser.add_argument("--samples", type=int, default=0,
                        help="Override samples_per_prompt from profiles file")
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Fire one short_chat call before sampling to warm up KV cache / CUDA graphs")
    args = parser.parse_args()

    with open(PROFILES_PATH) as f:
        cfg = json.load(f)
    defaults = cfg.get("defaults", {})
    samples_per_prompt = args.samples or defaults.get("samples_per_prompt", 3)

    wanted = set(p.strip() for p in args.profiles.split(",") if p.strip())
    profiles = [p for p in cfg["profiles"] if not wanted or p["id"] in wanted]
    if not profiles:
        print(f"❌ no matching profiles (wanted: {wanted})")
        sys.exit(2)

    headers = login(args.endpoint)

    # Warmup — fire one short call to stabilize decode path
    if args.warmup:
        print("▶ Warmup...")
        _ = call_generate(
            args.endpoint, headers,
            "Say 'ready' and nothing else.",
            {"max_tokens": 5, "use_rag": False, "rag_k": 0},
            defaults,
        )

    out_profiles = []
    for profile in profiles:
        out_profiles.append(run_profile(profile, defaults, args.endpoint, headers, samples_per_prompt))

    # Try to surface a model identifier from the first valid sample
    model_reported = None
    for prof in out_profiles:
        for s in prof["samples"]:
            tm = s.get("telemetry") or {}
            if tm.get("model_name"):
                model_reported = tm["model_name"]
                break
        if model_reported:
            break

    out = {
        "name": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.endpoint,
        "samples_per_prompt": samples_per_prompt,
        "model_reported": model_reported,
        "defaults": defaults,
        "profiles": out_profiles,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = results_dir / f"sizer_bakeoff_{args.name}_{stamp}.json"
    md_path = results_dir / f"sizer_bakeoff_{args.name}_{stamp}.md"
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(render_markdown(out))
    print(f"\n📄 {json_path}")
    print(f"📄 {md_path}")


if __name__ == "__main__":
    main()
