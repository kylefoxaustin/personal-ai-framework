#!/usr/bin/env python3
"""Merge sizer bake-off JSONs into a single sizer_bundle.json for the
personal-ai-assistant-sizer repo.

Bundle schema (LLM-first, per [sizer]'s 16:14 recommendation 2026-04-22):

    {
      "meta": {...},
      "models":    { <model_key>: {constants: total_params, active_params, ...} },
      "workloads": { <profile_id>: { <model_key>: {p50/p90/p95 per metric} } },
      "tiers_measured": { ... }      # tier hardware specs (Phase 1 placeholder)
    }

Usage:
    python3 eval/build_sizer_bundle.py \\
        --bakeoff eval/results/sizer_bakeoff_qwen3-moe-2026-04-22_.json \\
        --bakeoff eval/results/sizer_bakeoff_qwen25-14b-2026-04-22_.json \\
        --out eval/results/sizer_bundle.json

Per-model intrinsic constants are declared in MODEL_CONSTANTS below; this script
does not infer them from the bake-off (the bake-off only measures timing).
Update MODEL_CONSTANTS when adding a new model.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# Model intrinsic constants (architecture + GGUF metadata).
# Sources: Qwen model cards, GGUF file sizes on disk. Verify against your
# actual GGUF before shipping a public bundle.
MODEL_CONSTANTS = {
    "qwen2.5-14b-q4-dense": {
        "display_name": "Qwen 2.5 14B Instruct (Q4_K_M, dense)",
        "family": "qwen2.5",
        "is_moe": False,
        "total_params": 14_700_000_000,
        "active_params": 14_700_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 8_986_070_304,
        "hidden_dim": 5120,
        "num_layers": 48,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "ctx_len_extended": 131072,
    },
    "qwen3-30b-a3b-q4-moe": {
        "display_name": "Qwen3 30B A3B (Q4_K_M, MoE)",
        "family": "qwen3",
        "is_moe": True,
        "total_params": 30_500_000_000,
        "active_params": 3_300_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 18_556_684_448,
        "hidden_dim": 2048,
        "num_layers": 48,
        "num_attention_heads": 32,
        "num_kv_heads": 4,
        "num_experts": 128,
        "experts_per_token": 8,
        "vocab_size": 151936,
        "ctx_len_trained": 262144,
        "ctx_len_extended": 262144,
    },
}


# Tier hardware presets — placeholder for Phase 1 before the sizer repo exists.
# These mirror keyhole-sizer's LOW_LP4 / LOW_LP5_32BIT / LOW_LP5_64BIT / LOW_LP5X
# / MID / HIGH post 2026-04-22 split, at 0.70 bandwidth efficiency. Final values
# will be pulled from the sizer repo's npu_model.py once scaffolding is copied.
TIERS_PLACEHOLDER = {
    "note": "Placeholder — copy from sizer repo's npu_model.py TIERS dict once crib is done. Values below are mirrors of keyhole-sizer as of 2026-04-22.",
    "presets": {
        "NPU_LOW_LP4":      {"tops_int8": 2.0,  "bus_width_bit": 32, "mem_speed_gtps": 4.266, "bw_eff": 0.70},
        "NPU_LOW_LP5_32BIT": {"tops_int8": 2.0,  "bus_width_bit": 32, "mem_speed_gtps": 6.4,   "bw_eff": 0.70},
        "NPU_LOW_LP5_64BIT": {"tops_int8": 2.0,  "bus_width_bit": 64, "mem_speed_gtps": 6.4,   "bw_eff": 0.70},
        "NPU_LOW_LP5X":     {"tops_int8": 2.0,  "bus_width_bit": 64, "mem_speed_gtps": 8.533, "bw_eff": 0.70},
        "NPU_MID":          {"tops_int8": 40.0, "bus_width_bit": 128, "mem_speed_gtps": 6.4,   "bw_eff": 0.70},
        "NPU_HIGH":         {"tops_int8": 200.0,"bus_width_bit": 128, "mem_speed_gtps": 8.4,   "bw_eff": 0.70},
    },
}


# Map a bake-off's reported model_name (from the gguf filename) to a canonical
# MODEL_CONSTANTS key. Extend as new models are bake-offed.
MODEL_NAME_PATTERNS = [
    ("qwen2.5-14b", "qwen2.5-14b-q4-dense"),
    ("qwen25-14b",  "qwen2.5-14b-q4-dense"),
    ("kyle-14b",    "qwen2.5-14b-q4-dense"),  # Kyle's fine-tuned 14B dense
    ("qwen3-30b",   "qwen3-30b-a3b-q4-moe"),
    ("kyle-30b-a3b", "qwen3-30b-a3b-q4-moe"),
    ("30b-a3b",     "qwen3-30b-a3b-q4-moe"),
]


def canonicalize_model_name(reported: str) -> str:
    if not reported:
        return "unknown"
    low = reported.lower()
    for pattern, canonical in MODEL_NAME_PATTERNS:
        if pattern in low:
            return canonical
    return reported  # fall through — user can fix in-place


def extract_workload_row(profile_summary: dict) -> dict:
    """Flatten a profile's summary into the per-(workload,model) cell shape."""
    row = {"n": profile_summary.get("n_valid", 0)}
    for field in ("host_ms", "prefill_ms", "decode_ms", "total_ms",
                  "prefill_tok_per_s", "decode_tok_per_s",
                  "prompt_tokens", "completion_tokens"):
        v = profile_summary.get(field)
        if not v:
            continue
        row[f"{field}_p50"] = v["p50"]
        row[f"{field}_p90"] = v["p90"]
        row[f"{field}_p95"] = v["p95"]
    return row


def merge_bakeoff(bundle: dict, bakeoff_path: Path) -> None:
    with open(bakeoff_path) as f:
        data = json.load(f)
    reported = data.get("model_reported")
    canonical = canonicalize_model_name(reported or "")
    if canonical == "unknown":
        print(f"⚠️  {bakeoff_path.name}: no model_reported — using 'unknown' key")
    elif canonical == reported:
        print(f"⚠️  {bakeoff_path.name}: reported model '{reported}' has no MODEL_NAME_PATTERNS match — keeping as-is")
    else:
        print(f"✅ {bakeoff_path.name}: reported '{reported}' → canonical '{canonical}'")
    bundle["meta"]["source_bakeoff_files"].append(str(bakeoff_path))
    bundle["meta"]["models_measured"].append({
        "canonical": canonical,
        "reported": reported,
        "bakeoff_file": str(bakeoff_path.name),
        "timestamp": data.get("timestamp"),
    })
    for prof in data.get("profiles", []):
        pid = prof["profile_id"]
        row = extract_workload_row(prof["summary"])
        row["target_prefill_tokens"] = prof.get("target_prefill_tokens")
        row["target_decode_tokens"] = prof.get("target_decode_tokens")
        row["critical_metric"] = prof.get("critical_metric")
        row["request_params"] = prof.get("request_params")
        bundle["workloads"].setdefault(pid, {})[canonical] = row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bakeoff", action="append", required=True,
                        help="Path to a sizer bake-off JSON. Repeat for multiple models.")
    parser.add_argument("--out", required=True,
                        help="Output path for sizer_bundle.json")
    args = parser.parse_args()

    bundle = {
        "meta": {
            "schema_version": 1,
            "methodology_version": "2026-05-11-semantic-regrade-shipped",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "eval/build_sizer_bundle.py",
            "source_bakeoff_files": [],
            "models_measured": [],
            "notes": (
                "Phase 1 bundle — timing measured from Skippy prod /generate "
                "with include_telemetry=True. Tier hardware constants below are "
                "placeholders; final values inherit from personal-ai-assistant-sizer/"
                "sizer/npu_model.py once that repo is stood up."
            ),
        },
        "models": MODEL_CONSTANTS,
        "workloads": {},
        "tiers_measured": TIERS_PLACEHOLDER,
    }

    for bf in args.bakeoff:
        path = Path(bf)
        if not path.exists():
            print(f"❌ bake-off not found: {bf}")
            sys.exit(2)
        merge_bakeoff(bundle, path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"\n📦 {out_path}  ({len(bundle['workloads'])} workloads, "
          f"{len(bundle['meta']['models_measured'])} model bake-offs merged)")


if __name__ == "__main__":
    main()
