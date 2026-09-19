#!/usr/bin/env python3
"""Assemble sizer_bundle.json **v2** — adds per-board measured LLM perf and a
feasibility gate to the v1 (models/workloads) bundle.

v1 (eval/build_sizer_bundle.py) is LLM-perf-only on abstract tier placeholders,
built from 5090 telemetry bake-offs. v2 is ADDITIVE: it keeps the v1 sections and
adds two new ones the fleet measured since May that v1 has no slot for —
`boards_measured` (real silicon: 5090 / Orin / iq9) and `feasibility_gates`
(placeable/unplaceable predicate, keyed part+toolchain+version). Schema spec:
docs/sizer-bundle-v2-schema.md. Consumer: personal-ai-assistant-sizer (pai-sizer).

Every figure carries a provenance tag (MEASURED / DERIVED / SOURCED) per Fleet Law;
a DERIVED/SOURCED value is never presented as MEASURED. Provenance travels as a
STRUCTURED FIELD so a consumer can filter on it.

Usage:
    python3 eval/build_sizer_bundle_v2.py \
        --v1 eval/results/sizer_bundle.json \
        --skippy5090 eval/results/ladder/rtx5090/bench_7bv4_5090_20260915.json \
        --out eval/results/sizer_bundle.json
"""
import argparse, json, sys
from pathlib import Path

# Reuse the single source of truth for model intrinsics.
sys.path.insert(0, str(Path(__file__).parent))
from build_sizer_bundle import MODEL_CONSTANTS  # noqa: E402


# ── boards_measured: real silicon. Curated from measured data (bus, 2026-09-15). ──
# Perf for the 5090 is folded in from the bench JSON at build time (see below); the
# Orin and iq9 blocks are transcribed from their measured sources with provenance.
BOARDS_MEASURED = {
    "rtx5090": {
        "part": "NVIDIA GeForce RTX 5090 (32 GB GDDR7)",
        "arch": "Blackwell (sm_120)",
        "role": "SOURCE OF TRUTH — every other rung is a delta against this",
        "provenance": {"tier": "MEASURED", "source": "scripts/bench_skippy.py",
                       "caveat": "local desktop 5090, Skippy prod /generate/stream"},
        "perf": {},   # filled from --skippy5090 at build time
    },
    "orin-agx": {
        "part": "NVIDIA Jetson AGX Orin (204.8 GB/s LPDDR5)",
        "arch": "Ampere (sm_87)",
        "provenance": {"tier": "MEASURED", "source": "eval/results/orin/GGUF_LADDER.md",
                       "caveat": "llama.cpp CUDA, llama-bench -p0 -n128; decode-only, no RAG"},
        "perf": {
            "qwen25-7b-v4-q4-dense": {
                "q4_k_m": {"decode_tps": 27.82, "achieved_gb_s": 121.6,
                           "tag": "MEASURED", "date": "2026-07-09"}},
            "qwen2.5-14b-q4-dense": {
                "q4_k_m": {"decode_tps": 13.76, "achieved_gb_s": 123.7, "tag": "MEASURED"},
                "q8_0":   {"decode_tps": 10.79, "achieved_gb_s": 169.4, "tag": "MEASURED"}},
            "qwen3-30b-a3b-q4-moe": {
                "q4_k_m": {"decode_tps": 43.74, "achieved_gb_s": 91.6, "tag": "MEASURED"}},
        },
        "precision_facts": [
            {"claim": "achieved bandwidth is size-invariant within a precision but "
                      "precision-dependent (fp16/Q8 ~80% bus, Q4 ~60%, MoE-Q4 ~45%)",
             "tag": "MEASURED"},
            {"claim": "Q4 draws MORE power than fp16 (32.5W vs 27.1W) while moving fewer "
                      "bytes — dequant is compute, not free", "tag": "MEASURED"},
        ],
    },
    "iq9": {
        "part": "Qualcomm SA8775P / QCS9075 (Hexagon v73 HTP, 2×NSP)",
        "arch": "Hexagon HTP v73",
        "provenance": {"tier": "MEASURED", "source": "qualcomm memory iq9-npu-llm-genie-deploy; "
                       "genie-t2t-run --profile on qwen25vl_bundle",
                       "caveat": "single-NSP unless noted, clocks pinned"},
        "perf": {
            # CORRECTED 2026-09-15: fresh on-silicon enumeration superseded a ~3x-off
            # memory line (prefill 197.5 → 596.3). Use the measured-today text-prefill.
            "qwen2.5-7b-w4a16": {
                "w4a16": {"prefill_tps": 596.3, "ttft_ms": 287, "decode_tps": 9.45,
                          "tag": "MEASURED", "date": "2026-09-15",
                          "note": "text prompt, ≤256 input tokens (see context_cap)"}},
            "qwen3-vl-4b-w4a16": {
                "w4a16": {"prefill_tps": 1003, "decode_tps": 15.75, "tag": "MEASURED",
                          "note": "PERF-ONLY / ACCURACY-INVALID — vendor harness zeroes deepstack"}},
        },
        "context_cap": {
            "npu_max_input_tokens": 256,
            "beyond_cap": {"backend": "CPU llama.cpp", "decode_tps": 19,
                           "ttft_s_at_2000tok": 108, "tag": "MEASURED", "date": "2026-09-15"},
            "note": "BINDING constraint for RAG/long-context on iq9 — NPU prefill 596 t/s "
                    "applies ONLY ≤256 input tokens; encode as a length gate, not a cost curve.",
        },
        "npu_capability": {
            "nsp_count": 2, "vtcm_mb_per_nsp": 8, "vtcm_poolable": False,
            "datatypes": ["w4a16", "w8a16", "fp16", "int8"],
            "fp8": "v79+ only", "mxfp4_fp4": "rejected",
            "tag": "SOURCED (on-silicon capability query)"},
        "precision_facts": [
            {"claim": "decode is DDR-bandwidth-bound (confirmed 3 ways); prefill is "
                      "compute/TOPS-bound + parallel — scarce resource for decode = DDR BW, not TOPS",
             "tag": "MEASURED"},
            {"claim": "dual-NSP = ~1.94x AGGREGATE throughput (2 policies) but 0% on single-policy "
                      "decode latency (2nd NSP adds compute, not bandwidth)", "tag": "MEASURED"},
            {"claim": "bare int8 MatMul is 1.14–1.23x SLOWER than fp16 on this HTP (HMX wants "
                      "conv-shaped ops, not GEMM) — do NOT assume int8>fp16 for transformers",
             "tag": "MEASURED"},
            {"claim": "INT4 decode ~1.3x over fp16-ish with a +6-8% TTFT dequant tax "
                      "(Thor +5-9%, 5090 flat)", "tag": "MEASURED"},
        ],
        "gaps": [
            {"axis": "decode_tps_vs_context_length", "status": "NOT_RUN", "tag": "GAP",
             "note": "gated on qualcomm's VSLAM push; DERIVED expectation: decode t/s FALLS with "
                     "context on a BW-bound decode (KV re-read every step) — slope unmeasured"},
        ],
    },
}


# ── feasibility_gates: placeable/unplaceable predicate. Runs BEFORE the roofline. ──
FEASIBILITY_GATES = {
    "imx95__eIQ-Neutron__3.2.0": {
        "part": "NXP i.MX95 eIQ Neutron NPU",
        "toolchain": "eIQ Neutron", "version": "3.2.0",
        "external_ref": "95emulator tree: docs/imx95-feasibility-gate.json (schema imx95/feasibility-gate/v1)",
        "summary": {"models": 41, "placeable": 26, "unplaceable": 15},
        "gating_ops": [
            {"op": "NeutronAdd", "outcome": "absent",
             "excludes": "every residual architecture (elementwise-add across two quant scales)",
             "control": "micro elementwise-MUL places bit-exact while micro ADD SIGILLs — isolates the OP",
             "tag": "MEASURED"},
            {"op": "NeutronMulSymByAsym", "outcome": "faults",
             "excludes": "SE-block families", "tag": "MEASURED"},
        ],
        "version_stability": {"stable_across": ["3.1.3", "3.2.0"], "tag": "MEASURED",
                              "note": "gate did not move across these versions — key can be coarser than every point release"},
        "correlated_failure": {"claim": "agents sharing a backbone family fail placement together, "
                               "not independently — a scheduler treating placement as per-agent "
                               "probability is wrong in a correlated direction", "tag": "REASONING"},
        "provenance": {"tier": "MEASURED", "scope": "toolchain-not-silicon",
                       "caveat": "emulated NPU path in QEMU vs vendor golden kernels; one board, "
                                 "one toolchain, 37→41 models — DO NOT extrapolate to other parts"},
        "design_notes": {
            "key": "(part, toolchain, VERSION) — toolchain versions move independently of silicon",
            "predicate_before_roofline": "feasibility is a GATE (placeable/fallback/unplaceable), "
                "not a cost; folding it into a cost model yields a confident wrong number",
            "post_fusion": "the op set that matters is POST-FUSION (converter behaviour), not the "
                "model file — MobileNet-v1→29 fused ops, DenseNet-121→246",
            "cheap_form": "a short list of LOAD-BEARING gating ops (residual-add) excludes whole "
                "arch families in one lookup, no graph walk",
        },
    },
}


def fold_5090_bench(path: Path) -> dict:
    """Map scripts/bench_skippy.py output → boards_measured perf rows."""
    d = json.load(open(path))
    label = {"cold_start": "cold_start", "plain_chat": "short_chat",
             "rag_heavy": "rag_long_context", "reasoning": "long_decode"}
    perf = {"qwen25-7b-v4-q4-dense": {"q4_k_m": {}}}
    row = perf["qwen25-7b-v4-q4-dense"]["q4_k_m"]
    for cat, v in d.items():
        wl = label.get(cat, cat)
        row[wl] = {"decode_tps_p50": v.get("tps_p50"), "decode_tps_p95": v.get("tps_p95"),
                   "ttft_s_p50": v.get("ttft_p50"), "n": v.get("n")}
    row["tag"] = "MEASURED"
    row["date"] = "2026-09-15"
    return perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="existing v1 sizer_bundle.json")
    ap.add_argument("--skippy5090", required=True, help="bench_skippy.py 7B v4 5090 JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v1 = json.load(open(a.v1))

    boards = json.loads(json.dumps(BOARDS_MEASURED))  # deep copy
    boards["rtx5090"]["perf"] = fold_5090_bench(Path(a.skippy5090))

    # models: carry v1 + ensure the production 7B v4 constants are present.
    models = dict(v1.get("models", {}))
    if "qwen25-7b-v4-q4-dense" in MODEL_CONSTANTS:
        models["qwen25-7b-v4-q4-dense"] = MODEL_CONSTANTS["qwen25-7b-v4-q4-dense"]

    bundle = {
        "meta": {
            "schema_version": 2,
            "methodology_version": "2026-09-15-boards-and-feasibility",
            "generated_at": "2026-09-15",   # stamped by hand; Date.now() unavailable in some runners
            "generator": "eval/build_sizer_bundle_v2.py",
            "spec": "docs/sizer-bundle-v2-schema.md",
            "supersedes": v1.get("meta", {}).get("methodology_version"),
            "notes": "v2 = v1 (models/workloads) + boards_measured (real silicon) + "
                     "feasibility_gates (placeable/unplaceable). All figures provenance-tagged; "
                     "iq9 perf CORRECTED 2026-09-15 (fresh enumeration superseded a stale memory line).",
        },
        "models": models,
        "workloads": v1.get("workloads", {}),          # v1 5090 telemetry bake-offs (14b/moe)
        "tiers_measured": v1.get("tiers_measured", {}),  # v1 abstract-tier placeholder (kept)
        "boards_measured": boards,
        "feasibility_gates": FEASIBILITY_GATES,
    }
    json.dump(bundle, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    print(f"  models: {list(models)}")
    print(f"  boards_measured: {list(boards)}")
    print(f"  feasibility_gates: {list(FEASIBILITY_GATES)}")


if __name__ == "__main__":
    main()
