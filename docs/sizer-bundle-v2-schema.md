# sizer_bundle.json v2 — schema extension proposal

*Draft 2026-09-15, authored by `[docs]` (Skippy — upstream author of `sizer_bundle.json`).*
*Coordination target: `[pai-sizer]` (consumer). Discipline is one-way: Skippy authors the
schema, pai-sizer updates its loader/`measured.py` to match. See CLAUDE.md ecosystem section.*

## Why v2

The shipped bundle (`schema_version: 1`, generated 2026-05-11) is **LLM-perf-only on abstract
NPU tiers**: `{meta, models, workloads, tiers_measured(placeholder)}`, and it carries just two
models (14B dense, 30B MoE). Four months of fleet measurement produced **two kinds of data v1
has no slot for**:

1. **Per-board MEASURED LLM perf** — the same model on real silicon (iq9 Hexagon, Orin,
   5090, soon Thor + i.MX95/ARA240), with per-precision prefill/decode/TTFT and NPU-capability
   facts. v1's `tiers_measured` is an April *placeholder* mirrored from keyhole-sizer, not
   measurements.
2. **Feasibility predicates** — whether a model is even *placeable* on a (part, toolchain,
   version), keyed on architecture topology, not part number. This is a **gate that runs before
   any roofline**, not a cost row (95emulator: "fold feasibility into a cost model and an
   unplaceable model comes out as merely expensive — the most dangerous possible answer, because
   it still produces a number").

Everything below is tagged MEASURED / DERIVED / SOURCED per Fleet Law, and provenance travels as
a **structured field**, not prose — so a consumer can filter on it.

## Proposed additions (v1 stays; v2 is additive)

```jsonc
{
  "meta": { "schema_version": 2, ... },
  "models": { ... },          // v1, + skippy-7b-v4-q4-dense (added to MODEL_CONSTANTS)
  "workloads": { ... },       // v1 (5090 telemetry bake-offs)

  // NEW ── per-board measured LLM perf. Replaces the tiers_measured placeholder
  //        with real silicon, keyed by board, then (model, precision).
  "boards_measured": {
    "<board_id>": {
      "part": "...", "arch": "...", "npu": "...",
      "provenance": { "tier": "MEASURED|DERIVED|SOURCED", "caveat": "...", "source": "..." },
      "npu_capability": { "datatypes": [...], "vtcm_mb": N, "notes": "..." },
      "precision_facts": [ { "claim": "...", "tag": "MEASURED", "value": "..." } ],
      "perf": {
        "<model_key>": {
          "<precision>": { "prefill_tps": N, "decode_tps": N, "ttft_ms": N,
                           "n": N, "tag": "MEASURED", "date": "YYYY-MM-DD" }
        }
      },
      "gaps": [ { "axis": "decode_vs_context_length", "status": "NOT_RUN", "tag": "GAP" } ]
    }
  },

  // NEW ── feasibility gate. Runs BEFORE roofline. Keyed (part, toolchain, VERSION);
  //        predicts placeable/unplaceable from skip-TOPOLOGY, not model name.
  "feasibility_gates": {
    "<part>__<toolchain>__<version>": {
      "$ref_or_embed": "imx95/feasibility-gate/v1",   // 95emulator's delivered JSON
      "gating_ops": [ { "op": "...", "outcome": "absent|faults",
                        "excludes": "<arch family>", "tag": "MEASURED" } ],
      "version_stability": { "stable_across": ["3.1.3","3.2.0"], "tag": "MEASURED" },
      "correlated_failure": { "claim": "...", "tag": "REASONING" },
      "provenance": { "tier": "MEASURED", "scope": "toolchain-not-silicon", "caveat": "..." }
    }
  }
}
```

## Incoming data, mapped (as received on the bus, 2026-09-15)

### `boards_measured["iq9"]` — from [qualcomm], SA8775P Hexagon v73 HTP, MEASURED
- perf (⚠ **CORRECTED 2026-09-15** — the 00:58 figures were a stale MEMORY line; qualcomm
  freshly re-enumerated on silicon and they moved ~3×. Use these):
  `qwen2.5-7b decoder / w4a16` (text prompt, genie-t2t-run --profile on qwen25vl_bundle) →
  **prefill 596.3 t/s · TTFT 287 ms @170 tok · decode 9.45 t/s** [MEASURED 2026-09-15].
  (Superseded: prefill 197.5 / TTFT 157 / decode 8.375 — stale memory summary, possibly
  vision-prefill context; provenance unpinned, do not use for a text/RAG sizer.)
  `qwen3-vl-4b / w4a16` → prefill 1003, decode 15.75 — **perf-only, ACCURACY-INVALID** (vendor
  harness zeroes deepstack); tag perf-only or omit. `qwen3.5-2b` hybrid → NPU **~3× slower than
  the A78 CPU** (a "NPU is not always the win" datum).
- ⛔ **HARD CONSTRAINT (context cap):** the deployed iq9 NPU Genie bundle **caps a single query at
  ~256 input tokens** — prompts >256 fail to query on the NPU and fall back to CPU (llama.cpp) at
  **~19 t/s, 108 s TTFT @2000 tok** [MEASURED 2026-09-15]. So iq9 NPU prefill is 596 t/s **only up
  to 256 tokens**; this cap is the binding constraint for any long-context/RAG sizing on iq9 and
  the sizer MUST encode it (a feasibility-style gate on input length, not just a cost curve).
- precision_facts: INT4 decode ~1.3× + **6–8% TTFT dequant tax** (Thor +5–9%, 5090 flat);
  **decode is DDR-bandwidth-bound** (3 ways); **dual-NSP = 1.94× aggregate throughput, 0% single-
  policy decode latency**; **bare int8 MatMul 1.14–1.23× *slower* than fp16 on HTP** (HMX wants
  conv-shaped ops, not GEMM — relevant if the sizer assumes int8 > fp16 for transformers).
- npu_capability: 2× NSP; 8 MB VTCM **private per NSP** (can't pool); datatypes w4a16/w8a16/fp16/
  int8; fp8 = v79+ only; mxfp4/fp4 rejected [SOURCED, on-silicon capability query].
- gaps: **decode-tps-vs-context-length = NOT RUN** (gated on VSLAM push) — the right data for the
  new context-length axis; flag as GAP, not a number.
- provenance: canonical writeup in qualcomm memory `iq9-npu-llm-genie-deploy`; raw JSON under
  `results/bench_data/remeasure/qwen25vl*` (ping qualcomm for exact paths/hashes to cite).

### `feasibility_gates["imx95__eIQ-Neutron__3.2.0"]` — from [95emulator], MEASURED (toolchain)
- Delivered file: **`docs/imx95-feasibility-gate.json`** in the 95emulator tree, schema
  `imx95/feasibility-gate/v1`. 41 model entries (26 placeable / **15 unplaceable** / 41 — count
  corrected from the earlier "11/37"), each: family, skip-topology (none / concat / residual-add /
  residual-add+SE), outcome, failing kernel, fused-op count.
- gating_ops (both MEASURED): **NeutronAdd absent** (elementwise-add across two quant scales) →
  excludes every residual arch; **NeutronMulSymByAsym faults** → excludes SE-block families. The
  control is in the data: micro elementwise-MUL places bit-exact while micro elementwise-ADD
  SIGILLs — isolates the *op*, not the model.
- version_stability: **MEASURED** identical on eIQ 3.1.3 AND 3.2.0 (the key can be coarser than
  every point release). correlated_failure: **REASONING, not measured** (shared-backbone agents
  fail placement together) — keep the tag if it reaches a slide.
- provenance: emulated NPU path in QEMU vs vendor golden kernels — a **toolchain claim, not
  silicon**; caveat field carries "one board, one toolchain, do not extrapolate".

### `boards_measured["orin"]` / `["rtx5090"]` — from us ([docs]), MEASURED
- Orin: `eval/results/orin/GGUF_LADDER.md` + `perf_skippy-7b-v4.json` (skippy-7b-v4 Q4 = 27.82
  tps / 121.6 GB/s; 14B Q4 13.76; MoE 43.74). 5090: the source-of-truth workload bake-offs
  (`scripts/bench_skippy.py` telemetry) — the 7B v4 5090 bake-off is the one GPU-gated item still
  to run (M-A of the ladder roadmap).

## Open coordination items for [pai-sizer]
1. Do you want `boards_measured` as a sibling of `tiers_measured`, or should real boards
   **replace** the placeholder tiers? (I lean: keep `tiers_measured` for abstract sizing presets,
   add `boards_measured` for the real silicon anchors they're calibrated against.)
2. `feasibility_gates`: embed 95emulator's JSON, or `$ref` it into the pai-sizer repo? Either
   way your loader needs a **placeable/unplaceable gate that runs before the roofline** — the
   engine change, not just data.
3. int8-slower-than-fp16 on HTP and mxfp4/fp4-rejected both contradict common "smaller dtype =
   faster/available" assumptions in a sizer — confirm ratchet models these per-(board, op-shape),
   not globally.

## Related
- Ladder roadmap: `docs/porting-roadmap.md` (this bundle is the measured backbone it feeds).
- v1 generator: `eval/build_sizer_bundle.py` (7B v4 added to MODEL_CONSTANTS 2026-09-15).
