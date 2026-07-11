# Skippy Quantization vs Quality — Summary for NPU Team

**TL;DR** — we measured Skippy's pass rate on a {N_PROMPTS}-prompt
eval (mix of RAG-over-datasheets, reasoning, coding, refusal, multi-hop,
numerical-precision, persona, long-form knowledge) at three precision
levels. Findings below.

## What we measured

- **Prompt set**: `eval/prompts_v2.json` — {N_PROMPTS} prompts across
  {N_CATEGORIES} categories, each sampled {N_SAMPLES}× at temperature=0.
- **Scoring**: gold-substring match (`all` mode) or any-of refusal phrases
  (`any` mode for refusal category). Deterministic noise floor = 0.0
  (two identical fp16 runs produce byte-identical outputs).
- **Skippy stack**: 6,731-doc ChromaDB (datasheets + blogs + transcripts),
  hybrid BM25+semantic retrieval, standard system prompt. RAG enabled
  for `v2+RAG` runs, off for `v1` LLM-only runs.
- **Hardware**: RTX 5090 (reference silicon for the bandwidth math).

## Precision axis — the three variants

| Variant | Weight bits | Compute dtype | Simulates what silicon? |
|---|---|---|---|
| **Q4_K_M** | 4-bit (k-means) | fp16 | fp16-capable NPU / GPU (Ampere+, Mid/High tier) |
| **Q8_0** | 8-bit (per-tensor) | fp16 | same fp16-capable NPU, more weight precision |
| **W8A8** | 8-bit | **int8** | **pure int8 NPU** (NXP Neutron class, low-tier) |

Q4 vs Q8 isolates **weight precision**. Q8 vs W8A8 isolates **activation
precision / compute dtype** — which is what the int8-only silicon
architecture debate is actually about.

## Headline numbers (dense Qwen 2.5 14B Instruct, RAG enabled)

| Variant | Pass rate | Δ vs Q4_K_M |
|---|---|---|
| Q4_K_M (baseline — current Skippy) | {Q4_PASS}/{TOT} ({Q4_PCT}%) | — |
| Q8_0 | {Q8_PASS}/{TOT} ({Q8_PCT}%) | {Q8_DELTA} pp |
| **W8A8** (int8 compute) | **{W8A8_PASS}/{TOT} ({W8A8_PCT}%)** | **{W8A8_DELTA} pp** |

## By category — where does int8 compute hurt (if anywhere)

| Category | Q4_K_M | Q8_0 | W8A8 | Interpretation |
|---|---|---|---|---|
| `coding` | -/- | -/- | -/- | deterministic tasks — usually quant-insensitive |
| `reasoning` | -/- | -/- | -/- | multi-step arithmetic — quant-sensitive canary |
| `rag_datasheet` | -/- | -/- | -/- | fact extraction from retrieved context |
| `numerical_precision` | -/- | -/- | -/- | exact-number answers — activation quant risk |
| `multihop` | -/- | -/- | -/- | requires cross-chunk synthesis |
| `refusal` | -/- | -/- | -/- | hallucination vs "I don't know" |
| `rag_blog` / `rag_email` | -/- | -/- | -/- | open-form knowledge extraction |
| `persona` / `general` | -/- | -/- | -/- | style/voice — usually quant-insensitive |

## Failure-mode signatures (sample outputs)

### Break mode A: conservative refusal
> **Prompt:** "What memory interface does the i.MX 93 support for external DRAM?"
> - Q4_K_M: "The i.MX 93 supports a **16-bit DRAM controller**, specifically **LPDDR4X**..."
> - W8A8: "The retrieved excerpts do not cover the specific memory interface..."

### Break mode B: wrong-chunk hallucination
> **Prompt:** "Which peripheral on the i.MX 93 handles UART..."
> - Q4_K_M: "**LPUART**. IP block: **IOMUXC**."
> - W8A8: "**FlexIO** module handles UART..." ← wrong peripheral

### Break mode C: borderline reasoning drift
> - Q4_K_M: 3/3 correct on multi-step arithmetic
> - W8A8: 2/3 — one sample's greedy token path diverged

## What this means for the architecture decision

(To be filled once W8A8 data lands. Template for the verdict format:)

> **If W8A8 loses ≤2pp vs Q4_K_M and fails only in stylistic ways:** int8
> silicon is viable for Skippy. Architecture win on area/power, minimal
> quality cost. Recommend building the int8 engine.
>
> **If W8A8 loses 3-7pp concentrated in one or two categories:** int8 is
> viable for workloads that avoid those categories (e.g. fine for chat,
> not fine for datasheet Q&A). Architecture decision depends on Skippy's
> intended use split.
>
> **If W8A8 loses ≥8pp or shows catastrophic category collapse:** int8
> compute is the wrong target for this use case. fp16/bf16 tensor support
> matters enough to justify the silicon cost.

## Caveats / honest notes

- W8A8 was produced via llm-compressor (SmoothQuant + GPTQ), which is
  closer to production-grade INT8 quantization than naive per-tensor.
  Real silicon compilers (NXP eIQ, Qualcomm AIMET, etc.) may produce
  better or worse results depending on their calibration pipeline.
- Calibration used {CALIB_SAMPLES} samples from ultrachat. Skippy-specific
  calibration (using Kyle's training corpus) might improve W8A8 results
  on Skippy's own prompts at the cost of general capability.
- RAG retrieval itself is deterministic across variants, but the LLM-
  driven query rewrite step DOES vary between quants. Some of the
  observed degradation may be "different quant retrieved different chunks"
  rather than "different quant answered the same chunks differently."
  To fully separate these effects, a future experiment could lock the
  rewritten query and retrieved chunks identical across variants.
- Only dense 14B tested for W8A8 (vs Q4/Q8). MoE W8A8 quantization is
  a separate experiment — llm-compressor's MoE support is less mature
  and 30B-A3B calibration pushes RAM limits on the 5090 workstation.

## Regenerate these numbers

```bash
# Reference — current Skippy
python3 eval/run_accuracy_eval.py --name reference-dense-q4km-v2-rag \
  --prompts eval/prompts_v2.json --with-rag --samples 3

# Q8_0 candidate
# (swap config to models/qwen2.5-14b-q8/Qwen2.5-14B-Instruct-Q8_0.gguf, restart)
python3 eval/run_accuracy_eval.py --name candidate-dense-q8-v2-rag \
  --prompts eval/prompts_v2.json --with-rag --samples 3

# W8A8 candidate (HF path, no Skippy dependency — pure LLM comparison)
python3 eval/run_accuracy_eval_hf.py --model-path models/qwen2.5-14b-w8a8 \
  --name candidate-dense-w8a8 --prompts eval/prompts.json --samples 3

# Diff
python3 eval/compare_accuracy_runs.py \
  --reference eval/results/acc_reference-dense-q4km-v2-rag_*.json \
  --candidate eval/results/acc_candidate-dense-w8a8_*.json \
  --out eval/results/acc_diff_dense_q4km_vs_w8a8.md
```
