# Accuracy comparison: `reference-dense-fp16-v1-vllm` vs `candidate-dense-fp8-v1`

- Reference: reference-dense-fp16-v1-vllm — 2026-04-23T10:56:24
- Candidate: candidate-dense-fp8-v1 — 2026-04-23T10:50:45
- Prompt set: v1 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 58.3% → 58.3% (Δ = +0.0 pp)
- **Total passes:** 21 → 21 (Δ = +0)
- **Median Jaccard overlap** (word-level): 0.756 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.38 | 0.0 |
| persona | 1 | +0 | 0.788 | 0.0 |
| rag_blog | 1 | +0 | 0.724 | 0.0 |
| rag_datasheet | 3 | +0 | 0.431 | 0.0 |
| rag_email | 1 | +0 | 1.0 | 1.0 |
| reasoning | 2 | +0 | 0.812 | 0.5 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_imx93_cortex` | rag_datasheet | +0 | 0.2 | 0.0 | 2.951 |
| `general_embedded_book` | general | +0 | 0.353 | 0.0 | 0.65 |
| `general_qwen_moe` | general | +0 | 0.408 | 0.0 | 0.908 |
| `rag_imx93_uart` | rag_datasheet | +0 | 0.431 | 0.0 | 1.088 |
| `reason_multistep` | reasoning | +0 | 0.624 | 0.0 | 1.355 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.724 | 0.0 | 1.02 |
| `persona_skippy_voice` | persona | +0 | 0.788 | 0.0 | 0.991 |
| `rag_imx93_ddr_width` | rag_datasheet | +0 | 0.8 | 0.0 | 0.983 |
| `code_fibonacci` | coding | +0 | 1.0 | 1.0 | 1.0 |
| `code_reverse_string` | coding | +0 | 1.0 | 1.0 | 1.0 |