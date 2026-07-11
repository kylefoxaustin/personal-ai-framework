# Accuracy comparison: `reference-dense-fp16-v2-vllm` vs `candidate-dense-fp8-v2`

- Reference: reference-dense-fp16-v2-vllm — 2026-04-23T11:34:08
- Candidate: candidate-dense-fp8-v2 — 2026-04-23T11:31:03
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 30.3% → 31.8% (Δ = +1.5 pp)
- **Total passes:** 40 → 42 (Δ = +2)
- **Median Jaccard overlap** (word-level): 0.841 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.433 | 0.0 |
| multihop | 3 | +0 | 0.915 | 0.0 |
| numerical_precision | 2 | +0 | 0.989 | 0.5 |
| persona | 2 | +0 | 0.591 | 0.5 |
| rag_blog | 1 | +0 | 0.667 | 0.0 |
| rag_datasheet | 26 | -1 | 0.814 | 0.0 |
| rag_email | 1 | +0 | 1.0 | 1.0 |
| reasoning | 2 | +0 | 0.889 | 0.5 |
| refusal | 3 | +3 | 0.812 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `refusal_fictional_chip` | refusal | +3 | 0.154 | 0.0 | 3.143 |
| `persona_skippy_voice` | persona | +0 | 0.182 | 0.0 | 0.897 |
| `rag_ds_src_gpr5_offset` | rag_datasheet | +0 | 0.244 | 0.0 | 4.563 |
| `rag_imx93_cortex` | rag_datasheet | +0 | 0.265 | 0.0 | 3.22 |
| `general_embedded_book` | general | +0 | 0.301 | 0.0 | 1.311 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +0 | 0.35 | 0.0 | 3.697 |
| `rag_ds_cortex_a55_l2_size` | rag_datasheet | +0 | 0.353 | 0.0 | 1.226 |
| `rag_ds_ecc_curve` | rag_datasheet | -1 | 0.455 | 0.0 | 0.863 |
| `rag_imx93_ddr_width` | rag_datasheet | +0 | 0.486 | 0.0 | 2.376 |
| `rag_ds_mipi_dsi_max_pixel_clock` | rag_datasheet | +0 | 0.491 | 0.333 | 1.092 |