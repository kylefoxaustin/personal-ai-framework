# Accuracy comparison: `reference-fp16-v2-rag-pure` vs `candidate-fp8-v2-rag-pure`

- Reference: reference-fp16-v2-rag-pure — 2026-04-23T18:43:01
- Candidate: candidate-fp8-v2-rag-pure — 2026-04-23T18:53:55
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 63.6% → 63.6% (Δ = +0.0 pp)
- **Total passes:** 84 → 84 (Δ = +0)
- **Median Jaccard overlap** (word-level): 0.851 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.536 | 0.0 |
| multihop | 3 | +0 | 0.674 | 0.0 |
| numerical_precision | 2 | +0 | 0.834 | 0.334 |
| persona | 2 | +0 | 0.322 | 0.0 |
| rag_blog | 1 | +0 | 0.3 | 0.0 |
| rag_datasheet | 26 | +0 | 0.993 | 0.5 |
| rag_email | 1 | +0 | 0.759 | 0.0 |
| reasoning | 2 | +0 | 0.867 | 0.5 |
| refusal | 3 | +0 | 0.93 | 0.333 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.214 | 0.0 | 1.308 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.3 | 0.0 | 0.865 |
| `general_embedded_book` | general | +0 | 0.304 | 0.0 | 1.249 |
| `persona_brief_role` | persona | +0 | 0.429 | 0.0 | 0.925 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.467 | 0.0 | 1.322 |
| `multihop_peripheral_count` | multihop | +0 | 0.505 | 0.0 | 1.316 |
| `rag_ds_usb_host_reset_wait` | rag_datasheet | +0 | 0.529 | 0.0 | 1.203 |
| `rag_imx93_cortex` | rag_datasheet | +0 | 0.579 | 0.333 | 1.281 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +0 | 0.605 | 0.0 | 0.939 |
| `rag_imx93_uart` | rag_datasheet | +0 | 0.62 | 0.0 | 1.29 |