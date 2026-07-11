# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `candidate-dense-q8-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: candidate-dense-q8-v2-rag — 2026-04-23T09:20:43
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 62.9% (Δ = -5.3 pp)
- **Total passes:** 90 → 83 (Δ = -7)
- **Median Jaccard overlap** (word-level): 0.559 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.328 | 0.0 |
| multihop | 3 | +0 | 0.388 | 0.0 |
| numerical_precision | 2 | +0 | 0.602 | 0.0 |
| persona | 2 | +0 | 0.056 | 0.0 |
| rag_blog | 1 | +0 | 0.398 | 0.0 |
| rag_datasheet | 26 | -3 | 0.664 | 0.0 |
| rag_email | 1 | -3 | 0.242 | 0.0 |
| reasoning | 2 | -1 | 0.807 | 0.5 |
| refusal | 3 | +0 | 0.567 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 1.747 |
| `persona_skippy_voice` | persona | +0 | 0.111 | 0.0 | 0.806 |
| `rag_imx93_ddr_width` | rag_datasheet | -3 | 0.19 | 0.0 | 1.739 |
| `rag_angry_birds_demo` | rag_email | -3 | 0.242 | 0.0 | 0.477 |
| `rag_imx93_uart` | rag_datasheet | -3 | 0.25 | 0.0 | 1.368 |
| `general_qwen_moe` | general | +0 | 0.318 | 0.0 | 2.167 |
| `general_embedded_book` | general | +0 | 0.337 | 0.0 | 0.782 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.348 | 0.0 | 1.23 |
| `multihop_peripheral_count` | multihop | +0 | 0.375 | 0.0 | 0.524 |
| `refusal_fictional_chip` | refusal | +0 | 0.375 | 0.0 | 1.186 |