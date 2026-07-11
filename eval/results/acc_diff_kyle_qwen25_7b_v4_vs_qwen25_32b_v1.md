# Accuracy comparison: `candidate-kyle-qwen25-7b-v4-v2-rag` vs `candidate-kyle-qwen25-32b-v1-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Candidate: candidate-kyle-qwen25-32b-v1-v2-rag — 2026-05-06T22:25:36
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 70.5% → 63.6% (Δ = -6.9 pp)
- **Total passes:** 93 → 84 (Δ = -9)
- **Median Jaccard overlap** (word-level): 0.472 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.95 | 0.666 |
| general | 2 | +0 | 0.188 | 0.0 |
| multihop | 3 | +0 | 0.55 | 0.0 |
| numerical_precision | 2 | +0 | 0.514 | 0.0 |
| persona | 2 | +0 | 0.09 | 0.0 |
| rag_blog | 1 | +0 | 0.517 | 0.0 |
| rag_datasheet | 26 | -12 | 0.516 | 0.0 |
| rag_email | 1 | +0 | 0.333 | 0.0 |
| reasoning | 2 | +3 | 0.69 | 0.5 |
| refusal | 3 | +0 | 0.211 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `general_embedded_book` | general | +0 | 0.083 | 0.0 | 3.887 |
| `persona_brief_role` | persona | +0 | 0.086 | 0.0 | 1.453 |
| `persona_skippy_voice` | persona | +0 | 0.095 | 0.0 | 2.564 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.097 | 0.0 | 4.512 |
| `refusal_out_of_scope` | refusal | +0 | 0.1 | 0.0 | 5.0 |
| `rag_ds_package_pitch` | rag_datasheet | +0 | 0.208 | 0.0 | 1.093 |
| `rag_imx93_uart` | rag_datasheet | +0 | 0.21 | 0.0 | 5.025 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.211 | 0.0 | 0.388 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.25 | 0.0 | 0.495 |
| `refusal_fictional_chip` | refusal | +0 | 0.259 | 0.0 | 0.653 |