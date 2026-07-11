# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `reference-moe-q4km-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 68.9% (Δ = +0.7 pp)
- **Total passes:** 90 → 91 (Δ = +1)
- **Median Jaccard overlap** (word-level): 0.402 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.336 | 0.0 |
| multihop | 3 | +0 | 0.263 | 0.0 |
| numerical_precision | 2 | +0 | 0.383 | 0.0 |
| persona | 2 | +0 | 0.0 | 0.0 |
| rag_blog | 1 | +0 | 0.329 | 0.0 |
| rag_datasheet | 26 | +3 | 0.585 | 0.0 |
| rag_email | 1 | +0 | 0.309 | 0.0 |
| reasoning | 2 | +0 | 0.674 | 0.5 |
| refusal | 3 | -2 | 0.192 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 0.867 |
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 0.417 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.342 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.089 | 0.0 | 0.091 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.143 | 0.0 | 0.145 |
| `refusal_fictional_chip` | refusal | +0 | 0.172 | 0.0 | 0.15 |
| `refusal_out_of_scope` | refusal | +0 | 0.192 | 0.0 | 0.23 |
| `multihop_imx93_vs_95_cores` | multihop | +0 | 0.2 | 0.0 | 0.246 |
| `general_embedded_book` | general | +0 | 0.233 | 0.0 | 0.57 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.25 | 0.0 | 0.244 |