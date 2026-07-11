# Accuracy comparison: `baseline-qwen3-30b-a3b-instruct-2507-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-v4-v2-rag`

- Reference: baseline-qwen3-30b-a3b-instruct-2507-v2-rag — 2026-05-04T17:00:41
- Candidate: candidate-kyle-qwen3-30b-a3b-v4-v2-rag — 2026-05-04T16:15:54
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 71.2% → 61.4% (Δ = -9.8 pp)
- **Total passes:** 94 → 81 (Δ = -13)
- **Median Jaccard overlap** (word-level): 0.350 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | -3 | 0.058 | 0.0 |
| multihop | 3 | -6 | 0.061 | 0.0 |
| numerical_precision | 2 | +0 | 0.297 | 0.0 |
| persona | 2 | +0 | 0.192 | 0.0 |
| rag_blog | 1 | +0 | 0.176 | 0.0 |
| rag_datasheet | 26 | -4 | 0.465 | 0.0 |
| rag_email | 1 | +0 | 0.235 | 0.0 |
| reasoning | 2 | +0 | 0.604 | 0.5 |
| refusal | 3 | +0 | 0.088 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `general_qwen_moe` | general | -3 | 0.018 | 0.0 | 0.089 |
| `multihop_peripheral_count` | multihop | -3 | 0.053 | 0.0 | 0.125 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.054 | 0.0 | 0.086 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.059 | 0.0 | 0.116 |
| `multihop_imx93_vs_95_cores` | multihop | -3 | 0.061 | 0.0 | 0.12 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.062 | 0.0 | 0.112 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.067 | 0.0 | 0.132 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.079 | 0.0 | 0.154 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.079 | 0.0 | 0.156 |
| `refusal_fictional_chip` | refusal | +0 | 0.088 | 0.0 | 0.146 |