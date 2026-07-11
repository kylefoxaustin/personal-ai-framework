# Accuracy comparison: `candidate-moe-thinking-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-v4-v2-rag`

- Reference: candidate-moe-thinking-v2-rag — 2026-04-24T09:35:25
- Candidate: candidate-kyle-qwen3-30b-a3b-v4-v2-rag — 2026-05-04T16:15:54
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 63.6% → 61.4% (Δ = -2.2 pp)
- **Total passes:** 84 → 81 (Δ = -3)
- **Median Jaccard overlap** (word-level): 0.115 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +1 | 0.111 | 0.0 |
| general | 2 | -3 | 0.037 | 0.0 |
| multihop | 3 | -6 | 0.052 | 0.0 |
| numerical_precision | 2 | -3 | 0.097 | 0.0 |
| persona | 2 | +0 | 0.075 | 0.0 |
| rag_blog | 1 | +0 | 0.108 | 0.0 |
| rag_datasheet | 26 | +5 | 0.139 | 0.0 |
| rag_email | 1 | +3 | 0.039 | 0.0 |
| reasoning | 2 | +0 | 0.103 | 0.0 |
| refusal | 3 | +0 | 0.045 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.006 | 0.0 | 0.021 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.015 | 0.0 | 0.024 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.015 | 0.0 | 0.018 |
| `reason_math_fraction` | reasoning | +0 | 0.017 | 0.0 | 0.004 |
| `persona_skippy_voice` | persona | +0 | 0.02 | 0.0 | 0.032 |
| `multihop_imx93_vs_95_cores` | multihop | -3 | 0.028 | 0.0 | 0.021 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.029 | 0.0 | 0.024 |
| `general_embedded_book` | general | +0 | 0.031 | 0.0 | 0.026 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.031 | 0.0 | 0.025 |
| `rag_angry_birds_demo` | rag_email | +3 | 0.039 | 0.0 | 0.1 |