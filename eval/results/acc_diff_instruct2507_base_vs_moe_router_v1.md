# Accuracy comparison: `baseline-qwen3-30b-a3b-instruct-2507-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag`

- Reference: baseline-qwen3-30b-a3b-instruct-2507-v2-rag — 2026-05-04T17:00:41
- Candidate: candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag — 2026-05-06T19:45:12
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 71.2% → 67.4% (Δ = -3.8 pp)
- **Total passes:** 94 → 89 (Δ = -5)
- **Median Jaccard overlap** (word-level): 0.339 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.108 | 0.0 |
| multihop | 3 | +0 | 0.255 | 0.0 |
| numerical_precision | 2 | +0 | 0.316 | 0.0 |
| persona | 2 | +0 | 0.144 | 0.0 |
| rag_blog | 1 | +0 | 0.074 | 0.0 |
| rag_datasheet | 26 | -4 | 0.451 | 0.0 |
| rag_email | 1 | -1 | 0.235 | 0.0 |
| reasoning | 2 | +0 | 0.713 | 0.5 |
| refusal | 3 | +0 | 0.088 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `refusal_made_up_peripheral` | refusal | +0 | 0.054 | 0.0 | 0.086 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.059 | 0.0 | 0.119 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.067 | 0.0 | 0.132 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.074 | 0.0 | 0.081 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.079 | 0.0 | 0.154 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.079 | 0.0 | 0.156 |
| `refusal_fictional_chip` | refusal | +0 | 0.088 | 0.0 | 0.146 |
| `general_embedded_book` | general | +0 | 0.098 | 0.0 | 0.174 |
| `persona_brief_role` | persona | +0 | 0.115 | 0.0 | 0.894 |
| `general_qwen_moe` | general | +0 | 0.119 | 0.0 | 0.192 |