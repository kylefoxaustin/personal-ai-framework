# Accuracy comparison: `baseline-qwen3-30b-a3b-instruct-2507-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-full-v1-v2-rag`

- Reference: baseline-qwen3-30b-a3b-instruct-2507-v2-rag — 2026-05-04T17:00:41
- Candidate: candidate-kyle-qwen3-30b-a3b-full-v1-v2-rag — 2026-05-07T10:17:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 71.2% → 62.9% (Δ = -8.3 pp)
- **Total passes:** 94 → 83 (Δ = -11)
- **Median Jaccard overlap** (word-level): 0.365 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.295 | 0.0 |
| multihop | 3 | +0 | 0.229 | 0.0 |
| numerical_precision | 2 | +0 | 0.304 | 0.0 |
| persona | 2 | +0 | 0.187 | 0.0 |
| rag_blog | 1 | -3 | 0.022 | 0.0 |
| rag_datasheet | 26 | -8 | 0.459 | 0.0 |
| rag_email | 1 | +0 | 0.241 | 0.0 |
| reasoning | 2 | +0 | 0.69 | 0.5 |
| refusal | 3 | +0 | 0.088 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ev_sustainability` | rag_blog | -3 | 0.022 | 0.0 | 0.034 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.054 | 0.0 | 0.086 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.059 | 0.0 | 0.119 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.062 | 0.0 | 0.112 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.067 | 0.0 | 0.132 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.079 | 0.0 | 0.154 |
| `refusal_fictional_chip` | refusal | +0 | 0.088 | 0.0 | 0.146 |
| `rag_imx93_ddr_width` | rag_datasheet | +2 | 0.161 | 0.0 | 0.155 |
| `persona_skippy_voice` | persona | +0 | 0.174 | 0.0 | 1.879 |
| `general_embedded_book` | general | +0 | 0.176 | 0.0 | 0.299 |