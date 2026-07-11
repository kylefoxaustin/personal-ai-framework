# Accuracy comparison: `candidate-kyle-qwen25-14b-v1-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-v4-v2-rag`

- Reference: candidate-kyle-qwen25-14b-v1-v2-rag — 2026-05-02T20:14:00
- Candidate: candidate-kyle-qwen3-30b-a3b-v4-v2-rag — 2026-05-04T16:15:54
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 72.7% → 61.4% (Δ = -11.3 pp)
- **Total passes:** 96 → 81 (Δ = -15)
- **Median Jaccard overlap** (word-level): 0.520 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | -3 | 0.058 | 0.0 |
| multihop | 3 | -6 | 0.056 | 0.0 |
| numerical_precision | 2 | -3 | 0.3 | 0.0 |
| persona | 2 | +0 | 0.24 | 0.0 |
| rag_blog | 1 | +0 | 0.125 | 0.0 |
| rag_datasheet | 26 | -9 | 0.627 | 0.0 |
| rag_email | 1 | +3 | 0.312 | 0.0 |
| reasoning | 2 | +0 | 0.641 | 0.5 |
| refusal | 3 | +3 | 0.071 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 1.867 |
| `general_qwen_moe` | general | -3 | 0.021 | 0.0 | 0.111 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +0 | 0.03 | 0.0 | 0.237 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.339 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.35 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.042 | 0.0 | 0.408 |
| `multihop_peripheral_count` | multihop | -3 | 0.048 | 0.0 | 0.477 |
| `refusal_made_up_peripheral` | refusal | +3 | 0.053 | 0.0 | 0.494 |
| `multihop_imx93_vs_95_cores` | multihop | -3 | 0.056 | 0.0 | 0.451 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.056 | 0.0 | 0.661 |