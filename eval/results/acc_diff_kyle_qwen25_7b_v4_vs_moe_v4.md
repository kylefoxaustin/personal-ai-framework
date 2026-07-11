# Accuracy comparison: `candidate-kyle-qwen25-7b-v4-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-v4-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Candidate: candidate-kyle-qwen3-30b-a3b-v4-v2-rag — 2026-05-04T16:15:54
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 70.5% → 61.4% (Δ = -9.1 pp)
- **Total passes:** 93 → 81 (Δ = -12)
- **Median Jaccard overlap** (word-level): 0.526 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | -3 | 0.179 | 0.0 |
| multihop | 3 | -6 | 0.056 | 0.0 |
| numerical_precision | 2 | +0 | 0.296 | 0.0 |
| persona | 2 | +0 | 0.218 | 0.0 |
| rag_blog | 1 | +0 | 0.11 | 0.0 |
| rag_datasheet | 26 | -6 | 0.662 | 0.0 |
| rag_email | 1 | +0 | 0.75 | 0.0 |
| reasoning | 2 | +3 | 0.637 | 0.5 |
| refusal | 3 | +0 | 0.13 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.023 | 0.0 | 0.145 |
| `general_qwen_moe` | general | -3 | 0.024 | 0.0 | 0.133 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.029 | 0.0 | 0.156 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.339 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.35 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.042 | 0.0 | 0.408 |
| `multihop_peripheral_count` | multihop | -3 | 0.048 | 0.0 | 0.41 |
| `multihop_imx93_vs_95_cores` | multihop | -3 | 0.056 | 0.0 | 0.494 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.11 | 0.0 | 3.975 |
| `numprec_mcu_boot_rom` | numerical_precision | +0 | 0.111 | 0.0 | 0.477 |