# Accuracy comparison: `candidate-kyle-qwen25-14b-v1-v2-rag` vs `candidate-kyle-qwen25-32b-v4-clean-v2-rag`

- Reference: candidate-kyle-qwen25-14b-v1-v2-rag — 2026-05-02T20:14:00
- Candidate: candidate-kyle-qwen25-32b-v4-clean-v2-rag — 2026-05-07T02:55:44
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 72.7% → 63.6% (Δ = -9.1 pp)
- **Total passes:** 96 → 84 (Δ = -12)
- **Median Jaccard overlap** (word-level): 0.544 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.362 | 0.0 |
| multihop | 3 | -3 | 0.41 | 0.0 |
| numerical_precision | 2 | -3 | 0.369 | 0.0 |
| persona | 2 | +0 | 0.144 | 0.0 |
| rag_blog | 1 | +0 | 0.075 | 0.0 |
| rag_datasheet | 26 | -12 | 0.677 | 0.0 |
| rag_email | 1 | +3 | 0.214 | 0.0 |
| reasoning | 2 | +0 | 0.663 | 0.5 |
| refusal | 3 | +3 | 0.273 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.339 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 0.35 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.042 | 0.0 | 0.418 |
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.05 | 0.0 | 0.466 |
| `rag_ds_src_gpr4_offset` | rag_datasheet | -3 | 0.053 | 0.0 | 0.594 |
| `rag_ds_src_gpr5_offset` | rag_datasheet | -3 | 0.053 | 0.0 | 0.594 |
| `refusal_made_up_peripheral` | refusal | +3 | 0.053 | 0.0 | 0.494 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.056 | 0.0 | 0.661 |
| `persona_skippy_voice` | persona | +0 | 0.059 | 0.0 | 2.233 |
| `numprec_mcu_boot_rom` | numerical_precision | -3 | 0.062 | 0.0 | 0.82 |