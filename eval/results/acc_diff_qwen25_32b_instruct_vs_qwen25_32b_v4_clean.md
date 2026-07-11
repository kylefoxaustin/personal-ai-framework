# Accuracy comparison: `baseline-qwen25-32b-instruct-v2-rag` vs `candidate-kyle-qwen25-32b-v4-clean-v2-rag`

- Reference: baseline-qwen25-32b-instruct-v2-rag — 2026-05-07T12:35:52
- Candidate: candidate-kyle-qwen25-32b-v4-clean-v2-rag — 2026-05-07T02:55:44
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 63.6% (Δ = -4.6 pp)
- **Total passes:** 90 → 84 (Δ = -6)
- **Median Jaccard overlap** (word-level): 0.396 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.938 | 0.5 |
| general | 2 | +0 | 0.269 | 0.0 |
| multihop | 3 | -3 | 0.22 | 0.0 |
| numerical_precision | 2 | -3 | 0.394 | 0.0 |
| persona | 2 | +0 | 0.335 | 0.0 |
| rag_blog | 1 | +0 | 0.103 | 0.0 |
| rag_datasheet | 26 | -3 | 0.458 | 0.0 |
| rag_email | 1 | +0 | 0.36 | 0.0 |
| reasoning | 2 | +0 | 0.772 | 0.5 |
| refusal | 3 | +3 | 0.037 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +0 | 0.0 | 0.0 | 1.051 |
| `refusal_made_up_peripheral` | refusal | +3 | 0.0 | 0.0 | 4.556 |
| `refusal_out_of_scope` | refusal | +0 | 0.037 | 0.0 | 0.1 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.042 | 0.0 | 0.107 |
| `rag_ds_src_gpr5_offset` | rag_datasheet | -3 | 0.048 | 0.0 | 0.519 |
| `numprec_mcu_boot_rom` | numerical_precision | -3 | 0.05 | 0.0 | 0.08 |
| `rag_ds_src_gpr4_offset` | rag_datasheet | -3 | 0.053 | 0.0 | 0.577 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.07 | 0.0 | 0.083 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.071 | 0.0 | 0.095 |
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.082 | 0.0 | 0.111 |