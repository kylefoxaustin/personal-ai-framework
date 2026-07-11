# Accuracy comparison: `candidate-kyle-qwen25-32b-v1-v2-rag` vs `candidate-kyle-qwen25-32b-v4-clean-v2-rag`

- Reference: candidate-kyle-qwen25-32b-v1-v2-rag — 2026-05-06T22:25:36
- Candidate: candidate-kyle-qwen25-32b-v4-clean-v2-rag — 2026-05-07T02:55:44
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 63.6% → 63.6% (Δ = +0.0 pp)
- **Total passes:** 84 → 84 (Δ = +0)
- **Median Jaccard overlap** (word-level): 0.429 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.938 | 0.334 |
| general | 2 | +0 | 0.382 | 0.0 |
| multihop | 3 | -3 | 0.328 | 0.0 |
| numerical_precision | 2 | +0 | 0.506 | 0.0 |
| persona | 2 | +0 | 0.272 | 0.0 |
| rag_blog | 1 | +0 | 0.5 | 0.0 |
| rag_datasheet | 26 | +3 | 0.417 | 0.0 |
| rag_email | 1 | +0 | 0.373 | 0.0 |
| reasoning | 2 | +0 | 0.777 | 0.5 |
| refusal | 3 | +0 | 0.1 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.077 | 0.0 | 0.301 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.097 | 0.0 | 0.222 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.1 | 0.0 | 0.402 |
| `refusal_out_of_scope` | refusal | +0 | 0.1 | 0.0 | 0.2 |
| `persona_skippy_voice` | persona | +0 | 0.115 | 0.0 | 0.67 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.143 | 0.0 | 0.191 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.2 | 0.0 | 0.287 |
| `numprec_mcu_boot_rom` | numerical_precision | +0 | 0.217 | 0.0 | 0.272 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +0 | 0.217 | 0.0 | 0.266 |
| `rag_ds_src_gpr4_offset` | rag_datasheet | +0 | 0.238 | 0.0 | 0.275 |