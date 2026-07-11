# Accuracy comparison: `reference-moe-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v4-v2-rag`

- Reference: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Candidate: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.9% → 70.5% (Δ = +1.6 pp)
- **Total passes:** 91 → 93 (Δ = +2)
- **Median Jaccard overlap** (word-level): 0.474 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.18 | 0.0 |
| multihop | 3 | +0 | 0.429 | 0.0 |
| numerical_precision | 2 | +0 | 0.37 | 0.0 |
| persona | 2 | +0 | 0.202 | 0.0 |
| rag_blog | 1 | +0 | 0.103 | 0.0 |
| rag_datasheet | 26 | +3 | 0.647 | 0.0 |
| rag_email | 1 | +0 | 0.404 | 0.0 |
| reasoning | 2 | -3 | 0.7 | 0.5 |
| refusal | 3 | +2 | 0.158 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 3.025 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 2.925 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +3 | 0.042 | 0.0 | 2.45 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.059 | 0.0 | 0.788 |
| `persona_skippy_voice` | persona | +0 | 0.1 | 0.0 | 1.3 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.103 | 0.0 | 0.387 |
| `general_embedded_book` | general | +0 | 0.107 | 0.0 | 0.241 |
| `numprec_mcu_boot_rom` | numerical_precision | +0 | 0.111 | 0.0 | 2.098 |
| `refusal_fictional_chip` | refusal | +0 | 0.13 | 0.0 | 2.951 |
| `refusal_made_up_peripheral` | refusal | +2 | 0.158 | 0.0 | 3.207 |