# Accuracy comparison: `reference-moe-q4km-v2-rag` vs `candidate-moe-thinking-v2-rag`

- Reference: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Candidate: candidate-moe-thinking-v2-rag — 2026-04-24T09:35:25
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.9% → 63.6% (Δ = -5.3 pp)
- **Total passes:** 91 → 84 (Δ = -7)
- **Median Jaccard overlap** (word-level): 0.115 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | -1 | 0.111 | 0.0 |
| general | 2 | +0 | 0.161 | 0.0 |
| multihop | 3 | +0 | 0.11 | 0.0 |
| numerical_precision | 2 | +3 | 0.104 | 0.0 |
| persona | 2 | +0 | 0.072 | 0.0 |
| rag_blog | 1 | +0 | 0.101 | 0.0 |
| rag_datasheet | 26 | -8 | 0.144 | 0.0 |
| rag_email | 1 | -3 | 0.095 | 0.0 |
| reasoning | 2 | +0 | 0.2 | 0.0 |
| refusal | 3 | +2 | 0.045 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.014 | 0.0 | 57.833 |
| `reason_math_fraction` | reasoning | +0 | 0.017 | 0.0 | 228.5 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.029 | 0.0 | 41.875 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +3 | 0.031 | 0.0 | 40.325 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +0 | 0.031 | 0.0 | 48.439 |
| `refusal_fictional_chip` | refusal | +0 | 0.045 | 0.0 | 49.463 |
| `refusal_out_of_scope` | refusal | +0 | 0.045 | 0.0 | 50.293 |
| `refusal_made_up_peripheral` | refusal | +2 | 0.057 | 0.0 | 29.854 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.062 | 0.0 | 38.325 |
| `numprec_mcu_boot_rom` | numerical_precision | +3 | 0.067 | 0.0 | 40.244 |