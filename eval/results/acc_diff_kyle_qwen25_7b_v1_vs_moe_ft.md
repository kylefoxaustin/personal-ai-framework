# Accuracy comparison: `reference-moe-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v1-v2-rag`

- Reference: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Candidate: candidate-kyle-qwen25-7b-v1-v2-rag — 2026-05-01T23:27:46
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.9% → 75.0% (Δ = +6.1 pp)
- **Total passes:** 91 → 99 (Δ = +8)
- **Median Jaccard overlap** (word-level): 0.119 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.165 | 0.0 |
| multihop | 3 | +0 | 0.151 | 0.0 |
| numerical_precision | 2 | +3 | 0.081 | 0.0 |
| persona | 2 | +0 | 0.054 | 0.0 |
| rag_blog | 1 | +0 | 0.154 | 0.0 |
| rag_datasheet | 26 | +7 | 0.121 | 0.0 |
| rag_email | 1 | +0 | 0.162 | 0.0 |
| reasoning | 2 | -1 | 0.211 | 0.0 |
| refusal | 3 | -1 | 0.047 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.007 | 0.0 | 48.075 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +3 | 0.007 | 0.0 | 48.975 |
| `persona_skippy_voice` | persona | +0 | 0.008 | 0.0 | 67.9 |
| `reason_math_fraction` | reasoning | +0 | 0.013 | 0.0 | 965.5 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +3 | 0.018 | 0.0 | 44.15 |
| `refusal_fictional_chip` | refusal | +0 | 0.019 | 0.0 | 58.244 |
| `refusal_out_of_scope` | refusal | +0 | 0.047 | 0.0 | 53.146 |
| `numprec_mcu_boot_rom` | numerical_precision | +3 | 0.048 | 0.0 | 52.61 |
| `refusal_made_up_peripheral` | refusal | -1 | 0.079 | 0.0 | 29.122 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.086 | 0.0 | 30.486 |