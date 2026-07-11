# Accuracy comparison: `candidate-kyle-qwen25-7b-v2-v2-rag` vs `candidate-kyle-qwen25-7b-v3-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v2-v2-rag — 2026-05-02T01:28:35
- Candidate: candidate-kyle-qwen25-7b-v3-v2-rag — 2026-05-02T16:35:14
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 71.2% → 58.3% (Δ = -12.9 pp)
- **Total passes:** 94 → 77 (Δ = -17)
- **Median Jaccard overlap** (word-level): 0.177 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.088 | 0.0 |
| multihop | 3 | +0 | 0.136 | 0.0 |
| numerical_precision | 2 | +1 | 0.371 | 0.0 |
| persona | 2 | +0 | 0.141 | 0.0 |
| rag_blog | 1 | +0 | 0.225 | 0.0 |
| rag_datasheet | 26 | -17 | 0.179 | 0.0 |
| rag_email | 1 | +0 | 0.161 | 0.0 |
| reasoning | 2 | -1 | 0.151 | 0.0 |
| refusal | 3 | +0 | 0.088 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.038 | 0.0 | 0.103 |
| `reason_math_fraction` | reasoning | +0 | 0.043 | 0.0 | 0.01 |
| `refusal_out_of_scope` | refusal | +0 | 0.048 | 0.0 | 0.045 |
| `general_qwen_moe` | general | +0 | 0.079 | 0.0 | 0.052 |
| `refusal_fictional_chip` | refusal | +0 | 0.088 | 0.0 | 0.055 |
| `general_embedded_book` | general | +0 | 0.098 | 0.0 | 0.056 |
| `rag_imx93_uart` | rag_datasheet | -3 | 0.103 | 0.0 | 0.139 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.11 | 0.0 | 0.058 |
| `rag_ds_vdd_ana_0p8_max_voltage` | rag_datasheet | -3 | 0.112 | 0.0 | 0.035 |
| `rag_ds_cortex_m33_max_freq` | rag_datasheet | +0 | 0.12 | 0.0 | 0.062 |