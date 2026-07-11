# Accuracy comparison: `candidate-kyle-qwen25-7b-v2-v2-rag` vs `candidate-kyle-qwen25-7b-v4-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v2-v2-rag — 2026-05-02T01:28:35
- Candidate: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 71.2% → 70.5% (Δ = -0.7 pp)
- **Total passes:** 94 → 93 (Δ = -1)
- **Median Jaccard overlap** (word-level): 0.177 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.158 | 0.0 |
| multihop | 3 | +0 | 0.182 | 0.0 |
| numerical_precision | 2 | +0 | 0.287 | 0.0 |
| persona | 2 | +0 | 0.065 | 0.0 |
| rag_blog | 1 | +0 | 0.25 | 0.0 |
| rag_datasheet | 26 | +0 | 0.178 | 0.0 |
| rag_email | 1 | +0 | 0.161 | 0.0 |
| reasoning | 2 | -1 | 0.192 | 0.0 |
| refusal | 3 | +0 | 0.149 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.026 | 0.0 | 0.21 |
| `refusal_out_of_scope` | refusal | +0 | 0.027 | 0.0 | 0.019 |
| `reason_math_fraction` | reasoning | +0 | 0.043 | 0.0 | 0.01 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.073 | 0.0 | 0.016 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.079 | 0.0 | 0.038 |
| `rag_ds_cortex_m33_max_freq` | rag_datasheet | +0 | 0.103 | 0.0 | 0.044 |
| `persona_brief_role` | persona | +0 | 0.104 | 0.0 | 0.051 |
| `rag_imx93_ddr_width` | rag_datasheet | -3 | 0.104 | 0.0 | 0.042 |
| `rag_ds_package_pitch` | rag_datasheet | +0 | 0.117 | 0.0 | 0.087 |
| `rag_ds_vdd_ana_0p8_max_voltage` | rag_datasheet | +0 | 0.121 | 0.0 | 0.035 |