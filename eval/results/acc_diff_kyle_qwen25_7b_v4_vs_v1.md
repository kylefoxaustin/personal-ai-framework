# Accuracy comparison: `candidate-kyle-qwen25-7b-v1-v2-rag` vs `candidate-kyle-qwen25-7b-v4-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v1-v2-rag — 2026-05-01T23:27:46
- Candidate: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 75.0% → 70.5% (Δ = -4.5 pp)
- **Total passes:** 99 → 93 (Δ = -6)
- **Median Jaccard overlap** (word-level): 0.134 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.128 | 0.0 |
| multihop | 3 | +0 | 0.167 | 0.0 |
| numerical_precision | 2 | -3 | 0.132 | 0.0 |
| persona | 2 | +0 | 0.083 | 0.0 |
| rag_blog | 1 | +0 | 0.056 | 0.0 |
| rag_datasheet | 26 | -4 | 0.134 | 0.0 |
| rag_email | 1 | +0 | 0.105 | 0.0 |
| reasoning | 2 | -2 | 0.175 | 0.0 |
| refusal | 3 | +3 | 0.13 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.012 | 0.0 | 0.017 |
| `reason_math_fraction` | reasoning | +0 | 0.013 | 0.0 | 0.001 |
| `persona_skippy_voice` | persona | +0 | 0.017 | 0.0 | 0.019 |
| `refusal_out_of_scope` | refusal | +0 | 0.047 | 0.0 | 0.019 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.056 | 0.0 | 0.055 |
| `general_embedded_book` | general | +0 | 0.058 | 0.0 | 0.044 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.086 | 0.0 | 0.033 |
| `rag_imx93_ddr_width` | rag_datasheet | -1 | 0.087 | 0.0 | 0.053 |
| `rag_ds_cortex_a55_l2_size` | rag_datasheet | -3 | 0.093 | 0.0 | 0.032 |
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.099 | 0.0 | 0.037 |