# Accuracy comparison: `baseline-qwen25-7b-base-v2-rag` vs `candidate-kyle-qwen25-7b-v2-v2-rag`

- Reference: baseline-qwen25-7b-base-v2-rag — 2026-05-01T21:18:35
- Candidate: candidate-kyle-qwen25-7b-v2-v2-rag — 2026-05-02T01:28:35
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 67.4% → 71.2% (Δ = +3.8 pp)
- **Total passes:** 89 → 94 (Δ = +5)
- **Median Jaccard overlap** (word-level): 0.206 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.165 | 0.0 |
| multihop | 3 | +1 | 0.26 | 0.0 |
| numerical_precision | 2 | +0 | 0.29 | 0.0 |
| persona | 2 | +0 | 0.055 | 0.0 |
| rag_blog | 1 | +0 | 0.144 | 0.0 |
| rag_datasheet | 26 | +3 | 0.221 | 0.0 |
| rag_email | 1 | +3 | 0.146 | 0.0 |
| reasoning | 2 | -2 | 0.283 | 0.0 |
| refusal | 3 | +0 | 0.203 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `reason_math_fraction` | reasoning | +0 | 0.043 | 0.0 | 101.0 |
| `persona_skippy_voice` | persona | +0 | 0.047 | 0.0 | 2.09 |
| `persona_brief_role` | persona | +0 | 0.063 | 0.0 | 17.42 |
| `refusal_out_of_scope` | refusal | +0 | 0.068 | 0.0 | 9.507 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.11 | 0.0 | 18.806 |
| `rag_imx93_ddr_width` | rag_datasheet | +3 | 0.112 | 0.0 | 6.877 |
| `rag_ds_src_gpr4_offset` | rag_datasheet | +0 | 0.123 | 0.0 | 24.58 |
| `general_embedded_book` | general | +0 | 0.127 | 0.0 | 2.828 |
| `rag_ds_cortex_m33_max_freq` | rag_datasheet | +0 | 0.137 | 0.0 | 12.982 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.144 | 0.0 | 2.262 |