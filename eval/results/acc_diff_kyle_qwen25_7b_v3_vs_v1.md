# Accuracy comparison: `candidate-kyle-qwen25-7b-v1-v2-rag` vs `candidate-kyle-qwen25-7b-v3-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v1-v2-rag — 2026-05-01T23:27:46
- Candidate: candidate-kyle-qwen25-7b-v3-v2-rag — 2026-05-02T16:35:14
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 75.0% → 58.3% (Δ = -16.7 pp)
- **Total passes:** 99 → 77 (Δ = -22)
- **Median Jaccard overlap** (word-level): 0.135 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.054 | 0.0 |
| multihop | 3 | +0 | 0.149 | 0.0 |
| numerical_precision | 2 | -2 | 0.142 | 0.0 |
| persona | 2 | +0 | 0.059 | 0.0 |
| rag_blog | 1 | +0 | 0.061 | 0.0 |
| rag_datasheet | 26 | -21 | 0.161 | 0.0 |
| rag_email | 1 | +0 | 0.105 | 0.0 |
| reasoning | 2 | -2 | 0.137 | 0.0 |
| refusal | 3 | +3 | 0.084 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `reason_math_fraction` | reasoning | +0 | 0.013 | 0.0 | 0.001 |
| `general_qwen_moe` | general | +0 | 0.051 | 0.0 | 0.038 |
| `persona_brief_role` | persona | +0 | 0.054 | 0.0 | 0.12 |
| `general_embedded_book` | general | +0 | 0.057 | 0.0 | 0.051 |
| `refusal_made_up_peripheral` | refusal | +3 | 0.057 | 0.0 | 0.061 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.061 | 0.0 | 0.051 |
| `persona_skippy_voice` | persona | +0 | 0.064 | 0.0 | 0.05 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.071 | 0.0 | 0.056 |
| `refusal_out_of_scope` | refusal | +0 | 0.084 | 0.0 | 0.044 |
| `rag_ds_cortex_a55_l2_size` | rag_datasheet | -3 | 0.086 | 0.0 | 0.03 |