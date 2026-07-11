# Accuracy comparison: `baseline-qwen25-7b-base-v2-rag` vs `candidate-kyle-qwen25-7b-v3-v2-rag`

- Reference: baseline-qwen25-7b-base-v2-rag — 2026-05-01T21:18:35
- Candidate: candidate-kyle-qwen25-7b-v3-v2-rag — 2026-05-02T16:35:14
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 67.4% → 58.3% (Δ = -9.1 pp)
- **Total passes:** 89 → 77 (Δ = -12)
- **Median Jaccard overlap** (word-level): 0.360 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.093 | 0.0 |
| multihop | 3 | +1 | 0.204 | 0.0 |
| numerical_precision | 2 | +1 | 0.413 | 0.0 |
| persona | 2 | +0 | 0.059 | 0.0 |
| rag_blog | 1 | +0 | 0.146 | 0.0 |
| rag_datasheet | 26 | -14 | 0.5 | 0.0 |
| rag_email | 1 | +3 | 0.246 | 0.0 |
| reasoning | 2 | -3 | 0.626 | 0.5 |
| refusal | 3 | +0 | 0.233 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.034 | 0.0 | 1.146 |
| `general_qwen_moe` | general | +0 | 0.071 | 0.0 | 0.282 |
| `persona_brief_role` | persona | +0 | 0.083 | 0.0 | 1.253 |
| `general_embedded_book` | general | +0 | 0.116 | 0.0 | 0.159 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | -3 | 0.14 | 0.0 | 0.293 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.146 | 0.0 | 0.153 |
| `rag_imx93_cortex` | rag_datasheet | -3 | 0.146 | 0.0 | 0.302 |
| `refusal_fictional_chip` | refusal | +0 | 0.155 | 0.0 | 0.188 |
| `rag_imx93_ddr_width` | rag_datasheet | +2 | 0.164 | 0.0 | 0.979 |
| `multihop_peripheral_count` | multihop | +1 | 0.2 | 0.0 | 0.329 |