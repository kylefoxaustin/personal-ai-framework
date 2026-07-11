# Accuracy comparison: `reference-moe-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v3-v2-rag`

- Reference: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Candidate: candidate-kyle-qwen25-7b-v3-v2-rag — 2026-05-02T16:35:14
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.9% → 58.3% (Δ = -10.6 pp)
- **Total passes:** 91 → 77 (Δ = -14)
- **Median Jaccard overlap** (word-level): 0.458 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.096 | 0.0 |
| multihop | 3 | +0 | 0.524 | 0.0 |
| numerical_precision | 2 | +1 | 0.28 | 0.0 |
| persona | 2 | +0 | 0.024 | 0.0 |
| rag_blog | 1 | +0 | 0.119 | 0.0 |
| rag_datasheet | 26 | -14 | 0.571 | 0.0 |
| rag_email | 1 | +0 | 0.404 | 0.0 |
| reasoning | 2 | -3 | 0.658 | 0.5 |
| refusal | 3 | +2 | 0.105 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 3.4 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.02 | 0.0 | 7.925 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.026 | 0.0 | 6.425 |
| `persona_brief_role` | persona | +0 | 0.047 | 0.0 | 2.958 |
| `general_qwen_moe` | general | +0 | 0.073 | 0.0 | 0.292 |
| `refusal_fictional_chip` | refusal | +0 | 0.105 | 0.0 | 2.122 |
| `refusal_out_of_scope` | refusal | +0 | 0.105 | 0.0 | 2.341 |
| `general_embedded_book` | general | +0 | 0.119 | 0.0 | 0.278 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.119 | 0.0 | 0.358 |
| `numprec_mcu_boot_rom` | numerical_precision | +1 | 0.136 | 0.0 | 2.61 |