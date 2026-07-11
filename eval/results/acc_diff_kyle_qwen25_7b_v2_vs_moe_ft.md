# Accuracy comparison: `reference-moe-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v2-v2-rag`

- Reference: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Candidate: candidate-kyle-qwen25-7b-v2-v2-rag — 2026-05-02T01:28:35
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.9% → 71.2% (Δ = +2.3 pp)
- **Total passes:** 91 → 94 (Δ = +3)
- **Median Jaccard overlap** (word-level): 0.144 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.15 | 0.0 |
| multihop | 3 | +0 | 0.184 | 0.0 |
| numerical_precision | 2 | +0 | 0.186 | 0.0 |
| persona | 2 | +0 | 0.029 | 0.0 |
| rag_blog | 1 | +0 | 0.103 | 0.0 |
| rag_datasheet | 26 | +3 | 0.147 | 0.0 |
| rag_email | 1 | +0 | 0.207 | 0.0 |
| reasoning | 2 | -2 | 0.188 | 0.0 |
| refusal | 3 | +2 | 0.027 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 6.2 |
| `refusal_fictional_chip` | refusal | +0 | 0.02 | 0.0 | 38.683 |
| `refusal_out_of_scope` | refusal | +0 | 0.027 | 0.0 | 51.707 |
| `reason_math_fraction` | reasoning | +0 | 0.043 | 0.0 | 101.0 |
| `refusal_made_up_peripheral` | refusal | +2 | 0.056 | 0.0 | 20.829 |
| `persona_brief_role` | persona | +0 | 0.059 | 0.0 | 28.792 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.061 | 0.0 | 50.2 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.062 | 0.0 | 39.125 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.079 | 0.0 | 26.329 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +0 | 0.087 | 0.0 | 50.425 |