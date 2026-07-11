# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v2-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: candidate-kyle-qwen25-7b-v2-v2-rag — 2026-05-02T01:28:35
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 71.2% (Δ = +3.0 pp)
- **Total passes:** 90 → 94 (Δ = +4)
- **Median Jaccard overlap** (word-level): 0.190 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.164 | 0.0 |
| multihop | 3 | +0 | 0.15 | 0.0 |
| numerical_precision | 2 | +0 | 0.335 | 0.0 |
| persona | 2 | +0 | 0.024 | 0.0 |
| rag_blog | 1 | +0 | 0.151 | 0.0 |
| rag_datasheet | 26 | +6 | 0.194 | 0.0 |
| rag_email | 1 | +0 | 0.2 | 0.0 |
| reasoning | 2 | -2 | 0.255 | 0.0 |
| refusal | 3 | +0 | 0.107 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 24.976 |
| `reason_math_fraction` | reasoning | +0 | 0.043 | 0.0 | 101.0 |
| `persona_skippy_voice` | persona | +0 | 0.047 | 0.0 | 2.583 |
| `refusal_out_of_scope` | refusal | +0 | 0.069 | 0.0 | 11.91 |
| `rag_ds_cortex_m33_max_freq` | rag_datasheet | +0 | 0.082 | 0.0 | 26.444 |
| `general_embedded_book` | general | +0 | 0.107 | 0.0 | 2.817 |
| `refusal_fictional_chip` | refusal | +0 | 0.107 | 0.0 | 5.788 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.11 | 0.0 | 18.806 |
| `rag_ds_package_pitch` | rag_datasheet | +0 | 0.111 | 0.0 | 37.836 |
| `rag_ds_vdd_ana_0p8_max_voltage` | rag_datasheet | +0 | 0.111 | 0.0 | 28.171 |