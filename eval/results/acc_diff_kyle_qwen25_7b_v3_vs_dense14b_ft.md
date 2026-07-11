# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v3-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: candidate-kyle-qwen25-7b-v3-v2-rag — 2026-05-02T16:35:14
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 58.3% (Δ = -9.9 pp)
- **Total passes:** 90 → 77 (Δ = -13)
- **Median Jaccard overlap** (word-level): 0.406 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.096 | 0.0 |
| multihop | 3 | +0 | 0.25 | 0.0 |
| numerical_precision | 2 | +1 | 0.425 | 0.0 |
| persona | 2 | +0 | 0.017 | 0.0 |
| rag_blog | 1 | +0 | 0.174 | 0.0 |
| rag_datasheet | 26 | -11 | 0.5 | 0.0 |
| rag_email | 1 | +0 | 0.215 | 0.0 |
| reasoning | 2 | -3 | 0.631 | 0.5 |
| refusal | 3 | +0 | 0.176 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 1.337 |
| `persona_skippy_voice` | persona | +0 | 0.034 | 0.0 | 1.417 |
| `general_qwen_moe` | general | +0 | 0.078 | 0.0 | 0.34 |
| `general_embedded_book` | general | +0 | 0.113 | 0.0 | 0.158 |
| `refusal_fictional_chip` | refusal | +0 | 0.162 | 0.0 | 0.318 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.174 | 0.0 | 0.28 |
| `refusal_out_of_scope` | refusal | +0 | 0.176 | 0.0 | 0.539 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +2 | 0.213 | 0.0 | 2.316 |
| `multihop_imx93_vs_95_cores` | multihop | +0 | 0.214 | 0.0 | 0.293 |
| `rag_angry_birds_demo` | rag_email | +0 | 0.215 | 0.0 | 0.203 |