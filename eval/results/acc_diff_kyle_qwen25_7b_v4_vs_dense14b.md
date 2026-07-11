# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v4-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 70.5% (Δ = +2.3 pp)
- **Total passes:** 90 → 93 (Δ = +3)
- **Median Jaccard overlap** (word-level): 0.502 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.227 | 0.0 |
| multihop | 3 | +0 | 0.231 | 0.0 |
| numerical_precision | 2 | +0 | 0.538 | 0.0 |
| persona | 2 | +0 | 0.0 | 0.0 |
| rag_blog | 1 | +0 | 0.145 | 0.0 |
| rag_datasheet | 26 | +6 | 0.632 | 0.0 |
| rag_email | 1 | +0 | 0.215 | 0.0 |
| reasoning | 2 | -3 | 0.675 | 0.5 |
| refusal | 3 | +0 | 0.228 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 1.277 |
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 0.542 |
| `general_embedded_book` | general | +0 | 0.09 | 0.0 | 0.138 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.145 | 0.0 | 0.302 |
| `multihop_imx93_vs_95_cores` | multihop | +0 | 0.17 | 0.0 | 0.179 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.172 | 0.0 | 0.192 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +3 | 0.19 | 0.0 | 1.16 |
| `refusal_out_of_scope` | refusal | +0 | 0.192 | 0.0 | 0.23 |
| `rag_angry_birds_demo` | rag_email | +0 | 0.215 | 0.0 | 0.203 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.228 | 0.0 | 0.898 |