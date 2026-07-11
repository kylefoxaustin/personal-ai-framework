# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `candidate-kyle-qwen25-14b-v1-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: candidate-kyle-qwen25-14b-v1-v2-rag — 2026-05-02T20:14:00
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 72.7% (Δ = +4.5 pp)
- **Total passes:** 90 → 96 (Δ = +6)
- **Median Jaccard overlap** (word-level): 0.516 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.303 | 0.0 |
| multihop | 3 | +0 | 0.255 | 0.0 |
| numerical_precision | 2 | +3 | 0.532 | 0.0 |
| persona | 2 | +0 | 0.067 | 0.0 |
| rag_blog | 1 | +0 | 0.141 | 0.0 |
| rag_datasheet | 26 | +9 | 0.663 | 0.0 |
| rag_email | 1 | -3 | 0.143 | 0.0 |
| reasoning | 2 | +0 | 0.645 | 0.5 |
| refusal | 3 | -3 | 0.192 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 1.988 |
| `refusal_fictional_chip` | refusal | +0 | 0.106 | 0.0 | 0.526 |
| `persona_skippy_voice` | persona | +0 | 0.133 | 0.0 | 0.417 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.141 | 0.0 | 0.803 |
| `rag_angry_birds_demo` | rag_email | -3 | 0.143 | 0.0 | 0.193 |
| `multihop_imx93_vs_95_cores` | multihop | +0 | 0.192 | 0.0 | 0.196 |
| `refusal_out_of_scope` | refusal | +0 | 0.192 | 0.0 | 0.23 |
| `multihop_peripheral_count` | multihop | +0 | 0.255 | 0.0 | 0.164 |
| `general_qwen_moe` | general | +0 | 0.281 | 0.0 | 1.132 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.281 | 0.0 | 0.291 |