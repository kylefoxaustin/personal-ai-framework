# Accuracy comparison: `candidate-kyle-qwen25-14b-v1-v2-rag` vs `candidate-kyle-qwen25-32b-v1-v2-rag`

- Reference: candidate-kyle-qwen25-14b-v1-v2-rag — 2026-05-02T20:14:00
- Candidate: candidate-kyle-qwen25-32b-v1-v2-rag — 2026-05-06T22:25:36
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 72.7% → 63.6% (Δ = -9.1 pp)
- **Total passes:** 96 → 84 (Δ = -12)
- **Median Jaccard overlap** (word-level): 0.477 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.938 | 0.334 |
| general | 2 | +0 | 0.294 | 0.0 |
| multihop | 3 | +0 | 0.632 | 0.0 |
| numerical_precision | 2 | -3 | 0.459 | 0.0 |
| persona | 2 | +0 | 0.077 | 0.0 |
| rag_blog | 1 | +0 | 0.077 | 0.0 |
| rag_datasheet | 26 | -15 | 0.561 | 0.0 |
| rag_email | 1 | +3 | 0.161 | 0.0 |
| reasoning | 2 | +0 | 0.669 | 0.5 |
| refusal | 3 | +3 | 0.2 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 3.333 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.077 | 0.0 | 0.445 |
| `refusal_out_of_scope` | refusal | +0 | 0.1 | 0.0 | 5.0 |
| `persona_brief_role` | persona | +0 | 0.154 | 0.0 | 0.933 |
| `rag_angry_birds_demo` | rag_email | +3 | 0.161 | 0.0 | 2.726 |
| `refusal_made_up_peripheral` | refusal | +3 | 0.2 | 0.0 | 1.229 |
| `rag_imx93_uart` | rag_datasheet | +0 | 0.218 | 0.0 | 5.81 |
| `general_qwen_moe` | general | +0 | 0.247 | 0.0 | 1.081 |
| `multihop_cpu_plus_npu` | multihop | +0 | 0.293 | 0.0 | 0.521 |
| `refusal_fictional_chip` | refusal | +0 | 0.31 | 0.0 | 0.549 |