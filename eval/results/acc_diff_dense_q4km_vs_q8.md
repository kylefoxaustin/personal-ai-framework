# Accuracy comparison: `reference-qwen25-14b-vanilla` vs `candidate-qwen25-14b-q8`

- Reference: reference-qwen25-14b-vanilla — 2026-04-22T23:07:29
- Candidate: candidate-qwen25-14b-q8 — 2026-04-23T08:21:43
- Prompt set: v1 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 52.8% → 58.3% (Δ = +5.5 pp)
- **Total passes:** 19 → 21 (Δ = +2)
- **Median Jaccard overlap** (word-level): 0.428 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.403 | 0.0 |
| persona | 1 | +0 | 0.159 | 0.0 |
| rag_blog | 1 | +0 | 0.422 | 0.0 |
| rag_datasheet | 3 | +0 | 0.395 | 0.0 |
| rag_email | 1 | +3 | 0.299 | 0.0 |
| reasoning | 2 | -1 | 0.969 | 0.5 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.159 | 0.0 | 0.802 |
| `rag_angry_birds_demo` | rag_email | +3 | 0.299 | 0.0 | 3.01 |
| `rag_imx93_uart` | rag_datasheet | +0 | 0.318 | 0.0 | 0.618 |
| `general_embedded_book` | general | +0 | 0.373 | 0.0 | 1.172 |
| `rag_imx93_cortex` | rag_datasheet | +0 | 0.395 | 0.0 | 0.624 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.422 | 0.0 | 0.937 |
| `general_qwen_moe` | general | +0 | 0.434 | 0.0 | 0.899 |
| `rag_imx93_ddr_width` | rag_datasheet | +0 | 0.583 | 0.0 | 0.983 |
| `reason_multistep` | reasoning | -1 | 0.937 | 0.0 | 0.99 |
| `code_fibonacci` | coding | +0 | 1.0 | 1.0 | 1.0 |