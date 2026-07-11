# Accuracy comparison: `reference-moe-q4km` vs `reference-moe-q4km-rerun`

- Reference: reference-moe-q4km — 2026-04-22T22:22:40
- Candidate: reference-moe-q4km-rerun — 2026-04-22T22:23:30
- Prompt set: v1 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 58.3% → 58.3% (Δ = +0.0 pp)
- **Total passes:** 21 → 21 (Δ = +0)
- **Median Jaccard overlap** (word-level): 1.000 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 1.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 1.0 | 1.0 |
| persona | 1 | +0 | 1.0 | 1.0 |
| rag_blog | 1 | +0 | 1.0 | 1.0 |
| rag_datasheet | 3 | +0 | 1.0 | 1.0 |
| rag_email | 1 | +0 | 1.0 | 1.0 |
| reasoning | 2 | +0 | 1.0 | 1.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `code_fibonacci` | coding | +0 | 1.0 | 1.0 | 1.0 |
| `code_reverse_string` | coding | +0 | 1.0 | 1.0 | 1.0 |
| `general_embedded_book` | general | +0 | 1.0 | 1.0 | 1.0 |
| `general_qwen_moe` | general | +0 | 1.0 | 1.0 | 1.0 |
| `persona_skippy_voice` | persona | +0 | 1.0 | 1.0 | 1.0 |
| `rag_angry_birds_demo` | rag_email | +0 | 1.0 | 1.0 | 1.0 |
| `rag_ev_sustainability` | rag_blog | +0 | 1.0 | 1.0 | 1.0 |
| `rag_imx93_cortex` | rag_datasheet | +0 | 1.0 | 1.0 | 1.0 |
| `rag_imx93_ddr_width` | rag_datasheet | +0 | 1.0 | 1.0 | 1.0 |
| `rag_imx93_uart` | rag_datasheet | +0 | 1.0 | 1.0 | 1.0 |