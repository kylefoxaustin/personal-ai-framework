# Accuracy comparison: `baseline-qwen25-7b-base-v2-rag` vs `candidate-kyle-qwen25-7b-v4-v2-rag`

- Reference: baseline-qwen25-7b-base-v2-rag — 2026-05-01T21:18:35
- Candidate: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 67.4% → 70.5% (Δ = +3.1 pp)
- **Total passes:** 89 → 93 (Δ = +4)
- **Median Jaccard overlap** (word-level): 0.407 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.228 | 0.0 |
| multihop | 3 | +1 | 0.196 | 0.0 |
| numerical_precision | 2 | +0 | 0.507 | 0.0 |
| persona | 2 | +0 | 0.145 | 0.0 |
| rag_blog | 1 | +0 | 0.125 | 0.0 |
| rag_datasheet | 26 | +3 | 0.502 | 0.0 |
| rag_email | 1 | +3 | 0.246 | 0.0 |
| reasoning | 2 | -3 | 0.659 | 0.5 |
| refusal | 3 | +0 | 0.18 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.047 | 0.0 | 0.122 |
| `persona_skippy_voice` | persona | +0 | 0.056 | 0.0 | 0.438 |
| `refusal_out_of_scope` | refusal | +0 | 0.065 | 0.0 | 0.184 |
| `general_embedded_book` | general | +0 | 0.101 | 0.0 | 0.138 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.125 | 0.0 | 0.165 |
| `multihop_peripheral_count` | multihop | +1 | 0.18 | 0.0 | 0.247 |
| `refusal_fictional_chip` | refusal | +0 | 0.18 | 0.0 | 0.262 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +3 | 0.19 | 0.0 | 1.189 |
| `multihop_imx93_vs_95_cores` | multihop | +0 | 0.196 | 0.0 | 0.186 |
| `rag_imx93_ddr_width` | rag_datasheet | +0 | 0.205 | 0.0 | 0.29 |