# Accuracy comparison: `candidate-kyle-qwen25-7b-v4-v2-rag` vs `candidate-kyle-qwen25-14b-v1-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Candidate: candidate-kyle-qwen25-14b-v1-v2-rag — 2026-05-02T20:14:00
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 70.5% → 72.7% (Δ = +2.2 pp)
- **Total passes:** 93 → 96 (Δ = +3)
- **Median Jaccard overlap** (word-level): 0.710 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +0 | 0.185 | 0.0 |
| multihop | 3 | +0 | 0.778 | 0.0 |
| numerical_precision | 2 | +3 | 0.643 | 0.0 |
| persona | 2 | +0 | 0.221 | 0.0 |
| rag_blog | 1 | +0 | 0.062 | 0.0 |
| rag_datasheet | 26 | +3 | 0.784 | 0.0 |
| rag_email | 1 | -3 | 0.312 | 0.0 |
| reasoning | 2 | +3 | 0.695 | 0.5 |
| refusal | 3 | -3 | 0.222 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.056 | 0.0 | 1.512 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.062 | 0.0 | 2.656 |
| `general_embedded_book` | general | +0 | 0.116 | 0.0 | 5.208 |
| `refusal_made_up_peripheral` | refusal | -3 | 0.158 | 0.0 | 0.316 |
| `persona_skippy_voice` | persona | +0 | 0.2 | 0.0 | 0.769 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.218 | 0.0 | 0.611 |
| `refusal_fictional_chip` | refusal | +0 | 0.222 | 0.0 | 1.19 |
| `persona_brief_role` | persona | +0 | 0.242 | 0.0 | 1.557 |
| `general_qwen_moe` | general | +0 | 0.254 | 0.0 | 1.2 |
| `rag_ds_package_pitch` | rag_datasheet | +0 | 0.258 | 0.0 | 0.28 |