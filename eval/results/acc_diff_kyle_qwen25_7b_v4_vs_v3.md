# Accuracy comparison: `candidate-kyle-qwen25-7b-v3-v2-rag` vs `candidate-kyle-qwen25-7b-v4-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v3-v2-rag — 2026-05-02T16:35:14
- Candidate: candidate-kyle-qwen25-7b-v4-v2-rag — 2026-05-02T17:52:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 58.3% → 70.5% (Δ = +12.2 pp)
- **Total passes:** 77 → 93 (Δ = +16)
- **Median Jaccard overlap** (word-level): 0.543 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.231 | 0.0 |
| multihop | 3 | +0 | 0.478 | 0.0 |
| numerical_precision | 2 | -1 | 0.516 | 0.0 |
| persona | 2 | +0 | 0.045 | 0.0 |
| rag_blog | 1 | +0 | 0.419 | 0.0 |
| rag_datasheet | 26 | +17 | 0.673 | 0.0 |
| rag_email | 1 | +0 | 1.0 | 1.0 |
| reasoning | 2 | +0 | 0.692 | 0.5 |
| refusal | 3 | +0 | 0.25 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 0.382 |
| `general_qwen_moe` | general | +0 | 0.083 | 0.0 | 2.778 |
| `persona_brief_role` | persona | +0 | 0.091 | 0.0 | 0.498 |
| `refusal_out_of_scope` | refusal | +0 | 0.105 | 0.0 | 0.427 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.125 | 0.0 | 0.315 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +1 | 0.149 | 0.0 | 0.501 |
| `rag_imx93_cortex` | rag_datasheet | +3 | 0.17 | 0.0 | 1.348 |
| `rag_imx93_ddr_width` | rag_datasheet | -2 | 0.208 | 0.0 | 0.296 |
| `rag_ds_usb_host_reset_wait` | rag_datasheet | +3 | 0.236 | 0.0 | 3.531 |
| `refusal_made_up_peripheral` | refusal | +0 | 0.25 | 0.0 | 1.814 |