# Accuracy comparison: `candidate-kyle-qwen3-30b-a3b-v4-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag`

- Reference: candidate-kyle-qwen3-30b-a3b-v4-v2-rag — 2026-05-04T16:15:54
- Candidate: candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag — 2026-05-06T19:45:12
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 61.4% → 67.4% (Δ = +6.0 pp)
- **Total passes:** 81 → 89 (Δ = +8)
- **Median Jaccard overlap** (word-level): 0.778 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 0.887 | 0.0 |
| general | 2 | +3 | 0.555 | 0.5 |
| multihop | 3 | +6 | 0.043 | 0.0 |
| numerical_precision | 2 | +0 | 0.85 | 0.5 |
| persona | 2 | +0 | 0.205 | 0.0 |
| rag_blog | 1 | +0 | 0.103 | 0.0 |
| rag_datasheet | 26 | +0 | 0.778 | 0.0 |
| rag_email | 1 | -1 | 1.0 | 0.667 |
| reasoning | 2 | +0 | 0.642 | 0.5 |
| refusal | 3 | +0 | 1.0 | 1.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `multihop_peripheral_count` | multihop | +3 | 0.036 | 0.0 | 4.927 |
| `multihop_imx93_vs_95_cores` | multihop | +3 | 0.043 | 0.0 | 2.78 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +3 | 0.071 | 0.0 | 1.0 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.103 | 0.0 | 0.182 |
| `general_qwen_moe` | general | +3 | 0.111 | 0.0 | 2.15 |
| `persona_skippy_voice` | persona | +0 | 0.125 | 0.0 | 1.982 |
| `reason_multistep` | reasoning | +0 | 0.284 | 0.0 | 1.281 |
| `persona_brief_role` | persona | +0 | 0.286 | 0.0 | 1.403 |
| `rag_ds_usb_host_reset_wait` | rag_datasheet | +0 | 0.304 | 0.0 | 0.438 |
| `rag_imx93_cortex` | rag_datasheet | +0 | 0.345 | 0.0 | 2.98 |