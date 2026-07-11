# Accuracy comparison: `candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag` vs `candidate-kyle-qwen3-30b-a3b-full-v1-v2-rag`

- Reference: candidate-kyle-qwen3-30b-a3b-router-v1-v2-rag — 2026-05-06T19:45:12
- Candidate: candidate-kyle-qwen3-30b-a3b-full-v1-v2-rag — 2026-05-07T10:17:13
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 67.4% → 62.9% (Δ = -4.5 pp)
- **Total passes:** 89 → 83 (Δ = -6)
- **Median Jaccard overlap** (word-level): 0.732 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.325 | 0.0 |
| multihop | 3 | +0 | 0.25 | 0.0 |
| numerical_precision | 2 | +0 | 0.718 | 0.5 |
| persona | 2 | +0 | 0.756 | 0.0 |
| rag_blog | 1 | -3 | 0.095 | 0.0 |
| rag_datasheet | 26 | -4 | 0.725 | 0.0 |
| rag_email | 1 | +1 | 0.667 | 0.0 |
| reasoning | 2 | +0 | 0.888 | 0.5 |
| refusal | 3 | +0 | 1.0 | 1.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.071 | 0.0 | 1.0 |
| `rag_ev_sustainability` | rag_blog | -3 | 0.095 | 0.0 | 0.422 |
| `rag_imx93_cortex` | rag_datasheet | +0 | 0.125 | 0.0 | 0.089 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.148 | 0.0 | 3.439 |
| `general_qwen_moe` | general | +0 | 0.149 | 0.0 | 3.779 |
| `multihop_peripheral_count` | multihop | +0 | 0.182 | 0.0 | 0.455 |
| `multihop_cpu_plus_npu` | multihop | +0 | 0.25 | 0.0 | 0.239 |
| `rag_ds_usb_host_reset_wait` | rag_datasheet | +0 | 0.304 | 0.0 | 2.283 |
| `rag_ds_ecc_curve` | rag_datasheet | +0 | 0.4 | 0.0 | 0.537 |
| `numprec_lpddr_rate` | numerical_precision | +0 | 0.435 | 0.0 | 0.708 |