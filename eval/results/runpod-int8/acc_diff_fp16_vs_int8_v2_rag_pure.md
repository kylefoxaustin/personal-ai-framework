# Accuracy comparison: `reference-fp16-v2-rag-pure-int8pod` vs `candidate-int8-v2-rag-pure`

- Reference: reference-fp16-v2-rag-pure-int8pod — 2026-04-24T20:09:23
- Candidate: candidate-int8-v2-rag-pure — 2026-04-24T20:14:14
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 65.2% → 61.4% (Δ = -3.8 pp)
- **Total passes:** 86 → 81 (Δ = -5)
- **Median Jaccard overlap** (word-level): 0.748 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.489 | 0.0 |
| multihop | 3 | +0 | 0.447 | 0.0 |
| numerical_precision | 2 | +0 | 0.675 | 0.0 |
| persona | 2 | +0 | 0.65 | 0.5 |
| rag_blog | 1 | +0 | 0.331 | 0.0 |
| rag_datasheet | 26 | -2 | 0.864 | 0.0 |
| rag_email | 1 | -3 | 0.635 | 0.0 |
| reasoning | 2 | +0 | 1.0 | 0.834 |
| refusal | 3 | +0 | 0.586 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.294 | 0.0 | 0.882 |
| `persona_skippy_voice` | persona | +0 | 0.3 | 0.0 | 0.949 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.331 | 0.0 | 0.649 |
| `multihop_peripheral_count` | multihop | +0 | 0.413 | 0.0 | 0.938 |
| `multihop_imx93_vs_95_cores` | multihop | +0 | 0.447 | 0.0 | 1.004 |
| `general_qwen_moe` | general | +0 | 0.469 | 0.0 | 1.146 |
| `numprec_mcu_boot_rom` | numerical_precision | +0 | 0.489 | 0.0 | 1.245 |
| `rag_ds_imx93_gpio_count` | rag_datasheet | +0 | 0.495 | 0.0 | 0.891 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | -2 | 0.507 | 0.0 | 0.914 |
| `general_embedded_book` | general | +0 | 0.509 | 0.0 | 1.094 |