# Accuracy comparison: `candidate-kyle-qwen25-7b-v1-v2-rag` vs `candidate-kyle-qwen25-7b-v2-v2-rag`

- Reference: candidate-kyle-qwen25-7b-v1-v2-rag — 2026-05-01T23:27:46
- Candidate: candidate-kyle-qwen25-7b-v2-v2-rag — 2026-05-02T01:28:35
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 75.0% → 71.2% (Δ = -3.8 pp)
- **Total passes:** 99 → 94 (Δ = -5)
- **Median Jaccard overlap** (word-level): 0.246 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.167 | 0.0 |
| multihop | 3 | +0 | 0.285 | 0.0 |
| numerical_precision | 2 | -3 | 0.209 | 0.0 |
| persona | 2 | +0 | 0.15 | 0.0 |
| rag_blog | 1 | +0 | 0.079 | 0.0 |
| rag_datasheet | 26 | -4 | 0.262 | 0.0 |
| rag_email | 1 | +0 | 0.162 | 0.0 |
| reasoning | 2 | -1 | 0.383 | 0.0 |
| refusal | 3 | +3 | 0.178 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `rag_ev_sustainability` | rag_blog | +0 | 0.079 | 0.0 | 0.755 |
| `rag_ds_cortex_a55_l2_size` | rag_datasheet | -3 | 0.086 | 0.0 | 1.167 |
| `persona_skippy_voice` | persona | +0 | 0.088 | 0.0 | 0.091 |
| `general_embedded_book` | general | +0 | 0.112 | 0.0 | 0.906 |
| `refusal_made_up_peripheral` | refusal | +3 | 0.129 | 0.0 | 0.715 |
| `numprec_mcu_boot_rom` | numerical_precision | -3 | 0.14 | 0.0 | 0.137 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | -3 | 0.15 | 0.0 | 0.799 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | +0 | 0.157 | 0.0 | 1.08 |
| `rag_angry_birds_demo` | rag_email | +0 | 0.162 | 0.0 | 0.878 |
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.167 | 0.0 | 0.899 |