# Accuracy comparison: `reference-moe-q4km-v2-rag` vs `candidate-kyle-qwen25-14b-v1-v2-rag`

- Reference: reference-moe-q4km-v2-rag — 2026-04-23T09:10:11
- Candidate: candidate-kyle-qwen25-14b-v1-v2-rag — 2026-05-02T20:14:00
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.9% → 72.7% (Δ = +3.8 pp)
- **Total passes:** 91 → 96 (Δ = +5)
- **Median Jaccard overlap** (word-level): 0.633 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.297 | 0.0 |
| multihop | 3 | +0 | 0.5 | 0.0 |
| numerical_precision | 2 | +3 | 0.346 | 0.0 |
| persona | 2 | +0 | 0.24 | 0.0 |
| rag_blog | 1 | +0 | 0.125 | 0.0 |
| rag_datasheet | 26 | +6 | 0.724 | 0.0 |
| rag_email | 1 | -3 | 0.159 | 0.0 |
| reasoning | 2 | +0 | 0.697 | 0.5 |
| refusal | 3 | -1 | 0.75 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_skippy_voice` | persona | +0 | 0.0 | 0.0 | 1.0 |
| `rag_ds_thermal_rja_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 3.025 |
| `rag_ds_thermal_rjc_11x11` | rag_datasheet | +0 | 0.033 | 0.0 | 2.925 |
| `rag_ds_ddr_rate_eiq_usecase` | rag_datasheet | +3 | 0.042 | 0.0 | 2.45 |
| `numprec_mcu_boot_rom` | numerical_precision | +3 | 0.062 | 0.0 | 1.22 |
| `refusal_fictional_chip` | refusal | +0 | 0.071 | 0.0 | 3.512 |
| `rag_ev_sustainability` | rag_blog | +0 | 0.125 | 0.0 | 1.029 |
| `rag_angry_birds_demo` | rag_email | -3 | 0.159 | 0.0 | 0.312 |
| `multihop_cpu_plus_npu` | multihop | +0 | 0.221 | 0.0 | 5.339 |
| `rag_ds_imx93_149_gpio` | rag_datasheet | -3 | 0.233 | 0.0 | 4.22 |