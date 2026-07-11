# Accuracy comparison: `baseline-qwen25-7b-base-v2-rag` vs `candidate-kyle-qwen25-7b-v1-v2-rag`

- Reference: baseline-qwen25-7b-base-v2-rag — 2026-05-01T21:18:35
- Candidate: candidate-kyle-qwen25-7b-v1-v2-rag — 2026-05-01T23:27:46
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 67.4% → 75.0% (Δ = +7.6 pp)
- **Total passes:** 89 → 99 (Δ = +10)
- **Median Jaccard overlap** (word-level): 0.193 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.206 | 0.0 |
| multihop | 3 | +1 | 0.242 | 0.0 |
| numerical_precision | 2 | +3 | 0.139 | 0.0 |
| persona | 2 | +0 | 0.064 | 0.0 |
| rag_blog | 1 | +0 | 0.155 | 0.0 |
| rag_datasheet | 26 | +7 | 0.196 | 0.0 |
| rag_email | 1 | +3 | 0.117 | 0.0 |
| reasoning | 2 | -1 | 0.25 | 0.0 |
| refusal | 3 | -3 | 0.16 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `reason_math_fraction` | reasoning | +0 | 0.013 | 0.0 | 965.5 |
| `persona_skippy_voice` | persona | +0 | 0.024 | 0.0 | 22.888 |
| `numprec_mcu_boot_rom` | numerical_precision | +3 | 0.101 | 0.0 | 11.535 |
| `persona_brief_role` | persona | +0 | 0.103 | 0.0 | 14.975 |
| `rag_ds_cortex_a55_l2_size` | rag_datasheet | +3 | 0.106 | 0.0 | 26.333 |
| `rag_ds_src_gpr5_offset` | rag_datasheet | +0 | 0.108 | 0.0 | 22.841 |
| `rag_ds_mac_addr34_high_offset` | rag_datasheet | +0 | 0.11 | 0.0 | 7.877 |
| `rag_angry_birds_demo` | rag_email | +3 | 0.117 | 0.0 | 5.044 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.118 | 0.0 | 21.776 |
| `rag_ds_src_gpr4_offset` | rag_datasheet | -3 | 0.121 | 0.0 | 31.71 |