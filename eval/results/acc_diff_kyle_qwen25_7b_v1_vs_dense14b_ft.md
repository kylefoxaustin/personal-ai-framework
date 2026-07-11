# Accuracy comparison: `reference-dense-q4km-v2-rag` vs `candidate-kyle-qwen25-7b-v1-v2-rag`

- Reference: reference-dense-q4km-v2-rag — 2026-04-23T09:14:57
- Candidate: candidate-kyle-qwen25-7b-v1-v2-rag — 2026-05-01T23:27:46
- Prompt set: v2 · samples/prompt: 3

## Headline numbers

- **Pass rate:** 68.2% → 75.0% (Δ = +6.8 pp)
- **Total passes:** 90 → 99 (Δ = +9)
- **Median Jaccard overlap** (word-level): 0.175 (1.0 = identical word bags, 0.0 = no words shared)
- **Median exact-match rate** across samples: 0.000 (fraction of prompts where candidate produced byte-identical text)

## By category

| Category | Prompts | ΣΔ pass | Median Jaccard | Median exact-match |
|---|---:|---:|---:|---:|
| coding | 2 | +0 | 1.0 | 1.0 |
| general | 2 | +0 | 0.195 | 0.0 |
| multihop | 3 | +0 | 0.203 | 0.0 |
| numerical_precision | 2 | +3 | 0.153 | 0.0 |
| persona | 2 | +0 | 0.017 | 0.0 |
| rag_blog | 1 | +0 | 0.184 | 0.0 |
| rag_datasheet | 26 | +10 | 0.175 | 0.0 |
| rag_email | 1 | +0 | 0.129 | 0.0 |
| reasoning | 2 | -1 | 0.269 | 0.0 |
| refusal | 3 | -3 | 0.134 | 0.0 |

## Most-divergent prompts (lowest Jaccard = most drift)

| Prompt | Category | Δpass | Jaccard p50 | Exact rate | Length ratio |
|---|---|---:|---:|---:|---:|
| `persona_brief_role` | persona | +0 | 0.0 | 0.0 | 21.47 |
| `reason_math_fraction` | reasoning | +0 | 0.013 | 0.0 | 965.5 |
| `persona_skippy_voice` | persona | +0 | 0.033 | 0.0 | 28.292 |
| `numprec_mcu_boot_rom` | numerical_precision | +3 | 0.103 | 0.0 | 26.962 |
| `rag_ds_eiq_neutron_npu` | rag_datasheet | +0 | 0.105 | 0.0 | 15.54 |
| `rag_ds_cortex_a55_l2_size` | rag_datasheet | +3 | 0.106 | 0.0 | 26.333 |
| `rag_ds_vdd_usb_3p3_max_voltage` | rag_datasheet | +0 | 0.11 | 0.0 | 24.829 |
| `rag_ds_vdd_ana_0p8_max_voltage` | rag_datasheet | +0 | 0.117 | 0.0 | 27.643 |
| `rag_ds_rsa_key_size` | rag_datasheet | +0 | 0.118 | 0.0 | 21.776 |
| `rag_ds_cortex_m33_max_freq` | rag_datasheet | +0 | 0.119 | 0.0 | 29.321 |