# Eval results — methodology + reconciliation notes

## 2026-05-08 — Post-regrade Mistral baseline reconciliation (per external Claude review)

After SK-P0-001 quarantined the persona category (denominator: 132 → 126), all eval JSONs were regraded via `eval/regrade_for_broken_categories.py`. Confirming the load-bearing Mistral numbers:

| Run | Pre-regrade (132-basis) | Post-regrade (126-basis) | Δ-vs-base in absolute passes |
|---|---:|---:|---:|
| Mistral 7B v0.3 Instruct (stock) | 80/132 = 60.6% | **80/126 = 63.5%** | reference |
| Skippy Mistral v4 (assistant_only_loss + template patch) | 75/132 = 56.8% | 75/126 = **59.5%** | **−5 passes** |
| Skippy Mistral v4-fullseq (no template patch, full-seq loss) | 0/132 = 0.0% | 0/126 = 0.0% | −80 passes — see DIAGNOSIS.md |

**Delta in absolute passes is preserved across the regrade (−5 passes vs base for the assistant-only variant); percentage delta shifts slightly from −3.80pp to −3.97pp because the denominator shrank from 132 to 126.**

The percentage shift does not change the direction of the gotcha #7 finding, but it **does** shift the variance-bounds bar: a −4.0pp delta is more or less likely to clear ≥2σ than a −3.8pp delta would, depending on what σ comes in at after SK-P0-002 lands.

**No re-eval was performed.** Pre-regrade JSONs include both the original `summary_v1_legacy` block (preserved verbatim for audit) and the regraded `summary` block. The `acc_candidate-kyle-mistral-7b-v4-fullseq_*.json` is preserved per the failure-data memory rule even though it scored 0; see `training/output/mistral-v4-fullseq/DIAGNOSIS.md` for root-cause analysis (the empty output is a Skippy inference-pipeline bug on unpatched-template GGUFs, not a training failure).
