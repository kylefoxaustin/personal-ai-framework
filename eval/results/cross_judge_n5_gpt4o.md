# Cross-judge corroboration on N=5 — Sonnet 4.6 vs GPT-4o

**Date:** 2026-05-10
**Trigger:** Reviewer Q1 follow-up — "Cross-judge corroboration with a non-Anthropic model is the strongest single methodology hardening." Promoted from future-work to executed.
**Judges:** `claude-sonnet-4-6` (Anthropic, original run 2026-05-09) + `gpt-4o-2024-08-06` (OpenAI, this run 2026-05-10).
**Apples-to-apples:** same `JudgeScore` schema, same `JUDGE_SYSTEM_PROMPT`, same `select_held_out_samples()` seed=42 sampler, same 4-dim rubric (correctness / instruction-following / faithfulness / conciseness, each 0–2). Only the judge model differs.
**Cost:** ~$5 OpenAI (GPT-4o) + ~$4 Anthropic (Sonnet, prior). 100 model-judge calls total across 10 (eval JSON, judge) pairs.

## TL;DR

| Family | Substring Δ | Sonnet Δ | GPT-4o Δ | Direction agree? |
|---|---:|---:|---:|---|
| Qwen 2.5 7B | +3.1pp | −0.350 | **−0.690** | ✓ both ≤ 0 |
| Qwen 2.5 14B | +8.7pp | ±0.000 | **−0.214** | ✓ both ≤ 0 |
| Gemma 2 9B | +3.2pp | −0.620 | **+0.119** | **✗ DISAGREE** |
| Mistral 7B v0.3 | −3.8pp | −0.218 | −0.048 | ✓ both ≤ 0 |
| Llama 3.1 8B | −3.2pp | −1.165 | **−1.524** | ✓ both ≤ 0 |

**9 of 10 judge passes confirm v4 ≤ base.** The lone exception is Gemma 2 9B under GPT-4o, which gives a marginal +0.119 lift (vs Sonnet's strong −0.620 erasure). The "every judge-Δ is ≤ 0" headline from the Sonnet-only run is **partially preserved**: the directional claim holds for 4/5 cells under both judges, but Gemma 2 9B shows judge-sensitivity that the single-judge result didn't expose.

## Mechanism comparison (per-dimension)

### Sonnet 4.6 lifts (recap)

| Cell | Correct (b/v) | Instruct (b/v) | Faithful (b/v) | Concise (b/v) |
|---|---|---|---|---|
| Qwen 7B | 1.476 / 1.462 | 1.690 / 1.769 | 1.762 / 1.333 | 1.857 / 1.872 |
| Qwen 14B | 1.526 / 1.500 | 1.711 / 1.868 | 1.711 / 1.447 | 1.868 / 2.000 |
| Gemma 2 9B | 1.487 / 1.366 | 1.667 / 1.634 | 1.564 / 1.366 | 2.000 / 1.732 |

Sonnet's mechanism for lifts: **faithfulness drops on v4** (Qwen 7B −0.43, Qwen 14B −0.26, Gemma −0.20).

### GPT-4o lifts

| Cell | Correct (b/v) | Instruct (b/v) | Faithful (b/v) | Concise (b/v) | Notes |
|---|---|---|---|---|---|
| Qwen 7B | 1.452 / 1.452 | 1.929 / 1.857 | 1.738 / 1.143 | 1.881 / 1.857 | Faithful drops sharply (−0.595) |
| Qwen 14B | 1.500 / 1.476 | 1.881 / 1.738 | 1.310 / 1.262 | 1.762 / 1.762 | Faithful drops slightly (−0.048); instruct drops (−0.143) |
| Gemma 2 9B | 1.476 / 1.452 | 1.857 / 1.738 | 1.024 / 1.262 | 1.500 / 1.524 | **Faithful LIFTS (+0.238) on v4 — opposite of Sonnet's reading** |

GPT-4o's reading of Gemma is what flips the Δ sign: GPT-4o judges Gemma v4 as *more* faithful to RAG context than the Gemma base (1.262 vs 1.024), while Sonnet judged the opposite (1.366 vs 1.564). Both judges agree on conciseness near identity; both agree v4 has slightly worse correctness + instruction-following. The disagreement is concentrated on faithfulness scoring of Gemma's RAG-cited responses.

### Sonnet vs GPT-4o regressions

| Cell | Sonnet base | Sonnet v4 | GPT-4o base | GPT-4o v4 | Direction |
|---|---:|---:|---:|---:|---|
| Mistral 7B v0.3 | 5.718 | 5.500 | 5.619 | 5.571 | both judges agree on small regression |
| Llama 3.1 8B | 5.951 | 4.786 | 6.429 | 4.905 | both judges agree on **strong** regression |

Both judges corroborate the Llama regression as the sharpest cell in the dataset (Sonnet −1.165, GPT-4o −1.524). Mistral is small under both judges.

## Asymmetry verdict — refined under cross-judge

The Sonnet-only verdict was: "across N=5, every judge-Δ is ≤ 0; the v4 recipe produced no LLM-judge-corroborated capability gain." With GPT-4o cross-judge added:

- **9 of 10 judge passes confirm v4 ≤ base.** The "no judge-corroborated capability gain" reading holds for the 4 cells where both judges agree (Qwen 7B, Qwen 14B, Mistral, Llama).
- **Gemma 2 9B is judge-sensitive on the borderline.** Sonnet judges v4 strongly worse (−0.620); GPT-4o judges v4 marginally better (+0.119). The disagreement concentrates on the faithfulness dimension of Gemma's RAG-cited responses.
- **The "lift erases on judge" reading is judge-fragile for Gemma specifically.** It is robust for Qwen 7B and Qwen 14B (both judges agree, with GPT-4o bearish more strongly than Sonnet on Qwen 7B and slightly less on Qwen 14B).
- **Regression cells are robustly capability damage** — both judges corroborate the regression direction for Mistral and Llama, with Llama being the strongest regression in the dataset on both metrics.

## Customer-template wording — refined

The Sonnet-only customer-template wording was:

> "Across N=5 cells, the v4 recipe produced no LLM-judge-corroborated capability gain. ... Lift magnitude on substring does not correlate with judge-Δ."

With cross-judge data, a tighter version:

> "Across N=10 judge passes (5 cells × 2 judges, Sonnet 4.6 + GPT-4o), 9 of 10 confirm v4 ≤ base. The substring grader at temp=0 measures format fidelity, not capability lift, for fine-tunes on this recipe; substring lifts of +3.1 to +8.7pp do not correlate with judge-Δ. Two of the three substring lifts (Qwen 2.5 7B, Qwen 2.5 14B) are corroborated by both judges as judge-flat-or-negative; the third (Gemma 2 9B) is judge-sensitive (Sonnet judges v4 worse, GPT-4o judges v4 marginally better, divergence concentrated on RAG faithfulness scoring). Substring regressions on Mistral and Llama are corroborated as real capability damage by both judges. Customers should expect the v4 recipe to teach voice and refusal patterns reliably across base families, but should not expect underlying capability lift on bases that already perform competently on the eval — and should run cross-judge corroboration for any cell whose deployment decision rests on a marginal judge result."

## Methodology note — judge-sensitivity is real

The Gemma 2 9B cell is exactly the kind of borderline result where single-judge findings carry interpretation risk that cross-judge exposes. Reviewer was right that this is the highest-value methodology hardening.

For future fine-tune evaluations on this recipe:

- **Two judges are minimum for any cell whose deployment turns on a marginal Δ.** A cell where both judges report Δ within ±0.2 should be treated as judge-sensitive and a third judge (or a return to the substring grader's per-category breakdown) should be brought in.
- **Cross-judge cost is not a barrier.** $5–10 per N=5 judge pass is in the noise compared to fine-tune compute; running both Sonnet and GPT-4o by default is feasible.
- **Where judges disagree, examine the per-dimension breakdown.** The Gemma disagreement is concentrated on faithfulness scoring, which suggests the two judges weight RAG-citation faithfulness differently for this model's response style. Worth understanding rather than averaging away.

## Files / artefacts

- 10 judge JSONs in `eval/results/judge_xj_*` (this run, GPT-4o)
- 10 judge JSONs from the Sonnet 4.6 run (2026-05-09, mostly `judge_n5_*` and `judge_baseline-*` / `judge_candidate-*` from 2026-05-08)
- This analysis: `eval/results/cross_judge_n5_gpt4o.md`

All artefacts to be pushed to `gdrive:skippy_files/personal-ai-assistant/eval-results/` per artifact-auto-push rule.
