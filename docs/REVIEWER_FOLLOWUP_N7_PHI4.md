# Reviewer FYI — N=7 Phi-4 Corroborates Two-Factor Model

**Date:** 2026-05-10
**Builds on:** `docs/REVIEWER_FOLLOWUP_N6_YI.md` (your two-factor sign-off + Phi-4 queued recommendation + pre-staged sequencing for the falsification outcomes)
**Cross-references:** `eval/results/phi4_n7_corroborates_two_factor.md` (full per-cell + per-dimension breakdown)
**Asking for:** nothing — your pre-staged sequencing covered this outcome. This is a closure FYI.

---

You queued Phi-4 with explicit pre-staged sequencing:

> "If it regresses, two-factor model gets stronger evidence and you update the published doc with the N=7 corroboration. If it lifts, the two-factor model breaks and you update with 'Yi-specific quirk' framing."

**Phi-4 regressed.** Two-factor model corroborated at 7/7 cells.

## N=7 verdict

| Family | Reasoning | Family-match | Substring Δ | Sonnet Δ | GPT-4o Δ | Predicted | Observed |
|---|---:|---|---:|---:|---:|---|---|
| Qwen 7B | 6/6 | ✓ Qwen | +3.1pp | −0.350 | −0.690 | lift | substring ↑ judge erases ✓ |
| Qwen 14B | 3/6 | ✓ Qwen | +8.7pp | ±0.000 | −0.214 | lift | substring ↑ judge erases ✓ |
| Gemma 9B | 6/6 | ✗ | +3.2pp | −0.620 | +0.119 | lift | substring ↑ judge-sensitive ✓ |
| Mistral 7B | 0/6 | ✗ | −3.8pp | −0.218 | −0.048 | regress | both ✓ |
| Llama 8B | 1/6 | ✗ | −3.2pp | −1.165 | −1.524 | regress | both ✓ |
| Yi 9B | 3/6 | ✗ | −28.6pp | −0.848 | −0.714 | regress | both ✓ |
| **Phi-4 (14B)** | **3/6** | **✗** | **−1.6pp** | **−0.627** | **−0.834** | **regress** | **both ✓** |

**13 of 14 judge passes confirm v4 ≤ base** (the one positive is still Gemma + GPT-4o at +0.119, unchanged from N=6).

## Three things worth flagging beyond your pre-staged response

### 1. Yi is no longer an outlier on direction

The N=6 question "is Yi-specific quirk or true cross-family effect" is now answered: Phi-4 is the second cross-family 3/6 base, and it also regresses on substring with judges corroborating. **The two-factor model's falsifiable prediction held.**

### 2. Substring magnitude is unreliable on cross-family intermediate-reasoning bases

Yi −28.6pp substring → −0.8 judge. Phi-4 −1.6pp substring → −0.8 judge. **Same judge regression magnitude, ~18× difference in substring magnitude.** Phi-4's substring −1.6pp is within the temp=0 noise floor (σ≈1.4–2.3pp), but the judge sees a clear regression. A team running substring-only would have shipped Phi-4 v4 thinking it was "close enough to stock." **Substring grader's reliability as a regression detector varies by base type** — emphatic vote for the "two judges by default" standing rule on cross-family deployments.

### 3. Phi-4 introduces a hybrid damage profile

Prior cells split into two damage profiles: **lift cells lose faithfulness** (Qwen 7B/14B, Gemma) while keeping correctness; **regress cells lose correctness** (Mistral, Llama, Yi) with various faithfulness effects. **Phi-4 loses both correctness AND faithfulness** at moderate magnitudes (Sonnet correctness −0.190, faithfulness −0.369; GPT-4o correctness −0.381, faithfulness −0.429). First cell to mix the two mechanisms. Worth noting for future characterization; not load-bearing for the customer-template framing.

### Bonus observation — Phi-4 is the highest-scoring stock base in the dataset

Sonnet 7.077, GPT-4o 7.310 — better than Qwen 2.5 7B (the production model's base) by 0.291 / 0.524 on each judge. This means Phi-4 stock would be a strong candidate to *replace* Qwen as a production base if voice/persona were transferred via a different mechanism (DPO + small persona-only corpus?) — but the v4 recipe damages it. Not actionable now; flagging because it's a non-obvious finding from the data.

## Customer-template wording update

Per your "update the published doc with the N=7 corroboration" sequencing, `recipe-taxonomy.md` now reads:

> "Across N=14 judge passes (7 cells × 2 judges, Sonnet 4.6 + GPT-4o), 13 of 14 confirm v4 ≤ base. ... Bases at intermediate stock reasoning (3/6, N=3) split by family-match: Qwen 14B (Qwen-family) lifted +8.7pp on substring; Yi-1.5-9B-Chat (cross-family) regressed −28.6pp (Sonnet −0.848, GPT-4o −0.714); Phi-4 (cross-family, 14B, Microsoft) regressed −1.6pp on substring (Sonnet −0.627, GPT-4o −0.834). Both cross-family intermediate-reasoning bases regressed, corroborating the two-factor model: lift requires either ceiling reasoning OR family-match to the corpus source distribution. Substring magnitude does not predict capability damage magnitude on this base type — Yi's catastrophic −28.6pp and Phi-4's noise-floor −1.6pp both produced similar judge regression magnitudes (−0.6 to −0.9). Run cross-judge for any cross-family intermediate-reasoning deployment decision."

The Yi-specific "biggest substring regression" framing stays visible (your "don't soften" rule). The Phi-4 nuance about substring-magnitude-unreliability is added as customer-actionable methodology guidance.

## Status

- Customer-template publication shipped at N=6 (commit `6398944`); updated to N=7 (this commit).
- White paper § 7, GOTCHA Addendum, recipe-taxonomy all updated.
- 4 new judge JSONs + analysis MD pushed to Drive.
- [backend] notified for keyhole § 5.5 + deck mirror (separate bus message).
- Phi-4 v4 GGUF (8.5G Q4) on Drive.

Holds: none. Nothing pending from your side.

Optional next step (not asked): a fourth 3/6 cross-family base would consolidate Yi/Phi-4 from N=2 to N=3 in that band. We'd recommend NOT pursuing unless you flag a specific question — current N=7 corroboration is preliminary-but-strong, and additional within-band data has diminishing marginal information value vs. e.g. characterizing 2/6 or 4/6 reasoning bands (which are currently uncharacterized).

---

*Document location: `docs/REVIEWER_FOLLOWUP_N7_PHI4.md`*
