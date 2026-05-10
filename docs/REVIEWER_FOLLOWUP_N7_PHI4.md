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

## Reviewer Closure (2026-05-10)

The reviewer declared closure on the gotcha #7 thread and added four substantive observations that we've folded into the published doc set.

### Closure declared

> "This is genuinely the right place to close. 7/7 cells fitting a two-factor model with a pre-registered falsifier corroborating is meaningfully stronger evidence than most internal-review-grade industry work, and the team has built the case methodically across the entire arc. Closure acknowledged."

> "Gotcha #7 thread closed from my side. Customer-template publication confirmed at N=7."

### Observation 1 — Substring-magnitude-unreliable finding deserves promotion

Reviewer's verbatim:

> "The substring-magnitude-unreliable finding ... is sharper than they're framing it. ... on cross-family intermediate-reasoning bases, the substring noise floor can completely hide real capability damage. A team without judges shipping Phi-4 v4 would have looked at a −1.6pp result within the σ≈1.4-2.3pp temp=0.3 noise floor and concluded 'essentially equivalent to base.' It's not. Both judges see a clear regression. This is a methodology finding worth promoting from 'buried in N=7 narrative' to the methodology section itself, alongside the temperature-sensitivity finding from earlier. The two together (substring fragility under temperature perturbation; substring can hide capability damage at noise-floor magnitudes on certain base types) tell a coherent story about when substring grading is reliable and when it isn't. **That story is the most valuable methodology contribution this whole campaign produced — bigger than gotcha #7 itself.**"

**Adopted.** White paper § Grader-methodology findings now has a Finding 3 (substring unreliable on cross-family intermediate-reasoning bases) paired with Finding 1 (temperature sensitivity) and Finding 2 (LLM-judge reversal). The paired-interpretation paragraph is rewritten to include a "when substring is reliable" matrix across all three regimes.

### Observation 2 — One untested cell in the two-factor space

Reviewer's verbatim:

> "The two-factor model covers all observed combinations except one: family-match base with low stock reasoning. Qwen 2.5 doesn't ship a base with 0-1/6 stock reasoning in the cells you've measured. The model predicts 'lift' via the family-match gate, but it's not tested. Smaller Qwen 2.5 sizes (1.5B, 0.5B) might fall in that band — or might not. Not a publication blocker. The current N=7 framing is honest about being preliminary. But if someone later asks 'have you tested all four corners of your two-factor space,' the answer is 'three of four corners directly; the fourth is predicted but unmeasured.' Worth a footnote in the methodology section so the model's coverage is transparent."

**Adopted.** GOTCHA_7_RESOLUTION.md N=7 update subsection now has a coverage-transparency table showing the three measured corners + the one untested (family-match × low-reasoning) corner.

### Observation 3 — No more 3/6 cross-family data needed

> "The agent's recommendation against more 3/6 cross-family data is correct. Diminishing returns — N=3 to N=4 in the same band buys very little additional confidence. If anything else gets run later, the higher-information experiments are 2/6 or 4/6 reasoning band characterization (untested) or the Qwen-low-reasoning cell above. Defer all of these unless a specific question forces them."

Confirmed. No further within-band 3/6 cross-family fine-tunes queued.

### Observation 4 — Phi-4-as-alt-production-base is a v5 question

> "Bonus observation on Phi-4 stock scoring highest is worth keeping in the working notes but I agree with the agent it's not actionable now. The implication ('Phi-4 might be a stronger production base if voice/persona could be transferred without the v4 recipe') opens a whole different campaign — DPO, small persona corpus, lighter-weight personality transfer methods. That's a Skippy v5 design question for another quarter, not a gotcha #7 question."

Noted. Phi-4-as-alt-base observation parked in working notes for v5 design discussion.

### NXP-internal framing recommendation

> "For NXP-internal: this entire arc is itself a credibility story. 'Team identified a preliminary finding, applied increasingly rigorous methodology, falsified one branch, refined the model, corroborated the refinement with a pre-registered falsifier' is the kind of process narrative that builds trust. If the deck or briefing has room to surface the arc itself (not just the final result), that's worth doing. Reviewers care about whether the team will catch its own over-claims; this arc demonstrates yes."

**Recorded for deck regen.** When the deck/briefing is next updated, consider surfacing the arc itself (N=2 architecture-coupling over-claim → N=5 reasoning-floor predictor → N=6 reframe with Yi falsifying it → N=7 two-factor model corroborated by Phi-4) as a process narrative alongside the final result. The arc demonstrates the team's self-correction discipline.

### What this arc actually produced (reviewer's summary)

> "The original gotcha #7 was a flawed N=1 over-claim that would have shipped to NXP. The current N=7 framing is a falsifiable two-factor model with cross-judge corroboration, surfaced methodology findings on substring fragility, an asymmetry hypothesis tested across families, and a customer-actionable rule for pre-deployment evaluation. That's a significantly more credible deliverable than what started this thread, and most of the value-add isn't in gotcha #7 itself — it's in the secondary methodology findings that came out of the rigor."

### Final status

- **Customer-template publication confirmed at N=7.** Reviewer-final wording in `recipe-taxonomy.md`.
- **Grader-methodology section** of white paper now contains three findings (temperature-sensitivity, LLM-judge reversal, substring-unreliability-on-cross-family-intermediate) with paired interpretation as a "when substring is reliable" matrix.
- **Coverage transparency** noted in GOTCHA Addendum (3/4 corners of two-factor space measured).
- **Phi-4-as-alt-base** parked as v5 design question.
- **Arc-as-credibility-story** recommendation logged for deck regen.

Gotcha #7 thread closed. No further holds, no pending reviewer asks.

---

*Document location: `docs/REVIEWER_FOLLOWUP_N7_PHI4.md`*
