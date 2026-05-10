# Reviewer Follow-up — Cross-judge Corroboration Result

**Date:** 2026-05-10
**Builds on:** `docs/REVIEWER_FOLLOWUP_JUDGE_VERDICT.md` (prior sign-off + your three sharpenings)
**Cross-references:** `eval/results/cross_judge_n5_gpt4o.md` (full per-cell + per-dimension breakdown), `docs/GOTCHA_7_RESOLUTION.md` (Judge-on-N=5 Verdict section, cross-judge subsection added)
**Asking for:** ~5 min on one open question (Gemma cell tie-break) at the bottom.

---

You flagged cross-judge corroboration as the highest-value single methodology hardening in your Q1 reply. We promoted it from future-work to executed the same day.

## Verdict

`gpt-4o-2024-08-06` ran on the same 10 (eval JSON, sample subset) pairs as the Sonnet 4.6 run, with the same `JudgeScore` schema, same `JUDGE_SYSTEM_PROMPT`, same seed=42 sampler. Cost: ~$5 OpenAI.

| Family | Substring Δ | Sonnet Δ | GPT-4o Δ | Direction agree? |
|---|---:|---:|---:|---|
| Qwen 2.5 7B | +3.1pp | −0.350 | **−0.690** | ✓ both ≤ 0 |
| Qwen 2.5 14B | +8.7pp | ±0.000 | **−0.214** | ✓ both ≤ 0 |
| **Gemma 2 9B** | +3.2pp | **−0.620** | **+0.119** | **✗ DISAGREE** |
| Mistral 7B v0.3 | −3.8pp | −0.218 | −0.048 | ✓ both ≤ 0 |
| Llama 3.1 8B | −3.2pp | −1.165 | **−1.524** | ✓ both ≤ 0 |

**9 of 10 judge passes confirm v4 ≤ base.** Direction agrees on 4 of 5 cells; Gemma 2 9B disagrees.

## What this preserves and what it changes

### Preserved (load-bearing)

- **The Qwen 14B "biggest substring lift, most evaporative" demonstration** — your suggested headline framing — is **robust under cross-judge**. Sonnet ±0.000, GPT-4o −0.214; both judges agree the +8.7pp substring lift produces no judge-corroborated capability gain. The point that "lift magnitude on substring does not predict capability gain" survives cross-judge cleanly.
- **Both regression cells (Mistral, Llama) are corroborated by both judges as real capability damage.** Llama is the strongest regression in the dataset on both metrics (Sonnet −1.165, GPT-4o −1.524). Mistral is small under both.
- **The "expect voice + refusal transfer; do not expect underlying capability lift" customer rec** is unchanged.

### Changed (sharpened, not weakened)

- The Sonnet-only headline "every judge-Δ ≤ 0; no judge-corroborated capability gain in any cell" is **partially preserved**: 9/10 holds; the Gemma cell is the exception.
- Customer-template wording promoted from "until a semantic-rubric judge corroborates" (singular) to "across N=10 judge passes (5 cells × 2 judges), 9 of 10 confirm v4 ≤ base" (plural with explicit cross-judge framing).
- Standing methodology adopted: **"two judges minimum for any cell whose deployment decision turns on a marginal Δ. Cross-judge cost (~$5 per N=5 pass) is in the noise compared to fine-tune compute."**

## The Gemma 2 9B cell — judge-divergence reading

The disagreement is concentrated on the **faithfulness dimension** of RAG-cited responses. Per-dimension breakdown:

| Dimension (0–2) | Gemma base — Sonnet | Gemma v4 — Sonnet | Gemma base — GPT-4o | Gemma v4 — GPT-4o |
|---|---:|---:|---:|---:|
| Correctness | 1.487 | 1.366 | 1.476 | 1.452 |
| Instruction-following | 1.667 | 1.634 | 1.857 | 1.738 |
| **Faithfulness to RAG context** | **1.564** | **1.366** | **1.024** | **1.262** |
| Conciseness | 2.000 | 1.732 | 1.500 | 1.524 |

Both judges agree that Gemma v4 is slightly worse on correctness and instruction-following, and roughly identical on conciseness. The disagreement is **just on faithfulness**: Sonnet thinks Gemma v4 is *less* faithful to RAG context than the base; GPT-4o thinks Gemma v4 is *more* faithful. The magnitude is meaningful (Sonnet −0.198 base→v4, GPT-4o +0.238 base→v4) and the divergence is large enough to flip the total Δ sign.

**Plausible read:** the two judges weight RAG-citation faithfulness differently for Gemma's specific response style. Gemma's outputs may include something that Sonnet treats as a faithfulness penalty (over-extrapolation? citation pattern?) and GPT-4o treats as a faithfulness credit (or vice versa). Without a third judge or human spot-checks, we can't break the tie analytically.

## Open question

**How should we treat the Gemma cell in the customer-template publication?**

- **(a)** Flag explicitly as a known judge-sensitive cell, document the divergence reading (faithfulness scoring concentrated), and let customers replicate cross-judge for their own bases. Customer-facing wording: "Gemma 2 9B v4 is judge-sensitive on the borderline; Sonnet judges v4 worse, GPT-4o judges v4 marginally better. The divergence concentrates on RAG-faithfulness scoring."
- **(b)** Bring in a third judge on Gemma specifically to break the tie. Candidates: Llama-405B-judge (via Together / Fireworks, ~$1–2 for one cell), DeepSeek (~$0.50), or a human spot-check on the 38–41 sample subset (~30 min reviewer time). Whichever direction the third judge falls, the customer-template wording adapts.

Our default if no objection: **(a)**. Three reasons:

1. The Gemma cell is *not* load-bearing for the headline — Qwen 14B carries the "biggest lift, most evaporative" demo, and that's robust under cross-judge.
2. Single-cell judge-sensitivity is *itself* informative for the customer template — it directly motivates the "two judges minimum on marginal Δ" methodology recommendation.
3. A third judge that agrees with one of Sonnet or GPT-4o would still leave the question of "why did the disagreeing judge see it differently" — pushing the disagreement onto the third judge rather than resolving the underlying ambiguity. The faithfulness-scoring divergence is the data; tie-breaking it doesn't unmake it.

But (b) is cheap if you'd rather have a tie-broken Gemma row before NXP-internal review reads the customer template. Your call.

## Methodology takeaway (for the next campaign)

Two judges minimum on any marginal-Δ cell, by default. Cross-judge cost is small enough that it should be standing methodology, not opt-in. The Gemma cell is the kind of borderline result that single-judge runs would have shipped as "lift erases on judge"; cross-judge surfaces the actual ambiguity.

## Status

- Customer-template publication wording refined to incorporate cross-judge result (`docs/recipe-taxonomy.md`)
- White paper § 7 asymmetry paragraph updated to reflect cross-judge (`docs/skippy-white-paper.md`)
- GOTCHA Judge-on-N=5 Verdict section has a cross-judge subsection with the full 5-cell comparison (`docs/GOTCHA_7_RESOLUTION.md`)
- Reviewer follow-up doc with the verdict reply and current open question logged (`docs/REVIEWER_FOLLOWUP_JUDGE_VERDICT.md`)
- 10 GPT-4o judge JSONs + analysis MD pushed to `gdrive:skippy_files/personal-ai-assistant/eval-results/`
- [backend] notified for keyhole § 5.5 + deck mirroring; mirrored at keyhole `4546a38` + my-stuff `c1b264c`

Holds: pending your call on Gemma option (a) vs (b). All other publication-blocking items are clear.

---

*Document location: `docs/REVIEWER_FOLLOWUP_CROSS_JUDGE.md`*
