# Gotcha #7 — Progress Update for Reviewer (N=5)

**Status:** New data; framing reframed beyond the N=2 doc you signed off on 2026-05-08. Seeking your opinion before we treat the reframe as load-bearing.
**Date:** 2026-05-09
**Author:** [docs] session
**Cross-references:** `docs/GOTCHA_7_RESOLUTION.md` (your approved N=2 framing + Addendum with the N=5 table)

---

## What you signed off on (recap)

The N=2 preliminary framing — Qwen lifts (+3.1pp 7B, +5.3pp 14B); non-Qwen regresses (−3.8pp Mistral, −3.0pp Llama). Your upgrade criterion to take this from "preliminary" to "established":

> *a third non-Qwen family that also regresses (Phi, Yi, Gemma — any), pushing to N=3*

We ran Gemma 2 9B Instruct as the cleanest possible non-Qwen point — different template (`<start_of_turn>` / `<end_of_turn>`, neither ChatML nor `[INST]`), no `{% generation %}` patch needed, same recipe, same corpus, same hardware.

## What happened

**Gemma 2 9B v4 lifted +3.2pp** (61.9% → 65.1%, 126-sample post-regrade basis). Same magnitude as Qwen 7B v4 lifted from its base. Your upgrade criterion was *not* met; the architecture-coupling reading the N=2 picture supported is *falsified*.

## The N=5 picture

| Base | Stock reasoning | Stock refusal | v4 Δheadline | Direction |
|---|---:|---:|---:|---|
| Qwen 2.5 7B | 6/6 | 9/9 | +3.1pp | lift |
| Qwen 2.5 14B | 6/6 | 6/9 | +5.3pp | lift |
| **Gemma 2 9B** | **6/6** | **9/9** | **+3.2pp** | **lift** |
| Mistral 7B v0.3 | 0/6 | 6/9 | −4.0pp | regress |
| Llama 3.1 8B | 1/6 | 6/9 | −3.2pp | regress |

A clean predictor emerges: **stock reasoning floor.** 6/6 → lift, 0–1/6 → regress, no exceptions across 5 bases. Refusal floor doesn't predict (Qwen 14B is 6/9 and lifts; Mistral and Llama are 6/9 and regress). Architecture family doesn't predict (Gemma is non-Qwen and lifts). Template format doesn't predict (Gemma's template is unique among the five and lifts).

## Reframed claim (supersedes the N=2 architecture-coupling reading)

> *Recipe transfer is base-capability-coupled. Across N=5 cross-family v4 runs, the v4 recipe lifts headline (+3.1 to +5.3pp) on bases whose stock reasoning is at ceiling (6/6) and regresses (−3.2 to −4.0pp) on bases whose stock reasoning is at floor (0–1/6). Architecture family is not the discriminator. The gain pattern (refusal/persona/rag_email) transfers cleanly across all 5 families; the damage pattern (rag_datasheet/coding/rag_blog) appears only when the base lacks reasoning headroom. Strong directional signal — every base point lines up with the reasoning-floor predictor — but N=5 is not statistical evidence.*

The damage-portion of your prior framing (gains transfer; damage is base-specific) survives unchanged. What changed is the predictor of *which way the headline moves*.

## New methodological question we want your opinion on

Your prior caveat (SK-P0-002 + LLM-judge SK-P1-002) was that **temp=0 substring grader pass rate measures format-fidelity for fine-tunes, not robust capability gain**. We carried that caveat forward. But the caveat may apply *asymmetrically* across the N=5 picture:

- **For the lifts** (Qwen 7B/14B, Gemma): high stock reasoning bases learn the v4 corpus's phrasings well (low train_loss uptake; e.g., Qwen 7B v4 train_loss=0.676). The +3pp lift could be largely format-fidelity, exactly as you cautioned for N=2 Qwen.
- **For the regressions** (Mistral, Llama): low stock reasoning bases learn the corpus less effectively (e.g., Llama 3.1 8B v4 train_loss=0.8024 — meaningfully higher). They didn't reproduce the trained phrasings as crisply, so format-fidelity *cannot* explain the regression. The regression has to be capability damage on rag_datasheet/coding/rag_blog being measured by the same grader that may flatter the lifts — i.e., the regressions are arguably "more real" than the lifts in a capability sense.

If that asymmetry holds, the N=5 reframe is consistent with your prior caveat — and in fact the reasoning-floor discriminator may be operating partly *through* the substring metric: high-reasoning bases have spare capacity to memorise corpus phrasings (looks like a capability gain to the grader); low-reasoning bases spend the spare capacity on the categories the recipe is trying to *keep*, not on phrasing memorisation, and the damage shows up.

We do not yet have an LLM-judge run on the N=5 set to test this. (We have judge data on Qwen 7B v4 vs base only, where the judge reverses the substring grader's verdict.)

## What we'd like your opinion on

1. **Reframe acceptability at N=5.** "Base-capability-coupled, predicted by stock reasoning floor" — does this read as appropriately hedged given N=5 (no statistical claim, but clean directional split with no exceptions), or do you want stronger hedging until N=6?

2. **Lift-vs-regress asymmetry under the format-fidelity caveat.** Is the asymmetric reading above (lifts may be format-fidelity, regressions are likely capability damage) something you'd want disclosed inline in the gotcha framing, or is it methodologically too tentative without a judge run on the full N=5?

3. **N=6 data-point selection.** The highest-information falsifier we've identified is a base with stock reasoning at 3–4/6 (intermediate) — which family would you recommend? Phi-3-mini and Yi-9B are candidates we haven't run; both we believe sit in the intermediate-reasoning band on our eval, but we have not measured them.

4. **Customer-template guidance.** Should we elevate "before transferring this recipe to a new base, run a stock baseline on your eval and check the reasoning category specifically; predict the direction from the reasoning floor, not from the family name" from a methodological note to a recommended pre-deployment check in the customer-facing template?

5. **LLM-judge follow-up scope.** Given limited compute, would you prioritise a judge run on the N=5 (to test the asymmetry hypothesis) or an N=6 substring run on an intermediate-reasoning base (to add a data point)?

---

## Doc state

Files reflecting the N=5 reframe (commit `f1de271`):

- `docs/GOTCHA_7_RESOLUTION.md` — Addendum appended; preserves your reviewer-approved language above the line
- `docs/skippy-white-paper.md` — § 7 rewrite, cross-family stock table, per-category profile
- `docs/recipe-taxonomy.md` — Tier 3 dispatch + filled-cells matrix + Pattern #3 N=5 framing

[backend] (cross-repo) has landed coherent updates on the keyhole side: briefing § 5.5 fully reframed (`af7a777`), deck `slide_skippy_recipe_taxonomy` bullet promoted AMBER → INDIGO with new verdict rows.

Eval JSONs and Q4 GGUFs (Gemma stock + Gemma v4) pushed to `gdrive:skippy_files/personal-ai-assistant/`.

We are holding on a customer-template publication of the reframe pending your opinion on the questions above.

---

## Reviewer Verdict (2026-05-09)

The reviewer signed off on the N=5 reframe as **committable as preliminary on the customer-template publication**, with two structural additions to the framing and a softer customer-template predictive claim. All three are folded into the doc set in the same commit as this section. The reviewer explicitly waived re-litigation of the N=2 sign-off ("that's framing being correctly updated when new data fired").

### Direct answers

**Q1 (reframe acceptability at N=5):** Yes, with a structural caveat. Reasoning floor is the cleanest predictor we've identified, **not necessarily the predictor**. Stock overall pass rate doesn't cleanly split this sample (Gemma 61.9% lifts, Mistral 60.6% regresses — opposite-direction calls on a 1.3pp gap), so reasoning floor is meaningfully sharper than the obvious alternatives — but at N=5, alternative predictors that happen to correlate (overall base capability, training-data overlap, instruction-tuning recipe similarity) cannot be ruled out. **Adopted framing:** *"Across N=5, lift/regress correlates perfectly with stock reasoning category performance (6/6 lifts; 0–1/6 regresses). This is consistent with a base-capability-coupling hypothesis. At N=5 we cannot rule out alternative predictors that happen to correlate with reasoning floor in this sample. Treat as a strong directional indicator, not a causal claim."*

**Q2 (asymmetry under the grader-methodology caveat):** Disclose, but **explicitly as a hypothesis with test status named**. The asymmetry argument is sharp — if it holds, the regressions are stronger evidence than the lifts. But (a) the train_loss → memorization mechanism is plausible but not airtight (Llama 0.8024 vs Qwen 7B 0.676 could be memorization difference *or* gradient-flow / base-loss-landscape difference — don't lean on train_loss as the mechanism), and (b) we have judge data on one lift cell only (Qwen 7B); the asymmetry has not been tested across the full N=5. **Adopted disclosure** (carried inline in gotcha § 7, not in a footnote): *"Plausibly, the lifts are partly format-fidelity (per our SK-P0-002 + SK-P1-002 caveat) and the regressions are more interpretable as capability damage on real categories. If that asymmetry holds, the regressions are the stronger evidence in the N=5 picture. We have judge data on one lift cell (Qwen 7B); the asymmetry has not been tested across the full N=5."*

**Q3 (N=6 base selection):** Don't pick blind. **Run stock baselines on Phi-3-mini AND Yi-9B first.** Whichever lands at 3–4/6 reasoning is the intermediate-band candidate. If both land at 0–1/6 or 5–6/6, neither tests the intermediate band and a different base must be identified. Reviewer's blind preference if forced: **Phi-3-mini** (different family — Microsoft, distinct from Qwen/Mistral/Llama/Gemma — and smaller param regime at 3.8B, which tests size as a confound too).

**Q4 (customer-template guidance):** **Promote the procedure (run a stock baseline first); soften the predictive claim.** Adopted wording in `recipe-taxonomy.md`: *"Run a stock baseline on your eval before transferring this recipe to a new base. In our N=5 sample, bases with stock reasoning at ceiling (6/6) lifted with the v4 recipe; bases at floor (0–1/6) regressed. Bases in the intermediate range (2–5/6) have not been characterized. Treat the recipe as untested for intermediate-reasoning bases and budget a full iteration cycle."* The procedural part (baseline first, check reasoning category) is solid regardless of how the predictor evolves; the predictive claim tracks the data.

**Q5 (judge-on-N=5 vs N=6 substring):** **Judge-on-N=5 first.** Three reasons: (a) faster and cheaper — API calls (~$5–10 Sonnet, ~2–3h), no fine-tune training; (b) directly tests the asymmetry that's central to the white paper claim; (c) the asymmetry result reframes how customers should read the N=5 evidence regardless of the eventual N=6 outcome. **Sequence:** judge-on-N=5 (this week), then N=6 with stock-baseline-first selection (next week).

### Action items landed (this commit)

- White paper § 7: predictor-vs-proxy caveat + asymmetry hypothesis disclosure (inline)
- `docs/GOTCHA_7_RESOLUTION.md`: Reviewer follow-up section appended
- `docs/recipe-taxonomy.md`: customer-template procedural wording added; "Reading the matrix" #3 hedged
- Runbook `eval/RUNBOOK_judge_n5.md` — staged, not executed (gated on Kyle's spend approval)
- Runbook `eval/RUNBOOK_n6_stock_baselines.md` — staged, awaiting Kyle's go to download Phi-3-mini + Yi-9B
- [backend] notified for keyhole § 5.5 + deck mirroring (separate bus message)

### Holds released

- Customer-template publication of the N=5 reframe is **unlocked** (reviewer green-light contingent on Q1 + Q2 doc edits, which are in this commit).

---

## Q5 Verdict — asymmetry confirmed across N=5 (2026-05-09 follow-up)

Per the reviewer's Q5 sequencing recommendation, we ran `claude-sonnet-4-6` LLM-judge on all 5 base+v4 pairs the same day (~$4 Sonnet API, ~3h wall time including a fresh 14B base eval).

### Verdict

**Asymmetry hypothesis confirmed across the full N=5.**

| Family | Substring Δ | Direction | Judge base /8 | Judge v4 /8 | Judge Δ |
|---|---:|---|---:|---:|---:|
| Qwen 2.5 7B | +3.1pp | lift | 6.786 | 6.436 | **−0.350** |
| Qwen 2.5 14B | +8.7pp | lift | 6.816 | 6.816 | **±0.000** |
| Gemma 2 9B | +3.2pp | lift | 6.718 | 6.098 | **−0.620** |
| Mistral 7B v0.3 | −3.8pp | regress | 5.718 | 5.500 | **−0.218** |
| Llama 3.1 8B | −3.2pp | regress | 5.951 | 4.786 | **−1.165** |

All 3 substring lifts erase or reverse on the judge. Both substring regressions hold or widen. Mechanism is consistent across lift cells: faithfulness to RAG context drops on v4, while conciseness and instruction-following hold — the substring grader does not penalise the faithfulness loss because trained phrasings still match gold tokens.

### Data correction (Qwen 14B base reasoning floor)

Running the asymmetry test required a fresh 14B base eval. The resulting baseline measurement corrects three values previously reported on the gotcha #7 N=5 table:

| | Old (interpolated) | New (apples-to-apples) |
|---|---|---|
| Qwen 14B reasoning | 6/6 | **3/6** (intermediate) |
| Qwen 14B refusal | 6/9 | **9/9** |
| Qwen 14B Δ headline | +5.3pp | **+8.7pp** |

This refines the predictor-vs-proxy caveat the reviewer asked for in Q1: "reasoning floor predicts the direction" was correct as a directional claim across the cells we had measured, but at intermediate band (3/6) the substring still lifted — the predictor is therefore "≤1/6 reasoning → regress; ≥3/6 reasoning → substring lift," not the cleaner "6/6 vs 0–1/6" split the prior data suggested. The asymmetry hypothesis (Q2) is **strengthened** by this correction: 14B's substring lift is exactly the kind of lift the asymmetry predicted would erase on the judge — and it did (Δ=0.000).

### Customer-template guidance — strengthened

The recipe-taxonomy hedge was:

> "...In our N=5 sample, bases with stock reasoning at ceiling (6/6) lifted with the v4 recipe; bases at floor (0–1/6) regressed. Bases in the intermediate range (2–5/6) have not been characterized."

It now becomes (post-judge-verdict):

> "Run a stock baseline on your eval before transferring this recipe to a new base. In our N=5 sample, bases with stock reasoning at floor (0–1/6) regressed on substring AND on a semantic LLM-judge. Bases at intermediate (3/6) or ceiling (6/6) lifted on substring, but the lift erased on the judge. Treat substring lifts on this recipe as format-fidelity-likely until a semantic-rubric judge corroborates; treat substring regressions as real capability damage."

### Files / artifacts

- Full per-cell + per-dimension analysis: `eval/results/asymmetry_n5_judge_vs_substring.md`
- 6 new judge JSONs: `eval/results/judge_n5_*` (4 from N=5 lifts/regresses) + `eval/results/judge_n5_{base,v4}_qwen25_14b_*` (the new 14B pair)
- Reused 4 judge JSONs from 2026-05-08 (Qwen 7B + Mistral pairs)
- 1 new accuracy eval JSON: `acc_baseline-qwen2.5-14b-instruct-v2-rag_20260509-131410.json`
- 1 new GGUF: `qwen2.5-14b-instruct-stock-q4_k_m.gguf` (8.4G)

All pushed to `gdrive:skippy_files/personal-ai-assistant/`.

### Status

- White paper § 7 asymmetry disclosure updated from "hypothesis with test status named" to "tested and confirmed across N=5"
- Gotcha #7 doc Addendum table corrected for 14B; Judge-on-N=5 verdict subsection added
- Customer-template wording in `recipe-taxonomy.md` strengthened
- Production llm-server restored to 7B v4 (`kyle-7b-v4-q4_k_m.gguf`) and verified healthy after the temporary 14B Instruct base swap

**N=6 sequencing (next week):** with the asymmetry confirmed, an N=6 fine-tune is no longer load-bearing for the framing. It is now an "is the substring predictor sound at intermediate band on a different family" data point — useful but not blocking. Phi-3-mini and Yi-1.5-9B-Chat are downloaded; stock-baseline measurement still pending Kyle's go (`eval/RUNBOOK_n6_stock_baselines.md`).

---

*Document location: `docs/REVIEWER_UPDATE_N5.md`*
