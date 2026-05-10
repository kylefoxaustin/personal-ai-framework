# Reviewer Follow-up — N=6 Yi-1.5-9B-Chat Falsifies "3/6 Lifts" Predictor

**Date:** 2026-05-10
**Builds on:** `docs/REVIEWER_FOLLOWUP_CROSS_JUDGE.md` (your three sharpenings sign-off + customer-template "reviewer-final" status)
**Cross-references:** `eval/results/yi_n6_falsifies_substring_predictor.md` (full per-cell + per-dimension breakdown), `docs/GOTCHA_7_RESOLUTION.md` (Judge-on-N=5 Verdict + N=6 update subsection)
**Asking for:** ~10 min on three questions at the bottom.

---

You blessed customer-template publication at N=5 with the cross-judge sharpenings. Per your Q3 hybrid (b), we measured stock baselines on three candidates (Phi-3-mini, Yi-1.5-9B-Chat, Gemma 2 2B) and picked Yi as the cleanest N=6 candidate. Then we ran the v4 recipe on Yi. **The result falsifies one branch of the predictor you blessed.** Heads up + a few specific questions.

## What happened

Yi-1.5-9B-Chat at 3/6 stock reasoning, 68.3% baseline, was the reviewer-recommended N=6 candidate (different family from anything in N=5, intermediate reasoning band, no context handicap). Trained 67 min on 5090 with the standard v4 recipe (r=64/α=128 LoRA on attention + dense FFN, 2 epochs, 6,517 alpaca examples, assistant_only_loss=True). train_loss=0.7509 — converged normally.

**Yi v4 substring regressed −28.6pp** (86/126 = 68.3% → 50/126 = 39.7%) — the **largest substring regression in any v4 fine-tune in the dataset**, larger than Mistral (−3.8pp) and Llama (−3.2pp) combined.

Per the "two judges by default" standing rule, ran both Sonnet 4.6 and GPT-4o on Yi base + Yi v4. **Both judges corroborate the regression.**

## N=6 picture (full)

| Family | Stock reasoning | Substring Δ | Sonnet Δ | GPT-4o Δ | Direction agree? |
|---|---:|---:|---:|---:|---|
| Qwen 2.5 7B | 6/6 | +3.1pp | −0.350 | −0.690 | ✓ both ≤ 0 |
| Qwen 2.5 14B | 3/6 | +8.7pp | ±0.000 | −0.214 | ✓ both ≤ 0 |
| Gemma 2 9B | 6/6 | +3.2pp | −0.620 | +0.119 | ✗ DISAGREE (faithfulness divergence) |
| Mistral 7B v0.3 | 0/6 | −3.8pp | −0.218 | −0.048 | ✓ both ≤ 0 |
| Llama 3.1 8B | 1/6 | −3.2pp | −1.165 | −1.524 | ✓ both ≤ 0 |
| **Yi-1.5-9B-Chat** | **3/6** | **−28.6pp** | **−0.848** | **−0.714** | **✓ both ≤ 0 (regression confirmed)** |

**11 of 12 judge passes confirm v4 ≤ base.** Gemma remains the one judge-divergence. Yi adds a sixth corroborated regression cell with both judges aligned.

## What this falsifies

Your sign-off wording at N=5 included this customer-template line:

> "Bases at higher stock reasoning (3/6, N=1: Qwen 14B; 6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased on judge."

The 3/6 row is now mixed:

| 3/6 cell | Substring Δ | Sonnet Δ | GPT-4o Δ |
|---|---:|---:|---:|
| Qwen 2.5 14B v4 | **+8.7pp** | ±0.000 | −0.214 |
| Yi-1.5-9B-Chat v4 | **−28.6pp** | **−0.848** | **−0.714** |

Within the same reasoning band, same recipe, same training pipeline, the v4 outcome flips from "biggest substring lift in the dataset" to "biggest substring regression in the dataset". The "≥3/6 → lift on substring" claim does not survive Yi.

The predictor still holds at the **floor** (0–1/6, N=2: both regress) and **ceiling** (6/6, N=2: both lift on substring). The **intermediate band (3/6, N=2) is empirically split** — one lift, one catastrophic regression.

## What survives — strengthened

- **Asymmetry hypothesis** (your Q2): unaffected and strengthened. All 3 substring lifts erase or reverse on at least one judge. All 3 substring regressions corroborated by both judges where measured (Mistral, Llama, Yi all both-judges-≤0). Yi adds a clean third corroborated regression cell.
- **"Lift magnitude does not predict capability gain"** (your sharpening): unaffected. Yi has the **biggest substring regression** and cleanly corroborating judge regression on both judges; Qwen 14B has the **biggest substring lift** and the most evaporative judge result. Substring magnitude tracks judge magnitude *only on the regression side*, not the lift side.
- **"Two judges by default" standing rule** (your Q1 sharpening): strongly reinforced. Yi at 3/6 reasoning, with a Sonnet Δ of −0.848 (well past "marginal"), would have been correctly flagged by either judge alone — but the cross-judge agreement gives the regression magnitude credibility no single judge could carry. The Gemma judge-divergence at the same Sonnet-magnitude level (Gemma −0.620 vs Yi −0.848) demonstrates exactly why "two by default" matters: similar-magnitude single-judge results can be cross-judge stable (Yi) *or* cross-judge unstable (Gemma).

## New finding — different damage profile on regression cells vs lift cells

Both judges agree on Yi's regression mechanism:

| Dimension (0–2) | Yi base — Sonnet | Yi v4 — Sonnet | Δ Sonnet | Yi base — GPT-4o | Yi v4 — GPT-4o | Δ GPT-4o |
|---|---:|---:|---:|---:|---:|---:|
| Correctness | 1.275 | 0.805 | **−0.470** | 1.048 | 0.833 | **−0.214** |
| Instruction-following | 1.600 | 1.098 | **−0.502** | 1.595 | 1.119 | **−0.476** |
| Faithfulness to RAG context | 1.250 | 1.341 | **+0.091** | 0.881 | 0.952 | **+0.071** |
| Conciseness | 1.625 | 1.659 | +0.034 | 1.833 | 1.738 | −0.095 |

**Yi v4 loses correctness + instruction-following; its RAG-faithfulness slightly improves.** Compare to the lift cells (Qwen 7B / 14B / Gemma) where faithfulness *drops* sharply (Sonnet sees −0.198 to −0.429 on the lift cells' faithfulness dimension).

**Customer-actionable refinement:** the v4 recipe damages *different things* on different bases. Lift cells lose RAG-citation discipline. Regression cells (especially intermediate-reasoning ones like Yi) lose capability on the question itself. The substring grader sees both kinds of damage, but for different reasons. A judge that weights faithfulness heavily will catch lift-cell damage; a judge that weights correctness will catch regression-cell damage.

## Proposed customer-template wording (replacing the falsified line)

Folded into `recipe-taxonomy.md` already in commit `38c258d`:

> "Bases at floor stock reasoning (0–1/6, N=2: Mistral 7B, Llama 8B) regressed on substring AND on judge. Bases at ceiling stock reasoning (6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased on judge. Bases at intermediate stock reasoning (3/6) produced **mixed substring direction**: Qwen 14B lifted +8.7pp (judge ±0.000 / −0.214); Yi-1.5-9B-Chat regressed −28.6pp (both cross-judges corroborate: Sonnet −0.848, GPT-4o −0.714). Within the intermediate band, *which base you pick matters as much as the band itself*."

The "Across N=12 judge passes, 11 of 12 confirm v4 ≤ base" headline is unchanged.

## Three questions

**Q1 — Is the falsification framing right, or do you want it framed differently?**
Our reading: "≥3/6 → lift" was over-fit to one data point (Qwen 14B); N=6 now has one data point each direction at 3/6, so the band is empirically mixed. We can't determine a within-band predictor from N=2. The right read is "predictor holds at floor and ceiling; intermediate is mixed; cross-judge is mandatory at any band". An alternative read would be "the predictor never held; Qwen 14B at 3/6 was an outlier/accident in N=5 and Yi exposes that". Both are consistent with the data; our framing leans toward the first (which preserves the predictor's narrow validity at the extremes). We're open to either if you prefer the second framing.

**Q2 — Do we need a third cross-family base at 3/6 to distinguish "Yi-specific quirk" from "true cross-family at 3/6 regresses"?**
Speculative: Qwen 14B is a *size-extension* of the same family the v4 corpus was built on (Qwen); Yi is genuinely cross-family at 3/6. Possible refinement: same-family-extension lifts on substring at 3/6; true cross-family at 3/6 regresses. We can't distinguish this from "Yi-specifically has a quirk" with N=1 cross-family at 3/6. Candidates for a third 3/6 cross-family measurement: Mistral-Nemo-12B, DeepSeek-V2-Lite, Phi-4 (newer Phi with 128K context, no context-saturation concern). Cost: ~1 day each (download + GGUF + stock baseline + FT + cross-judge). Worth it, or hold for now?

**Q3 — Customer-template publication shipping decision?**
The N=5 publication was reviewer-final and ready to ship. Yi falsifies one branch of that publication's predictor. Options:
- **(a)** Ship with the updated wording (commit `38c258d`) — the publication is *more* accurate with Yi than without, and the "two judges by default" rule is *more* justified. Acknowledge the Yi cell as part of the audit trail.
- **(b)** Pause publication until a third 3/6 cross-family base lands (Q2), so the framing is "established at intermediate band" rather than "mixed-band-known-after-N=2".
- **(c)** Pause publication until you've seen the new wording end-to-end and re-blessed.

Our default if no objection: **(c)** — quick re-look from you, then ship. (a) leaves a slightly under-reviewed customer-facing publication; (b) delays for data that may take a week.

## Status

- White paper § 7, GOTCHA Addendum, recipe-taxonomy customer-template — all updated with Yi cross-judge result (commit `38c258d` + this commit folding in Sonnet numbers)
- Customer-template wording (recipe-taxonomy.md) reflects mixed-direction at 3/6 with both judge numbers
- 4 new judge JSONs (2 Sonnet + 2 GPT-4o on Yi base + Yi v4) on Drive + analysis MD
- Yi v4 Q4 GGUF (5GB) on Drive
- Yi LoRA adapter (832MB safetensors) local; not yet pushed to Drive
- Production llm-server cycled to Yi v4 for eval, restored to 7B v4 (verified healthy)
- [backend] notified on the bus; mirroring held pending your re-look

Holds: customer-template publication paused pending your call on Q3.

---

*Document location: `docs/REVIEWER_FOLLOWUP_N6_YI.md`*
