# Semantic regrade — catalog-wide bulk re-score

**Date:** 2026-05-11
**Trigger:** Tier 3 item #2 per [pai-sizer] greenlight — "the highest-leverage remaining item because it corrects the substring-reliability problem the campaign identified."
**Tool:** `eval/regrade_semantic.py` (GPT-4o judge, binary pass/fail per sample, Pydantic-validated, prompt-cached).
**Cost:** 31 parallel regrades, ~$20 OpenAI (~$0.66/eval × 31), ~10 min wall time.
**Coverage:** 33 catalog entries total (2 prior + 31 bulk). All historical fine-tunes (v1/v2/v3/v4 across 7B/14B/MoE), N=7 cross-family campaign (Qwen 7B/14B/Gemma/Mistral/Llama/Yi/Phi-4), MoE variants (Thinking/Instruct-2507/v4/router-v1/full-v1), precision variants (fp8/q8), stock baselines (Phi-3-mini, Yi, Gemma 2 2B).

## TL;DR

Substring vs semantic on the same eval JSONs reveals **family-specific bias** that the gotcha #7 cycle predicted but had not directly demonstrated at scale. Across 33 entries:

- **Qwen-family fine-tunes regrade DOWN sharply** — production Skippy 7B v4: −10.3pp; Skippy 7B v1/v2/v3 + Skippy 14B v1 + MoE-router-v1: −3.2pp each; MoE-thinking: −12.7pp.
- **Non-Qwen stock bases regrade UP** — Gemma 9B +6.0pp, Gemma 2B +4.2pp, Qwen 14B Instruct +3.8pp, Mistral +2.4pp, Llama +1.6pp. Substring was being unfairly harsh.
- **Cross-family v4 fine-tunes split** — Gemma v4 +5.4pp (lifts under BOTH graders), Phi-4 v4 +0.8pp (flat-to-up), Llama v4 +1.0pp, Mistral v4 ±0.0pp, Yi v4 −1.4pp.

**The substring grader has Qwen-family format bias.** Cross-family lifts the campaign attributed to "family-match" in the two-factor model are predominantly substring artifacts — Gemma 9B v4 is the only cross-family lifter that survives semantic regrade.

## Full delta table (sorted by Δ regrade)

| Model | Substring | Semantic | Δ regrade |
|---|---:|---:|---:|
| candidate-moe-thinking-v2-rag (rerun) | 65.1% | 52.4% | **−12.7pp** |
| **★ Skippy 7B v4 (production)** | **73.8%** | **63.5%** | **−10.3pp** |
| candidate-dense-fp8-v2-rag | 66.7% | 58.7% | −7.9pp |
| candidate-moe-thinking-v2-rag (original) | 66.7% | 58.7% | −7.9pp |
| baseline-qwen3-30b-a3b-instruct-2507 | 74.6% | 69.0% | −5.5pp |
| candidate-kyle-qwen3-30b-a3b-router-v1 | 70.6% | 67.5% | −3.2pp |
| candidate-kyle-qwen25-14b-v1 (14B v4) | 76.2% | 73.0% | −3.2pp |
| candidate-kyle-qwen25-7b-v3 | 61.1% | 57.9% | −3.2pp |
| candidate-kyle-qwen25-7b-v1 | 78.6% | 75.4% | −3.2pp |
| candidate-kyle-qwen25-7b-v2 | 74.6% | 71.4% | −3.2pp |
| baseline-qwen25-7b-base | 70.6% | 68.2% | −2.4pp |
| candidate-kyle-yi-1.5-9b-v4 | 37.9% | 36.5% | −1.4pp |
| baseline-qwen25-32b-instruct | 71.4% | 70.6% | −0.8pp |
| candidate-kyle-qwen3-30b-a3b-full-v1 | 65.9% | 65.1% | −0.8pp |
| baseline-phi-4 | 68.2% | 67.5% | −0.7pp |
| candidate-kyle-mistral-7b-v4 | 59.5% | 59.5% | ±0.0pp |
| candidate-kyle-qwen3-30b-a3b-v4 | 64.3% | 64.3% | ±0.0pp |
| baseline-phi-3-mini-4k-instruct | 9.1% | 9.5% | +0.4pp |
| candidate-kyle-phi-4-v4 | 66.7% | 67.5% | +0.8pp |
| candidate-kyle-qwen25-32b-v4-clean | 66.7% | 67.5% | +0.8pp |
| candidate-dense-fp8-v2 | 33.3% | 34.1% | +0.8pp |
| acc_candidate-llama-3.1-8b-kyle-v4 | 53.8% | 54.8% | +1.0pp |
| baseline-yi-1.5-9b-chat | 65.2% | 66.7% | +1.5pp |
| candidate-kyle-qwen25-32b-v1 | 66.7% | 68.2% | +1.6pp |
| baseline-llama-3.1-8b-instruct | 59.5% | 61.1% | +1.6pp |
| baseline-mistral-7b-instruct-v0.3 | 63.5% | 65.9% | +2.4pp |
| candidate-dense-q8-v2-rag | 63.5% | 66.7% | +3.2pp |
| baseline-qwen2.5-14b-instruct | 64.4% | 68.2% | +3.8pp † |
| baseline-gemma-2-2b-it | 54.5% | 58.7% | +4.2pp |
| candidate-gemma-2-9b-v4 | 62.1% | 67.5% | **+5.4pp** |
| baseline-gemma-2-9b-it | 59.1% | 65.1% | **+6.0pp** |
| candidate-dense-fp8-v1 | 54.5% | 63.6% | +9.1pp |
| candidate-qwen25-14b-q8 | 54.5% | 63.6% | +9.1pp |

† 14B Instruct base substring number is on the raw 132-basis here (eval was run 2026-05-09 after the persona regrade pipeline, so `summary` was never re-scaled to 126). The semantic regrader excludes persona by default, so the semantic number is on 126-basis. This basis-mismatch slightly inflates the +3.8pp Δ; the true apples-to-apples delta is smaller. Affects only this one row — all other cells either went through `regrade_for_broken_categories.py` (`summary` already at 126) or were never given the persona prompts.

## Cross-family campaign — semantic-regraded N=7 picture

Recomputing the gotcha #7 N=7 table using SEMANTIC pass rates as the headline:

| Family | Stock substring | Stock semantic | v4 substring | v4 semantic | Substring Δ | Semantic Δ | Direction agrees? |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen 2.5 7B | 70.6% | 68.2% | 73.8% | 63.5% | **+3.2pp** | **−4.8pp** | **✗ REVERSES** |
| Qwen 2.5 14B | 64.4% / 67.5%‡ | 68.2% | 76.2% | 73.0% | +8.7pp‡ | +4.8pp / +5.5pp | ✓ both lift (smaller) |
| Gemma 2 9B | 59.1% | 65.1% | 62.1% | 67.5% | +3.0pp | +2.4pp | ✓ both lift |
| Mistral 7B v0.3 | 63.5% | 65.9% | 59.5% | 59.5% | −4.0pp | −6.4pp | ✓ both regress |
| Llama 3.1 8B | 59.5% | 61.1% | 53.8% | 54.8% | −5.7pp | −6.3pp | ✓ both regress |
| Yi-1.5-9B-Chat | 65.2% | 66.7% | 37.9% | 36.5% | −27.3pp | −30.2pp | ✓ both regress (catastrophic) |
| Phi-4 (14B) | 68.2% | 67.5% | 66.7% | 67.5% | −1.5pp | ±0.0pp | ✓ flat-to-down |

‡ Qwen 14B Instruct base substring is reported on the 132-basis raw `summary`; the 126-basis (post-regrade) value is 67.5% used in earlier docs. Either way the semantic regraded delta lands around +4–5pp.

**Direction reverses on Qwen 7B only.** Skippy 7B v4 — the production model — was the substring-most-impressive v4 in the dataset (+3.2pp lift); semantic regrade flips it to a regression (−4.8pp).

For the rest of the N=7 cells, semantic agrees with substring on direction; magnitudes shift but the picture survives:
- Gemma 9B v4 still lifts (the only true cross-family lift)
- Mistral / Llama / Yi / Phi-4 still flat-or-regress (Phi-4 goes from −1.5 to flat)
- Qwen 14B still lifts (smaller magnitude)

The two-factor model is **partially falsified** by the semantic-regrade view: family-match no longer predicts lift on the production base (Qwen 7B FT was supposed to lift via family-match, and on substring it does, but semantic says no). The two-factor model survives for Qwen 14B (still lifts under both graders) and for Gemma 9B (still lifts via ceiling-reasoning gate).

## Refined predictor (post-semantic-regrade)

The campaign's two-factor model:

> "Lift requires either ceiling stock reasoning (6/6) OR family-match to the corpus source distribution."

The semantic-regrade data refines this:

> "Substring lift requires either ceiling stock reasoning (6/6) OR family-match. **Semantic lift requires either ceiling stock reasoning OR Qwen-14B specifically (the only Qwen-family cell that lifts under semantic).** The production 7B v4 lifts on substring but not on semantic — its apparent capability gain is largely format-fidelity matching trained Qwen phrasings."

**The Skippy 7B v4 production model's headline +3.1pp gain over its base does not survive semantic regrade.** This was foreshadowed by the LLM-judge results (Sonnet −0.350, GPT-4o −0.690 on the v4-vs-base comparison) but the binary semantic regrade now makes it visible as a pass-rate delta consumers can interpret directly.

## Customer implications

The ship decision for Skippy 7B v4 was driven by the three-gate framework (capability + voice + safety), not by the substring headline. The model still passes all three gates (capability via substring + judge mid-range; voice ✓; safety ✓ refusal 9/9). The semantic regrade does not change the production decision; it sharpens what the +3.1pp lift was actually measuring.

For customers running this recipe:
- Don't trust substring +N.Npp lifts on Qwen-family fine-tunes — most or all of it is format-fidelity matching trained phrasings.
- Don't dismiss small substring regressions on non-Qwen bases (Phi-4 −1.6pp, Mistral −4.0pp) — semantic regrade either confirms or widens them.
- Run two judges OR semantic regrade by default. The substring grader is reliable for direction on cross-family-non-Qwen FTs, but its magnitudes are unreliable across the board for FT comparisons.

## Methodology takeaway (for the white paper)

The substring grader has Qwen-family format bias. This is a stronger statement than the campaign's existing "substring is reliable for direction; magnitude unreliable on cross-family intermediate-reasoning" framing — it identifies the SOURCE of the bias (training corpus phrasings come from Qwen, so the grader's gold tokens are Qwen-shaped) and predicts WHICH bases will be over- or under-graded.

**Future eval design:** rotate the corpus across families, OR run the semantic regrader as default. The substring grader's value is speed + determinism for base-vs-base comparisons; it should be augmented (not replaced) with semantic eval for FT comparisons.

## Files / artefacts

- 33 semantic-regraded JSONs in `eval/results/*_semantic.json`
- This analysis: `eval/results/semantic_regrade_catalog.md`
- Source script: `eval/regrade_semantic.py`

All pushed to `gdrive:skippy_files/personal-ai-assistant/eval-results/` for PAI sizer consumption.
