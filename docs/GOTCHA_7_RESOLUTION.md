# Gotcha #7 Resolution Document
**Status:** APPROVED — both reviewers signed off 2026-05-08; framing commits landed  
**Date:** 2026-05-08 (revised post-reviewer feedback)  
**Author:** [docs] session  

---

## Overview

This document summarises the four prerequisite tasks required before framing gotcha #7 ("recipe transfer may be base-family-coupled"). All four tasks are now complete. The evidence is more nuanced than the original framing anticipated — see Synthesis.

**Hold on framing commits remains in force** until: (a) pipeline-bug scope confirmed below, (b) Kyle + external reviewer sign off on revised framing.

---

## New Finding: Grader-Methodology Signal (Elevated from Task 3)

Two independent lines of evidence suggest the substring grader rewards **format fidelity at temp=0** rather than correctness robustness. These findings change how all headline numbers should be interpreted.

### A. Temperature sensitivity of fine-tunes (SK-P0-002)

Fine-tuned models are highly temperature-brittle at temp=0.3; base models are essentially flat:

| Model | temp=0 (production) | temp=0.3 (variance bounds) | Δ |
|---|---:|---:|---:|
| Qwen 7B base | 67.4% | 69.1% | +1.7pp |
| Skippy 7B v4 (Qwen FT) | 70.5% | 44.5% | **−26pp** |
| Mistral 7B base | 60.6% | 60.6% | 0pp |
| Skippy Mistral v4 (Mistral FT) | 56.8% | 54.0% | −2.8pp |

**Interpretation:** At temp=0 greedy decoding, fine-tunes produce the exact trained phrasings that the substring grader matches. At temp=0.3, stochastic sampling deviates from those phrasings — even when the semantic content is correct — and the grader fails the sample. Base models don't show this pattern because they weren't trained to reproduce specific substrings.

**What this is NOT:** A claim that the fine-tune is worse. It is a claim that the substring grader measures a narrow distributional property (phrasing consistency), not a broad capability property (correctness under variation).

**Methodological boundary:** The temp=0.3 variance bounds (σ ≈ 1.4–2.3pp) characterise sampling noise *within the temp=0.3 regime only*. They do not gate or invalidate temp=0 production deltas — those are measured in the same temp=0 regime. Comparing a temp=0 delta against a temp=0.3 noise floor would be a category error.

### B. LLM-judge corroboration (SK-P1-002)

The Sonnet 4.6 judge (4-dim rubric: correctness, instruction-following, faithfulness, conciseness) reaches a different conclusion than the substring grader on the v4-vs-base comparison:

- Substring grader: Skippy v4 beats Qwen base by +3.2pp
- LLM judge: Qwen base beats Skippy v4 by 0.35 points (small but opposite direction)

The judge rewards semantic quality; the grader rewards format match. They diverge on the fine-tune vs base comparison — not on the overall ladder ranking, but on the direction of the FT gain. This is a second independent signal that the +3.1pp headline may be a grader artifact.

**Together, A and B tell the same story:** temp=0 substring grader pass rate is a measure of training-induced phrasing consistency, not a robust measure of general capability gain.

**Critical nuance — differential application to base vs fine-tune models:** The format-fidelity characterisation applies specifically to fine-tune-vs-base comparisons. For base models (Qwen 7B base: +1.7pp at temp=0.3; Mistral 7B base: 0pp), the substring grader at temp=0 is reasonably stable — format-fidelity and correctness are not yet decoupled. For fine-tuned models (Skippy 7B v4: −26pp at temp=0.3), they decouple sharply. The correct unpacking: "The substring grader at temp=0 measures format-fidelity-or-correctness; for base models these correlate, but for fine-tunes they can decouple. A fine-tune that learned narrow output patterns matching the gold tokens at greedy decoding can score high on substring without underlying capability robustness." This prevents 'substring is bad' overclaiming — substring is reliable for base evaluation; it is specifically fine-tune-vs-base comparisons where the metric becomes fragile.

---

## Task 1: Mistral Full-Seq Falsification — Pipeline Bug Scope

**Critical question (reviewer directive 2):** Does the pipeline template bug affect the original Mistral v4 −3.8pp result, or only the full-seq falsification attempt?

**Answer: Full-seq only. The original Mistral v4 result is clean.**

Evidence:
- Original Mistral v4 (assistant_only_loss): `acc_candidate-kyle-mistral-7b-v4_20260508-095500.json` — **75/132 = 56.8%** at temp=0. Not zero.
- Full-seq Mistral v4: `acc_candidate-kyle-mistral-7b-v4-fullseq_20260508-132129.json` — **0/132 = 0.0%** at temp=0.
- Inspection of `models/mistral-7b-kyle/kyle-mistral-7b-v4-q4_k_m.gguf`: **no `{% generation %}` markers embedded.** The GGUF uses the stock Mistral template for inference.

The catastrophic 0/132 failure is isolated to the full-seq model. Root cause: the full-seq model was trained without the template patch and with a higher effective LR (same 2e-4 LR, but 2–4× more gradient-contributing tokens). It produces coherent outputs via HF inference but fails the Skippy pipeline — likely because it learned output distributions incompatible with how the pipeline formats prompts.

**Implication for gotcha #7:** The original Mistral v4 −3.8pp delta is a valid, unconfounded data point. Gotcha #7 has N=2 cross-family evidence (Mistral + Llama).

---

## Task 2: Post-Regrade Reconciliation

Persona category (6 prompts) quarantined as BROKEN_SUBSTRING_INCOMPATIBLE. Denominator 132 → 126. All 35+ eval JSONs regraded.

**Headline ladder (temp=0, 126-sample post-regrade basis):**

| Model | Passed /126 | Pass Rate |
|---|---:|---:|
| Skippy 7B v4 (Qwen FT) | 93 | 73.8% |
| Qwen 7B base | 89 | 70.6% |
| Mistral 7B base | 80 | 63.5% |
| Mistral 7B v4 (Mistral FT) | 75 | 59.5% |
| Llama 3.1 8B base | 75 | 59.5% |
| Llama 3.1 8B v4 (Llama FT) | 71 | 56.3% |

---

## Task 3: Variance Bounds (Noise Floor)

**Protocol:** 5 anchored models × 5 reps at temp=0.3. See Grader-Methodology section above for full interpretation.

| Model | Mean (temp=0.3) | σ | Range |
|---|---:|---:|---|
| qwen-7b-base | 69.1% | 2.27pp | 66.7–72.7% |
| mistral-7b-base | 60.6% | 1.42pp | 59.1–62.1% |
| qwen-32b-base | 67.0% | 0.67pp | 66.7–68.2% |
| skippy-7b-v4 | 44.5% | 2.81pp | 40.9–47.7% |
| skippy-mistral-v4 | 54.0% | 2.24pp | 50.8–56.1% |

σ ≈ 1.4–2.3pp for base models within the temp=0.3 regime.

---

## Task 4: Llama 3.1 8B v4 Cross-Family Outcome

**Training:** 2026-05-08, train_loss=0.8024, 2 epochs, 47.7 min on RTX 5090.

| Model | Passed /132 | Rate | vs Base (temp=0) | Δ/σ |
|---|---:|---:|---:|---:|
| Llama 3.1 8B base | 75 | 56.8% | — | — |
| Llama 3.1 8B v4 | 71 | 53.8% | −3.0pp | −1.3σ |

train_loss=0.8024 vs Qwen v4 0.676. Higher loss may indicate weaker signal uptake on the Skippy training distribution.

---

## Synthesis

### Cross-family recipe transfer (all comparisons at temp=0)

| Family | Base | FT | Δ (temp=0) | Δ/σ |
|---|---:|---:|---:|---:|
| Qwen 7B | 67.4% | 70.5% | **+3.1pp** | +1.4σ |
| Qwen 14B | ~67% | 72.7% | **+5.3pp** | +2.3σ |
| Mistral 7B | 60.6% | 56.8% | −3.8pp | −1.7σ |
| Llama 3.1 8B | 56.8% | 53.8% | −3.0pp | −1.3σ |

*Note: all deltas are temp=0 vs temp=0. σ values are from the temp=0.3 variance bounds and apply within that regime only; they are used here as a rough order-of-magnitude reference for what constitutes a meaningful delta, not as a strict statistical gate on temp=0 results.*

### What the data supports

**In the temp=0 substring-grader regime:**
- Qwen family gains (+3.1pp, +5.3pp); both are below 2σ individually but consistent N=2.
- Non-Qwen families regress (−3.8pp Mistral, −3.0pp Llama); both below 2σ individually but directionally consistent N=2.
- Pattern is architecturally split: Qwen benefits, non-Qwen does not.

**With grader-methodology caveat applied:**
- The Qwen gains may partly reflect format-fidelity learning rather than capability gain (temperature-sensitivity + LLM-judge both suggest this).
- The non-Qwen regressions are also measured by the same potentially-format-biased grader, so the magnitude is uncertain in both directions.
- The directional split (Qwen ↑, non-Qwen ↓) may still be real, but the mechanism is unclear: is it architectural, or is it that the training data's phrasing patterns are more Qwen-like?

### What this is NOT

- **Not "gotcha #7 stands" as an established fact.** Both individual cross-family deltas are below 2σ, and the grader methodology is under scrutiny.
- **Not "the fine-tune adds nothing."** The Qwen N=2 gains are real measurements; the question is what they measure.

### Proposed framing (reviewer-approved, awaiting Kyle sign-off)

> "Preliminary observation, N=2 within Qwen (lifts: +3.1pp 7B, +5.3pp 14B) and N=2 across non-Qwen (regressions: −3.8pp Mistral, −3.0pp Llama), directionally consistent within each group with a sign-pattern that varies by family group. Magnitudes are small (3–5pp). The underlying substring-lift premise is itself fragile under non-greedy sampling and alternative grading. Treat as preliminary signal worth flagging to customers, not as an established characterization of recipe transfer.
>
> **What would upgrade this to 'established':** a third non-Qwen family that also regresses (Phi, Yi, Gemma — any), pushing to N=3; or a Qwen v4 that regresses on a different corpus, which would falsify the simpler 'Qwen fine-tunes learn the substring grader's tells' alternative explanation.
>
> **Mistral template-confound disclosure:** Mistral required `{% generation %}` marker patching that Qwen and Llama did not. Whether the −3.8pp Mistral regression is a recipe-architecture interaction or a template-patch-architecture interaction is not fully disentangled (a Mistral assistant-only run with delimiter-based masking would separate them — not run here). Llama used ChatML-like templates and did not need patching, so the −3.0pp Llama result is the cleaner non-Qwen data point. Mistral is corroborating but template-confounded."

---

## Reviewer Questions

1. Is directional consistency across N=2 non-Qwen families (sub-2σ individually, same direction) sufficient to describe gotcha #7 as "preliminary-established" rather than "speculative"?
2. Does the grader-methodology caveat belong inline in the gotcha framing, or in a separate methodological note that gotcha #7 cross-references?
3. Should the Mistral train/inference template mismatch (trained with patched template, inferred with stock) be called out as a residual confound, or is it irrelevant given the assistant_only_loss model scored 56.8% (not catastrophically broken)?

---

## What Must Happen Before Framing Commits

- [x] All four tasks complete
- [x] Pipeline bug scope clarified (full-seq only; N=2 confirmed)
- [x] Temp regimes kept separate in all claims
- [x] "Gotcha #7 stands" language removed; preliminary framing substituted
- [x] Kyle reviews revised framing direction (2026-05-08 — blessed framing commits)
- [x] External reviewer signs off on revised doc (2026-05-08 — all 3 Qs answered; framing direction approved; Mistral confound disclosed; differential format-fidelity nuance added)
- [x] White paper gains a "Grader-Methodology Findings" section (temperature + LLM-judge paired) — commit 24b6ad4
- [x] [backend] SHARED-P0-001 un-held (2026-05-08 17:34)

---

## Addendum: N=3 update — architecture-coupling reading falsified (2026-05-08 ~23:50)

A third non-Qwen family was added per the reviewer-blessed upgrade criterion ("a third non-Qwen family that also regresses"). The selected base was **Gemma 2 9B Instruct** (Google) — chosen as the cleanest possible non-Qwen data point: different template format from both Qwen (ChatML) and Mistral/Llama (`[INST]`-style), uses `<start_of_turn>`/`<end_of_turn>` markers, and **no `{% generation %}` patch needed**. Same hyperparameters, same 6,517-example corpus, same assistant-only loss, same 5090 hardware.

**Result:** Gemma 2 9B v4 = **82/126 = 65.1%** vs stock 78/126 = 61.9%, a **+3.2pp lift** — same magnitude as Qwen 7B v4 lifted from its base.

This **falsifies the architecture-coupling reading at N=2**. Across N=5 cross-family v4 runs:

| Base | Stock reasoning | Stock refusal | v4 Δheadline |
|---|---:|---:|---:|
| Qwen 2.5 7B | 6/6 | 9/9 | +3.1pp |
| Qwen 2.5 14B | 6/6 | 6/9 | +5.3pp |
| Gemma 2 9B | 6/6 | 9/9 | **+3.2pp** |
| Mistral 7B v0.3 | 0/6 | 6/9 | −4.0pp |
| Llama 3.1 8B | 1/6 | 6/9 | −3.2pp |

The discriminator is **stock reasoning capability**, not architecture family or template format. The three bases that ship 6/6 reasoning all lift; the two that ship 0–1/6 reasoning both regress. Refusal floor and template format are not the discriminators (Qwen 14B is 6/9 stock refusal and lifts; Gemma is non-Qwen with non-ChatML template and lifts).

**Revised framing for the gotcha (supersedes the N=2 family-coupled reading above):**

> "Recipe transfer is base-capability-coupled. Across N=5 cross-family v4 runs, the v4 recipe lifts headline (+3.1 to +5.3pp) on bases whose stock reasoning is at ceiling (6/6) and regresses (−3.2 to −4.0pp) on bases whose stock reasoning is at floor (0–1/6). Architecture family is not the discriminator. The gain pattern (refusal/persona/rag_email) transfers cleanly across all 5 families; the damage pattern (rag_datasheet/coding/rag_blog) appears only when the base lacks reasoning headroom. Strong directional signal — every base point lines up with the reasoning-floor predictor — but N=5 is not statistical evidence. A sixth point with stock reasoning at 3–4/6 (intermediate) would be the highest-information next data point to falsify or confirm."

The damage-portion of the original gotcha (gains transfer, damage is base-specific) survives unchanged. What changed is the predictor of *which way the headline moves*.

White paper § 7 and § "Cross-family baselines" updated to reflect this revision. Recipe taxonomy Tier 3 dispatch table marked complete with Gemma row added. Customer template should advise: **before transferring this recipe to a new base, run a stock baseline on your eval and check the reasoning category specifically; predict the direction from the reasoning floor, not from the family name.**

---

*Document location: `docs/GOTCHA_7_RESOLUTION.md`*
