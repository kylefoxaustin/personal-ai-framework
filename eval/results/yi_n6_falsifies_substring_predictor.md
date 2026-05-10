# Yi-1.5-9B-Chat v4 (N=6) — substring predictor falsified; cross-judge corroborates regression

**Date:** 2026-05-10
**Trigger:** Reviewer Q3 hybrid (b) — measure stock baselines on intermediate-band candidates, then optionally fine-tune. Yi was the cleanest N=6 candidate: 68.3% stock, 3/6 reasoning, no context handicap, different family (01.AI).
**Apples-to-apples:** temp=0, RAG=on, `eval/prompts_v2.json`, 132-sample basis (post-regrade 126). Same training pipeline as the rest of the v4 cross-family campaign (SFTTrainer + assistant_only_loss=True + r=64/α=128 LoRA on attention + dense FFN, 2 epochs, 6,517 alpaca examples).
**Cost:** ~67 min training on 5090 (RTX 5090, bf16, no QLoRA), train_loss=0.7509 — comparable to Llama (0.8024) and Qwen 7B (0.676).

## TL;DR

**Yi v4 substring regressed −28.6pp** (86/126 = 68.3% → 50/126 = 39.7%) — the **largest substring regression in any v4 fine-tune in the dataset**. Both cross-judges corroborate: **Sonnet Δ −0.848, GPT-4o Δ −0.714**. The regression mechanism is consistent across judges: **correctness + instruction-following damage** (Sonnet correctness −0.470 / instruct −0.502; GPT-4o correctness −0.214 / instruct −0.476). Faithfulness *slightly improves* on v4 (Sonnet +0.091, GPT-4o +0.071) — a different mechanism from the lift cells (Qwen 7B / 14B / Gemma) where faithfulness *drops* on v4. **For Yi the recipe damages capability on the actual question; for the lift cells the recipe damages RAG-citation discipline.**

This **falsifies the "≥3/6 reasoning → lift on substring" branch of the N=5 predictor.** Yi at 3/6 stock reasoning produced a stronger substring regression than any 0–1/6 cell in the dataset. The 3/6 band is now mixed-direction:

| 3/6 reasoning cell | Substring Δ |
|---|---:|
| Qwen 2.5 14B v4 | **+8.7pp** (lift) |
| Yi-1.5-9B-Chat v4 | **−28.6pp** (regress) |

Within the same band, same recipe, the v4 outcome can flip from "biggest substring lift in the dataset" to "biggest substring regression in the dataset". The substring-direction predictor has narrow validity at intermediate reasoning bands.

## N=6 picture (updated)

| Family | Stock reasoning | Substring Δ | Sonnet Δ | GPT-4o Δ | Verdict |
|---|---:|---:|---:|---:|---|
| Qwen 2.5 7B | 6/6 | +3.1pp | −0.350 | −0.690 | judge erases lift |
| Qwen 2.5 14B | 3/6 | +8.7pp | ±0.000 | −0.214 | judge erases lift |
| Gemma 2 9B | 6/6 | +3.2pp | −0.620 | +0.119 | judge-sensitive (Sonnet erase, GPT-4o marginal lift) |
| Mistral 7B v0.3 | 0/6 | −3.8pp | −0.218 | −0.048 | regress confirmed |
| Llama 3.1 8B | 1/6 | −3.2pp | −1.165 | −1.524 | regress confirmed (strong) |
| **Yi-1.5-9B-Chat** | **3/6** | **−28.6pp** | **−0.848** | **−0.714** | **regress confirmed by both judges; largest substring regression in dataset** |

Eleven of twelve judge passes confirm v4 ≤ base. The Gemma cell remains the one judge-divergence (GPT-4o gives +0.119 vs Sonnet −0.620); Yi adds a sixth cell of corroborated regression with both judges aligned (Sonnet −0.848, GPT-4o −0.714).

## Per-category breakdown — Yi base vs Yi v4

| Category | Stock | v4 | Δ | Note |
|---|---:|---:|---:|---|
| coding | 4/6 | 0/6 | **−4** | recipe damage |
| general | 3/6 | 3/6 | 0 | |
| multihop | 6/9 | 0/9 | **−6** | recipe damage (catastrophic) |
| numerical_precision | 6/6 | 3/6 | **−3** | recipe damage |
| rag_blog | 3/3 | 3/3 | 0 | |
| rag_datasheet | 55/78 | 29/78 | **−26** | dominant damage |
| rag_email | 0/3 | 0/3 | 0 | |
| reasoning | 3/6 | 3/6 | 0 | unchanged at floor of intermediate band |
| refusal | 6/9 | 9/9 | **+3** | the only category that gained |

The recipe traded **+3 refusal calibration for −36 across coding / multihop / numerical_precision / rag_datasheet**. This is the same direction-of-trade as Mistral and Llama (gains on refusal/persona at the cost of retrieval/coding/multihop), but the magnitude is **roughly 7× larger** on the damage side than Mistral's regression.

## Per-dimension breakdown — both judges (the regression mechanism)

| Dimension (0–2) | Yi base — Sonnet | Yi v4 — Sonnet | Δ Sonnet | Yi base — GPT-4o | Yi v4 — GPT-4o | Δ GPT-4o |
|---|---:|---:|---:|---:|---:|---:|
| Correctness | 1.275 | 0.805 | **−0.470** | 1.048 | 0.833 | **−0.214** |
| Instruction-following | 1.600 | 1.098 | **−0.502** | 1.595 | 1.119 | **−0.476** |
| Faithfulness to RAG context | 1.250 | 1.341 | **+0.091** | 0.881 | 0.952 | **+0.071** |
| Conciseness | 1.625 | 1.659 | +0.034 | 1.833 | 1.738 | −0.095 |
| **Total /8** | **5.750** | **4.902** | **−0.848** | **5.357** | **4.643** | **−0.714** |

**Both judges agree on the mechanism:** Yi v4's regression is concentrated on **correctness + instruction-following** (each judge sees ~−0.5 on instruction-following; correctness drops sharper under Sonnet, smaller under GPT-4o). Faithfulness to RAG context *slightly improves* on v4 under both judges — the v4 recipe's RAG-citation pattern actually looks marginally more disciplined to both judges, even as the substring grader sees rag_datasheet collapse (55/78 → 29/78).

**Compare to the lift cells:** Qwen 7B / 14B / Gemma all *lose* faithfulness on v4 (Sonnet sees −0.198 to −0.429 on the lift cells' faithfulness dimension). **The regression mechanism differs from the lift mechanism.** Lift cells lose RAG-citation discipline (faithfulness drops, conciseness gains). Yi loses capability on the question itself (correctness + instruction-following drop, faithfulness slightly improves). Different damage profile, same recipe.

## Implications for the framing

### What's falsified

The N=5 customer-template wording said:

> "Bases at higher stock reasoning (3/6, N=1: Qwen 14B; 6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased on judge. Bases at floor (0–1/6) regressed on substring AND on judge."

With Yi added, the **3/6 row is now mixed**: Qwen 14B lifts, Yi regresses. The reasoning-floor predictor at intermediate band has narrow validity — *which* base at 3/6 you pick matters as much as the band itself. The predictor still holds at the floor (0–1/6 regress) and ceiling (6/6 lift), but the 3/6 middle is now an empirical split with one data point each direction.

### What survives (stronger)

- **The asymmetry hypothesis is unaffected and strengthened.** Lifts on substring don't hold up on judge (3 of 3 lift cells); regressions on substring are real capability damage (3 of 3 regression cells under Sonnet, 4 of 4 under GPT-4o now including Yi).
- **The "lift magnitude on substring does not predict capability gain" claim is unaffected.** Qwen 14B has the biggest substring lift (+8.7pp) and most evaporative judge result; Yi has the biggest substring regression (−28.6pp) and a corroborating judge regression. Substring magnitude tracks judge magnitude *only on the regression side*, not on the lift side.
- **The "two judges by default" standing methodology is strengthened.** Yi shows that even within a band that was "expected to lift," cross-judge corroboration is the only way to know what's real. A team that tested Yi v4 on substring alone and saw +0pp Yi (hypothetical) might have shipped it; a team running cross-judge would see the regression cleanly.

### What needs revising in the customer template

The "3/6 → lift on substring" claim drops. New wording:

> "Bases at floor stock reasoning (0–1/6, N=2: Mistral 7B, Llama 8B) regressed on substring AND on judge. Bases at ceiling stock reasoning (6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased on judge. Bases at intermediate stock reasoning (3/6) produced **mixed substring direction**: Qwen 14B lifted +8.7pp (judge ±0.000 / −0.214), Yi-1.5-9B regressed −28.6pp (GPT-4o judge corroborates at −0.714). Within the intermediate band, *which base you pick matters as much as the band itself*. Substring lifts are format-fidelity-likely until a semantic-rubric judge corroborates; substring regressions are real capability damage. Run two judges by default on any cross-family fine-tune evaluation."

### Mechanism note (speculative)

Yi-1.5-9B-Chat is genuinely cross-family relative to the v4 training corpus (which was built around Kyle's voice on a Qwen base). Qwen 14B is "same-family-extension" — different size of the same base family. Possible refined hypothesis: at 3/6 reasoning, *same-family extensions* lift on substring while *true cross-family* regresses. We cannot distinguish this from "Yi-specifically has some quirk" with N=1 cross-family at 3/6 — would need Mistral-Nemo-12B or DeepSeek-V2-Lite or similar to measure.

## Files / artefacts

- `eval/results/acc_candidate-kyle-yi-1.5-9b-v4-v2-rag_20260510-125703.json` — Yi v4 substring eval
- `eval/results/judge_xj_base_yi_20260510-125933.json` — Yi base GPT-4o judge
- `eval/results/judge_xj_v4_yi_20260510-125936.json` — Yi v4 GPT-4o judge
- `models/yi-1.5-9b-kyle/yi-1.5-9b-kyle-v4-q4_k_m.gguf` (5.0 GB) — pushed to Drive
- `training/output/yi-v4/final/` — LoRA adapter (832 MB)
- `training/logs/train_yi_v4_*.log` — training log

Both judges ran 2026-05-10 (Sonnet completed after Anthropic key was re-staged). Full cross-judge corroboration on Yi is now complete: both judges ≤ 0, both judges agree on mechanism (correctness + instruction-following damage, not faithfulness drop).

Production llm-server cycled to Yi v4 for the eval, then restored to 7B v4 (verified healthy).
