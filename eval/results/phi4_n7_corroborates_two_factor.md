# Phi-4 v4 (N=7) — corroborates two-factor model; Yi is no longer an outlier on direction

**Date:** 2026-05-10
**Trigger:** Reviewer 2026-05-10 Q2 verdict — "Phi-4 is the highest-information experiment available; the two-factor model is testable; if Phi-4 lifts, the two-factor model breaks and we're back to 'Yi has a specific quirk.'"
**Apples-to-apples:** temp=0, RAG=on, `eval/prompts_v2.json`, 132-sample basis (post-regrade 126). Same training pipeline as the N=6 cycle (SFTTrainer + assistant_only_loss=True + r=64/α=128 LoRA on attention + dense FFN, 2 epochs, 6,517 alpaca examples). **QLoRA nf4 base** (14B doesn't fit on 5090 in bf16; same approach as Qwen 14B v4).
**Cost:** ~111 min training, ~5 min merge, ~3 min GGUF + quantize, ~10 min eval, ~5 min judges. ~$2 OpenAI + ~$0.80 Anthropic for cross-judge.

## TL;DR

**Phi-4 v4 substring regressed −1.6pp** (90/126 = 71.4% → 88/126 = 69.8%). Both cross-judges corroborate the regression: **Sonnet Δ −0.627, GPT-4o Δ −0.834**. Per reviewer's pre-staged sequencing: this is the N=7 corroboration outcome — **the two-factor model holds at 7/7 cells**.

| Family | Reasoning | Family-match | Substring Δ | Sonnet Δ | GPT-4o Δ | Two-factor predicts | Observed |
|---|---:|---|---:|---:|---:|---|---|
| Qwen 2.5 7B | 6/6 | ✓ Qwen | +3.1pp | −0.350 | −0.690 | lift | substring ↑ judge erases ✓ |
| Qwen 2.5 14B | 3/6 | ✓ Qwen | +8.7pp | ±0.000 | −0.214 | lift | substring ↑ judge erases ✓ |
| Gemma 2 9B | 6/6 | ✗ | +3.2pp | −0.620 | +0.119 | lift | substring ↑ judge-sensitive ✓ |
| Mistral 7B v0.3 | 0/6 | ✗ | −3.8pp | −0.218 | −0.048 | regress | both ✓ |
| Llama 3.1 8B | 1/6 | ✗ | −3.2pp | −1.165 | −1.524 | regress | both ✓ |
| Yi-1.5-9B-Chat | 3/6 | ✗ | −28.6pp | −0.848 | −0.714 | regress | both ✓ |
| **Phi-4 (14B)** | **3/6** | **✗** | **−1.6pp** | **−0.627** | **−0.834** | **regress** | **both ✓** |

**Phi-4 was the falsification test for the two-factor model. It did not falsify.** Yi is no longer an outlier on direction — Phi-4 is the second cross-family base at 3/6 reasoning, and it regresses too. Yi remains an outlier on *substring magnitude* (−28.6pp catastrophic vs Phi-4's −1.6pp at the noise floor), but both substring regression direction and judge corroboration match.

**13 of 14 judge passes confirm v4 ≤ base** (7 cells × 2 judges = 14 passes; only Gemma + GPT-4o is positive at +0.119).

## Per-dimension breakdown — Phi-4 (both judges)

| Dimension (0–2) | Phi-4 base — Sonnet | Phi-4 v4 — Sonnet | Δ Sonnet | Phi-4 base — GPT-4o | Phi-4 v4 — GPT-4o | Δ GPT-4o |
|---|---:|---:|---:|---:|---:|---:|
| Correctness | 1.590 | 1.400 | **−0.190** | 1.738 | 1.357 | **−0.381** |
| Instruction-following | 1.744 | 1.825 | **+0.081** | 1.833 | 1.833 | ±0.000 |
| Faithfulness to RAG context | 1.769 | 1.400 | **−0.369** | 1.762 | 1.333 | **−0.429** |
| Conciseness | 1.974 | 1.825 | −0.149 | 1.976 | 1.952 | −0.024 |
| **Total /8** | **7.077** | **6.450** | **−0.627** | **7.310** | **6.476** | **−0.834** |

**Phi-4 stock is the highest-scoring base in the entire dataset on both judges** (Sonnet 7.077, GPT-4o 7.310). v4 erodes that to 6.450/6.476 — still good but no longer best.

## Damage profile — Phi-4 sits between lift and regress mechanisms

Per-dimension damage profiles across N=7:

| Family | Damage profile (both judges) | Mechanism |
|---|---|---|
| Qwen 7B / 14B / Gemma 9B (lift cells) | **faithfulness drops sharply**; correctness slight; instruction-following ±; conciseness ± | Lose RAG-citation discipline; substring grader doesn't penalise |
| Mistral 7B / Llama 8B (regress cells, low-reasoning) | correctness drops; instruction-following drops; faithfulness drops; conciseness drops (especially Llama) | Lose capability on the question itself; substring grader catches it |
| Yi 9B (regress cell, intermediate-reasoning, catastrophic) | **correctness + instruction-following drop sharply**; faithfulness *slightly improves*; conciseness flat | Loses capability on the question; *more disciplined* on citations |
| **Phi-4 (regress cell, intermediate-reasoning, mild)** | **correctness + faithfulness both drop**; instruction-following slight up (Sonnet) or flat (GPT-4o) | **Hybrid: lift-cell faithfulness-drop + regress-cell correctness-drop** |

**Phi-4 is the first cell where the damage profile mixes lift-cell and regress-cell mechanisms.** Lift cells lose faithfulness without losing correctness; regression cells (Yi, Llama, Mistral) lose correctness with various effects on faithfulness. Phi-4 loses both at moderate magnitudes — a hybrid that the substring grader sees as ~noise (−1.6pp), the GPT-4o judge sees as substantial (−0.834), and Sonnet sees as moderate (−0.627).

**Methodology takeaway emerging:** the substring grader's reliability *as a regression detector* varies by base. On low-reasoning cross-family bases (Mistral, Llama) the substring grader is reasonable. On intermediate-reasoning cross-family bases (Yi, Phi-4), the substring grader's regression magnitude is wildly inconsistent (−28.6pp vs −1.6pp) while the judges give consistent signal (−0.6 to −0.9 on both). **For cross-family deployment decisions, the judge is the reliable signal; the substring grader can over- or under-detect.**

## What this preserves (strengthened)

- **Two-factor model:** corroborated at N=7 (7/7 cells fit prediction). Yi is no longer an outlier on direction. The model's falsifiable prediction held.
- **Asymmetry hypothesis:** unaffected. 3 substring lifts erase or reverse on at least one judge; 4 substring regressions corroborated by both judges (where measured).
- **"Lift magnitude does not predict capability gain":** unaffected. Phi-4 adds a new wrinkle — **substring regression magnitude does not predict capability damage magnitude either**. Yi −28.6pp substring → −0.8 judge; Phi-4 −1.6pp substring → −0.8 judge. Same judge damage, very different substring magnitudes.
- **"Two judges by default" standing rule:** strongly reinforced. The substring grader saw Phi-4 as "essentially flat" (−1.6pp within noise floor σ≈1.4–2.3pp); both judges saw clear regression. A team relying on substring alone might have shipped Phi-4 v4 thinking it was "close enough to stock." Cross-judge prevents that mistake.

## What's new (worth noting)

- **Phi-4 stock is the highest-scoring base in the dataset on both judges** — better than Qwen 2.5 7B (the production model's base) by 0.291 on Sonnet and 0.524 on GPT-4o. This means Phi-4 stock would be a strong candidate to *replace* Qwen as a production base if voice/persona were transferred via a different mechanism (DPO + small persona-only corpus?) — but the v4 recipe damages it.
- **Hybrid damage profile is a new data point** — Phi-4 loses both correctness AND faithfulness, where prior cells lost one or the other. Worth flagging for future characterization.
- **Substring grader's reliability varies by base type** — not the primary finding here, but useful for customer guidance. We should recommend customers always run cross-judge, especially when their base is cross-family intermediate-reasoning.

## Customer-template wording update

The reviewer-blessed N=6 wording is unchanged in direction; we add a one-paragraph N=7 corroboration:

> "**N=7 update (Phi-4): two-factor model corroborated.** Phi-4 (Microsoft, 14B, 3/6 reasoning, cross-family) v4 substring regressed −1.6pp on the substring grader (within the temp=0 noise floor σ≈1.4–2.3pp), but both cross-judges corroborate the regression substantially (Sonnet Δ −0.627, GPT-4o Δ −0.834). This is the falsification test the two-factor model predicted: a third cross-family intermediate-reasoning base should regress. It did. Customer implication: **substring magnitude alone is unreliable for cross-family intermediate-reasoning bases** — Yi's −28.6pp and Phi-4's −1.6pp are both substring regressions, but the customer-relevant damage (judge Δ) is similar between them (~−0.7 to −0.9). Run cross-judge on any cell whose deployment decision turns on the v4 result."

## Open question for the reviewer

None — reviewer's pre-staged sequencing covered this outcome. *"If it regresses, two-factor model gets stronger evidence and you update the published doc with the N=7 corroboration."* That's what this analysis is.

Optional next step (not asked): a fourth 3/6 cross-family base would consolidate Yi/Phi-4 from N=2 to N=3 in that band, helping characterize the substring-magnitude variance within-band. Candidates: Mistral-Nemo-12B (Mistral-family but newer release, 3/6 reasoning unknown), DeepSeek-V2-Lite, Qwen3-8B-base (Qwen-family but a different generation). Cost ~1 day each. Not load-bearing.

## Files / artefacts

- `eval/results/acc_baseline-phi-4-v2-rag_20260510-143936.json` — stock baseline (71.4%)
- `eval/results/acc_candidate-kyle-phi-4-v4-v2-rag_20260510-165134.json` — v4 (69.8%)
- 4 judge JSONs: `judge_n5_{base,v4}_phi4_*` (Sonnet) + `judge_xj_{base,v4}_phi4_*` (GPT-4o)
- `models/phi-4-hf/phi-4-stock-q4_k_m.gguf` (8.5G, on Drive)
- `models/phi-4-kyle/phi-4-kyle-v4-q4_k_m.gguf` (8.5G, on Drive)
- `training/output/phi4-v4/final/` — LoRA adapter (163M; smaller than Yi's 832M because QLoRA-trained)
- `training/{train_lora_phi4_v4.py,merge_lora_phi4.py}` — committed for reproducibility

Production llm-server cycled to Phi-4 stock for the baseline + to Phi-4 v4 for the candidate eval, then restored to 7B v4 (verified healthy each time).
