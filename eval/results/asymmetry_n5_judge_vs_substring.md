# Lift-vs-regress asymmetry — judge-on-N=5 verdict

**Date:** 2026-05-09
**Trigger:** Reviewer follow-up Q5 (`docs/REVIEWER_UPDATE_N5.md`).
**Authoring:** [docs] session.

## TL;DR

The lift-vs-regress asymmetry hypothesis from gotcha #7 is **confirmed** at N=5. All three substring lifts erase or reverse on the LLM-judge; both substring regressions hold or widen.

| Family | Substring Δ | Substring direction | Judge base /8 | Judge v4 /8 | Judge Δ | Verdict |
|---|---:|---|---:|---:|---:|---|
| Qwen 2.5 7B | +3.1pp | lift | 6.786 | 6.436 | **−0.350** | substring lift reverses |
| Qwen 2.5 14B | +8.7pp | lift | 6.816 | 6.816 | **±0.000** | substring lift erased |
| Gemma 2 9B | +3.2pp | lift | 6.718 | 6.098 | **−0.620** | substring lift reverses (strong) |
| Mistral 7B v0.3 | −3.8pp | regress | 5.718 | 5.500 | **−0.218** | regress holds on judge |
| Llama 3.1 8B | −3.2pp | regress | 5.951 | 4.786 | **−1.165** | regress widens on judge |

**No lift cell is corroborated by the judge. Both regress cells are corroborated.** The metric-asymmetry reframing of gotcha #7 has empirical support across all five cells.

## Method

- **Judge model:** `claude-sonnet-4-6`. 4-dim rubric (correctness, instruction-following, faithfulness to RAG context, conciseness), each 0–2.
- **Sampling:** `eval/run_llm_judge_eval.py` selects up to 50 stable held-out (prompt, sample) pairs per eval JSON via seed=42, one sample per prompt across categories with `BROKEN_SUBSTRING_INCOMPATIBLE` (persona) excluded. n_judged after filtering ranges 38–42 across the 10 model-judge runs.
- **Pairs:**
  - **Qwen 2.5 7B + 14B + Mistral 7B v0.3:** existing apples-to-apples eval JSONs at temp=0, RAG-on, v2 prompts. Qwen 7B + Mistral judge JSONs from 2026-05-08; Qwen 14B base eval freshly run today (see "Data correction" below).
  - **Gemma 2 9B:** 2026-05-08 base + v4 eval JSONs.
  - **Llama 3.1 8B:** 2026-05-07/08 base + v4 eval JSONs (apples-to-apples; the full-seq broken Mistral was excluded).
- **Cost:** ~$0.40/run × 10 runs ≈ $4 total. Wall time ~3–5 min per run, all 10 runs done in ~3 hours of wall-clock with parallelism.

## Per-dimension breakdown (lift cells)

| Cell | Correct (b/v) | Instruct (b/v) | Faithful (b/v) | Concise (b/v) |
|---|---|---|---|---|
| Qwen 7B | 1.476 / 1.462 | 1.690 / 1.769 | 1.762 / 1.333 | 1.857 / 1.872 |
| Qwen 14B | 1.526 / 1.500 | 1.711 / 1.868 | 1.711 / 1.447 | 1.868 / 2.000 |
| Gemma 2 9B | 1.487 / 1.366 | 1.667 / 1.634 | 1.564 / 1.366 | 2.000 / 1.732 |

Pattern: across all three lifts, **faithfulness drops on v4** (Qwen 7B −0.43, Qwen 14B −0.26, Gemma −0.20). v4 trades RAG-context grounding for trained phrasing patterns. Conciseness and instruction-following hold or improve, but faithfulness loss offsets them in the totals.

## Per-dimension breakdown (regress cells)

| Cell | Correct (b/v) | Instruct (b/v) | Faithful (b/v) | Concise (b/v) |
|---|---|---|---|---|
| Mistral 7B | 1.231 / 1.125 | 1.538 / 1.675 | 1.333 / 1.025 | 1.615 / 1.675 |
| Llama 3.1 8B | 1.268 / 1.143 | 1.512 / 1.429 | 1.537 / 1.238 | 1.634 / **0.976** |

Pattern: regress cells lose **correctness** (Mistral −0.11, Llama −0.13) and **faithfulness** (Mistral −0.31, Llama −0.30) — the same faithfulness loss as the lifts, plus capability damage. Llama also collapses on conciseness (1.634 → 0.976), driving the −1.165 widening.

## Data correction — Qwen 14B base reasoning floor

The fresh apples-to-apples baseline measurement landed today: **Qwen 2.5 14B Instruct stock = 85/126 = 67.5%** on the v2-rag eval. This is the first clean 14B baseline against this exact prompt set.

Per-category breakdown of the new baseline:

| Category | Stock | v4 | Δ |
|---|---:|---:|---:|
| reasoning | **3/6** | 6/6 | +3 |
| multihop | 4/9 | 6/9 | +2 |
| rag_datasheet | 51/78 | 60/78 | +9 |
| coding | 6/6 | 6/6 | 0 |
| numerical_precision | 6/6 | 6/6 | 0 |
| rag_blog | 3/3 | 3/3 | 0 |
| rag_email | 0/3 | 0/3 | 0 |
| general | 3/6 | 3/6 | 0 |
| refusal | **9/9** | 6/9 | **−3** |
| (persona — quarantined) | 0/6 | 0/6 | n/a |

Three things this changes vs the gotcha #7 doc's prior N=5 framing:

1. **Stock reasoning floor was wrong.** The doc claimed 14B = 6/6 reasoning. Actual fresh measurement is **3/6** (intermediate band). The 6/6 number was likely interpolated from older runs or different prompt sets.
2. **Stock refusal floor was wrong.** Doc claimed 14B = 6/9 refusal. Actual is **9/9**.
3. **Substring Δ is bigger than reported.** Doc claimed +5.3pp; actual is **+8.7pp** (76.2% − 67.5%).

Implications for the predictor:

- The original "6/6 → lift, 0–1/6 → regress, intermediate uncharacterized" framing breaks: 14B at 3/6 stock reasoning **lifted on substring**. The intermediate band is no longer uncharacterized.
- But the asymmetry hypothesis **remains intact and is strengthened**: 14B's substring lift is exactly the kind of lift the asymmetry hypothesis predicted would erase on the judge — and it did (Δ=0.000).
- The reasoning-floor predictor is a clean predictor of *substring* direction across {≥3 lifts, ≤1 regresses} but is not a clean predictor of *judge* direction (judge says: no lift cell holds; both regresses hold).

## Asymmetry verdict

- **Lifts:** substring captures format-fidelity gains that do not survive a semantic-rubric judge. 3/3 lift cells erase or reverse on judge. The mechanism is consistent across families: faithfulness to RAG context drops on v4, and the substring grader does not penalise the loss because the trained phrasings still match gold tokens.
- **Regressions:** substring captures real capability damage that the judge corroborates. 2/2 regress cells hold or widen on judge. The damage profile is correctness + faithfulness (and on Llama, conciseness too).

**This refines the gotcha #7 framing in two ways:**

1. The "base-capability-coupled" hypothesis (what predicts *substring* direction) is partially broken by the 14B intermediate-reasoning lift. The reasoning-floor predictor is therefore weaker than reported at the pre-correction N=5 picture; with the corrected 14B floor, the predictor would need to be: "≤1/6 reasoning → regress; ≥3/6 reasoning → substring lift (but see judge caveat)."
2. The metric-asymmetry hypothesis (what predicts whether substring direction is *real*) is fully confirmed. **Substring lifts should not be treated as capability gains; substring regressions are real.**

## Customer-template implication

Stronger version of the recipe-taxonomy guidance, with the asymmetry-confirmed hedge:

> "Run a stock baseline on your eval before transferring this recipe to a new base. In our N=5 sample, bases with stock reasoning at floor (0–1/6) regressed on substring AND on a semantic LLM-judge. Bases at intermediate (3/6) or ceiling (6/6) lifted on substring, but the lift erased on the judge. Treat substring lifts on this recipe as format-fidelity-likely until a semantic-rubric judge corroborates; treat substring regressions as real capability damage. If you must ship from substring alone, ship only on a base that does not regress."

## Files / artefacts

- 10 judge JSONs in `eval/results/judge_n5_*` and `eval/results/judge_{baseline,candidate}-*-2026-05-08*` (existing pairs)
- 1 new accuracy eval JSON: `eval/results/acc_baseline-qwen2.5-14b-instruct-v2-rag_20260509-131410.json`
- 1 new GGUF: `models/qwen2.5-14b-hf/qwen2.5-14b-instruct-stock-q4_k_m.gguf` (8.4G; pushed to Drive)

All artefacts pushed to `gdrive:skippy_files/personal-ai-assistant/`.
