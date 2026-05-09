# Runbook: LLM-judge on N=5 cross-family

**Status:** staged, not executed. Gated on Kyle's API-spend approval. Reviewer-blessed as the next step (Q5, ahead of N=6).

## Purpose

Test the lift-vs-regress asymmetry hypothesis from gotcha #7 (`docs/GOTCHA_7_RESOLUTION.md` Reviewer follow-up). If lifts are partly format-fidelity (per SK-P0-002 + SK-P1-002) and regressions are capability damage, the LLM-judge — which scores on a 4-dim semantic rubric (correctness, instruction-following, faithfulness, conciseness) rather than substring match — should:

- **Lift cells (Qwen 7B, Qwen 14B, Gemma):** judge verdict moves *toward* base or shrinks the v4 advantage compared to the substring grader. (We already know Qwen 7B reverses on judge by 0.35 points.)
- **Regress cells (Mistral, Llama):** judge verdict either matches the substring grader's regression direction *or* widens it (real capability damage shows up under both metrics).

If those patterns hold, the asymmetry is supported. If lifts hold up under judge, format-fidelity is not the explanation for them and the reframe needs reconsideration. If regressions reverse on judge, the regression-as-capability-damage reading is challenged.

## Estimated cost

- ~5 (base, v4) pairs × ~126 prompts × Claude Sonnet 4.6 judge calls ≈ 630 API calls
- Empirical cost: ~$5–10
- Wall time: ~2–3h

## Prerequisites

- `ANTHROPIC_API_KEY` available in environment — **NOT** in `.claude/settings.local.json` (see `reference_settings_local_env_trap.md`). Source it from the user's shell or pass inline.
- Existing eval JSONs in `eval/results/` for all 5 (base, v4) pairs:
  - Qwen 2.5 7B base + Skippy 7B v4 (apples-to-apples temp=0, RAG-on)
  - Qwen 2.5 14B base + Skippy 14B v4
  - `acc_baseline-gemma-2-9b-it_20260508-233541.json` + `acc_candidate-gemma-2-9b-v4_20260508-215949.json`
  - Mistral 7B v0.3 base + Mistral 7B v4 (apples-to-apples — **NOT** the full-seq broken model)
  - Llama 3.1 8B base + Llama 3.1 8B v4
- `eval/run_llm_judge_eval.py` ready (modified per current branch — confirm pinned to `claude-sonnet-4-6`)

## Pre-flight checks

1. Confirm candidate JSONs are temp=0 (apples-to-apples with bases — judge run must hold temperature constant within a pair).
2. Confirm rubric weights match the prior judge run that produced the Qwen 7B reversal (so Qwen 7B's judge verdict here is reproducible as a sanity check).
3. Pin the judge model (`claude-sonnet-4-6`) explicitly in the run config; do NOT rely on a default that may drift.
4. Identify the apples-to-apples JSON pairs by listing `eval/results/acc_*` and matching base/candidate by date + RAG flag.

## Execution sketch (Kyle to fire — confirms API spend)

```bash
# Pair 1: Qwen 2.5 7B (sanity check — should reproduce known judge reversal)
python eval/run_llm_judge_eval.py \
  --baseline eval/results/<qwen2.5-7b-base-v2-rag>.json \
  --candidate eval/results/<kyle-qwen25-7b-v4-v2-rag>.json \
  --judge-model claude-sonnet-4-6 \
  --output eval/results/judge_n5_qwen25_7b_$(date +%Y%m%d-%H%M%S).json

# Pair 2: Qwen 2.5 14B (apples-to-apples 14B base vs 14B v4)
# Pair 3: Gemma 2 9B
#   --baseline eval/results/acc_baseline-gemma-2-9b-it_20260508-233541.json
#   --candidate eval/results/acc_candidate-gemma-2-9b-v4_20260508-215949.json
# Pair 4: Mistral 7B v0.3 (apples-to-apples; NOT the full-seq broken model)
# Pair 5: Llama 3.1 8B
```

(Replace placeholder paths with actual JSON filenames after pre-flight check 4.)

## Outputs

- One judge JSON per pair: `eval/results/judge_n5_<family>_<timestamp>.json`
- Aggregate comparison markdown: `eval/results/acc_diff_n5_judge_vs_substring.md`

## Interpretation gates

| Outcome | Reading | Action |
|---|---|---|
| Lifts shrink/reverse on judge AND regressions match/widen | **Asymmetry confirmed** | Fold into white paper § 7 as adopted finding (replaces "we have judge data on one lift cell" disclosure) |
| Mixed signal across the 5 pairs | **Asymmetry partially confirmed** | Keep the disclosure as hypothesis-with-mixed-test-result; surface the mixed pattern explicitly |
| Lifts hold up on judge OR regressions reverse | **Asymmetry falsified** | Reframe needs reconsideration; flag to reviewer before any customer-template publication |

## Post-actions

1. Push judge JSONs to `gdrive:skippy_files/personal-ai-assistant/eval-results/` per artifact-auto-push rule.
2. Update `docs/REVIEWER_UPDATE_N5.md` with the verdict.
3. Bus update to [backend] (so they can mirror the asymmetry verdict in keyhole § 5.5).
4. Send reviewer the resulting summary.
5. Decide on N=6 sequencing based on the asymmetry verdict.
