# Retriever fix A/B — the recall improvement made the PRODUCT WORSE

**Date:** 2026-07-10 · Qwen2.5-7B v4 (production), v2.1 prompts, --with-rag, temp 0
**Clean A/B:** same model, same prompts, ONLY the retriever code differs
(git-stash toggle of pipeline/advanced_rag.py + rag_service.py).

## Result — three graders agree, ~7-8pp REGRESSION

| grader | broken retriever | fixed retriever | Δ (fixed − broken) |
|---|--:|--:|--:|
| substring | 76.3% | 68.4% | **−7.9 pp** |
| GPT-4o judge | 74.6% | 67.5% | **−7.0 pp** |
| Sonnet judge | 75.4% | 67.5% | **−7.9 pp** |

114 samples graded cleanly by both judges in both arms. −7pp against a
σ≈1.4–2.3pp noise floor is 3–5σ. **Not a grading artifact** — two semantic judges
from different families confirm it.

## Why: recall@8 was the WRONG objective

Retrieval recall@8 (does a top-8 chunk contain the gold fact) went **1/7 → 6/7**.
That improvement is real. It also made answers worse, via three mechanisms, all
semantic-confirmed:

1. **Lost diversity / over-concentration.** `rag_imx93_cortex` ("what Arm cores?"
   → A55 AND M33): the broken retriever pulled chunks from *different* documents
   and the model covered both cores. The fixed retriever's over-fetch+rerank
   pulled all 3 chunks from the *same* section (Cortex-A55 detail); the model
   answered only about A55 and never mentioned M33. Optimizing for "most
   relevant" crowded out the coverage a multi-part question needs. (Also
   `multihop_peripheral_count`.)

2. **Over-grounding on narrow chunks.** `rag_imx93_uart`: the fixed retriever
   surfaced a specific pin-mux register chunk (`LPUART3_RTS_B → GPIO2_IO17`), and
   the model grounded on that narrow detail instead of naming the IOMUXC block the
   question asked about. A more specific answer to a *different* question.

3. **Reduced refusal → hallucination.** `rag_ds_imx93_149_gpio` (149 GPIO does not
   exist): broken retriever gave weak context, the model fell back on correct
   parametric knowledge and refused ("i.MX 93 has up to 128... doesn't have 149").
   Fixed retriever surfaced adversarial i.MX 93 context and the model confabulated
   "The i.MX 93 has 149 GPIO pins." Better retrieval reduced faithfulness.

Regressions (broken pass → fixed fail, semantic): `rag_imx93_149_gpio` 3→0,
`rag_imx93_cortex` 3→0, `rag_imx93_uart` 3→0, `rag_imx93_ddr_width` 1→0,
`multihop_peripheral_count` 2→1. Improvements: `rag_ds_standard_id_filters` 2→3,
`rag_ds_src_gpr5_offset` 2→3, `reason_multistep` 1→2. **Net −7pp.**

## The lesson

This is the day's theme inverted. Morning: *don't measure a layer through the
layer above it* (INT8 measured through a brittle grader → false −3.97pp). Evening:
*don't OPTIMIZE a layer's isolated metric and assume the system improves*
(recall@8 optimized → the system got 7pp WORSE). Both are the same truth: a
component metric is a proxy, and proxies diverge from the goal.

**recall@8 is not the objective. End-to-end answer accuracy is.** The retriever
changes must be gated on the A/B eval, and right now they fail it.

## Recommendation

**Do NOT ship the retriever changes as-is.** What the evidence says is actually
needed:
- **Diversity, not just recall** — MMR / max-marginal-relevance or source-dedup so
  multi-part questions get chunks that TOGETHER cover the answer, instead of 3
  near-duplicates of the single best match.
- **A faithfulness guard** — when the retrieved chunks don't actually contain the
  queried entity/number, prefer refusal over grounding (the 149-GPIO failure).
- **The end-to-end A/B (this doc) as the merge gate**, not recall@8.

The recall benchmark (`eval/bench_retrieval.py`) and this A/B harness stay — they
are the tools to iterate against. The current diff does not merge.

---

## Update — v2 (MMR diversity) recovers the regression to PARITY, not a win

Added Maximal Marginal Relevance to the final chunk selection (token-Jaccard +
same-source penalty, λ=0.6) so multi-part questions get diverse coverage instead
of near-duplicates of the single best match. Verified at the retrieval layer: the
"Arm cores" query now covers A55 AND M33 across 5 distinct sources (was 3 chunks
from one A55 section).

Clean 3-way A/B, Qwen 7B v4, v2.1, 111 samples both judges graded in all arms:

| grader | broken | recall-only | v2 (MMR) | v2 − broken |
|---|--:|--:|--:|--:|
| substring | 85/111 | 76/111 | 85/111 | **+0.0** |
| GPT-4o | 84/111 | 76/111 | 84/111 | **+0.0** |
| Sonnet | 86/111 | 77/111 | 86/111 | **+0.0** |

MMR fully undid the −7pp recall-only regression, but the retriever changes net to
**zero** end-to-end. Per-prompt it is a genuine trade (GPT-4o): MMR WON the obscure
facts — `thermal_rja` 0→3, `thermal_rjc` 0→3, `standard_id_filters` 2→3,
`reason_multistep` 1→3 — and LOST an equal number where retrieval derailed a
previously-correct answer — `uart` 3→0, `vdd_ana` 3→0, `multihop_peripheral` 1→0,
`cortex_a55_l2` 3→2, `src_gpr5_offset` 2→1. Nine up, nine down.

## The real finding: RAG is ~break-even for this model+corpus

The "broken" retriever scores 76% BECAUSE Qwen 7B v4 has strong parametric
knowledge of NXP parts. Every retrieval improvement trades wins on facts the model
can't know (thermal resistances, register offsets) against derailments of facts it
already knew. Net ≈ zero. **The model's parametric knowledge is doing most of the
work; RAG's dominant failure mode is over-grounding — trusting a retrieved chunk
over a correct parametric answer.**

## Recommendation (unchanged: do not ship; but now for a sharper reason)

The retriever changes are net-neutral, so they buy nothing while adding complexity
and the recall-only failure mode. Two things to try, in order, each gated on this
A/B:

1. **A no-RAG baseline run.** If Qwen 7B v4 with RAG OFF also scores ~76%, then RAG
   adds nothing on this corpus for this model, and the retrieval investment is
   miscast. This is the cheapest, highest-information next experiment. **Run it
   before any more retriever tuning.**
2. **A faithfulness / anti-over-grounding guard.** The losses (`uart`, `vdd_ana`)
   and the hallucination case (`149_gpio`) are all over-grounding: the model
   defers to a retrieved chunk that is narrower or wronger than what it knew.
   A guard that lets the model prefer its own answer when the chunk doesn't clearly
   contain the queried entity would target exactly the DOWN column.

Changes remain STASHED. config.yaml Mistral→Qwen stays (separate real fix).
