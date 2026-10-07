# Agentic-Edge Benchmark — Adversarial Review (fable protocol, 2026-10-07)

Three independent fable reviewers (methodology skeptic · provenance auditor · skeptical VP) read
FINDINGS.md, the roadmap, the deck, and the research JSON. **Verdict: a real, defensible core
(time-decomposition, bandwidth-bound physics, floor/ceiling frame) — but NOT VP-ready as written;
the headline is the weakest-provenance number.** Findings below, deduplicated, ranked, with fix
status. ✅=fixed tonight (safe/clear) · ⏳=needs Kyle/measurement (morning).

## BLOCKERS (slide-1 blast radius)

**B1. "Decode = 98.9% of LLM time" is denominator-inflated + from a non-agentic microbenchmark.**
(methodology B2/B3, VP #3) `total_ms`=2164 includes ~317 ms (~15%) that is neither prefill nor
decode (CPU-side tokenize/sample) — decode/total = **~85%**, not 98.9%. And decode_heavy is a
long-generation microbenchmark; the *actual* agentic tasks (the matrix) are **51–70%** decode.
Honest headline = a BAND: pure-gen ~85–90%, RAG-agentic 51–70%, +37–47% wall in CPU/DDR
orchestration — every component lands on bandwidth/CPU, none on TOPS. ✅ FINDINGS reworded to the band.

**B2. The stretch "validates the DERIVED ladder" is circular and inverts under its own numbers.**
(methodology B1, provenance M3) The retrieval proxy (5 ms embed+search) measured ~0.6% of the real
orchestration layer (~875 ms on the 5090). Scaling the FULL layer by the measured 6.6× gives ~5.8 s
retrieval vs ~5 s decode on Thor = **~50% decode, not ~99%** — the "bandwidth thesis gets stronger
down the ladder" conclusion reverses. ✅ Deleted the "validates the ladder" claim; kept the proxy as
"one NN component, 6.6×, scope-limited." Ladder stays DERIVED.

**B3. Accuracy void + a cherry-pick instruction in writing.** (VP #1, methodology S4) The benchmark
costs tasks without grading correctness; FINDINGS says "do not showcase a wrong-answer trace," and
the headline decode_heavy run is itself the fine-tune hallucination. The repo's own two-judges rule
was not applied. ⏳ NEEDS: a pass/fail task-success column (two-judge graded from the saved JSON
outputs), error-modes stated on a slide, cost-per-*successful*-task. (Morning — needs the grading run.)

## SERIOUS

**S1. Decode ladder blends instruments under a "same instrument" header.** (all three) 5090=full-stack
telemetry, Thor/Orin=llama-bench, i.MX95=INT4 Genie/dated-dossier. The 5.4× ratio is instrument-skewed.
✅ FINDINGS: labeled each rung's instrument + split the i.MX95 rung out. ⏳ run a 5090 llama-bench point
for a clean same-instrument curve (blocked: local CUDA 12.6 can't target sm_120; needs a CUDA-12.8
toolchain or in-container build — morning).

**S2. Deck contradicts FINDINGS + itself on provenance tags.** (VP #2, provenance H2/M6) Deck slide 12
calls i.MX95 "sourced" while slide 7 says MEASURED; slide 1 says "measured across 5090→Thor→Orin" when
only the 5090 ran the app; the i.MX95 7B number is undated in the deck though FINDINGS says cite the
date. ⏳ Deck rewrite (morning, with the .pptx build) — sync to FINDINGS, date the 7B, soften "measured
across the ladder" to "5090 real-app; Thor/Orin same model same instrument."

**S3. "No existing benchmark measures resource cost" is refuted by our OWN research.** (provenance H1)
The research JSON refuted this 0–3 (AgentSLABench/AgBench/AgentPerfBench/RooflineBench exist). ✅
Reworded: mainstream success-benchmarks don't; a 2025–26 cluster is starting; the whitespace is
specifically *edge per-phase CPU/BW/NPU/energy* decomposition.

**S4. H100 concurrency numbers deployed as edge costs without their condition.** (provenance H3) The
98%→16–26% / −66% figures are 2×H100-NVL/vLLM; the research caveat says the mechanism transfers but
**magnitudes do not transfer to LPDDR5X edge.** ✅ Caveat added wherever they appear.

**S5. Base-vs-finetune "rates identical" argued from decode_MS (time), not tok/s.** (methodology S5,
provenance M2) Time conflates rate × length; base doc_brief ran 170 tok/s vs the claimed "steady
217–247." ⏳ Redo the table in tok/s (n=3) from the existing JSONs (morning) — claim likely true, just
prove it in the right unit.

**S6. GPU-util sampler averages idle padding; the mem%>>sm% discriminator INVERTED and went unmentioned.**
(methodology S3) decode_heavy showed sm 52 / mem 34 — the opposite of the predicted bandwidth signature
("verify the variable moved" violation). ✅ FINDINGS: state the discriminator did not resolve on the
over-provisioned 5090; lean on time-decomposition + cross-board scaling + arithmetic, not nvidia-smi;
note utilization.memory is controller-busy %, not GB/s.

**S7. Multi-tool-chain "heavy profile" is a truncated run (empty telemetry, halted at confirm-gate);
tool-propensity finding is n=1.** (methodology S2, VP #5) ✅ FINDINGS: labeled task 9 "partial —
halted at safety gate, not comparable"; tool-propensity marked n=1 anecdote pending ≥10-prompt reps.

## MINOR / polish (✅ safe fixes applied where noted)
- M1 "every shipping agent" → "every *verified* (Google/Apple); Qualcomm/Samsung unconfirmed." ✅
- M2 "within 3% of 6.51 spec" → **3.2% below**. ✅
- M3 cold-start "50×" is n=1 desktop; the range is ~35–50× → say "~35–50×, 5090-host, edge cold path
  unmeasured." ✅
- M4 DERIVED Thor projections (~99%/~5 s) tagged inline. ✅ (folded into B1/B2)
- M5 i.MX95 header "now MEASURED (2026-10-06)" → measurement is 2026-07-16; header dated correctly. ✅
- M6 corpus count 31,939 vs script's 31,904 — reconcile. ⏳
- M7 same-task wall drift 25–30% between sections → one canonical n=3 table, min–max. ⏳

## The ONE thing missing (VP #7) — ⏳ morning
The deck has no **decision/ask**: no perf-W (joules/task — the sampler already has power), no competitor
datapoint (Snapdragon/Genio), no concrete "fund X to stand this up on i.MX95 silicon + accuracy gate."
Nothing here is *wrong* without it — it's *inert*. Add a closing ask slide.

## Morning priority order
1. B1 + B2 headline/stretch rewrite (✅ done tonight — verify in AM).
2. S1 5090 llama-bench point (cheapest high-value measurement; unblock the toolchain).
3. B3 + S5 two-judge accuracy grading + tok/s table (the repo's own rule).
4. S2 deck rewrite → .pptx, synced to the corrected FINDINGS.
5. VP #7 closing ask + perf-W + one competitor number.
