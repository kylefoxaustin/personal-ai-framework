# Reviewer Follow-up — Judge-on-N=5 Verdict + 14B Data Correction

**Date:** 2026-05-09
**Builds on:** `docs/REVIEWER_UPDATE_N5.md` (your Q1–Q5 sign-off earlier today)
**Cross-references:** `docs/GOTCHA_7_RESOLUTION.md` (Judge-on-N=5 Verdict section), `eval/results/asymmetry_n5_judge_vs_substring.md` (per-cell breakdown)
**Asking for:** ~10 min of your time on three specific questions at the bottom.

---

You suggested judge-first sequencing in Q5 (judge-on-N=5 this week, then N=6 next week with stock-baseline-first selection). We ran the judge the same day on all 5 base+v4 pairs. **The asymmetry hypothesis is confirmed.** Surfacing two things below, then three questions.

## 1. Verdict — asymmetry confirmed across the full N=5

| Family | Substring Δ | Direction | Judge base /8 | Judge v4 /8 | Judge Δ |
|---|---:|---|---:|---:|---:|
| Qwen 2.5 7B | +3.1pp | lift | 6.786 | 6.436 | **−0.350** |
| Qwen 2.5 14B | +8.7pp | lift | 6.816 | 6.816 | **±0.000** |
| Gemma 2 9B | +3.2pp | lift | 6.718 | 6.098 | **−0.620** |
| Mistral 7B v0.3 | −3.8pp | regress | 5.718 | 5.500 | **−0.218** |
| Llama 3.1 8B | −3.2pp | regress | 5.951 | 4.786 | **−1.165** |

**No lift cell is corroborated by the judge. Both regress cells are.** Mechanism is consistent across the lift cells: faithfulness to RAG context drops on v4 (Qwen 7B −0.43, Qwen 14B −0.26, Gemma −0.20 on the 0–2 dimension), while conciseness and instruction-following hold or improve. The substring grader does not penalise the faithfulness loss because trained phrasings still match gold tokens; the judge does. On the regress cells, the judge picks up correctness AND faithfulness drops, plus (Llama only) a conciseness collapse.

The disclosure language we adopted from your Q2 (then framed as "hypothesis with test status named — judge data on one lift cell only") has been promoted in the white paper to "tested and confirmed across the full N=5." Customer-template wording strengthened accordingly:

> "Run a stock baseline on your eval before transferring this recipe to a new base. In our N=5 sample, bases with stock reasoning at floor (0–1/6) regressed on substring AND on a semantic LLM-judge. Bases at intermediate (3/6) or ceiling (6/6) lifted on substring, but the lift erased on the judge. Treat substring lifts on this recipe as format-fidelity-likely until a semantic-rubric judge corroborates; treat substring regressions as real capability damage."

Cost: ~$4 in Sonnet API. Wall time ~3h including a fresh 14B base eval (see #2).

## 2. Surprise data correction — Qwen 14B base reasoning floor

To run the asymmetry test on 14B we needed an apples-to-apples baseline JSON, and the one we'd been quoting in the gotcha #7 N=5 table turned out to have been interpolated from older runs. The fresh measurement (87 min compute, GGUF-converted from `models/qwen2.5-14b-hf` and run through the same eval pipeline as the rest of the N=5) gives:

| | Old (interpolated) | New (apples-to-apples) |
|---|---|---|
| Qwen 14B reasoning | 6/6 | **3/6** (intermediate band) |
| Qwen 14B refusal | 6/9 | **9/9** |
| Qwen 14B Δ headline | +5.3pp | **+8.7pp** |

Three implications:

1. **The clean "6/6 → lift, 0–1/6 → regress, intermediate uncharacterized" predictor is partially refined.** 14B at 3/6 stock reasoning lifted on substring. So the substring-direction predictor across N=5 reads: **≤1/6 → regress; ≥3/6 → lift on substring (with judge erasure).** Intermediate is no longer "uncharacterized" — we have one data point in it (Qwen 14B), and it lifts on substring.
2. **Your Q1 predictor-vs-proxy caveat is strengthened, not weakened.** The reasoning-floor predictor is still the cleanest we've identified at N=5, but the corrected 14B sits in the band the predictor was previously silent on, and it lifted — meaning the predictor's "ceiling-only lifts" phrasing was over-fit to the pre-correction data.
3. **The asymmetry hypothesis is unaffected and arguably strengthened.** 14B's substring lift is exactly the kind of lift the asymmetry would predict erases on judge. It did (Δ=0.000). If the substring-direction predictor were the only thing we cared about, the data correction would be a methodology embarrassment; with the asymmetry verdict in hand, the correction is a confirmation.

The correction is footnoted in `docs/GOTCHA_7_RESOLUTION.md` and `docs/skippy-white-paper.md § 7`. The pre-correction Addendum quote-block is preserved as historical record; the corrected values supersede.

## 3. Three specific questions

**Q1 — Does this satisfy the Q5 test you asked for, or is there a follow-up you want before customer-template publication?**
You blessed publication-as-preliminary contingent on the Q1+Q2 doc edits, which landed before the judge result. The judge result strengthens the framing. Our read is publication is unlocked. If you'd like additional methodology hardening (e.g., judge run at temp=0.3 across N=5 to remove the temp-0 confound, or a second judge model for cross-judge corroboration) before publication, we can sequence that.

**Q2 — Wording adjustment for the 14B data correction?**
The customer-template wording above (# 1) reflects the corrected predictor. We removed the "intermediate range (2–5/6) have not been characterized" line because we now have one data point there. Should we keep the framing tighter ("3/6 specifically lifts on substring; 2/6, 4/6, 5/6 not yet measured") or accept that one data point speaks for the band?

**Q3 — N=6 sequencing?**
With asymmetry confirmed at N=5, the N=6 fine-tune is no longer load-bearing for the framing. It's now a "is the substring-direction predictor sound at intermediate band on a *different* family" data point — useful for the methodology, not for the customer-template publication. We can:
- (a) Hold N=6 indefinitely; ship customer-template now with the N=5 + judge data.
- (b) Run N=6 stock-baseline measurement only (~2h local, no spend) on Phi-3-mini and/or Yi-1.5-9B-Chat to characterise their reasoning floors, then decide whether to fine-tune.
- (c) Run N=6 fine-tune + judge as originally scoped.

Our default if you don't object: **(a)** — ship now, hold N=6 for if/when a customer asks about a specific intermediate-reasoning base.

---

## Status of artifacts

- 6 new judge JSONs + 1 new accuracy eval JSON + analysis MD on `gdrive:skippy_files/personal-ai-assistant/eval-results/`
- 14B Instruct stock Q4_K_M GGUF (8.4G) on `gdrive:skippy_files/personal-ai-assistant/ggufs/`
- All doc edits committed (`9081121`)
- [backend] notified for keyhole § 5.5 + deck mirroring (separate bus message, 2026-05-09 13:33)

Production llm-server temporarily swapped to 14B Instruct base for the eval and restored to 7B v4 (verified healthy).

---

## Reviewer Reply (2026-05-09) — sign-off + sharpenings

The reviewer signed off on customer-template publication and surfaced a sharper reading of the data plus three specific sharpenings.

### Sharper reading the reviewer surfaced

> "The agent framed it as 'no lift cell is corroborated by the judge; both regress cells are.' That's true. But every single judge-Δ value is ≤ 0. The v4 recipe produced zero judge-corroborated capability gain in any of the five cells tested. ... And the sharpest single point: Qwen 14B has the largest substring lift in the dataset (+8.7pp) and the most 'evaporative' judge result (Δ = ±0.000). A bigger substring lift didn't produce a bigger judge result. If anything, larger substring lifts are larger format-fidelity artifacts, not larger capability gains. That's a load-bearing point for the white paper — it directly demonstrates that substring-lift magnitude does not predict capability gain."

Folded into white paper § 7 + GOTCHA Judge-on-N=5 Verdict.

### Adopted customer-template wording (replaces the soft version)

> "Across N=5 cells, the v4 recipe produced no LLM-judge-corroborated capability gain. Substring lifts on Qwen 7B (+3.1pp), Qwen 14B (+8.7pp), and Gemma 9B (+3.2pp) all went to flat or negative on judge evaluation; substring regressions on Mistral 7B (−3.8pp) and Llama 8B (−3.2pp) were corroborated as real capability damage. The substring grader at temp=0 measures format fidelity, not capability lift, for fine-tunes on this recipe; lift magnitude on substring does not correlate with judge-Δ. Customers should expect the v4 recipe to teach voice and refusal patterns reliably across base families, but should not expect underlying capability lift on bases that already perform competently on the eval."

Folded into `docs/recipe-taxonomy.md`.

### Adopted predictor wording — granular by band

> "Bases with stock reasoning at floor (0–1/6, N=2: Mistral 7B, Llama 8B) regressed on substring and on judge. Bases at higher stock reasoning (3/6, N=1: Qwen 14B; 6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased on judge. Bases at 2/6, 4/6, or 5/6 stock reasoning have not been characterized."

The previous "intermediate band — N=1" wording was overgeneralizing. Granular N-per-band is now visible in the recipe-taxonomy customer-template subsection.

### Q1 — Methodology hardening

- **Don't run judge at temp=0.3.** Reviewer's reasoning: temp=0.3 already shows fine-tune fragility (SK-P0-002); rerunning judge there conflates two confounds rather than separating them. Stick with temp=0 judge as the orthogonal grader at production-decoding regime. Accepted.
- **Cross-judge corroboration with a non-Anthropic model** (GPT-4, DeepSeek, Llama-405B-judge) is the highest-value single hardening. Queued as future work, not blocking publication. Inline note added to white paper § 7 and GOTCHA Judge-on-N=5 Verdict.

### Q3 — N=6 hybrid recommendation accepted

Reviewer pushed back on the "ship now, hold N=6 indefinitely" default and recommended a hybrid:

- **(a)** Customer-template publication ships now with the strengthened framing — done in commit forthcoming.
- **(b)** In parallel, run stock baselines on **Phi-3-mini, Yi-1.5-9B-Chat, and Gemma 2 2B** (Gemma 2 2B for size-confound testing). Don't fine-tune yet; just measure stock reasoning floors. ~6h local 5090 time, $0 API cost. Positions us to respond fast if NXP-internal asks "what about Phi or Yi" or if a customer hits an intermediate-reasoning base.
- **(c)** Defer the actual N=6 fine-tune until a specific question emerges.

Phi-3-mini and Yi-1.5-9B-Chat are already downloaded (2026-05-09); Gemma 2 2B not yet downloaded. Stock-baseline measurement on all three is awaiting Kyle's go (per `eval/RUNBOOK_n6_stock_baselines.md`, with Gemma 2 2B added).

### Procedural concern (data provenance)

Reviewer asked whether any other N=5 cells were derived from interpolated rather than measured data. **Audit run 2026-05-09: no other cells used interpolated values.** All 5 base JSONs verified apples-to-apples (temp=0, RAG=on, prompts_v2, 132-sample basis) by inspection of each `config` block. Audit line added to GOTCHA Addendum.

### Sign-off recorded

> "Customer-template publication unlocked. Recommend the stronger framing on the headline (no judge-corroborated lift anywhere), the granular N-per-band wording on the predictor, and the parallel stock-baseline measurement track on Phi/Yi. Cross-judge corroboration is the highest-value future-work item; everything else is sharpening."

All three sharpenings folded in this commit. Cross-judge corroboration tracked as future work. Customer-template publication of the N=5 reframe is unlocked and the wording is reviewer-final.

---

## Q3 stock-baseline track — completed (2026-05-09)

Per the reviewer's Q3 hybrid recommendation (parallel to publication), we ran stock baselines on **Phi-3-mini-4k-instruct, Yi-1.5-9B-Chat, and Gemma 2 2B-it** — apples-to-apples (temp=0, RAG=on, `eval/prompts_v2.json`, 132-sample basis), local 5090, no API spend.

### Stock-baseline results

| Base | Total (post-regrade /126) | Reasoning /6 | Refusal /9 | Verdict |
|---|---:|---:|---:|---|
| Phi-3-mini-4k-instruct | 12/126 = 9.5% | **3/6** | 0/9 | **Disqualified for N=6 fine-tune** — 4K context window saturated by RAG (rag_datasheet 0/78). Any FT-vs-base comparison would be context-broken vs context-broken, not stock-vs-FT capability. A Phi-3-mini-128k variant would be needed. |
| Yi-1.5-9B-Chat | 86/126 = 68.3% | **3/6** | 6/9 | **Best N=6 fine-tune candidate.** Solid stock baseline, intermediate reasoning band, no context handicap, different family (01.AI). |
| Gemma 2 2B-it | 72/126 = 57.1% | **3/6** | 6/9 | Viable as a size-confound point within the Gemma family. Optional secondary N=6 candidate. |

**All three landed at 3/6 reasoning** — the same intermediate band as Qwen 14B. No 4/6 or 5/6 candidate emerged from this trio. The 4/6 and 5/6 reasoning bands remain uncharacterized.

### Updated recommendation on N=6 sequencing

If/when we run an N=6 fine-tune, **Yi-1.5-9B-Chat is the cleanest candidate** — it tests "does the v4 recipe lift on a *different family* at 3/6 reasoning, the way Qwen 14B did?" Same band, different family lineage (01.AI vs Alibaba), no context-window confound, comparable size to existing Gemma 9B baseline (clean size comparison).

The fine-tune itself is **not blocking customer-template publication** (already unlocked). It's queued as "useful when a specific question emerges" — e.g., NXP-internal review asking about Yi or Phi specifically, or a customer hitting an intermediate-band base.

**To fully characterize the predictor across all bands**, future work would still need at least one base measured at 2/6, 4/6, and 5/6 reasoning. None of our three candidates landed in those bands. A wider candidate sweep is downstream methodology hardening, not blocking.

### Files / artefacts

- `eval/results/stock_baselines_n6_candidates.md` — full per-cell + per-category breakdown
- 3 baseline eval JSONs in `eval/results/`
- 3 stock GGUFs in `models/{phi-3-mini-4k-hf,yi-1.5-9b-chat-hf,gemma-2-2b-it-hf}/`
- Pushed (or pushing) to `gdrive:skippy_files/personal-ai-assistant/`

Production llm-server temporarily swapped through Phi → Yi → Gemma 2 2B for the eval cycle, then restored to 7B v4 (verified healthy each time).

---

*Document location: `docs/REVIEWER_FOLLOWUP_JUDGE_VERDICT.md`*
