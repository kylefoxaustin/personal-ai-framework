# Skippy project — independent review brief for a fresh Claude

**For:** A Claude (browser, API, or fresh Code session) reviewing this work cold.
**From:** A Claude that's been embedded in the project. This document is structured to let you reach your own conclusions, not to convince you of mine.
**Purpose:** Give you enough self-contained context, methodology, and raw data to evaluate every claim independently.

> **Companion review on the keyhole side**: there is a sister briefing
> at `keyhole/docs/CLAUDE_REVIEW_BRIEFING.md` covering the silicon-tier
> pipeline (5090→edge BW projection, NPU tier dtype gating, 90× / 549×
> framing). The two briefings invite pushback on overlapping claims from
> different angles — a reviewer who reads both has stronger leverage on
> shared findings (especially the recipe-base-coupling finding).
>
> **Updated 2026-05-08** — All headline pass rates use the post-remediation
> 126-sample denominator (the persona category was quarantined as
> substring-incompatible per `eval/EVAL_SET_CHANGELOG.md`). Original
> 132-sample numbers preserved in each eval JSON's `summary_v1_legacy` block.
> Methodology version: `2026-05-08-post-remediation`.

---

## What I'm asking you to do

This document summarizes a one-week iteration campaign on a fully-local AI assistant called **Skippy**. The campaign produced:

- A production-shipped fine-tune
- Five fine-tunes that were *not* shipped, with documented reasons
- A claimed "recipe taxonomy" framework
- Several gotchas claimed to generalize beyond Skippy
- A claimed cross-family finding that "recipe transfer is base-family-coupled"

Read this document and evaluate the work along these axes:

1. **Are the conclusions supported by the data?** Or are they over-reach from a single eval setup?
2. **Are there alternative explanations for the headline findings** that the campaign team didn't consider?
3. **Are the methodology choices defensible** (substring grading, sample count, RAG configuration, eval set)?
4. **Are the generalizations claimed about "customer template" applicability** sound, or specific to Skippy's domain?
5. **What's missing?** What would you have run that this campaign didn't?

You should NOT take any conclusion at face value. Specifically I want you to push back on:
- The "recipe transfer is base-family-coupled" finding (one Mistral v4 data point — is one cell enough?)
- The "ship the smaller model" framing (is this a pragmatic choice or a rationalization?)
- The claim that fine-tuning works for "domain template" customers (does the eval design support that?)
- The choice of substring grading as the primary capability gate (the team admits it's gameable; do their compensating controls work?)

Reach your own verdict. If you agree with their conclusions, say so with reasoning. If you disagree, name the specific places.

---

## Project context — what Skippy is and isn't

### What it is

Skippy is a fully-local personal AI assistant. It runs on a single consumer GPU (RTX 5090, 32 GB) and answers questions over a personal knowledge base of ~61,000 documents (datasheets, emails, blog posts, code, internal notes). It uses:
- A quantized large language model (Q4_K_M GGUF via llama.cpp)
- Hybrid retrieval (ChromaDB semantic + BM25 + cross-encoder reranker)
- A FastAPI server that wraps inference + retrieval + tool-use
- A web UI

The codebase is at `personal-ai-framework`. The author (Kyle) is one engineer; this is not a vendor-scale project.

### What it is NOT

Skippy is **explicitly framed as a fine-tuning template / proof-of-concept**, NOT the best-Skippy-possible. Per project memory: "Skippy demonstrates a transferable fine-tuning recipe; customers swap Kyle's voice for their domain (defect tracking, internal codebase, NDA docs)."

This framing matters when evaluating the conclusions: claims like "v4 is the right recipe" are only meaningful in the context of "v4 is the right recipe *for a customer who wants to do what Skippy does*." A reviewer should weigh whether the claims generalize beyond that scope.

### Why "the deliverable is the recipe"

Skippy itself only matters to one person (Kyle). The framework is the deliverable. A customer with a 6,500-document defect-tracking corpus, a 7B-class base model, and a 4090 should be able to follow the documented recipe and get a similar lift to what Skippy got. That's the claim. The campaign was structured to validate that claim by running the recipe across multiple bases, sizes, and architectures, and seeing what transfers.

---

## Methodology

### Eval set: v2 prompts (132 samples)

Located at `eval/prompts_v2.json`. Structure:
- 44 prompts, each evaluated 3 times = 132 samples per run
- 10 categories: `coding`, `general`, `multihop`, `numerical_precision`, `persona`, `rag_blog`, `rag_datasheet`, `rag_email`, `reasoning`, `refusal`
- Each prompt has a list of "gold substrings" — case-insensitive substring matching, all-or-nothing per prompt

The team acknowledges this grading method is **gameable**: a verbose model can incidentally hit the gold tokens by saying many things. The compensating control is a separate "voice gate" (response length, bullet density, etc.) and a separate "safety gate" (made-up-product fabrication probes). Three independent gates, AND-combined for the shipping decision.

### RAG configuration

Hybrid retrieval pipeline:
1. Query rewrite (small LLM call, 64 tokens max)
2. Hybrid retrieval — ChromaDB cosine + BM25 keyword, top-20 from each, merged
3. Cross-encoder reranker — re-scores top-20, takes top-k chunks
4. Top-k chunks injected into the model's prompt

Same RAG config across every candidate (apples-to-apples). Differences in headline pass rate reflect the LLM, not retrieval.

Knowledge base: ~61K documents at the time of this campaign (the eval host showed `knowledge_base_documents=6731` after a recent re-ingestion — the 61K count includes all chunks across documents).

### Hardware

- **Inference**: RTX 5090 (32 GB VRAM, 1.79 TB/s memory BW), `llama-cpp-python`, `n_ctx=16384`, all eval runs.
- **Training (dense, ≤14B)**: RTX 5090, free local hours, QLoRA.
- **Training (32B and MoE)**: RunPod H100 SXM 80GB rentals, ~$15-35/run, 4-7 hours wall.

### Quantization

All shipped + measured candidates are Q4_K_M GGUF for apples-to-apples comparison. The team also ran FP8 and Q8 quantization comparisons (visible in the `[pre-v4]` rows of the data bundle); those are historical and not central to the v4 narrative.

### Categories of fine-tuning data

The Skippy training corpus has two parts:
1. **6,417 instruction/response pairs** in alpaca format — Skippy's voice and domain knowledge
2. **100 refusal exemplars** — explicit "I don't know" responses for out-of-scope or unanswerable questions

This corpus is fixed across all v4-recipe variants. What varies is the base model, LoRA target set, and hyperparameters.

---

## Results — the headline narrative

### Iteration arc on the same base (Qwen 2.5 7B): v1 → v4

The "v4 recipe" wasn't designed up front. It emerged from four iterations on a single base, fixing one problem at a time:

| Iteration | Headline | Status | What changed |
|---|---:|---|---|
| v1 | **78.6%** | Not shipped — rambles | Initial training script with two latent bugs |
| v2 | 74.6% | Worse | Fixed one bug; revealed the other |
| v3 | 61.1% | Not shipped — over-refuses | Architectural rewrite (loss masking + stop-token discipline); 300 refusal exemplars |
| **v4** | **73.8%** | **PRODUCTION** | Cut refusal exemplars 300→100, epochs 3→2 |

The key methodology moment: **v3 had a LOWER headline than v1, but was clearly the better model** (no over-generation, 9/9 refusal correctness, terse instruction-following). The team chose to *not* ship v1 despite its higher number, on the grounds that the substring grader was rewarding v1's verbosity for the wrong reasons.

If you're skeptical of this choice: the data bundle's `voice_metrics` sheet shows v1's average response was 1,912 chars vs v4's 157 chars, a 12.5× ratio. v4 produced concise, instruction-following answers; v1 produced three-paragraph essays that incidentally contained the gold tokens.

### Scaling the v4 recipe up

Same recipe at larger sizes:

| Base | Recipe | Headline | Status |
|---|---|---:|---|
| Qwen 2.5 7B | v4 (attention-only LoRA) | **73.8%** | production-shippable |
| Qwen 2.5 14B | v4 (attention + dense FFN) | **76.2%** | best headline; **fabricates fictional peripherals** — not shipped |
| Qwen 2.5 32B | v4 (2 epochs, clean) | **66.7%** | regresses −4.6pp vs 32B base; corpus too small |
| Qwen 3 30B-MoE | v4 attention-only | **64.3%** | catastrophic regression on multi-hop reasoning |
| Qwen 3 30B-MoE | v4 + router LoRA | **70.6%** | router LoRA recovers reasoning |
| Qwen 3 30B-MoE | v4 + router + experts | **65.9%** | expert LoRA over-fits 6.5K examples |

Two stated takeaways from this scan:
1. **Recipe is architecture-coupled**: dense + attention-only LoRA works at 7B/14B; MoE needs the router added; attention-only on MoE catastrophically breaks reasoning.
2. **Corpus size matters**: 6,500 examples enough to lift 7B/14B but at 32B the recipe trades capability for safety calibration, net regressive.

### Cross-family validation (the new finding)

The team ran the same recipe on a non-Qwen base — **Mistral 7B v0.3 Instruct** — to test whether the recipe is Qwen-specific:

| Base | Stock | + v4 recipe | Δ |
|---|---:|---:|---:|
| Qwen 2.5 7B Instruct | 70.6% | 73.8% | **+3.2pp** ✅ |
| Mistral 7B v0.3 Instruct | 63.5% | **59.5%** | **−4.0pp** ❌ |

Per-category, the picture is nuanced:

| Category | Qwen 7B v4 vs base | Mistral 7B v4 vs base |
|---|---|---|
| refusal | held 9/9 | **+3** (6/9 → 9/9) |
| rag_email | **+3** (0/3 → 3/3) | **+3** (0/3 → 3/3) |
| numerical_precision | flat | **+3** (3/6 → 6/6) |
| coding | held 6/6 | **−3** (6/6 → 3/6) |
| rag_blog | held 3/3 | **−3** (3/3 → 0/3) |
| rag_datasheet | **+3** (54/78 → 57/78) | **−8** (53/78 → 45/78) |

**The team's claim**: the *gain pattern* transferred cleanly across base families (refusal, rag_email, numerical_precision lifts are nearly identical) but the *damage pattern* is family-specific. The same fine-tune that improved retrieval on Qwen broke it on Mistral, while otherwise producing identical category-level gains.

The team named this "recipe transfer is base-family-coupled" and added it as gotcha #7 in their white paper. They have one untested hypothesis for the mechanism: Mistral's chat template required `{% generation %}` marker patching to enable `assistant_only_loss` masking, and that combination may interact with Mistral's `[INST]/[/INST]` formatting differently than with Qwen's ChatML. They have NOT run the falsification experiment (full-sequence loss instead of assistant-only).

### Cross-family fabrication is base-model-wide

A separate finding from the cross-family baseline runs:

| Stock model | `made_up_peripheral` pass | Comment |
|---|---:|---|
| Qwen 2.5 7B Instruct | 9/9 | Refuses correctly |
| Qwen 2.5 32B Instruct | 6/9 | Fabricates 3/9 |
| Qwen 3 30B-A3B Instruct-2507 | 9/9 | Instruction-tuned MoE refuses cleanly |
| Llama-3.1 8B Instruct | 6/9 | Cross-family — same failure mode |
| Mistral 7B v0.3 Instruct | 6/9 | Cross-family — same failure mode |

The team's conclusion: **confident fabrication is an industry-wide base-model property**, not a Skippy-recipe problem. Customers can't escape it by switching vendors. Their proposed solution is a layered defense (training-side: RAG-grounded refusal exemplars + adversarial training data; inference-side: cite-every-claim grounding enforcement; deployment-side: ship-smaller as the dodge that sidesteps it). Skippy production stacks ship-smaller (7B v4 cleanly passes 9/9) plus system-side citation enforcement; the 14B v4 has a documented unblock condition (must demonstrate 9/9 on a held-out probe via training-side + system-side fixes before promotion).

---

## The seven claimed gotchas

Each gotcha is an empirical finding from the campaign that the team claims generalizes to other AI fine-tuning projects. Evaluate whether each generalizes.

1. **The substring grader is gameable.** A verbose model incidentally hits gold tokens; a concise correct model misses. v1 (75%) vs v3 (61.1%) is the canonical example. **Compensating control:** track multiple metrics (capability + voice + safety as three independent gates). *Reviewer prompt:* is three-gate AND-combination sufficient, or is it just "three gameable measures"?

2. **Bugs that mask each other.** v1 and v2 had two interacting training-script bugs; fixing one revealed the other. **Lesson:** fix one knob at a time. *Reviewer prompt:* is this institutional discipline advice, or is there a deeper methodology insight?

3. **Confident fabrication on fictional inputs.** 14B v4 (highest headline) invents specs for non-existent peripherals. **Lesson:** add adversarial categories to your test suite *before* shipping. *Reviewer prompt:* the team's adversarial probe is "made_up_peripheral" with NXP-style names. Does this generalize to other domains, or is it tightly coupled to Skippy's hardware-engineer focus?

4. **Over-correction from data rebalancing.** v3 over-refused after the team added 300 refusal exemplars. v4 dialed back to 100 and the over-refusal stopped. **Lesson:** budget data-balance fixes as 2-3 iterations. *Reviewer prompt:* is this "data balance has a Goldilocks zone" insight specific to refusal data, or general?

5. **More capacity does NOT mean better fine-tunes.** MoE + experts LoRA over-fit at 6,500 examples. 32B with the same corpus regressed. **Lesson:** scale the data first, OR scale down the LoRA capacity. *Reviewer prompt:* the campaign's evidence is two data points (32B and MoE+experts). Is that enough to claim a general pattern?

6. **The unmodified base may already have problems.** The 32B v4 fine-tune appeared to introduce fabrication, but the 32B *base* already fabricated at the same rate. Cross-family confirmation: Llama and Mistral 7B-class also fabricate stock. **Lesson:** always run an apples-to-apples baseline of your unmodified base. *Reviewer prompt:* this seems like a clean methodology lesson — is there anything weak here?

7. **Recipe transfer is base-family-coupled, not just architecture-class-coupled.** *(NEW from Mistral v4.)* Same recipe + same corpus + only base changed → +3.2pp on Qwen, −4.0pp on Mistral. Gains transfer across families; damage is family-specific. **Lesson:** budget at least one corrective iteration for family-specific damage when transferring a recipe across families. *Reviewer prompt:* this is a one-data-point finding (one Mistral v4 cell). Is the conclusion warranted? Should they have run Llama v4 first to triangulate?

---

## Stated conclusions (with provenance)

The team has reached the following conclusions. Each links to where the conclusion is documented and what data backs it.

### 1. Skippy production is Qwen 2.5 7B v4 — "ship the smaller model"

**Where stated:** white paper "Iteration v4 — the dial-back" section; deck slide "What we shipped"; recipe taxonomy `validated cells`.

**Backing data:** v4 7B passes all three gates (capability 73.8%, voice 157-char avg, safety 9/9). 14B v4 fails the safety gate (6/9 on `made_up_peripheral`).

**Reviewer prompt:** is "ship-smaller" a pragmatic engineering choice, or a rationalization for not solving the 14B fabrication problem? Note that the team has documented a clear unblock condition (#1 + #3 layered defense, demonstrated 9/9 on a held-out probe), so the smaller-shipped state isn't permanent.

### 2. The recipe is validated for dense Qwen 7B-14B; needs the +router variant for MoE; doesn't extend cleanly to 32B with this corpus

**Where stated:** recipe taxonomy "Reading the matrix" section.

**Backing data:** 7B (+3.2pp) and 14B (+5.3pp) lift their bases. MoE attention-only loses −9.8pp; +router recovers to −4.0pp. 32B with the v4 corpus is net-negative (−4.6pp).

**Reviewer prompt:** the 32B regression is attributed to "param:data ratio" (6,500 examples not enough for 32B parameters). Is that the right attribution? Could it be hyperparameters (rank, learning rate)? The team has tested 2 vs 3 epochs and gotten the same headline; they have NOT tested at higher rank or different learning rate.

### 3. Recipe transfer is base-family-coupled (gotcha #7)

**Where stated:** white paper gotcha #7; recipe taxonomy Mistral v4 row; bus thread 2026-05-08 09:56.

**Backing data:** ONE cross-family fine-tune (Mistral 7B v4, −4.0pp). The Qwen 7B v4 (+3.2pp) provides the contrast.

**Reviewer prompt:** **this is the conclusion most worth pushing back on.** N=1 cross-family experiment with a structurally suspicious hypothesis (chat-template patching damaging retrieval). The Llama 8B v4 fine-tune has not yet been run. If Llama also regresses, the conclusion is more solid. If Llama lifts cleanly, the conclusion is wrong (Mistral may have idiosyncratic damage from the template patch, not a general "base-family" effect).

### 4. Confident fabrication is industry-wide (not Skippy-specific)

**Where stated:** deck slide "The fabrication problem"; cross-family baselines section in white paper.

**Backing data:** Stock Qwen 32B / Llama 8B / Mistral 7B all score 6/9 on the same `made_up_peripheral` prompts. Three families, same failure rate.

**Reviewer prompt:** N=3 families is a reasonable sample. The conclusion seems sound. Is there anything to push back on?

### 5. Voice transfer is recipe-robust across architectures

**Where stated:** deck slide "Voice as a separate gate"; recipe taxonomy "Voice transfer is recipe-robust" pattern.

**Backing data:** All four v4 fine-tunes (7B, 14B, MoE, MoE+router) preserved Skippy's voice — terse, no boilerplate, low formatting density. Voice metrics for stock vs FT models are roughly stable across architecture changes.

**Reviewer prompt:** "voice" is measured by length + formatting density + boilerplate-opener rate. Is that a sufficient proxy for "voice"? The team admits this is a substring-grader-like simplification.

### 6. Headline pass rate is one signal among several; multi-gate verification is required

**Where stated:** white paper "A verification framework" section.

**Backing data:** v1 (75% headline, voice failure) vs v3 (61.1% headline, voice ✅, safety ✅). Stock Qwen3-30B-A3B-Instruct-2507 (74.6% headline, voice failure). 14B v4 (76.2% headline, safety failure).

**Reviewer prompt:** the framework is internally consistent. Is it enough?

### 7. Customer template framework — the recipe taxonomy as a customer deliverable

**Where stated:** recipe taxonomy doc + deck slide "Recipe as 8-dimensional tuple".

**Backing data:** The 8 dimensions are claimed to deterministically locate any fine-tuning experiment. 8 cells filled in the Skippy matrix; each cell maps to a verdict (validated, failure-data, open).

**Reviewer prompt:** does this generalize, or is it specific to LoRA-on-Qwen-style fine-tunes? What about RLHF, DPO, full fine-tuning, continued pretraining?

---

## Specific questions for you to answer

The team would specifically value your independent assessment on:

1. **Is gotcha #7 sound?** N=1 experiment, structurally-suspect hypothesis. Specifically: should the team have run Llama v4 *before* writing up the "base-family-coupled" finding, to avoid premature generalization?

2. **Is the substring grading + voice gate + safety gate combination sufficient?** Or does the team need a semantic grader (cosine similarity, LLM-as-judge) to make capability-gate decisions trustworthy? Note that they document this as future work in the white paper but ship without it.

3. **Is the "ship the smaller model" framing intellectually honest?** They had a higher-headline candidate (14B v4 = 76.2% vs 7B v4 = 73.8%) but the 14B candidate fabricates. Is "we shipped the safer one" the right conclusion, or is it a way to avoid solving the harder problem?

4. **Does the recipe taxonomy generalize beyond Skippy?** The 8-dimensional framework is claimed to be a "customer template". Is it actionable for a customer with a defect-tracking corpus? An NDA-document corpus? A code corpus? What dimensions are missing?

5. **What's the strongest counter-claim** the team should consider? If you were trying to falsify the v4 recipe's transferability, what experiment would you run?

6. **Methodology critique** — anything specifically wrong with: 132-sample eval, 3-sample averaging, RAG configuration choice, hardware (single 5090 for inference), Q4_K_M as the only shipped quantization?

7. **What's missing?** What's the ONE experiment the team should have run that they didn't?

---

## Where the raw numbers live

For inspection of the actual JSON eval results + per-category breakdowns + voice metrics + 5090 perf:

**`docs/skippy-data-bundle.xlsx`** — single Excel workbook, 8 sheets:

| Sheet | Rows | What's there |
|---|---:|---|
| `index` | 7 | Sheet directory + row counts |
| `models` | 18 | Every base + fine-tune evaluated (arch, size, GGUF size, training cost, role) |
| `eval_headlines` | 35 | One row per eval run — pass rate, sample count, timestamp, raw JSON path |
| `eval_per_category` | 329 | Long format: (model × category) → pass / total / rate |
| `voice_metrics` | 6 | Length, bullets/resp, bolds/resp, emojis/resp, boilerplate opener rate |
| `perf_5090` | 5 | RTX 5090 throughput: decode tok/s, prefill tok/s, RAG total |
| `recipe_matrix` | 8 | 8 filled cells in the recipe taxonomy with all 8 dimensions |
| `methodology` | 16 | Eval setup, RAG config, hardware, gotchas, schema-versioning notes |

Sheets 3 + 4 are scraped fresh from `eval/results/acc_*.json` on every build. Sheets 2 + 5 + 6 + 7 + 8 are human-curated and updated when the data set changes.

Source for the bundle: `scripts/build_data_bundle.py` (re-runnable; takes ~2 seconds).

---

## Where the prose lives

If you want to read the full team narrative (not the brief), here are the source documents in dependency order:

1. **`docs/skippy-white-paper.md`** (445 lines) — the full prose write-up with the iteration arc, seven gotchas, three-gate verification framework, cost arc, recommendations, cross-family + Mistral falsification, and recipe taxonomy references.
2. **`docs/recipe-taxonomy.md`** (204 lines) — the 8-dimensional framework + 8 filled cells + open cells + customer template decision framework.
3. **`docs/personal-ai-use-cases.pptx`** (24 slides, binary) — the deck. Each slide has 2-3 specific findings; structured for an executive audience.
4. **`pipeline/llm_server.py`** + **`eval/run_accuracy_eval.py`** + **`eval/prompts_v2.json`** — the actual eval infrastructure.

Don't read all of these unless you need to. The summary in this document plus the data bundle should be enough for an independent review.

---

## What this brief is NOT

This brief is intentionally one-sided in a specific way: it presents the team's claims plus the evidence the team uses to support them, plus prompts where they want pushback. It does not pre-litigate the counter-arguments — that's your job.

If you find this document trying to convince you of something, that's a flaw in the document. Tell the team. The goal is to give you everything you need to reach your own conclusion, not to persuade you of theirs.

---

*Generated 2026-05-08 by the Skippy team's [docs] session for cross-Claude review.*
