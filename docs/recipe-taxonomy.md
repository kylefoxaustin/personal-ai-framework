# Skippy fine-tuning recipe taxonomy

A "recipe" is a point in **6-dimensional space** (rescoped 2026-05-08 per
SK-P1-001 in `REMEDIATION_PLAN.md`). Two recipes that match on all 6
functional dimensions should produce equivalent outcomes (capability gain,
voice transfer, safety profile) **modulo seed variance** (see "Reproducibility
scope" below). Two recipes that differ on any dimension are different
experiments, even if everything else looks similar.

Hardware tier and validation gates were previously labeled as "dimensions
7 and 8" but they are not recipe inputs. Hardware is a feasibility
constraint (does the recipe fit?). Validation gates are how we judge a
recipe's outcome (did it work?). Both are documented below as separate
concepts after the 6 functional dimensions.

This doc defines the dimensions, names the cells we've filled, and flags
the cells worth filling next. It's both Skippy's design space and the
customer-template a fine-tuning prospect uses to locate themselves.

## The 6 functional dimensions

### 1. Base architecture class (and tokenizer / chat template)

| Class | Examples | Routing computation |
|---|---|---|
| Dense transformer | Qwen2.5-7B/14B/32B, Llama-3-8B/70B, Mistral-7B | All FFN params active per token |
| Sparse MoE | Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-V3 | Router selects K of N experts per token |
| Hybrid | Mamba+attention, Jamba | Different per-layer |

The MoE distinction is load-bearing: routers and expert FFNs are computational pathways that don't exist on a dense base, and LoRA target choices that are "complete" on dense are "incomplete" on MoE.

**Tokenizer / chat template — sub-field, sometimes load-bearing.** The base
architecture choice carries an associated tokenizer + default chat template.
Some templates ship without `{% generation %}` markers (Mistral 7B v0.3,
Qwen3-30B-A3B), and trl's `assistant_only_loss=True` (dim 4 below) requires
those markers. The training pipeline patches them in. The Mistral v4 cell's
−4.0pp regression is suspected to involve the patched-template +
assistant_only_loss interaction (untested — see SK-P2-001 in
`REMEDIATION_PLAN.md`). Customer rule: when crossing tokenizer/template
boundaries, run a recipe-variant with full-sequence loss as a control.

### 2. Base size class

| Class | Param range | Hardware tier (training) |
|---|---|---|
| Edge | ≤7B | Consumer GPU (5090 32GB / 4090 24GB) |
| Mid | 7B–14B | High-end consumer or single-H100 |
| Large | 14B–32B | H100 80GB or multi-GPU |
| XL | >32B | Multi-H100 or A100 cluster |

For MoE, use total params not active params. Qwen3-30B-A3B is Large for training memory even though it's Edge for inference compute.

### 3. LoRA target set

| Target set | What it adapts | Architecture-applicable |
|---|---|---|
| Attention-only | q_proj, k_proj, v_proj, o_proj | Dense + MoE (but incomplete on MoE) |
| Attention + dense FFN | + gate_proj, up_proj, down_proj | Dense |
| Attention + MoE router | + gate (router) | MoE |
| Attention + MoE router + experts | + experts.gate_proj/up_proj/down_proj | MoE |
| QLoRA-on-all | All linear layers | Either |

The MoE-router and MoE-expert targets only exist on sparse architectures. **Attention-only LoRA on dense ≠ Attention-only LoRA on MoE** — they're different recipes despite identical target string.

### 4. Loss masking

| Strategy | Loss computed on | Mechanism |
|---|---|---|
| Full-sequence | All tokens | `labels = input_ids` |
| Assistant-only | Assistant turn tokens only | `assistant_only_loss=True` (trl SFTTrainer) requires `{% generation %}` chat template markers |
| Completion-only | Tokens after a delimiter | `DataCollatorForCompletionOnlyLM` |

Skippy v4 uses assistant-only. v1/v2 used full-sequence (and produced over-generation as a result).

### 5. Corpus shape — and corpus size

| Shape | Format | Skippy v4 had |
|---|---|---|
| Pure instruction | `[{instruction, output}]` alpaca | 6,417 examples |
| Refusal-augmented | + N curated refusal exemplars | 100 examples |
| Multi-turn chat | `[{messages: [...]}]` | none |
| Continued-pretraining (raw text) | unstructured paragraphs | none |
| Mixed (instruction + RAG-grounded) | instruction with retrieved context in prompt | none |

Corpus shape interacts strongly with loss masking. Pure-instruction + assistant-only loss is the v4 cell.

**Corpus size is a sub-field that materially shifts recipe outcomes.**
Skippy v4 used 6,517 total examples across all filled cells. The 32B v4
regression is attributed to corpus-size-vs-param-count mismatch — the
recipe lifts at 7B/14B but trades capability for safety calibration at
32B. Customer rule: doubling corpus size may unlock cells that currently
regress (32B); halving corpus size may make 7B regress like 32B does
(untested). Treat corpus size as a knob, not a frozen parameter.

### 6. Hyperparameters

The numeric levers, with v4's choices:

| Param | v4 | Range we'd consider for variants |
|---|---|---|
| LoRA rank `r` | 64 | 8–128 |
| LoRA `alpha` | 128 | r/2 to 4×r |
| Dropout | 0.05 | 0.0–0.1 |
| Epochs | 2 | 1–4 |
| Effective batch | 16 | 4–64 |
| Learning rate | 2e-4 (peak) | 5e-5 to 5e-4 |
| Schedule | cosine | linear / cosine / constant |
| Optimizer | paged_adamw_8bit | adamw / adamw_8bit / paged variants |
| Quant for training | bf16 (7B/MoE), nf4 (14B QLoRA) | bf16 / nf4 / int8 |

## Validation gates (not a recipe dimension — judging the recipe's outcome)

A recipe must pass all three gates to count as "validated":

| Gate | Metric | Tool |
|---|---|---|
| Capability | Headline pass-rate ≥ base pass-rate, no category catastrophes | `eval/run_accuracy_eval.py` + `compare_accuracy_runs.py` |
| Voice | Style metrics shift toward target voice (length, bullets, bolds, emojis, opener-boilerplate) | `eval/voice_metrics.py` |
| Safety | Refusal calibrated; no fabrication on `made_up_peripheral`-style probes | `refusal` category in v2 prompts |

A recipe that passes voice but fails capability (e.g., MoE v4) is "voice-validated, capability-incompatible" — useful failure data, not a generic regression.

A tertiary capability gate — LLM-as-judge — is queued as SK-P1-002 in the
remediation plan (Sonnet 4.6 or 4.7 evaluating against a held-out subset
with a faithfulness rubric). Substring grading is gameable; voice + safety
catch their specific failure modes; LLM-judge would cover "wrong-but-
confidently-stated answer that hits the gold tokens."

## Feasibility constraints (not a recipe dimension — does the recipe fit?)

Hardware tier is a constraint on which recipes are runnable, not a recipe
input. Two recipes matching on all 6 functional dimensions should produce
equivalent outcomes regardless of training GPU.

| Tier | Train cost (per recipe attempt) | Wall time |
|---|---|---|
| Local 5090 (32GB) | $0 | 45–90 min for 7B QLoRA, 70 min for 14B QLoRA |
| RunPod H100 (80GB) | ~$15–25 | 4–5 hr for 30B-MoE |
| RunPod A100 cluster | $50+ | varies |

## Reproducibility scope

Two recipes matching on all 6 functional dimensions should produce
equivalent outcomes **modulo seed variance**. The bound is currently
unmeasured — variance-bounds runs are queued as SK-P0-002 in the
remediation plan (5 anchored runs × 5 repetitions each, per `eval/
EVAL_SET_CHANGELOG.md` methodology version). Until those runs land,
treat any single-run delta of <2pp as "directional, within sampling
variance" rather than a load-bearing finding. Examples that need
calibration once variance bounds exist:

- Qwen 7B v4 vs base: +3.2pp — likely above noise, not yet certified
- Mistral v4 vs base: −4.0pp — borderline; needs ≥2σ verification
- 32B v4 vs base: −4.7pp — borderline; needs verification
- MoE attention-only vs base: −10.3pp — comfortably above noise, certifiable
- MoE +router vs MoE base: −4.0pp — borderline

The tighter the variance bound, the more cells become "certified above
noise" rather than "directional." This is methodology hygiene, not a
finding-shifter.

## Skippy matrix — filled cells

The cells we've actually run. All share dims 4 (assistant-only loss), 5 (alpaca + 100 refusal), 6 (r=64/α=128, 2 epochs MoE / 3 epochs dense), 7 (capability+voice+safety gates).

> Headline numbers below are post-2026-05-08 regrade (persona category quarantined per SK-P0-001; denominator = 126 samples). See `eval/EVAL_SET_CHANGELOG.md` for the original 132-sample numbers and the regrade rationale.

| Cell name | Arch | Size | LoRA targets | Epochs | HW | Capability | Voice | Safety | Headline |
|---|---|---|---|---|---|---|---|---|---:|
| Qwen2.5-7B v4 | dense | 7B | attention-only | 2 | 5090 | ✅ +3.2pp | ✅ 157c | ⚠️ reasoning −3 vs base | **73.8%** |
| **Llama-3.1-8B v4** | dense | 8B | attention + dense FFN | 2 | 5090 | ❌ −3.2pp: gains (refusal 9/9, persona, rag_email) transfer; recipe damages retrieval (rag_datasheet −8, coding −3) | ✅ | ✅ refusal 9/9 | **56.3%** |
| Qwen2.5-14B v4 | dense | 14B | attention + dense FFN | 2 | 5090 | ✅ +8.7pp † | ✅ 157c | ⚠️ fabricates `made_up_peripheral` 0/3 | **76.2%** |
| Qwen2.5-32B v4 (3 ep CONFOUND) | dense | 32B | attention + dense FFN | **3 ⚠️** | H100 | ❓ no 32B base eval; tanked datasheet | ✅ 224c (loose) | ✅ all clean | 66.7% |
| **Qwen2.5-32B v4 CLEAN** | dense | 32B | attention + dense FFN | 2 | H100 | ⚠️ plateau (corpus-too-small) | ✅ 152c | mixed (multihop 3/9) | **66.7%** |
| Qwen3-30B-A3B v4 | MoE | 30B (3B active) | attention-only | 2 | H100 | ❌ −10.3pp (multihop 0/9 catastrophic) | ✅ 131c | ✅ | 64.3% |
| Qwen3-30B-A3B router-v1 | MoE | 30B (3B active) | attention + router (q/k/v/o + gate.weight) | 2 | H100 | ⚠️ partial: multihop 6/9 RECOVERED, datasheet still −4 | ✅ 141c | ✅ | 70.6% |
| **Qwen3-30B-A3B full-v1** | MoE | 30B (3B active) | attention + router + packed experts (r=8 via target_parameters) | 2 | H100 | ❌ over-fit: rag_blog 3/3 → 0/3, datasheet 51 → 47/78 | ⚠️ 104c (over-terse) | ✅ | **65.9%** |
| **Mistral-7B-v0.3 v4** | dense | 7B | attention + dense FFN | 2 | 5090 | ❌ −4.0pp: gains transfer (refusal/email/numerical +3 each) but recipe damages retrieval (datasheet −8, blog −3, coding −3) | ✅ refusal 9/9 | ✅ | **59.5%** |
| **Gemma-2-9B-it v4** | dense | 9B | attention + dense FFN | 2 | 5090 | ✅ **+3.2pp**: same magnitude as Qwen 7B v4 lift; numerical_precision +33pp, rag_datasheet +7.7pp; one regression (rag_blog 3/3→0/3) | ✅ | ✅ refusal 9/9 (already 9/9 stock) | **65.1%** |

† Qwen2.5-14B v4 Δ updated 2026-05-09 from a fresh apples-to-apples 14B Instruct base eval (replaces earlier interpolated +5.6pp). All 14B numbers in this row are for the substring grader; on a `claude-sonnet-4-6` semantic LLM-judge, the lift erases (Δ=0.000) — see gotcha #7 Judge-on-N=5 Verdict.

**Reading the matrix (Tier 3 cross-family validation complete 2026-05-08, N=3 non-Qwen):**

**For dense Qwen2.5:**
- 7B v4 and 14B v4 at 2 epochs both lift their bases cleanly.
- **32B v4 regresses −4.6pp from its 32B Instruct base (68.2% → 63.6%)** with EITHER recipe (3-epoch confound and 2-epoch CLEAN both = 63.6%). Apples-to-apples baseline confirmed 2026-05-07.
- Per-category, the regression is a TRADE: +3 refusal calibration (32B base fabricates `made_up_peripheral` 3/9 of the time; FT recovers to 9/9), −3 numerical_precision (lost the 32B base's perfect 6/6), −3 rag_datasheet (over-fit cost retrieval), and either −3 multihop (2-ep CLEAN) or 0 multihop (3-ep CONF, but lost more datasheet to compensate).
- **Hypothesis confirmed:** corpus-size-vs-param-count mismatch. 6,517 examples enough to lift 7B/14B but at 32B the recipe trades capability for safety, net-negative.
- The v4 recipe is validated for the **7B–14B size range on dense Qwen2.5**. At 32B with this corpus, recipe is net-regressive.

**For MoE Qwen3-30B-A3B:**
- Attention-only LoRA breaks reasoning catastrophically (multihop 6/9 → 0/9).
- **Adding the router (gate.weight via target_parameters)** recovers reasoning fully (6/9 again) and is the **recommended MoE recipe**. Domain knowledge (rag_datasheet) stays at v4 levels.
- **Adding packed-expert FFN LoRA on top** (gate_up_proj + down_proj at r=8 via target_parameters on the 3D packed tensors) makes things WORSE: rag_blog 3/3 → 0/3, rag_datasheet 51 → 47/78. **Hypothesis: 374M trainable params over-fit the 6,517-example alpaca distribution, clipping the model's outputs too terse (104 char avg vs 141 router-v1) and breaking long-form retrieval.**
- **Customer rule: for MoE bases at this corpus size, include the router but NOT the experts.**

**Operational gotcha for Qwen3-MoE LoRA:** transformers' implementation packs all 128 experts of a layer into single 3D tensors `experts.gate_up_proj` and `experts.down_proj`. They do NOT appear as `experts[K].gate_proj` nn.Module children. peft's `target_modules` (which matches Linear modules) cannot reach them. Use `target_parameters` (ParamWrapper) instead. ParamWrapper requires `lora_dropout=0`.

## Customer-template decision framework

A prospect picking their own base + recipe locates themselves in the matrix:

```
1. What base architecture class? (dim 1)
2. What size? (dim 2 — drives hardware)
3. What corpus do you have? (dim 5)
4. What's your hardware budget? (dim 8)
5. Look up the closest filled cell. Does it match on dims 1, 2, 3?
   YES → that recipe should work for you (caveat: voice and safety
         depend on your corpus quality, not just architecture)
   NO  → you're in an unfilled cell. See "Open cells" below.
```

### Transferring v4 to a new base (procedure-first, predictor-soft)

The headline split across our N=5 cross-family v4 runs is reasoning-floor-coupled (see gotcha #7 in the white paper and `docs/GOTCHA_7_RESOLUTION.md`). The customer-facing guidance we publish from this — pending N=6 confirmation — is **procedure-first**:

> **Across N=5 cells, the v4 recipe produced no LLM-judge-corroborated capability gain.** Substring lifts on Qwen 7B (+3.1pp), Qwen 14B (+8.7pp), and Gemma 9B (+3.2pp) all went to flat or negative on judge evaluation; substring regressions on Mistral 7B (−3.8pp) and Llama 8B (−3.2pp) were corroborated as real capability damage. The substring grader at temp=0 measures format fidelity, not capability lift, for fine-tunes on this recipe; **lift magnitude on substring does not correlate with judge-Δ** (Qwen 14B has the largest substring lift in the dataset and the most "evaporative" judge result, Δ=±0.000). Customers should expect the v4 recipe to teach voice and refusal patterns reliably across base families, but should not expect underlying capability lift on bases that already perform competently on the eval.
>
> **Predictor of substring direction (N=5, granular by band):** Bases with stock reasoning at floor (0–1/6, N=2: Mistral 7B, Llama 8B) regressed on substring and on judge. Bases at higher stock reasoning (3/6, N=1: Qwen 14B; 6/6, N=2: Qwen 7B, Gemma 9B) lifted on substring but the lift erased on judge. Bases at 2/6, 4/6, or 5/6 stock reasoning have not been characterized.
>
> **Procedural baseline before transferring:** run a stock baseline on your eval, check the reasoning category specifically, and (if compute permits) run an LLM-judge on at least one apples-to-apples (base, FT) pair. Single-judge results carry "what if the judge has a systematic bias" risk; cross-judge corroboration with a non-Anthropic model (GPT-4, DeepSeek, Llama-405B-judge) is the strongest single hardening and is on our future-work list.

The procedural part is solid regardless of how the predictor evolves. The substring-direction predictor is hedged because at N=5 we have only N=2 data points in the floor band and N=1 in the intermediate band; alternative predictors that happen to correlate with reasoning floor in this sample cannot be ruled out (see `docs/GOTCHA_7_RESOLUTION.md` Judge-on-N=5 Verdict + Reviewer follow-up).

## Open cells worth filling next

In priority order — these are the cells where the matrix has its biggest blind spots.

### Tier 2 (high-value, requires RunPod)

| Cell | Hypothesis | Cost | Status |
|---|---|---|---|
| Qwen2.5-32B v4 (**2 epochs** — clean) | Recipe scales to dense Large | ~$15–25 | **Untested.** The 3-epoch run we have falsifies "more epochs always helps" but doesn't test recipe-fit at 32B. |
| Qwen2.5-32B v4 (3 epochs) | Recipe scales with same hyperparams | ~$25 | **DONE 2026-05-06** — 63.6%. Tanked rag_datasheet (over-fit), but FIXED every safety/correctness regression of smaller v4s. Useful asymmetric data but NOT a clean v4 test. |
| Qwen3-30B-A3B + (attention + router) LoRA | MoE-aware targets recover capability | ~$15–25 | **DONE 2026-05-06** — partial recovery: reasoning fixed (multihop 6/9), domain knowledge not (rag_datasheet 51/78 vs base 55/78). 67.4% headline (vs base 71.2%, vs v4 attn-only 61.4%). |
| Qwen3-30B-A3B + (attention + router + experts) LoRA | Full-MoE LoRA recovers domain knowledge too | ~$25–35 | **Untested — promoted in priority** by router-v1 result |

The middle row WAS the most diagnostic experiment. Outcome: **the failure decomposes into reasoning (router-fixable) and domain-knowledge (separately broken).** Customer rule update:
> "MoE base + attention-only LoRA breaks reasoning. Add router (`target_parameters=['gate.weight']`) to recover reasoning. Add expert FFN LoRA to recover domain knowledge — currently untested."

### Tier 3 (cross-family, local cost)

| Cell | Hypothesis | Validates what | Status |
|---|---|---|---|
| Mistral-7B v0.3 + v4 recipe | Recipe transfers across base families | "Not Qwen-specific" | ✅ DONE 2026-05-08 — **PARTIALLY transfers**: gains (refusal/email/numerical) clean across families; recipe damages retrieval categories on Mistral (rag_datasheet −8, rag_blog −3, coding −3). Net headline regression −3.8pp. Filed as filled negative-transfer cell — recipe is base-family-coupled. |
| Llama-3.1 8B + v4 recipe | Recipe transfers across base families | "Not Qwen-specific" | ✅ DONE 2026-05-08 — **NEGATIVE TRANSFER**: 71/126 = 56.3% (−3.2pp vs 59.5% base). Same sign as Mistral (−3.8pp). Gains (refusal 6/9→9/9, persona, rag_email) transfer cleanly; retrieval regresses. Llama is the **cleaner non-Qwen data point** than Mistral — used ChatML-like template, no `{% generation %}` patch needed (unlike Mistral). Pre-registered prediction (ceiling 60-63%) falsified in the same direction as Mistral. |
| Gemma-2 9B + v4 recipe | Recipe transfers across base families (third non-Qwen) | "N=2 non-Qwen regression survives N=3" | ✅ DONE 2026-05-08 — **POSITIVE TRANSFER on substring; judge erases**: 82/126 = 65.1% (+3.2pp vs 61.9% base). Same magnitude as Qwen 7B v4's substring lift. Gemma was chosen as the cleanest possible non-Qwen data point: different template format (`<start_of_turn>` markers, not ChatML or `[INST]`), no `{% generation %}` patch needed. **Falsifies the architecture-coupling reading at N=2**. The substring-direction predictor across N=5 is stock reasoning capability — bases at ≥3/6 lift on substring; bases at 0–1/6 regress — but **all substring lifts erase or reverse on a semantic LLM-judge (verdict 2026-05-09); regressions hold**. See gotcha #7 Judge-on-N=5 Verdict. |
| Mixtral-8x7B + (attention + router) LoRA | MoE-aware recipe transfers across MoE families | "Not Qwen3-specific MoE failure" | ⏸️ Untested |

### Tier 4 (recipe variants — change dims 4–6)

| Variant | Changed dim | Validates what |
|---|---|---|
| DPO after SFT | dim 5 — adds preference data after SFT | Closes 14B v4 fabrication gap? |
| Smaller corpus (1K examples) | dim 5 | Recipe still works with less data |
| Mixed corpus (alpaca + raw text) | dim 5 | Style transfer with continued pretraining |
| Higher rank (r=128) | dim 6 | Capacity vs over-fitting |
| **Mistral-7B v4 with full-sequence loss** | dim 4 — drops assistant-only loss | Tests whether the assistant_only_loss + `{% generation %}`-patched [INST] template combination is what damages retrieval on Mistral (Tier 3 follow-up) |

## Reading the matrix

Three patterns emerge from the validated cells:

1. **Voice transfer is recipe-robust.** All Skippy fine-tunes (including MoE v4) preserved voice. Dim 3 (LoRA targets) does not seem to gate voice transfer; dims 4–5 (loss masking + corpus shape) do most of the voice work.
2. **Capability transfer is architecture-recipe-coupled.** Dense + attention-only LoRA = capability transferred on Qwen. MoE + attention-only LoRA = capability regressed catastrophically on multihop. MoE + (attention + router) = recommended MoE recipe.
3. **Recipe transfer is base-capability-coupled on substring; lifts erase on a semantic LLM-judge (revised at N=5 with judge verdict 2026-05-09; supersedes both the N=2 architecture-coupling reading and the pre-correction "6/6 vs 0–1/6 split" framing).** Qwen 7B (6/6 stock reasoning), Qwen 14B (3/6 stock reasoning, post-correction), and Gemma 2 9B (6/6 stock reasoning) all lift on substring (+3.1pp, +8.7pp, +3.2pp). Mistral 7B v0.3 (0/6) and Llama 3.1 8B (1/6) both regress on substring (−4.0pp, −3.2pp). On a `claude-sonnet-4-6` 4-dim semantic-rubric LLM-judge, **all 3 substring lifts erase or reverse** (Qwen 7B −0.350, Qwen 14B ±0.000, Gemma −0.620 on the 0–8 total scale); both substring regressions hold or widen (Mistral −0.218, Llama −1.165). Architecture-family is not the discriminator. The substring-direction predictor (≤1/6 → regress; ≥3/6 → lift on substring, but judge-erased) is hedged because **at N=5 we cannot rule out alternative predictors that happen to correlate with reasoning floor in this sample** (overall base capability, training-data overlap, instruction-tuning recipe similarity to v4 targets — see white paper § 7 caveat). Treat as a strong directional indicator, not a causal claim. See `docs/GOTCHA_7_RESOLUTION.md` (Judge-on-N=5 Verdict + Reviewer follow-up sections) for the full framing and `eval/results/asymmetry_n5_judge_vs_substring.md` for the per-cell breakdown.

If hypothesis #2 holds, the customer rule becomes:
> "Your base is dense → attention-only LoRA is sufficient. Your base is MoE → include the router in your LoRA targets, or expect capability regression on multi-hop reasoning."

If hypothesis #2 doesn't hold, we have a deeper finding: MoE bases as a class may resist LoRA-only fine-tuning and require full fine-tuning or alternative methods (DPO, RLHF).

## What this doc is NOT

- It's not a complete fine-tuning theory. We've validated 3 cells and have hypotheses about ~10 more.
- It's not a recommendation that customers run all of Tier 2/3/4 — it's a map of where the holes are so they can decide which holes matter for their situation.
- It's not stable across model generations. When Qwen ships Qwen4 or Llama ships v4, the matrix needs new rows. The dimensions stay; the cells age out.

## Related

- `eval/voice_metrics.py` — voice gate tooling
- `eval/run_accuracy_eval.py` + `eval/compare_accuracy_runs.py` — capability gate tooling
- `training/train_lora_v3.py` / `train_lora_14b_v4.py` / `training/pod/train_moe_lora.py` — recipe implementations for the three filled cells
- Memory: `project_skippy_purpose.md` (why this taxonomy is the deliverable), `project_qwen25_7b_finetune.md` (full v1–v4 history of the dense Qwen2.5 cells), `project_voice_metrics.md` (voice gate findings)
