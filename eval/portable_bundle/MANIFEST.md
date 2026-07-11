# Skippy LLM eval corpus — portable bundle

**Author:** Skippy session `[docs]` (`~/Documents/GitHub/personal-ai-framework`)
**Date:** 2026-07-09
**Bundle size:** ~1.05 MB. Stdlib Python only. No CUDA, no GGUF, no torch, no ChromaDB, no network.

This is the LLM-accuracy counterpart to the vision/CNN corpora being run on the
Orin and IQ-9075. Where those measure *throughput on convolutions*, this measures
**whether a quantized language model still answers correctly** — on a real
retrieval task over NXP silicon documentation.

---

## 0. Why you might care

Your dossier's LLM section (`IQ9075_BOARD_DOSSIER.md` §4.4, §4.6) is **derived**,
not measured — per-token latency computed as `weight-bytes ÷ 76.8 GB/s`,
bus-validated against a single OpenVLA datapoint. §4.5 says outright: *"We did not
run a dedicated MoE benchmark... a concrete MoE decode measurement would sharpen
this section."*

This bundle turns those derivations into measurements.

> **⚠️ 2026-07-09 — read §5 first, and use `prompts_v2_1.json`.**
> An earlier version claimed INT8 W8A8 costs −3.97pp and suggested your §4.4
> needed a footnote. **Wrong; retracted.** A second judge (Claude Sonnet) finds
> **0.00pp** — zero flipped samples out of 120. The −3.97pp was an artifact of the
> **v2.0** eval set, 18 of whose 44 prompts carried an integrity defect.
>
> **The set is now repaired (v2.1).** 7 prompts got oracle chunks (their gold fact
> was in the corpus but the retriever never surfaced it); 6 whose gold fact is
> absent from the knowledge base entirely became **hallucination probes**; 4 weak
> golds tightened. `check_eval_integrity.py` exits 0 on v2.1.
>
> **Use `prompts_v2_1.json` + `rag_chunks_for_v2_1.json`** (now the defaults in
> `skippy_eval.py`). v2.0 files are retained for auditing historical runs only —
> the two are **not comparable** on the 17 touched prompts.

---

## 1. What the corpus is

| | |
|---|---|
| Prompts | 44 in `prompts_v2_1.json`; **41 scored** (3 quarantined, see §3) |
| Samples | 3 per prompt at temperature **0.0** → **123 scored samples** |
| Task | RAG question-answering over a 6,731-document knowledge base (i.MX reference manuals, datasheets, email, blog) |
| Grading | substring (offline) → semantic regrade (LLM) → LLM-judge rubric — **all three are compromised on 55% of prompts, see §5** |
| Reference HW | RTX 5090, llama.cpp CUDA, Q4_K_M GGUF |

**Category breakdown** (44 total):

```
rag_datasheet        23     faithfulness          6     refusal              3
coding                2     reasoning             2     general              2
numerical_precision   2     multihop              1     rag_blog             1
persona               2  ← QUARANTINED          general_embedded_book  1 ← QUARANTINED
```

**`faithfulness` is new in v2.1 and is the most interesting category.** Six
questions about **real** NXP silicon whose answers are provably *not* in the
knowledge base (`'Neutron'` appears in 0 of 6,731 chunks; no chunk states an
i.MX 93 GPIO count). The context cannot support an answer, so **refusing is
correct** and asserting the fact is parametric bleed-through. Harder than the
3 `refusal` prompts, which ask about fictional chips.

Sanity check: a model that always answers *"I don't have that information"*
passes exactly these 9 prompts (27/123) and nothing else.

The corpus is deliberately **retrieval-heavy**. 31 of 44 prompts require grounding
in retrieved context. This is not a trivia benchmark; it is a "can this model read
a datasheet excerpt and not make something up" benchmark. The 3 `refusal` prompts
ask about **fictional** chips and peripherals — a model that confidently answers
them is hallucinating, and scores zero.

---

## 2. RAG travels with the bundle — retrieval is held constant

`rag_chunks_for_v2_1.json` is a **frozen chunk cache**: the top k=8 chunks
Skippy's hybrid retriever returned for each prompt, plus **oracle chunks spliced in
for the 7 prompts the retriever missed** (see §5). Ship it and you reproduce the
exact context with **no vector DB and no document corpus**.

> **Correction (2026-07-09).** An earlier version of this document described the
> retriever as "BM25 + semantic + **cross-encoder** rerank." **It is not a
> cross-encoder.** It is `SimpleReranker`, a lexical heuristic
> (`overlap*0.4 + density*0.3 + phrase_bonus`), whose own docstring reads *"For
> production, consider using a cross-encoder model like ms-marco-MiniLM."* The
> semantic side is ChromaDB's default 384-dim `all-MiniLM-L6-v2`, and it carries
> **0.7** of the fusion weight against BM25's 0.3.
>
> **Measured consequence: recall@8 = 21/28 = 75%** on datasheet questions whose
> answer provably exists in the corpus. Every miss is an exact numeric or
> identifier lookup — `MAC_ADDRESS34_HIGH` / offset `410`, `RSA 4096`, `ECC P-521`,
> `RΘJA 22.5 °C/W`, `3733 MT/s` — precisely where a small sentence embedding is
> weakest and where BM25 would have succeeded had it not been downweighted.

This is a deliberate methodological choice, not a convenience. Freezing retrieval
means **every board sees byte-identical context**, so any accuracy difference is
attributable to the model and its quantization — never to a retriever that
behaved differently. Do not re-retrieve.

---

## 3. Scoring — and the traps in it

Three layers, in increasing cost and trustworthiness:

1. **Substring** (in `skippy_eval.py`, offline, exact). `match_mode: "all"` = every
   gold substring must appear; `match_mode: "any"` = one suffices (used for
   refusals, where many phrasings of "I don't know" are valid).
2. **Semantic regrade** (`regrade_semantic.py`, runs back on Skippy, needs an API key).
3. **LLM-judge** (4-dim rubric: correctness / instruction-following / faithfulness /
   conciseness, 0–2 each).

### Three traps. Please read these; we paid for all of them.

**(a) Substring grading has a model-family bias — and an LLM judge does NOT
automatically fix it.** Substring over-credits models whose surface phrasing
matches the gold strings (our Qwen fine-tunes swing −10.3pp substring→semantic).
But note: **our GPT-4o judge prompt is anchored to the same `gold_substrings`**, so
it is a *fuzzy substring matcher*, not an independent instrument. When it agreed
with substring to 0.01pp we briefly mistook that for corroboration. It was the same
measurement twice. A judge that reasons about whether the answer is *right*
(Sonnet, in our case) disagreed with both. **Use two judges from different
families, and treat agreement between a lexical grader and a gold-anchored judge as
no evidence at all.**

**(b) Temperature must be exactly 0.0.** Our fine-tunes drop ~26pp at temp=0.3.
That is a grading artifact, not a capability loss, but it will wreck your numbers.

**(c) `persona` is quarantined.** Both persona prompts ship with empty
`gold_substrings`, and the system prompt injects the "Skippy" identity into every
model — so the category scored 0/6 for *every* model ever tested, stock or tuned.
It is a constant −6 that differentiates nothing. Denominator is **126, not 132**.
(In our own older result JSONs, `summary.total_samples` is a stale `132` while
`summary.total` is the true `126`. Computing `passed/total_samples` silently yields
the pre-remediation number. We got bitten by this; now you won't.)

---

## 4. Reference numbers (RTX 5090, Q4_K_M, RAG on)

All figures **n=126, persona excluded, both graders on the same population**:

| Model | Params | substring | semantic | sem − sub |
|---|--:|--:|--:|--:|
| Qwen3-30B-A3B-Instruct-2507 (**MoE**, 3.3B active) | 30.5B | 74.6% | 69.0% | −5.6 |
| Skippy 7B v4 (fine-tune, **production**) | 7.6B | 73.8% | 63.5% | **−10.3** |
| Qwen2.5-32B-Instruct | 32.5B | 71.4% | 70.6% | −0.8 |
| Phi-4 | 14.7B | 71.4% | 67.5% | −4.0 |
| Qwen2.5-7B-Instruct | 7.6B | 70.6% | 68.2% | −2.4 |
| Yi-1.5-9B-Chat | 8.8B | 68.2% | 66.7% | −1.6 |
| Qwen2.5-14B-Instruct | 14.7B | 67.5% | 68.2% | +0.8 |
| Mistral-7B-Instruct-v0.3 | 7.2B | 63.5% | 65.9% | **+2.4** |
| Gemma-2-9B-it | 9.2B | 61.9% | 65.1% | **+3.2** |
| Llama-3.1-8B-Instruct | 8.0B | 59.5% | 61.1% | **+1.6** |

**Read the last column.** Non-Qwen models *gain* under semantic grading; the Qwen
fine-tune *loses 10.3pp*. That is the family bias of §3(a) made visible in one
table — substring grading systematically over-credits models whose surface
phrasing matches gold strings. It is why we do not quote substring numbers.

**Noise floor:** σ ≈ 1.4–2.3pp on repeated identical runs (5 models × 5 seeds).
Nothing under ~3pp is a finding. Two of our own independent fp16 reference runs on
different pods differ by 1.6pp — that is the floor, made visible.

---

## 5. ⚠️ RETRACTED: the "INT8 costs −3.97pp" result was an eval artifact

**An earlier version of this document claimed W8A8 INT8 costs ~4pp of task
accuracy, "confirmed by two graders agreeing to 0.01pp." That was wrong. Two
graders agreed because they were the same instrument twice.**

Corrected result, Qwen2.5-14B, v2+RAG, temp 0, 120 samples graded cleanly by both
judges:

| grader | fp16 | INT8 W8A8 | Δ |
|---|--:|--:|--:|
| substring | 66.7% | 62.5% | −4.17 pp |
| GPT-4o judge | 68.3% | 64.2% | −4.17 pp |
| **Claude Sonnet judge** | **62.5%** | **62.5%** | **0.00 pp** |

**Sonnet flips zero samples, in either direction, across all 120.** Its verdicts on
the fp16 and INT8 outputs are identical, sample for sample.

**Why the first two agreed.** The GPT-4o judge prompt is anchored to the same
`gold_substrings` the substring grader uses. It is a *fuzzy* substring matcher, not
an independent check of it. Agreement between them corroborates nothing.

**Every one of GPT-4o's 7 "regressions", read individually:**

- **3** on prompts whose gold fact is **absent from the frozen RAG context**. Both
  models correctly refuse; one refusal happens to contain a gold token.
- **2** on `rag_ds_imx93_149_gpio`. Gold is `["i.MX 93"]` — *a chip name, not the
  fact*. The context contains no "149" anywhere. Both models refuse. fp16's refusal
  reads *"...focuses on ... the i.MX 93 processors..."*; INT8's reads *"...focuses on
  the pin counts for different package types..."*. Substring: **pass vs fail.**
- **1** on `rag_ds_package_pitch`. Both refuse identically. GPT-4o passed one
  refusal and failed the other.
- **1** on `general_embedded_book`, which has **empty gold_substrings**. fp16 wrote
  *"David E. Simon"*; INT8 wrote *"David Simon"*. **A dropped middle initial.**

Zero residual evidence of capability loss — consistent with the independent fact
that coding and reasoning completions are **byte-identical** fp16 vs W8A8
(Jaccard 1.0). A model emitting identical code does not quietly get worse at
reading a datasheet.

**Corrected finding: at 8 bits, both FP8 and INT8 are free.** FP8 Δ = 0.0pp
(always was). INT8 Δ = 0.0pp under the stronger judge. The dynamic-range story
still has real teeth at **4 bits** — naive INT4 shatters backbones (KL 8–17) while
MXFP4 rescues them 112–328× — but do not extrapolate an 8-bit penalty that isn't
there.

### Run `check_eval_integrity.py` before you trust any pass rate

55% of this prompt set (24/44) has at least one defect:

```
  1. EMPTY gold_substrings (can never pass; still in denominator)     3 prompts
  2. Gold fact ABSENT from the frozen RAG context (unanswerable)     11 prompts
  3. WEAK gold: token appears >=20x in its own context               16 prompts
       rag_ds_imx93_149_gpio  'i.MX 93'  appears 100x
       rag_ds_cortex_m33_max_freq 'MHz'  appears  82x
```

A third of the retrieval eval asks questions the retrieved context cannot answer,
then scores on whether a gold string appears. **The correct behaviour — refusing —
scores zero unless the refusal happens to contain the token.**

**Do not fill an accuracy column from this prompt set until the defective items are
excluded.** Latency and decode work is unaffected: bandwidth math doesn't care
about gold strings.

## 6. Two asks

### Ask 1 — a real MoE decode measurement (fills your own §4.5 gap)

Run **stock `Qwen3-30B-A3B-Instruct-2507`** on v73. It is public on HuggingFace;
you need nothing from us to obtain it. 30.5B total / **3.3B active**. Resident
footprint: ~30.5 GB at int8 (tight in your 33 GB LPDDR5) or ~18.5 GB at Q4-class
4-bit (comfortable). Either precision answers the question.

It tests your §4.5 prediction directly: *decode latency should track **active**
params, memory footprint should track **total** params*. Taking int8 for the bus
math (1 byte/param), per-token latency should land near a 3.3B dense model's —
roughly **~43 ms/token** (3.3 GB ÷ 76.8 GB/s) — and **not** near the ~397 ms/token
its 30.5 GB total would imply. That is a **9× spread**; the measurement cannot come
back ambiguous. If it lands near 397 ms, the "MoE streams only active experts"
assumption is wrong on this silicon, and that is a much more interesting result.

We have a 5090 anchor for the same model: **74.6% substring / 69.0% semantic**.

### Ask 2 — score an int8 backbone by task accuracy, not KL

Run this bundle's 42 prompts through any int8 LLM backbone you can stand up on
v73, and score with `skippy_eval.py`. Then send `acc_*.json` back and we will run
the semantic regrade on Skippy (we have the API keys and the reference runs).

If your int8 backbone shows a refusal-specificity drop like ours did, §4.4's
"effectively lossless" needs a footnote. If it doesn't, we learn that the −3.8pp
was a vLLM/CUTLASS artifact rather than a property of int8 — which is **equally
valuable**, and is exactly why it should be measured on different silicon.

---

## 7. What we can and cannot hand you

| Artifact | Status | Note |
|---|---|---|
| `prompts_v2.json`, probes, RAG cache, scorer | ✅ in this bundle | board-agnostic |
| Stock baselines (Qwen2.5, Qwen3-MoE, Mistral, Llama-3.1, Gemma-2, Phi-4, Yi) | ✅ public | pull from HF, convert via QAIRT |
| **Skippy 7B v4** (production fine-tune) | ✅ available | full fp16 HF merge at `training/output/merged-v4-7b/` (~15 GB) |
| mistral / llama / gemma / phi-4 / yi **v4** fine-tunes | ✅ available | LoRA adapters (160–830 MB) + public base → merge → HF |
| **Qwen3-30B-A3B v4** (MoE fine-tune) | ❌ **GGUF-locked** | trained+merged on RunPod; only Q4_K_M GGUF came home. No safetensors, no adapter weights. Not QAIRT-convertible without retraining. |
| **Qwen2.5-32B v4** | ❌ **GGUF-locked** | same |

The two GGUF-locked fine-tunes are the reason Ask 1 targets the **stock** MoE.
You lose nothing: the stock model is the right control for a decode-latency
measurement anyway, since decode time depends on architecture and weight bytes,
not on what the weights were tuned to say.

---

## 8. Using the harness

```bash
# 1. render prompts (RAG on) — produces one JSON object per line
python3 skippy_eval.py build --rag --samples 3 -o prompt_pack.jsonl

# 2. generate with YOUR runtime. Apply the model's own chat template to
#    the `system` and `user` fields. temperature = 0.0, max_tokens = 512.
#    Append: {"uid": "<uid>", "prompt_id": "...", "sample_index": N, "text": "..."}
#    Optional telemetry per line: ttft_ms, decode_tok_s, tokens

# 3. score
python3 skippy_eval.py score completions.jsonl \
    --name iq9-qwen3-30b-a3b-int8 --rag \
    --hardware "IQ-9075 Hexagon v73" \
    --runtime "QAIRT 2.31 context-binary" \
    -o acc_iq9_moe_int8.json
```

`build` deliberately does **not** apply a chat template — that is the one thing
that legitimately differs per model, and getting it wrong silently destroys
accuracy. (We lost a Qwen3 run to stray `{% generation %}` markers left in a GGUF
chat template by the training toolchain. The model looked fine and scored like
noise.)

Send `acc_*.json` back to `[docs]` and we will semantic-regrade it against the
5090 reference runs so the comparison is apples-to-apples.
