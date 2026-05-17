# Personal AI Framework — Presenter Script
**Audience:** Technical management (engineering directors / VPs / senior managers).
**Deck:** `personal-ai-use-cases.pptx` (28 slides).
**Suggested length:** 35–45 minutes presentation + 10–15 minutes Q&A.
**Register:** Confident, evidence-grounded, willing to name uncertainty. No vendor-pitch energy.

---

## TL;DR for the audience (pre-meeting paragraph)

> Skippy is a fully-local AI assistant we built to validate a fine-tuning recipe — not a shipping product. The valuable artifact is the recipe + the matrix of where it works and where it doesn't. We've measured what edge silicon would have to look like to run this class of assistant; the answer is that memory bandwidth dominates, not compute, and that conclusion holds across three independent measurement stacks. The methodology arc behind these numbers — six framings in under 60 hours, each catching the prior one's over-claim — is the part of this story I want technical management to weigh, because it tells you whether the team will catch its own mistakes before customers do.

---

## Opening framing (~30 seconds)

> Thanks for the time. This is a use-case architecture review of an AI assistant we've built end-to-end on local hardware. The goals for the next 35 minutes: walk you through the system, hand you the bandwidth physics that drive edge sizing, show you what a fine-tuning recipe actually looks like across seven base models and three architectures, and be honest about where our methodology had to catch and correct itself. The deck ends with a customer-template framing — Skippy is a proof of concept for a transferable recipe, not a finished product.

---

## Slide 1 — Title

> "Personal AI Framework — a fully local AI assistant, use-case architecture." Two things to anchor: production runs Qwen 2.5 7B v4, a fine-tune. And everything I'll show you was measured on an RTX 5090, then projected to a 200-TOPS edge NPU using bandwidth physics I'll walk through.

## Slide 2 — Executive summary

> Eight things this assistant does. The two that matter for sizing: it runs 100% locally with no cloud calls, and it retrieves context from a 61,000-document personal corpus before every answer. The last bullet is the punchline of this deck — measured KPIs on the RTX 5090 let us size a 200-TOPS edge NPU to replace it.
>
> *Pause for any clarifying questions on scope. If asked "is this productized?" — say "no, it's a proof-of-concept for the recipe. I'll come back to that on the recipe-taxonomy slide near the end."*

## Slide 3 — System block diagram

> The shape of the system. FastAPI server in the middle handles auth, per-user context, and metrics. Tool-detection, RAG retrieval, main inference, and per-user memory each sit in front of the model. Storage and training are async paths — SQLite for conversation state, a separate ChromaDB collection per user for memory, LoRA training that produces a new model artifact, and Prometheus for observability. Color codes: green is AI/ML inference, amber is storage, indigo is transport.

## Slide 4 — Data flow: single-turn RAG inference

> What happens between Enter and the first token. Seven steps: rewrite the query for retrieval, hybrid-retrieve, merge BM25 with semantic, rerank, assemble the prompt, run inference, stream tokens back. The numbers below: query rewrite is 20–40 ms because it's a separate small LLM call with 64 max tokens. Retrieval is 60–120 ms — hybrid (BM25 plus HNSW plus a cross-encoder rerank on the top 20). Total wall clock for a 200-token answer is 1.0 to 2.0 seconds. Throughput is 180–215 tok/s sustained on 7B v4.

## Slide 5 — Data flow: agent tool pipeline

> The agentic layer. Two-pass design: first pass detects whether a tool is needed; second pass either runs safe tools in a bounded loop (up to 5 iterations, things like read_file or web_search) or surfaces a confirmation card for write tools (write_file, send_email, run_script). The user approves or denies write actions; safe tools execute without confirmation. Adds 200–800 ms per tool-using turn — usually invisible to the user.

## Slide 6 — Memory + RLHF feedback loop

> Two background paths. Top row: chat turns persist to SQLite, get ingested into a per-user ChromaDB collection, and the next conversation retrieves relevant memory turns alongside RAG. Bottom row: the user's thumbs-up / thumbs-down feedback feeds a training data collector that skips bad pairs, then a LoRA training job runs, the adapter is merged back into a GGUF, and the model is hot-swapped. The takeaway: inference and memory retrieval are hot paths, training is async — that distinction matters when you size for edge.

## Slide 7 — Measured KPIs

> What the RTX 5090 actually does on 7B v4 production. TTFT 30 to 120 milliseconds, throughput 180–215 tok/s sustained with a 183.9 median, 200-token end-to-end in 1 to 2 seconds. Peak VRAM is 5.5 gigabytes at 16K context. Knowledge base is 61.5K documents — 600 megabytes of RAM, 250 megabytes on disk. All these numbers come from Prometheus histograms on real production usage, not synthetic benchmarks.
>
> *If asked "why so much faster than what we hear about for edge?" — flag that 5090 has 1.8 TB/s memory bandwidth. The edge target has 100.8 GB/s. The next slide is the punchline.*

## Slide 8 — Memory bandwidth is the first-order edge constraint

> This is the key sizing insight. LLM inference is memory-bandwidth-bound, not compute-bound. Every token-decode reads the full active weight set at least once. So the bandwidth you need is approximately model size times tokens-per-second.
>
> Reference points on the table: RTX 5090 has 1.8 terabytes per second of bandwidth. Apple M3 Max has 400. Our target NPU at 75% utilization gives us 100.8 gigabytes per second. There's an upgrade ladder shown — LPDDR5T, LPDDR6 — that walks the same 128-bit bus up to 224 gigabytes per second peak.
>
> The bottom bullets show what each model size can decode at 100.8 GB/s. The 7B v4 production model gets about 21 tokens per second on the target NPU versus 184 on the 5090. The Qwen 3 30B-A3B MoE model — 16 gigabytes total but only 1.5 gigabytes active per token — measured at 37 tokens per second on Edge NPU 2. We'll come back to that.

## Slide 9 — Model catalog: dense vs MoE vs quantization

> The full landscape of edge-relevant models. Three things to notice. First, our production model — the highlighted row — is Qwen 2.5 7B v4, with 73.8% on our internal evaluation. Second, dense models from 7B to 14B fit in 16GB edge RAM; 32B doesn't. Third, MoE models change the math entirely — DeepSeek-V2-Lite has 16 gigabytes total but only 1.2 gigabytes active per token, projecting to about 84 tokens per second decode ceiling. MoE is where the headroom is for edge.

## Slide 10 — Target NPU case study

> The specific target spec: 200 TOPS INT8, 128-bit LPDDR5X at 8.4 gigatransfers per second, 75% utilization budget. Peak memory bandwidth works out to 134.4, usable to 100.8.
>
> The right-hand box is the bandwidth-vs-compute punchline. For a Llama 2 7B model at Q4, the decode loop needs about 3.8 gigabytes per token. At 100.8 GB/s, that's 27 tokens per second. The compute requirement is 14 GOPs per token. At 200 TOPS INT8, that's a compute ceiling of 14,000 tokens per second. We're bandwidth-bound by a factor of 520. Compute is wildly over-provisioned for LLM decode.
>
> Bottom table is the same math at different utilization scenarios — 50% conservative, 75% target, 85% aggressive. The deck uses 75%.

## Slide 11 — Compute tiers on the same bus

> This is the slide I'd anchor on if you remember one. The vendor offers three tiers: Low, Mid, High. Low has half the bus width. Mid and High share the **same 128-bit LPDDR5X bus at 8.4 GT/s**. The compute differs — Mid is 200 INT8 TOPS, High is 400 INT8 TOPS plus 200 floating-point TOPS.
>
> The middle table is the empirical proof. Vendor measured Qwen 3 30B-A3B on all three tiers. Notice: Mid and High both post 37.85 tokens per second decode. Same number. Because decode is bandwidth-bound and they share the bus.
>
> Where the compute jump pays off is TTFT — time to first token. That's a prefill workload, dense matmul over a thousand-token prompt. Mid does it in 351 milliseconds, High in 176. Two-times speedup. Compute scales TTFT linearly.
>
> The customer-decision box on the right summarizes: LLM-only workloads should pick Mid; the High tier is for mixed LLM-plus-CNN or for floating-point precision requirements. To raise decode rate, upgrade memory — not compute.
>
> *This is the slide where you want technical management to ask the question "so how much TOPS do we actually need?" — and your answer is "less than you think for LLM, more if you're running CNNs."*

## Slide 12 — MoE vs dense memory model

> Why MoE changes the math. Dense models read every weight per token; MoE models route to a small subset of experts per token. So a Mixtral 8x7B at Q4 is 26 gigabytes total but only 6.5 gigabytes active per token. That gives you 15 tokens per second on the same bandwidth where a 14B dense would give you 12 — at richer total capacity. The catch is the 26 gigabyte RAM requirement excludes most 16GB edge SKUs.
>
> Qwen 3 30B-A3B is the edge-friendly MoE: 16 gigabytes total, 1.5 active. That's the model that measured at 37.85 TPS.

## Slide 13 — MoE vs dense on three NPU tiers

> The expanded vendor-measured table plus a memory upgrade ladder. Pay attention to the middle five rows: those are projections of what happens if you upgrade Mid's memory to LPDDR5T-11.2, LPDDR6-12, LPDDR6-14. Decode scales linearly with bandwidth — 8.4 GT/s gives 37.85 TPS, 14 GT/s would give 63 TPS. TTFT stays constant because prefill is compute-bound, not bandwidth-bound.
>
> Takeaway underneath: money's better spent on faster memory than on more TOPS if your workload is MoE decode.

## Slide 14 — Vendor claim reconciliation

> This is a methodology slide. We had a vendor claim of 60 tokens per second on Llama 2 7B. The physics ceiling at 75% utilization is 26.5. That's a 2.26-times gap.
>
> The right-hand box enumerates the paths that could close that gap: INT4 weight-only quantization halves the bandwidth need, speculative decoding adds 1.3 to 1.5 times, flash-attention kernels add marginal wins. The bottom panel is the set of questions you ask the vendor to figure out which combination they're using. This is the slide where I'd show technical management how to gut-check any vendor TPS claim against memory physics — and walk away knowing what to ask.

## Slide 15 — Which workloads fit on the target

> The verdict table. Single-user 7B dense chat: fits comfortably, 27 tokens per second. 14B dense at 16K context: tight but acceptable, 12 tokens per second. The 30B-A3B MoE at 37.85 TPS is the strong edge MoE play.
>
> Tools, RAG, OCR, transcription — all CPU-bound, not on the NPU. The interesting "no" row: LoRA retraining needs about 28 gigabytes peak RAM at full precision. That offloads to the host. So the NPU is for inference; training is a separate path.

## Slide 16 — Platform sizing reference points

> Where the target NPU lands on the spectrum. RTX 5090 desktop is the upper bound — 209 INT8 TOPS, 1.8 TB/s bandwidth, runs 7B v4 at 180–215 TPS, 450 watts. Jetson AGX Orin is the closest spirit-cousin: similar compute, twice our target's bandwidth. M3 Max for context — Apple's high-end. The target NPU sits at 200 TOPS compute but phone-class memory bandwidth, which is the genuinely novel position. Mobile SoCs at 60 GB/s land at the 3B model tier.

## Slide 17 — Dense vs MoE on the same host: BW physics validation

> This slide does two things. First, head-to-head on our 5090 desktop: 14B dense (the previous production candidate) versus Qwen 3 30B-A3B MoE. The dense reads 9.2 gigabytes per token, the MoE reads about 1.5 — so the MoE decodes faster even though it has more total parameters.
>
> Second, the cross-reference table — this is the credibility check. We have a synthetic benchmark (Keyhole project) that measures pure decode and prefill in isolation. Their 5090 number for Q4 decode is 250 tokens per second; their bandwidth-math projection to the NPU is about 16.5 TPS. Our production measurement on 7B v4 with full RAG is 183.9 TPS — same hardware, different workload shapes, but the bandwidth physics agree within three percent across two independent measurement stacks. **That's our evidence that the BW-bound projection is a real constraint, not an artifact of either tool.**

## Slide 18 — v4 fine-tune campaign: final results

> Now we shift from the framework to the fine-tuning work. This is the punchline table.
>
> The highlighted row is what shipped: Qwen 2.5 7B v4 at 73.8%, plus 3.1 percentage points over the stock base. Voice passes, safety passes. That's production.
>
> The 14B candidate scored higher (76.2%) but fabricates fictional peripherals on three out of nine adversarial probes — we'll get to that on slide 22. Didn't ship.
>
> The MoE story: attention-only LoRA breaks reasoning catastrophically (minus 10.3 points), adding the router recovers reasoning to within four points of base, adding expert FFN LoRA over-fits and regresses. The recommended MoE recipe is attention-plus-router, no experts.
>
> Cross-family at the bottom: Mistral, Llama, Yi, and Phi-4 all regress. We initially read this as architecture-coupled at N=2. Yi at −28.6pp is catastrophic and was the moment that falsified the single-factor reasoning predictor. Phi-4 is the pre-registered falsifier we added at N=7 — and it corroborates the two-factor model. The next slide shows how that reading evolved.

## Slide 19 — Cross-family baselines on Skippy's eval

> Seven stock bases, evaluated cold without any fine-tuning. Two patterns visible.
>
> First, the left table shows decode performance on the RTX 5090 is essentially family-invariant — Qwen, Mistral, Llama, all within seven percent of each other at the 7-8B class, around 170 to 185 tokens per second. The variation tracks model size in gigabytes, not vendor or family. Bandwidth physics again.
>
> Second, evaluation quality is **not** invariant. Qwen 7B Instruct lands at 70.6%, Llama 3.1 8B at 59.5%, Yi-1.5-9B at 68.3%, Phi-4 at 71.4%, Gemma 9B at 61.9%. Eleven points of spread across families. Pick your base for quality, not for tokens-per-second.
>
> The right-hand box previews the two-factor predictor: cross-family v4 fine-tunes lift when the base has ceiling-stock-reasoning OR family-match to the corpus. Otherwise they regress. Phi-4 is the corroboration — pre-registered falsifier, and it did regress as predicted.

## Slide 20 — Two-factor methodology + substring grading reliability

> This is the methodology punchline of the campaign. Two stories on one slide.
>
> Left: the two-factor predictor. We started with a single-factor model (architecture-coupled) at N=2. Adding Gemma falsified that — Gemma is non-Qwen but lifted. New single-factor model (reasoning floor). Adding Yi falsified that — Yi has mid-range reasoning but regressed catastrophically. Two-factor model proposed: lift requires ceiling reasoning OR family-match. Adding Phi-4 corroborated. Seven of seven cells fit. Three of four corners of the matrix measured.
>
> Right: when is substring grading reliable? Substring works fine for base-vs-base comparisons. For fine-tunes versus base, substring is only directionally reliable at temperature zero — magnitude is unreliable. The Yi-versus-Phi-4 example at the bottom: Yi's substring drop is 28.6 percentage points, Phi-4's is 1.6 points — eighteen-times spread. But both LLM judges read the same regression magnitude. A team running substring-only would have shipped Phi-4 thinking "essentially equivalent to stock." It isn't. **Standing methodology rule that came out of this campaign: two LLM judges by default on every cross-family fine-tune.**

## Slide 21 — The arc itself: six framings in sixty hours

> If technical management remembers one slide for credibility reasons, I'd want it to be this one. This is what catching your own mistakes looks like.
>
> Six framings in under 60 hours, each correctly capturing what the data supported at the time, each pre-registering a falsifier, each superseded when the next data point falsified it. The original gotcha framing at N=2 would have shipped as "architecture-coupling" — that's an overclaim. By N=7, we have a two-factor model that survived a pre-registered falsifier.
>
> The Yi catastrophic regression was a moment where you could have buried the data — minus 28.6 points is embarrassing. We promoted it to a load-bearing data point instead. That decision changed the framing.
>
> The substring-reliability finding emerged from the rigor of running two judges. Our reviewer flagged it as more valuable than the gotcha-7 framing itself.
>
> *This is the slide that distinguishes "team that ships fast" from "team that ships fast and catches itself." Technical management cares about the latter.*

## Slide 22 — The fabrication problem

> A separate story from cross-family results: confident hallucination as a base-model property.
>
> Adversarial probe: nine prompts about real-but-non-existent peripheral features (think "tell me about the i.MX 93's QuantumFlow Engine"). The peripherals don't exist. We measure pass rate — does the model refuse or fabricate?
>
> Three families at 7B-and-above scale across different vendors all fabricate: Qwen 32B (3 out of 9 fabricated), Llama 8B (3 out of 9), Mistral 7B (3 out of 9). Same failure mode across the industry. Notice the within-7B split — stock Qwen 7B passes 9 of 9 while stock Mistral 7B fabricates 3 of 9. Same parameter count, different vendor. That tells us the mechanism is not pure parameter scaling — it's base-model training-data and instruct-tuning specific. This is **not a Skippy-recipe problem**. It's a base-model failure mode customers can't escape by switching vendors.

## Slide 23 — Eight options for fabrication defense

> The customer playbook. Eight layers, ranked cheapest to most aggressive. The two highlighted entries: option 1, RAG-grounded refusal exemplars in training data — add about 200 examples where retrieval returns nothing and the model learns to refuse. Option 3, system-level grounding enforcement at inference time — cite every claim or reject. These two stack well: training-side teaches behavior; system-side catches whatever the model still produces.
>
> Option 5, RLHF or DPO for grounding, is what Anthropic and OpenAI use. It works but it's expensive — hundreds to thousands of dollars per pass plus labeling. Option 6, the highlighted ★, is "just ship the smaller model" — pragmatic, free, sidesteps the failure for models that already pass.
>
> The customer-rule headline: for teams without RLHF budget, layers 1 + 3 are the sweet spot. **Don't rely on a single layer.** System-prompt disclaimers alone fail at scale.

## Slide 24 — What we shipped: layered defense by deployment choice

> Skippy production stacks two layers — option 6 (ship-smaller, 7B v4 instead of 14B v4) plus option 3 (system-level grounding via citation-required RAG).
>
> The right column documents the conditions under which we'd promote 14B v4 to production: layer 1 (RAG-grounded refusal data) plus layer 3 (system grounding enforcement) both demonstrating 9 out of 9 on a held-out adversarial probe. Estimated about 200 synthetic refusal examples plus a day of engineering work.
>
> The bottom block is the customer-template: three tiers depending on RLHF budget and how high-stakes the deployment is.

## Slide 25 — Recipe as a six-dimensional tuple

> The framework slide for what Skippy is. Every fine-tune is a point in a six-dimensional space: architecture class, size class, LoRA target set, loss masking, corpus shape, hyperparameters. Plus three validation gates (capability, voice, safety) and a hardware tier.
>
> The right-hand matrix lays out which cells we've validated. Three cells filled — dense 7B works with attention-only LoRA, dense 14B works with attention+FFN, MoE 30B-A3B works with attention-plus-router. The cross-family N=7 results give us the two-factor predictor for which un-evaluated cells will lift versus regress.
>
> The customer template at the bottom: measure your stock-base reasoning, check family match, predict your outcome. **Standing methodology: run two judges by default on cross-family evaluations.**

## Slide 26 — Voice as a separate gate

> One more thing substring evaluation can't see: voice. We measure response length, bullet density, bold density, emoji density, opener boilerplate. The stock Qwen Instruct cadence averages 672 characters per response with formatting and boilerplate. Skippy 7B v4 production averages 157 characters — terse, target voice.
>
> The interesting row is stock Qwen 3 30B-A3B Instruct-2507. It scored 74.6% on our headline evaluation — the highest of any model we tested. But it fails voice gate completely. 335-character default cadence, 1.65 bolds per response, emojis, marketing tone. If voice matters in your deployment, the headline winner can be wrong.
>
> All four of our v4 fine-tunes preserved Skippy's voice — voice is recipe-robust and architecture-independent. That's the persona-gate value proposition for customer templates.

## Slide 27 — Headline erosion: six methodology improvements retired the capability claim

> The most uncomfortable slide. The original headline was 7B v4 at +3.1 percentage points over base. Looked like a clean capability lift. Six methodology improvements moved it.
>
> Add an LLM judge: the lift erases on one independent semantic grader. Add temperature 0.3 sampling: the fine-tune collapses 29 points. Add a second judge (GPT-4o cross-judge): both judges now read it negative. Refine to the two-factor predictor: family-match is load-bearing for what looked like a capability lift. Bulk semantic regrade across the catalog: the lift flips sign — substring +3.1 becomes semantic minus 4.8. **It was a Qwen-shaped format-fidelity artifact in the training data, not a capability gain.**
>
> The production decision was unaffected because we never relied on the headline alone. Three-gate framework: capability gate failed silently, voice gate passed, safety gate passed. Skippy 7B v4 shipped because voice and safety carried real signal.
>
> *This is more credible than a clean +3 headline would be. That's what I'd ask technical management to take from the whole campaign.*

## Slide 28 — Key takeaways

> The summary. Skippy is a fine-tuning template, not a finished product — the recipe is the deliverable, customers swap our voice for their domain.
>
> Production is 7B v4 because it passes all three gates. The 14B candidate scored higher on headline but fabricates.
>
> The dense recipe is validated 7B to 14B; doesn't extend to 32B with our corpus size. The MoE recipe is attention-plus-router, no experts.
>
> Bandwidth physics dominates: NPU Mid and High share the same bus and post identical decode TPS; the compute jump halves TTFT but doesn't move decode. To raise decode, upgrade memory.
>
> Cross-family substring lift requires ceiling reasoning OR family-match. The recipe's value is voice and safety calibration, not capability lift. The "v4 lifts capability" framing was retired through six methodology improvements.
>
> Substring grading has Qwen-family format bias — biggest methodology output of the whole campaign. Customers running cross-family campaigns should semantic-grade by default. The tooling and per-eval cost are on the slide.

---

## Anticipated Q&A

### Strategic / business questions

**Q: Why "fine-tuning template" instead of "AI product"?**
> Because the fine-tune is what's valuable. The chassis — RAG, agent loop, memory, observability — is well-understood at this point. What customers can't easily get is a validated recipe for adapting a base model to their voice and domain with their own corpus, plus a clear methodology for evaluating whether it worked. That's the deliverable. Skippy is the working example.

**Q: How long would it take a customer to apply this recipe to their own corpus?**
> Recipe cost is roughly two days of engineering plus a few dollars of compute for a 7B-class fine-tune on H100. The evaluation methodology — the two-judge cross-family validation — is another half day of engineering plus about $1 per evaluation pass. Total to validated fine-tune: under a week, well under $100 in cloud compute.

**Q: What's the moat?**
> Honestly, not much in the recipe itself — Qwen-family fine-tuning is well-documented. The moat is in the methodology: the two-factor predictor, the substring-reliability framework, the layered fabrication defense, the three-gate validation. Those are the durable artifacts.

### Technical / sizing questions

**Q: Why 200 TOPS if we're bandwidth-bound?**
> Two reasons. Prefill, which is compute-bound — TTFT scales with TOPS. And mixed workloads — CNN, vision, anything that does dense matmul without autoregressive weight re-reads. If your edge SKU only runs LLM decode, you can absolutely under-provision compute. If it runs vision pipelines too, you need it.

**Q: What about memory bandwidth upgrades — LPDDR5T, LPDDR6?**
> Those buy you decode tokens-per-second linearly, on the same compute. Slide 13 shows the projection: same 200-TOPS Mid tier with LPDDR6-14 lifts Qwen 30B-A3B decode from 38 to 63 TPS. The economics of memory upgrades versus compute upgrades favor memory for any LLM-heavy workload.

**Q: How confident are you in the 100.8 GB/s usable number?**
> Reasonably. It's peak bandwidth at 75% utilization. Our anchor-measurement infrastructure on the actual NPU silicon validates that the bandwidth projection is in the right neighborhood — measured Qwen 3 30B-A3B decode of 37.85 TPS implies achieved bandwidth around 57 GB/s, which sits between the conservative two-factor projection (70 GB/s) and the headline number (100.8). The deck uses 100.8 because it matches the implementation defaults; the more conservative two-factor number would be 70 if we wanted to under-promise.

**Q: Why not just use a bigger model?**
> Two reasons. Bandwidth — 32B Q4 doesn't fit in 16GB RAM and even on a 24GB SKU runs at five tokens per second. And capability — our recipe overruns the 32B model on a 6.5K-example corpus; we get a net regression at that scale. Bigger isn't free.

### Methodology / credibility questions

**Q: How do you know the substring grading bias is real and not just statistical noise?**
> We ran a bulk semantic regrade across 33 catalog entries with GPT-4o as the semantic judge. Qwen-family fine-tunes regrade down sharply — our production 7B v4 dropped 10.3 percentage points under the semantic grader. Non-Qwen stock bases regrade up. The mechanism is mechanical: the gold-answer substrings in our eval were shaped by the corpus, which is Qwen-shaped. Format match counts as a "pass" on substring even when semantics are wrong. The regrade tool is at `eval/regrade_semantic.py` and runs at about 66 cents per eval.

**Q: The two-factor predictor — "ceiling reasoning OR family-match" — fits seven points. Isn't that post-hoc? How would you falsify the un-tested Qwen-floor cell?**
> Fair concern, and we flag it on slide 20 — the Qwen × floor-reasoning corner is the one we'd most expect to break the predictor. We can't measure it directly because Qwen 2.5 doesn't ship a base in the 0–1/6 reasoning band at the sizes we're evaluating. The cleanest falsifier would be running v4 against a Qwen-family base with deliberately constrained reasoning capability — either an older smaller Qwen variant or a Qwen distill. If it lifted under both judges and under semantic regrade, the two-factor model survives. If it regressed, then family-match alone isn't enough and we need a stronger predictor. We've prioritized the cells most useful for customer guidance over filling the corner that's least likely to drive a real customer decision.

**Q: Six framings in under 60 hours sounds like a lot. Is the team thrashing?**
> Fair question. The pace was driven by a tight reviewer round — each framing was correctly capturing what the data supported at the time, and each one pre-registered a falsifier. We added the next data point that would either corroborate or break the framing, and we shipped the next framing within hours of the result. The arc is preserved in `docs/GOTCHA_7_RESOLUTION.md` if you want to see the timestamps. It's not thrashing — it's six successive supersessions, each load-bearing.

**Q: What's the production-state-changing operations protocol?**
> We have a three-gate framework: capability, voice, safety. A model promotion requires explicit pass on all three. 14B v4 had the highest capability score but failed safety on fabrication probes, so it didn't ship. That decision predated the headline-erosion discovery — it would have held either way. The three gates are the load-bearing thing, not any single headline number.

### Operational questions

**Q: How do you train? Where? What does it cost?**
> Fine-tuning is QLoRA at rank 64 for 7B/14B dense — fits on a single 5090, takes about three to four hours, electricity cost. For 32B and MoE we rent H100 on RunPod for the run — about $15 per training run including data prep and evaluation. The runbook is at `docs/cloud-training-runbook.md`. Auto-training kicks off when users mark enough conversations as good; the merge-and-deploy is automated.

**Q: How do you handle data privacy?**
> Everything is per-user. Each user gets their own SQLite conversation store, their own ChromaDB memory and facts collections, their own RAG index. Auth is bcrypt. Nothing leaves the host unless the user explicitly enables web search or sends an email through the agent. Training data is per-user; the LoRA adapter trains on one user's conversations.

---

## Closing remarks (~30 seconds)

> Three takeaways. One: edge LLM sizing is bandwidth-bound, and the NPU compute tier you pick depends on whether you're running mixed workloads — not on LLM decode rate. Two: a fine-tuning recipe is a six-dimensional artifact with a tractable matrix of where it works and where it doesn't, and we've validated three recipe cells and characterized seven cross-family bases against a two-factor predictor. Three: the headline number on any single fine-tune evaluation is unreliable unless you've checked it against cross-family bases, multiple judges, semantic grading, and three independent validation gates. The methodology arc is the credibility story; the recipe is the deliverable.
>
> Happy to take questions.

---

## Time budget reference

| Section | Slides | Suggested time |
|---|---|---:|
| Opening framing | — | 30 s |
| Framework intro | 1–7 | 8 min |
| Bandwidth physics + NPU sizing | 8–14 | 10 min |
| Workload + platform | 15–17 | 5 min |
| Fine-tune campaign + methodology | 18–28 | 12 min |
| Q&A | — | 10–15 min |
| **Total** | | **45–60 min** |

Slides 11 (compute tiers), 17 (BW physics validation), 21 (process arc), and 27 (headline erosion) are the load-bearing slides for technical management credibility. If time runs short, compress the data-flow slides (4, 5, 6) and the fabrication-options slide (23) before compressing the load-bearing ones.
