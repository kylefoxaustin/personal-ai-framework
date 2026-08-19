# Skippy Hardware-Ladder Porting Roadmap

**5090 → Thor → Orin → i.MX95 (+ Neutron / 2× ARA240)**

*Status: v0.1 draft, 2026-08-17. Owner: Kyle + docs (Skippy session).*
*Decisions locked: fleet-reservable boards · LLM ladder first · inference+eval core first · RTX 5090 = source of truth.*

---

## 1. Why

Skippy is a full-featured agentic AI (LLM + RAG over a 31.9K-chunk KB + tool/agent loop).
Its **capability shape** is valuable well beyond a desktop 5090 — but the customers who want
it won't buy a $5k GPU or carry a tower into a store or a customer site:

- **Retail vision:** a Skippy trained to count people in a fitting-room line or a post-sale
  crowd. That's *images*, not emails — and it wants to live in the ceiling of a store, cheaply.
- **Field-sales reference:** a Skippy trained on an industrial customer's manuals + training
  videos, so a salesperson can answer "what's wrong at this site" without a big cloud AI —
  running on something they can carry.

So we port Skippy **down the hardware ladder** and measure, at each rung, what it costs and
what you lose — always against the **5090 as source of truth**. The output is a set of simple
curves (perf, model size, recall accuracy, capability set) that let a customer building or
buying an SoC know *exactly* what silicon they need for the job — and what they give up.
This is the sizer thesis, grounded in a **real running app** instead of projections.

This directly realizes Skippy's stated purpose: *Skippy is a template; the fine-tune is the
deliverable.* Each use-case swaps Kyle's voice for the customer's domain (crowd images,
industrial manuals) and each board tells the customer which hardware tier that domain needs.

---

## 2. Source of truth + the measurement framework

**The 5090 running Skippy 7B v4 is the reference.** Every other rung is reported as a delta
against it. For each (board × model × quant) we capture **four curves**:

| Curve | Instrument (already exists) | Unit |
|---|---|---|
| **Performance** | `scripts/bench_skippy.py` (5-workload TTFT + decode) | tok/s, TTFT ms |
| **Largest model that fits** | GGUF/`.dvm` load + memory headroom | params, quant, GiB |
| **Recall accuracy** | `eval/` harness + `eval/regrade_semantic.py` (two-judge, **differential — see §2.1**) | Δ vs 5090, semantic |
| **Capability set retained** | which of {chat, RAG, agent-loop, vision} runs on-device | checklist |

### 2.1 Accuracy is measured *differentially*, and thresholded *later*

A thousand hours taught us that a single absolute pass-rate is a fragile scalar that lies
(substring vs semantic gave opposite signs; the INT8 "−3.97pp" was really 0.0; denominator
traps; a ±1.4–2.3pp noise floor). Two things survived that scrutiny and anchor accuracy here:
the **three-gate framework** (capability / voice / safety — production shipped on passing all
three, *not* on the highest headline) and **"measure the variable directly, not through the
layer above it."** For a porting study the variable under test is the **hardware/runtime/quant**,
not the model's intrinsic smarts — so accuracy is measured **differentially against the 5090**,
in two matching columns (ties directly to the "both columns" model decision, §7.1):

- **Column A — port fidelity (same model on both).** 5090-`7Bv4` vs board-`7Bv4`, same prompts.
  **Bar = answer agreement within the σ≈1.4–2.3pp noise floor, across all three gates.** This
  isolates the port: if Orin-Q4 or an ARA240-INT8 compile changes the answers, *that* is the
  attributable loss, and we name which gate broke (wrong number / refusal-turned-fabrication /
  voice collapse).
- **Column B — deployment accuracy (best-fit model per board).** Best-board-model vs the 5090
  reference. **Report the degradation curve; do NOT pre-declare a threshold** — set the
  customer-facing "still Skippy" line *after* the curves exist and can support one.

**Safety (no fabrication) is the hard gate** — a port that starts inventing peripherals fails
regardless of its headline, the same rule that cut the higher-scoring 14B from production.
Always: two judges from different families + semantic regrade + provenance tags.

**Provenance discipline (Fleet Law).** Every number in this roadmap and its results is tagged
**MEASURED** (ran it on a censused board, with proof), **DERIVED** (computed from measurements,
labelled), or **SOURCED** (vendor/datasheet). A DERIVED or SOURCED number is *never* compared
against a MEASURED one in a headline. The whole point of this exercise is honest curves.

---

## 3. Two axes (keep them separate)

- **Axis A — hardware scaling (this roadmap's spine).** The *same* Skippy (LLM + RAG + agent),
  progressively smaller silicon. Produces the scaling curves. **We do this first.**
- **Axis B — modalities (Phase 2+).** Vision (crowd-counting CNN on Neutron), multimodal RAG
  over manuals+video. New capabilities that *motivate* the small hardware. Layered on after the
  LLM ladder is characterized, because they need different models and different accelerators.

---

## 4. The measurement rig (do this once, use it everywhere)

Clean curves require **identical** measurement across boards. Before rung work, freeze:

- **One eval set** (the current datasheet/persona eval, semantic-regraded, two judges).
- **One bench protocol** (`bench_skippy.py` 5 workloads, fixed prompts, `-r` repeats).
- **One reporting schema** (JSON: board, model, quant, tok/s p50/p95, TTFT, pass%, capability
  checklist, provenance tags) so results drop straight into a comparison table / the sizer bundle.
- **A results home:** `eval/results/ladder/<board>/…`, mirrored to Drive per the artifact rule.

Milestone **M1** below builds this rig by re-establishing the 5090 baseline through it.

---

## 5. Per-board dossier

### 5090 — source of truth ✅ instruments ready
- **Access:** local, the fleet `gpu` (reserve `/reserve gpu`).
- **State:** production Skippy 7B v4 live; accuracy corpus exists (**73.8% pass**, MEASURED,
  prior campaign — to be re-confirmed as *the* canonical baseline in M1). Perf envelope measured
  across 5 workloads (**3.6 → 222 tok/s** decode range, MEASURED, `FEATURES.md`).
- **Work:** M1 — lock the canonical baseline through the frozen rig. Small.

### Thor (Jetson AGX Thor, Blackwell) — ⚠ not yet in the fleet
- **Access:** **none yet — no fleet asset card exists** (`bus.sh asset info thor` → "No card").
  Blocker for reservation/SSH. **M0 = register + drill Thor.**
- **Hypothesis (yours):** "just runs" — Blackwell arch like the 5090, so CUDA + llama.cpp
  should work unchanged. Likely true. Real question = the perf *delta* (fewer SMs, lower memory
  bandwidth than the 5090's 32 GB GDDR7) and how large a model its unified memory holds.
- **Work:** register/drill (M0) → run the *same* GGUF + rig → report Thor-vs-5090 slowdown. This
  is mostly a **benchmarking exercise**. Difficulty: **days.**

### Orin (Jetson AGX Orin, Ampere sm_87) — LLM perf half already MEASURED ✅
- **Access:** fleet `orin-agx`. `ssh orin` (USB, 35 MB/s) or LAN 10.0.1.124 (77 MB/s). Reserve
  `/reserve orin-agx`.
- **State:** LLM decode ladder **done** (`eval/results/orin/GGUF_LADDER.md`, MEASURED):
  `skippy-7b-v4` Q4 = **27.82 tok/s** / 121.6 GB/s; 14B Q4 = 13.76; MoE-30B-A3B Q4 = 43.74.
  Weight-streaming decode law characterized. Accuracy-at-the-fitting-model **not** yet run.
- **Work:** the *port* work is the **full inference+RAG+eval stack on ARM** (ChromaDB, the
  retriever, the agent loop) + measuring accuracy of whatever model clears the bar on-device.
  The "small vs 5090" pain is real but the LLM engine is proven. Difficulty: **weeks.**

### i.MX95 FRDM-PRO — dual-NPU, the hard + most interesting rung
- **Access:** fleet `imx95-frdm`. `ssh imx95` (root, no pw). Reserve `/reserve imx95-frdm`.
  ⚠ **`class: opaque`, NEVER DRILLED** — ARA240 occupancy is *host-undetectable*; never reap,
  coordinate on the bus. **Step 0 = drill + `verify-imx95-frdm.sh`.**
- **Two accelerators, clean architectural split:**
  - **Neutron** (~2–3 TOPS, on-SoC, CNN-only; SOURCED, datasheet) → **vision** (Axis B / Phase 2:
    crowd-counting). TFLite delegate `/usr/lib/libneutron_delegate.so`; convert on host w/ eIQ SDK.
  - **2× Kinara ARA240** over PCIe (~40 TOPS each, **LLM-optimized**; SOURCED, fleet card) → **LLM.**
    A **Qwen2.5-7B is already staged** (`/usr/share/llm/Qwen2.5-7B-Instruct/model.dvm`), driven by
    the Kinara runtime (`optimum-ara`: LLaMA/Qwen/LLaVA/Whisper).
- **The real work = the Kinara toolchain:** compile/quantize Skippy's *fine-tuned* 7B v4 into a
  `.dvm` for the ARA240 dataflow architecture (stock Qwen-7B is staged; the *fine-tune* is not).
  This is the "sucks rocks" part — but it's a toolchain effort, **not** a feasibility wall.
- **Work:** drill → stock-Qwen-7B smoke on ARA240 → compile Skippy-7B-v4 → rig perf+accuracy →
  (Phase 2) crowd-count CNN on Neutron. Difficulty: **a genuine research project.**

---

## 6. Phased roadmap + milestones

**Phase 0 — Rig + access (unblocks everything)**
- **M0** Register + drill **Thor** (asset card, SSH/verify). Drill **i.MX95** (opaque board).
- **M1** Freeze the measurement rig; re-establish the **5090 canonical baseline** through it
  (perf + accuracy + capability checklist). This *is* the reference all deltas cite.

**Phase 1 — the "easy" NVIDIA rungs (fast curve points)**
- **M2** **Thor:** same GGUF, same rig → Thor-vs-5090 perf delta + max-model-fit. (Tests "just runs.")
- **M3** **Orin:** stand up the full inference+RAG+eval stack on ARM; run accuracy at the
  fitting model. (LLM perf already in hand from `GGUF_LADDER.md` — fold it in.)
- **Checkpoint:** first real curve — 5090 → Thor → Orin on perf, model-fit, accuracy.

**Phase 2 — the i.MX95 LLM rung (the hard one)**
- **M4** ARA240 smoke: run the *staged* stock Qwen-7B `.dvm`, learn the Kinara runtime + rig it.
- **M5** Compile **Skippy 7B v4** → `.dvm` (the toolchain milestone) → perf + accuracy on ARA240.
- **Checkpoint:** the full LLM ladder 5090 → Thor → Orin → i.MX95-ARA240, one honest curve.

**Phase 3 — modalities (Axis B, the "why")**
- **M6** Vision pilot: a crowd-counting CNN on the i.MX95 **Neutron** (the use-case the NPU is
  built for) — establishes the vision capability + its own tiny-hardware curve.
- **M7** (optional) Field-sales multimodal RAG over manuals+video on the biggest rung that fits.
- **M8** Synthesize: the customer-facing sizing curves + a short write-up (feeds sizer / deck /
  white paper).

**Rough effort:** Phase 0–1 = the fast wins (days–weeks). Phase 2 = the deep one (Kinara). Phase
3 = new-capability builds. Each milestone is bus-coordinated (boards are shared) and its results
land in `eval/results/ladder/` + Drive.

---

## 7. Open decisions (to resolve as we go — none block M0/M1)

1. **Model per rung.** ✅ **RESOLVED: both columns.** Column A = the *same* 7B v4 everywhere
   (clean hardware-scaling curve + the §2.1 port-fidelity measurement); Column B = the best-fit
   model each board can run (the deployment recommendation + degradation curve).
2. **Accuracy bar.** ✅ **RESOLVED (see §2.1): differential, thresholded later.** Column A bar =
   reproduces the 5090's answers within the σ≈1.4–2.3pp noise floor across capability/voice/safety.
   Column B = report the degradation curve; set the customer-facing "still Skippy" line *after* the
   data exists. Safety (no fabrication) is the non-negotiable gate on every rung.
3. **i.MX95 accelerator breakout.** LLM→ARA240, vision→Neutron (recommended). Two ARA240s → later
   choice: two LLM instances vs LLM+vision-at-scale. Not needed until Phase 2/3.
4. **Full-app scope creep.** Inference+RAG+eval is the committed scope. Web UI / email / calendar
   only where they fit (Thor yes; Orin maybe; i.MX95 unlikely) — decide per rung, don't force it.

---

## 8. What already exists (we are not starting from zero)

- **Instruments:** `scripts/bench_skippy.py`, `eval/bench_retrieval.py`, `eval/regrade_semantic.py`,
  `eval/repair_eval_set.py`, `eval/compare_accuracy_runs.py`, `eval/build_sizer_bundle.py`.
- **5090 accuracy corpus:** `eval/results/acc_*.json` (7B v4 = 73.8%, semantic-regraded).
- **Orin LLM ladder:** `eval/results/orin/GGUF_LADDER.md` (MEASURED — Milestone M3's perf half).
- **Board access + cautions:** fleet cards via `bus.sh asset info <board>`.
- **The sizer connection:** these curves are exactly the measured backbone the PAI/keyhole sizers
  consume — the roadmap feeds `eval/build_sizer_bundle.py`.
