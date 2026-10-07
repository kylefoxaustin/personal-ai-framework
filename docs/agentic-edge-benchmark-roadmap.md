# Agentic-at-the-Edge Benchmark — VP Deliverable Roadmap

**Goal:** answer the VP's question — *"what does agentic-on-the-edge mean, and what are its
performance requirements?"* — with a defensible, measured-where-possible artifact.
**Deadline:** Friday. Authored 2026-10-06 (Tue early eve, Austin). Owner: docs (Skippy session) + Kyle.
**Status key:** ✅ done · 🔨 in progress · ⏳ queued · 🎯 stretch.

---

## The frame (locked with Kyle 2026-10-06)

Two things are both called "agentic," and the delta IS the answer:
- **(a) bounded skill-orchestrator** — fixed, pre-tested skills; intent→plan→execute→synthesize;
  no new code. **Shippable on edge. = Skippy.** → the product FLOOR.
- **(b) autonomous code-gen agent** — writes/runs novel code, full system access; wants a
  frontier brain. **= openclaw / the fleet.** → the CEILING (and why it's cloud/biggest-chip).

VP takeaway shape: *edge-agentic = bounded orchestration driven by an on-device LLM; here's the
per-phase CPU/DDR/NPU cost across the hardware ladder, with the autonomous ceiling shown for contrast.*

Deliverable carries Fleet-Law provenance tags: **MEASURED / DERIVED / SOURCED**, never mixed in a headline.

---

## Scope discipline (what fits in 2.5 days — read this first)

- **MEASURED spine = the 5090.** Skippy runs natively there; we measure real agentic tasks +
  decompose CPU/DDR/NPU per phase. This is the headline data and it is genuinely measured.
- **Ladder = DERIVED.** Thor + Orin have MEASURED raw LLM physics (llama-bench prefill/decode);
  full Skippy-on-ARM is a multi-day port we will NOT finish by Friday. So Thor/Orin agentic cost
  is **projected** from the 5090 phase-mix × each board's measured prefill/decode — clearly labelled.
- **i.MX95 = SOURCED + feasibility.** ARA240 runs a 7B (SOURCED); full agentic on it is post-Friday.
  Coordinate with the fleet's `agentic-skills-imx` session for what's real there.
- **openclaw ceiling = 🎯 stretch.** One autonomous task on Thor w/ the local 7B if Thu allows;
  otherwise framed as the designed next experiment.

**The honest Friday story:** a measured 5090 agentic decomposition + an honest derived ladder +
the a/b spectrum + a credible methodology — a strong *first* VP touch, not the final word.

---

## Milestones

### Phase 0 — Frame + scope (Tue eve, TODAY)
- **M0.1 ✅ Frame locked** (above). Edge-agentic = bounded orchestration; Skippy floor / openclaw ceiling.
- **M0.2 🔨 Define the 5 benchmark tasks** — market × skill-mix × dominant resource (draft below).
- **M0.3 ✅ Format = BLUF deck.** Slide 1 tells the ENTIRE story standalone (the definition, the
  one headline chart, and the answer); every following slide is supporting detail for exactly one
  slide-1 claim. Target ~10-14 slides. (Kyle 2026-10-06: "slide 1 tells everything all in one slide.")
- **M0.4 ⏳ Deep research lands** (running `wzhvzipaq`) → harden the definition, taxonomy, and the
  "existing benchmarks + the gap" slide. Not blocking — the frame stands on reasoning if it's late.

### Phase 1 — MEASURE on the 5090 (Wed) ← the critical path
- **M1.1 Instrument the agentic trace.** Extend Skippy's existing telemetry (`/metrics`, per-gen
  prefill_ms/decode_ms/tok-s) with a per-*task* harness that, for one agentic task, logs every LLM
  call's prefill/decode + every tool's wall-time + a sampled CPU% / GPU-util / memory-BW / power
  trace (nvidia-smi dmon + /proc) tagged to each phase. Output: one JSON per task run.
- **M1.2 Run the 5 tasks on the 5090** → the MEASURED agentic resource decomposition (the headline).
  Each task: end-to-end latency, TTFT, tok/s, tool-call count, and per-phase CPU/DDR/NPU share.

### Phase 2 — Extend across the ladder (Wed PM → Thu)
- **M2.1 DERIVED ladder projection.** Project each task's cost onto Thor + Orin using the 5090
  phase-mix × their MEASURED prefill/decode (Thor 1599/40.9, Orin ~/27.8). Label DERIVED; state the
  assumptions (tool-call costs held, decode BW-bound scaling).
- **M2.2 🎯 openclaw-on-Thor ceiling.** Stand up openclaw w/ the local 7B, run ONE autonomous task,
  measure cost + whether small-model autonomy even works. If Thu allows; else document as designed.
- **M2.3 i.MX95 feasibility note** — ARA240-runs-7B (SOURCED) + the Neutron feasibility gate (we
  already have `imx95-feasibility-gate.json`); 1 paragraph + coordinate w/ `agentic-skills-imx`.

### Phase 3 — Synthesize the VP deliverable (Thu PM → Fri AM)
- **M3.1 Build the artifact.** TWO forms (decided 2026-10-06):
  - ✅ **Web deck** (review form) — `docs/agentic-edge-deck.html`, published as an Artifact for
    phone review. BLUF slide 1 + 11 supporting slides. Draft, prelim data.
  - ⏳ **`.pptx`** (delivery form, Thu) — generate from the SAME content via the repo's python-pptx
    tooling (`scripts/build_use_case_deck.py` + template converter) once the data is final, so it's
    cut once from blessed numbers. This is what the VP presents/edits from Friday.
  Content: definition (a/b), the 5 tasks, MEASURED 5090 decomposition, DERIVED ladder, openclaw
  ceiling (measured or designed), i.MX95 note, and the one-line "which NXP tier serves agentic."
- **M3.2 Two-judge review** (Fleet Law — nothing leaves the repo on one judge), fix, finalize.
  Deliver to Kyle **Fri AM** for the VP.

---

## Draft 5-task benchmark set (M0.2 — refine with research)

| # | Task | Market | Skill mix | Dominant resource |
|---|---|---|---|---|
| 1 | Voice cmd → find-in-camera-archive → notify ("find the orange cat, text me") | consumer | ASR + vision-CNN + RAG + LLM-plan + tool | NPU (vision) + LLM decode |
| 2 | Meeting → transcribe → summarize → draft follow-up emails | consumer/enterprise | ASR(Whisper) + LLM multi-step gen | NPU(ASR) + LLM prefill/decode |
| 3 | Field-service: photo + manual-RAG → diagnosis | industrial | vision + RAG(long-ctx) + LLM synth | DDR-BW (RAG+decode) |
| 4 | Inbox triage → classify → draft replies | consumer/enterprise | LLM multi-step, light tools | LLM decode (serial, BW) |
| 5 | Camera feed → detect event → reason → alert/log | industrial/auto | CNN(continuous) + LLM-on-trigger | NPU(CNN) sustained + LLM burst |

Each exercises a *different* CPU/DDR/NPU signature — the point of the set.

---

## Critical path + risks
- **Long pole = M1.1** (the agentic-trace harness). Everything headline depends on it; start Wed first.
- **Risk: full-stack-on-ARM temptation.** Do NOT try to get Skippy's docker stack on Thor/Orin by
  Friday — that's the trap. Ladder stays DERIVED; say so plainly.
- **Risk: research late** → frame stands on reasoning; fold citations in M3 if they land in time.
- **Dependency: GPU/boards** — 5090 for M1 (reserve), Thor for M2.2 (held soft now).
</content>
