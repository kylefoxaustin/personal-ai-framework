# Agentic at the Edge — VP Deck Blueprint (BLUF)

*Format per Kyle: Slide 1 tells the ENTIRE story standalone; every later slide is supporting detail
for one slide-1 claim. Target ~12 slides. Build the real deck (pptx/HTML) Thu from this blueprint.
All numbers carry MEASURED / DERIVED / SOURCED tags. Source of truth: `FINDINGS.md`.*

---

## SLIDE 1 — THE WHOLE STORY (BLUF, standalone)

**Title: "Agentic at the Edge is Bandwidth-Bound, Not TOPS-Bound"**

**What agentic means on-device (the part everyone gets wrong):** a bounded agent that orchestrates
a *fixed, pre-tested* set of on-device skills (transcribe, retrieve, see, draft) — intent → plan →
execute → synthesize. **It does NOT write and run new code.** Every shipping on-device agent today
(Apple, Google Gemini Nano, Qualcomm) works this way. [SOURCED]

**The one chart** (decode tok/s vs memory bandwidth across our hardware ladder — 5090 / Thor / Orin,
all running the same 7B agent brain): decode throughput tracks **memory bandwidth**, and an agentic
workload is **98.8% decode** [MEASURED]. ⇒ **the 500-eTOPS NPU sits idle during agentic work; the
binding resource is LPDDR5X bandwidth + CPU.**

**The three takeaways:**
1. **Size for bandwidth, not TOPS.** Agentic decode is memory-bound; a huge NPU doesn't help it.
   The NPU earns its keep on *perception* (camera/CNN), not on *the agent*.
2. **The agent is cheap; the orchestration isn't.** On a real task the LLM is ~⅔ of the time;
   retrieval + orchestration (CPU/DDR) is ~⅓ — and **cold-start is 50× warm** (89 s → 2 s). [MEASURED]
3. **One bounded agent, not an agent swarm.** Multi-agent collapses cache + throughput on constrained
   silicon; the edge answer is a single well-scoped agent. [SOURCED]

**Bottom line for NXP:** an agentic-edge SoC is sized by **memory bandwidth + CPU + a modest LLM
accelerator**, with the big NPU reserved for perception. We measured this on a real agentic app
(Skippy) across 5090→Thor→Orin; here's the data.

---

## Supporting slides (each backs one slide-1 claim)

**S2 — "Two things are called 'agentic'; only one ships on-device."**
The a/b split: (a) fixed-skill orchestrator [Skippy, Apple, Gemini Nano] vs (b) autonomous code-gen
[openclaw / cloud agents]. Table of shipping products, all (a). [SOURCED] Backs slide-1 definition.

**S3 — "Here's a bounded edge agent, concretely." (Skippy)**
Skippy's agent loop: fixed tool registry (RAG, ASR, OCR, email-draft, calendar) + multi-step
orchestration. The reference edge agent we benchmark. Screenshot + the 5-task set.

**S4 — "Agentic work is decode-dominated." [MEASURED]**
decode_heavy trace: decode 1826 ms vs prefill 21 ms = 98.8%. Why: autoregressive, KV re-read every
token; cross-turn reuse means little re-prefill. Backs slide-1 "98.8% decode."

**S5 — "Decode is bandwidth-bound → the NPU is idle." [MEASURED + SOURCED]**
Prefill compute-bound vs decode bandwidth-bound; the decode ladder (5090 220 / Thor 41 / Orin 28
tok/s) tracks bandwidth. The 500-eTOPS NPU does nothing for decode. THE headline chart.

**S6 — "The agent is cheap; orchestration + cold-start aren't." [MEASURED]**
Warm agentic task 2.5 s = ~65% LLM / ~35% retrieval+orchestration (CPU/DDR). Cold-start 89 s (50×).
Edge implication: keep models resident; size CPU/DDR for the orchestration layer.

**S7 — "The hardware ladder." [MEASURED/DERIVED]**
5090 → Thor → Orin → (i.MX95) — what fits, decode tok/s, where agentic becomes painful. Thor's
123 GB unified pool vs 5090's 32 GB. i.MX95 = feasibility (ARA240 runs a 7B). [SOURCED for i.MX95]

**S8 — "One agent, not a swarm." [SOURCED]**
Prefix-cache 98%→16-26%, throughput −66% under concurrent agents. Single bounded agent is the edge
answer; the fleet/openclaw multi-agent model is cloud/biggest-chip. Backs slide-1 takeaway 3.

**S9 — "5 edge-agentic tasks & their resource signatures."**
The 5-task table (voice→camera-find, meeting→draft, field-service RAG, inbox-triage, camera→reason)
× dominant resource (NPU-perception vs LLM-decode-BW vs CPU-orchestration).

**S10 — "Nobody measures this yet." [SOURCED]**
AgentBench/τ-bench/WebArena measure task *success*, not on-device *resource cost*. The gap = the
opportunity. An edge-agentic *resource* benchmark is whitespace NXP can own.

**S11 — "What this means for NXP silicon."**
Sizing guidance by market: consumer (voice assistant) / industrial (field-service) / automotive
(in-cabin) — each's CPU/DDR-BW/NPU mix. The big-NPU high-end part is sized for *perception*, the
agent rides on bandwidth + CPU.

**S12 — "Method + honesty."**
MEASURED (5090, real Skippy traces) / DERIVED (ladder projection) / SOURCED (literature). What's
not yet done (full-stack on ARM = next), retrieval answer-quality is a separate axis. Two-judge
reviewed. Credibility slide.

---

## Still-TODO before the deck is final (Wed/Thu)
- The one clean headline chart (S5): add a 5090 llama-bench decode point for apples-to-apples.
- 5-task resource signatures (S9): run the ones with available assets; derive the rest.
- i.MX95 line (S7/S11): coordinate with `agentic-skills-imx` for what's real.
- Stretch: full Skippy on Thor/Orin → convert S7 ladder from DERIVED to MEASURED.
