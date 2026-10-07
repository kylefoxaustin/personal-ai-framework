# Agentic-at-the-Edge — Measured Findings (deck source of truth)

> ⚠ **UNDER REVISION after adversarial review (fable protocol, 2026-10-07) — see
> `docs/agentic-edge-review-2026-10-07.md`.** The clearly-correct overclaim fixes are applied
> below; items needing a measurement or Kyle's call (two-judge accuracy grading, a same-instrument
> 5090 llama-bench point, the deck rewrite, a closing ask) are flagged ⏳ in the review doc. Do NOT
> quote the old "98.9% decode" single number — it is replaced by the measured band (see Headline).

*2026-10-06, 5090 source-of-truth measurements + deep-research grounding. Provenance tags per
Fleet Law: MEASURED (ran it) / DERIVED (computed) / SOURCED (literature). Harness:
`scripts/agentic_bench/trace.py`. Research: `docs/research/agentic-edge-deep-research-20261006.json`.*

## The frame (what "agentic at the edge" means)
- **(a) bounded skill-orchestrator** — fixed, pre-tested skills; intent→plan→execute→synthesize;
  no code-gen. **Shippable on edge. = Skippy.** SOURCED: *every* shipping on-device agent today is
  this model — VERIFIED for Google + Apple (Qualcomm/Samsung specifics unconfirmed at primary-source level) (Gemini Nano = 4 fixed LoRA skills; Apple 3B, "not a general chatbot," constrained
  tool-calling that can't hallucinate tool names). **No shipped product runs on-device code-gen.**
- **(b) autonomous code-gen agent** — writes/runs novel code; wants a frontier brain. = openclaw /
  the fleet. The CEILING; cloud or biggest-chip territory.

## ★ Headline (the VP's premise, refined — SOURCED + MEASURED)
**For the agent's token generation, TOPS do not set the rate — memory bandwidth and CPU do.** A big
NPU (e.g. 500 eTOPS) does little for batch-1 decode; it earns its area on *perception* (vision/ASR)
and on *prefill/TTFT*, not on the agent's generation loop. (Reframed from the earlier absolute "NPU
is idle" per VP review — the NPU is not idle, it's doing a different job; same sizing conclusion.)
- SOURCED: agentic LLM work is decode-heavy, and batch-1 decode is memory-bandwidth-bound while
  prefill is compute-bound (roofline; the research cites decode = 91–98.6% of LLM *compute* time in
  long high-reuse sessions). Mainstream agent benchmarks (AgentBench/τ-bench/WebArena/AndroidWorld)
  score only task SUCCESS, not on-device resource cost; a 2025–26 cluster (AgentSLABench, AgBench,
  AgentPerfBench, RooflineBench) is just starting on resources. The specific whitespace: an
  **edge-silicon per-phase CPU/BW/NPU/energy** decomposition.
- MEASURED (ours, 5090) — report the BAND, not one number: the decode *share of LLM time* runs
  **~85–90% on pure-generation tasks** (decode/total_ms; prefill ~1%, ~10–15% is CPU-side
  tokenize/sample) down to **~51–70% on RAG-grounded tasks**, which additionally spend **37–47% of
  wall-time in retrieval+orchestration (CPU/DDR)**. Every component lands on bandwidth or CPU; none
  on TOPS. (The earlier "98.9%" was decode/(prefill+decode) on a long-generation microbenchmark —
  denominator-inflated and not representative of agentic mixes; superseded by this band.)

## MEASURED — 5090 (source of truth)
| workload | wall | prefill | decode | decode % of LLM | note |
|---|--:|--:|--:|--:|---|
| decode_heavy (400 tok gen) | 2.24 s | 21 ms | 1826 ms | **98.8%** | decode dominates generation |
| prefill_heavy (big ctx, 1 tok) | 1.26 s | 997 ms | ~0 | 0% | prefill = compute-bound (SM 99% peak) |
| rag_multistep (WARM) | **2.5 s** | 520 ms | ~1100 ms | — | agentic task: ~**65% LLM / ~35% retrieval+orchestration** |
| rag_multistep (COLD, 1st call) | **89 s** | 519 ms | 312 ms | — | **cold-cache: ~88 s loading embed model + BM25 index + ChromaDB** |

- **Cold-start is a first-class edge metric.** 89 s cold → ~1.7–2.5 s warm (~35–50×, n=1, 5090-host). A device that
  sleeps/wakes pays the cold cost every cold path — must be designed for (keep models resident).
- **The agentic task's cost is not just the LLM.** Warm, retrieval+orchestration is ~1/3 of
  wall-time (and ~all of it when cold). The 500-eTOPS NPU is idle twice over: decode is BW-bound,
  and retrieval/orchestration is CPU/DDR-bound. Caveat: the LLM/orchestration split varies with
  answer length; report as a band, not a point.
- 5090 GPU-util signatures (SM%/mem%) are NOISY and over-provisioned for a 7B — do NOT headline
  them. The robust evidence is the time decomposition + the per-board decode ladder below.
- **n=3 confirmation (2026-10-06):** decode_heavy decode-share = **98.9%** (stable); 5090 decode
  rate **~220 tok/s** consistent across decode_heavy (220.0) and rag_multistep (202.8). The
  decode-dominance headline is stable, not a single-sample fluke.

## MEASURED — the decode ladder (⚠ MIXED instruments — see per-row tags; 7B v4 Q4_K_M)
| board | mem bandwidth | decode tok/s | prefill tok/s |
|---|--:|--:|--:|
| RTX 5090 (Blackwell, desktop) | ~1792 GB/s | ~220 (full-stack chat) | 4846 (telemetry) |
| **Thor (Jetson, Blackwell sm_110)** | TBD | **40.9** | **1599** |
| Orin AGX (Ampere sm_87) | 204.8 GB/s | 27.8 | — |
- DERIVED: decode tok/s tracks memory bandwidth across the ladder — the bandwidth-bound law, on
  real NXP-relevant-class silicon. (5090 full-stack vs llama-bench are different instruments; a
  5090 llama-bench point is a TODO for a clean apples-to-apples decode curve.)

## Multi-agent vs single agent (SOURCED)
Adding concurrent agents is expensive: prefix-cache hit rate collapses ~98% → 16–26%, throughput
drops up to 66% under concurrent agent load. ⚠ **These magnitudes are from a 2×H100-NVL / vLLM
datacenter measurement — the MECHANISM transfers to edge, the NUMBERS do NOT; edge magnitudes must
be re-measured on LPDDR5X silicon** (per the research caveat). Direction stands: **a single
well-scoped bounded agent (Skippy model) is the edge answer; multi-agent is justified only when the
mission needs genuinely parallel independent context.** Confirms the floor (Skippy) / ceiling
(openclaw, the fleet) split.

## Open / honest caveats (do not hide in the deck)
- Ladder agentic-task numbers are DERIVED (full Skippy on ARM is the early-finish stretch, not done).
- **i.MX95 ARA240 — now MEASURED (fleet, 2026-10-06; corrected after qualcomm + agentic-skills-imx review):**
  - **LLM decode, Qwen2.5-7B on ARA240: 6.3 tok/s decode · TTFT 2.0 s** [MEASURED 2026-07-16,
    agentic-skills-imx `IMX95_BOARD_DOSSIER.md` §5] — load 29.6 s / 8.19 GB, coherent output; **within
    3% of NXP's published 6.51 tok/s spec** [SOURCED]. ⚠ dated: the staged `.dvm` has since been
    deleted, so this is MEASURED-*then*, not re-confirmable today — cite with the date. (e2e short-prompt
    5.36 tok/s is [UNVERIFIED] user-rate; the gap vs 6.3 is prefill folded into e2e, not a slower engine.)
  - **LLM decode, Qwen2.5-3B on ARA240: 12.9 tok/s decode · 46.6 prefill · TTFT 0.88 s · die 52 °C**
    [MEASURED ~2026-09, qualcomm via optimum-ara, INT4 Genie .dvm] — fresh, re-confirmable. 3B not 7B,
    but a clean scoped architectural number.
  - ⚠ **The 71% host / 15% accelerator split is a VISION number, NOT LLM** (yolov8n single-frame CNN
    e2e 42.8 ms / ORB-SLAM3) — it's the right "off-SoC M.2 handoff dominates" story for **perception**,
    but must NOT sit on an LLM-decode slide (decode is bandwidth/dequant-bound; that frame is
    host-pre/post-bound). Pair the two as separate rungs: i.MX95 has a MEASURED generative rung (3B/7B
    decode) AND a MEASURED perception rung (handoff-dominated detection).
  - **Neutron is a prefill/TTFT story, never a decode one:** prefill offloads 8.43× @L=512 (→~2.15×
    @16k) but decode is a WASH (1.08×, explicitly do-not-offload). So "LLM decode on i.MX95" = the
    ARA240, full stop. [agentic-skills-imx]
  - **Toolchain status corrected (qualcomm's finding, not agentic-skills-imx's):** the r1.3 "Qwen3
    can't compile / paused pending v3.0" gate is now STALE — v3.0/r3.0 is downloaded, qualcomm compiled
    Qwen3-VL-4B to an ARA240 `.dvm` (0 errors, ~2 h), and r3.0 supports a Qwen2.5-7B GGUF path, so a
    fresh Skippy-7B ARA240 number is reachable (multi-hour compile; board-side VLM runtime still pending).
- Retrieval ANSWER-quality is a separate axis from resource cost: warm+skip_agent_loop IDs the
  right chip, but TOPS is sometimes wrong (said 0.5, truth is 2.0 TOP/s) and the agent-loop path
  degraded one answer to "no NPU"; plus residual i.MX93 leakage. Known over-grounding headroom —
  flagged, not fixed here. Pick clean demo phrasings; do not showcase a wrong-answer trace.

## MEASURED — 5-task matrix (5090, n=3, 2026-10-06, warm)
| task | type | wall | decode % of LLM | orchestration % of task | decode tok/s |
|---|---|--:|--:|--:|--:|
| decode_heavy | long generation | 1.7 s | 98.8% | 14% | 230.8 |
| inbox_triage | multi-step LLM, no RAG | 0.6 s | 94.5% | 36% | 246.5 |
| field_service | RAG + synthesis | 2.3 s | 70.1% | 37% | 217.1 |
| rag_multistep | RAG + synthesis | 1.9 s | 51.3% | 47% | 221.4 |
| prefill_heavy | big-context control | 1.2 s | 0% | 20% | — |

**Two-layer confirmation:** pure-LLM agentic tasks are decode-dominated (94–99% of LLM time);
RAG-backed tasks carry 37–47% of wall-time in retrieval+orchestration (CPU/DDR, GPU near-idle).
Decode rate is steady ~217–247 tok/s on the 5090 across task types. (GPU sm%/mem% remain noisy
on the over-provisioned 5090 — time decomposition is the robust signal; see per-task JSONs.)

## MEASURED — the stretch: retrieval layer on Thor ARM vs 5090 x86 (2026-10-06)
Same `scripts/agentic_bench/retrieval_proxy.py` on both hosts (Skippy's all-MiniLM-L6-v2 embedder
+ cosine search over the real 31,939-chunk corpus size). KB copied to Thor; m6venv (torch 2.14+cu130).

| retrieval component | 5090 host (x86) | Thor (ARM) | ARM / x86 |
|---|--:|--:|--:|
| CPU query-embed | 4.49 ms | 25.94 ms | 5.8× |
| GPU query-embed | 1.97 ms | 5.04 ms | 2.6× |
| cosine search (31,939×384) | 0.51 ms | 6.92 ms | 13.6× |
| **embed + search (CPU path)** | **5.0 ms** | **32.9 ms** | **6.6×** |

**What this is (scoped — corrected after review, does NOT "validate the ladder"):** one NN
component of retrieval — the all-MiniLM query-embed + a cosine search — measured on Thor ARM at
**6.6× the 5090 (CPU path)**. That is a real, useful datapoint: the embedding layer is modestly
slower on ARM, not catastrophically.
⚠ **It does NOT validate the full derived ladder, and the earlier "~99% decode on Thor" was wrong.**
This proxy is ~0.6% of the real orchestration layer (the 5090's warm RAG task spends ~875 ms in
retrieval+orchestration — ChromaDB/BM25/rerank/agent-loop — of which embed+search is ~5 ms). Doing
the honest projection: scale the *full* ~875 ms layer by 6.6× → ~5.8 s on Thor, vs ~5 s decode (200
tok ÷ 40.9 tok/s) ⇒ **~50% decode on Thor, not ~99%** [DERIVED]. So the phase mix does NOT shift
dramatically toward decode at the edge; both layers scale ~together. The components also ranged
2.6×–13.6× (cosine search alone 13.6×), so "~6.6× ≈ decode's 5.4×" is one point in a wide band.
**The ladder stays DERIVED** with stated assumptions; the real validation is running the actual
ChromaDB+hybrid retrieval path on Thor (TODO). Also note the cosine search here is a brute-force
numpy GEMV on a random matrix, not ChromaDB's HNSW — a different algorithm; treat the search ratio
as indicative only.

## MEASURED — the 10-task agentic benchmark (5090, 2026-10-06, warm, coherence-gated)
Grounded in OpenClaw's capability set (A/B); run via `scripts/agentic_bench/trace_bench.py`.
**Coherence gate applied: every output was read; a number only counts if the output was coherent**
(a broken output is a cheaper, unrepresentative computation — "broken is faster").

| # | task | OpenClaw cap | engine | wall | profile (sm%/mem%/W) | coherent? |
|---|---|---|---|--:|---|---|
| 1 | email (intro i.MX 95) | Email | LLM+RAG | 3.5 s | 45/29/312 decode-heavy | ✅ |
| 2 | web-search + summarize | Web browse | DDG+LLM | 4.6 s | 11/5/140 **network-bound** (search 3.7 s) | ✅ |
| 3 | spec_rag (Neutron features) | Memory | RAG+LLM | 2.5 s | 69/39/321 retrieval+decode | ✅ |
| 4 | file read→summarize→write | File R/W | file tools+LLM | 1.0 s | 14/7/148 | ✅ (after read_file fix) |
| 5 | transcribe meeting | *(edge-only)* | **Whisper base** | 0.97 s | **7.3× realtime** | ✅ (engine direct) |
| 6 | OCR screenshot→text | *(edge-only)* | **Tesseract** | 0.6 s | **4/0/78 — CPU, GPU idle** | ✅ |
| 7 | meeting→transcribe→summarize | *(edge-only)* | Whisper→LLM | ~ASR+LLM | multi-stage | ✅ (composed) |
| 8 | run sandboxed script | **Shell** | run_script | 0.06 s | process/compute | ✅ (correct output) |
| 9 | multi-tool chain (web→file→email) | Orchestration | agent loop | 11.4 s | **74/50/412 — heavy** | ⚠ paused at confirm-gate |
| 10 | doc brief (rag_k=8) | Knowledge+gen | RAG+LLM | 4.5 s | 61/38/409 **prefill-heavy** (pf 1419 ms) | ✅ |

**Profiles span:** decode/bandwidth (1,3,10) · network I/O (2) · filesystem (4) · ASR-compute (5,7) ·
**CPU-only vision/OCR (6)** · shell/process (8) · multi-step orchestration (9) · prefill/long-context (10).
The perception tasks (5,6) finally give the "NPU/accelerator earns its keep on perception, not the
agent" claim measured legs — OCR runs with the GPU at **4%** (pure CPU), ASR hammers it.

**Task 9 is a finding, not a failure:** the multi-tool chain engaged the agent loop (11.4 s, 74% SM,
412 W — a real heavy multi-step profile) then **halted at Skippy's `write_file` confirm-gate** ("approve
or deny"). That's the bounded agent's *safety* behavior — it will not autonomously write/act without
human approval. The floor/ceiling contrast in one datapoint: the edge agent stops; OpenClaw-style
autonomy would just run it.

### Skippy bugs found while building this (real app bugs, not benchmark artifacts)
- **`read_file` workspace-resolution bug — FIXED** (`pipeline/agent_tools.py`): `read_file` resolved
  relative paths against CWD while `write_file` used the workspace, so `write_file("x")` then
  `read_file("x")` disagreed. Now both resolve to the workspace (verified). 
- **`/upload/transcribe` mislabels a dependency conflict as "Whisper not installed"**: whisper +
  torch+cuda ARE installed and Whisper loads/runs standalone (7.3× RT), but `import whisper` fails
  *inside the server process* (dep conflict with already-loaded libs) and the `except ImportError`
  prints a misleading message. ASR measured directly as the honest workaround; the endpoint fix
  (lazy-load whisper in a subprocess, or pin the conflicting dep) is a documented TODO.
- **OCR demo image is a UI screenshot** (text extracted fine, but low-semantic-value); swap for a
  datasheet page for a cleaner demo.

## MEASURED — vanilla base vs fine-tune (5090, 2026-10-06): reproducibility validated
Swapped Skippy's brain to the PUBLIC base **Qwen2.5-7B-Instruct Q4** (same arch as the fine-tune),
reran all 10, restored production after. Three results:

**1. Resource rates ~identical (architecture-determined) — the benchmark is reproducible on the
public model, no private brain needed.**
| task | fine-tune decode_ms | base decode_ms |
|---|--:|--:|
| email | 1970 | 2007 |
| spec_rag | 1072 | 1080 |
| doc_brief | 2075 | 2947 (more verbose) |
| file_ops | 564 | 391 |
Decode *rate*, prefill, memory, GPU profiles track the model ARCHITECTURE, not the fine-tuning —
so anyone with the public Qwen2.5-7B GGUF reproduces our numbers. (This is why the README repro
can say "pull the base model.")

**2. Base outputs are CLEANER.** Proper "Subject:/Hi [name]" emails, well-structured markdown
briefs, correct specs — and NONE of the fine-tune artifacts (the decode_heavy hallucination, the
Marvell-signature leak were fine-tune behaviors). ⇒ **publish the benchmark numbers on the base
model**: reproducible AND more coherent.

**3. NUANCE — tool-calling propensity differs (matters for the agentic/orchestration task).**
On `multi_tool_chain`: the **fine-tune actually engaged the agent loop** (11.4 s, paused at the
write_file confirm-gate — it tried to *act*); the **base model narrated the steps in prose** (4.2 s,
coherent but did NOT invoke tools). So for measuring genuine multi-step *orchestration*, the
fine-tune (trained on tool-using conversations) is the more representative agent; the base
under-triggers tools. Report the orchestration profile with that caveat, or force tool-use.

**Takeaway for the deliverable:** canonical benchmark numbers = base model (reproducible + coherent);
note the fine-tune is production Skippy and is the better *tool-caller* for the orchestration task.

## MEASURED — canonical all-10 sweep on BASE model (5090, 2026-10-07, ASR fixed)
After the transcribe subprocess fix, ALL 10 tasks run cleanly on the public base
Qwen2.5-7B-Instruct Q4 — the definitive reproducible dataset. Representative wall/decode:
email 3.6s (dc 1916) · web_summarize 1.3s (network, sm 8%) · spec_rag 2.3s (dc 1023) ·
file_ops 0.8s · transcribe 1.9s (ASR, CPU subprocess, sm 3%) · ocr 0.6s (CPU, sm 2.6%) ·
meeting_summarize 1.9s · run_script 0.05s · multi_tool_chain 4.2s (base narrates) ·
doc_brief 5.5s (prefill-heavy, pf 1442, dc 2815). All coherent (base avoids the fine-tune
artifacts). Perception tasks (transcribe/ocr) run with the GPU at 2-3% — pure CPU, the
"accelerator is idle for the agent; it earns its keep on perception" point, measured.
Raw per-task JSONs: eval/results/ladder/agentic/bench_*.json.

## MEASURED — task-success / accuracy (base model, 2026-10-07) — ⚠ SINGLE-JUDGE DRAFT
Addresses the #1 VP objection ("fast — but RIGHT?"). A fable judge graded the canonical base-model
outputs for coherence + task-completion + factual correctness vs the i.MX 95 datasheet.
**⚠ single-judge — needs the second judge (Sonnet/GPT-4o) to confirm per Fleet Law before it ships.**

| task | success | note |
|---|---|---|
| email | PASS | coherent, facts correct (truncated mid-sign-off — token cap) |
| spec_rag | PASS | 2.0 TOP/s, 1024 MACs, 2 OPS/MAC all correct |
| file_ops | PASS | chained read→summarize→write (write_result confirms); minor LPDDR label nit |
| transcribe | PASS | transcript exact ("Thunderstorms could produce large hail...") |
| run_script | PASS | correct compute (2.0 TOPS), rc=0 |
| web_summarize | PARTIAL | coherent but content-free ("details not in snippets") — didn't summarize the subject |
| meeting_summarize | PARTIAL | transcript correct but **summary=null — the summarize step never ran** (real bug) |
| doc_brief | PARTIAL | well-structured but **"256 MACs" (should be 1024)** + invented "PCIe x12 / 2×10GbE" |
| ocr | FAIL | tesseract returned ~90% gibberish from the UI screenshot (low-contrast source) |
| multi_tool_chain | FAIL | base narrated instead of invoking tools; + TRDC/VFCCU expansions wrong, "1 A55" vs 6 |

**5/10 PASS → the honest metric is cost-per-SUCCESSFUL-task, not cost-per-task.** Most failures are
FIXABLE deployment issues, not fundamental: (a) OCR demo image is a low-contrast UI screenshot — swap
for a datasheet page; (b) **meeting_summarize summary=null is a real bug** — the summarize=true path
returns the transcript without invoking the summarizer; (c) **doc_brief/multi_tool_chain hallucinate
specifics** (256 vs 1024 MACs, invented interfaces, wrong acronym expansions) — the over-grounding /
small-model-knowledge gap, worse on long-form; (d) 4 tasks truncate mid-sentence — a harness
max_tokens cap, not model quality. Arbitration needed (2nd judge): ocr (legibility vs pipeline-ran),
meeting_summarize (transcript-correct vs summary-missing), email (truncation cost).
**Deck must show this column** — cost numbers on tasks the agent fails are not sellable without it.

---

## ACCURACY vs MODEL SIZE — base 7B vs base 14B, same 5090, same tasks (MEASURED, 2026-10-07)

The edge-sizing decision isn't just "does it fit + how fast" — it's **"is the smaller brain a
good enough agent."** Ran base Qwen2.5-**7B** vs base Qwen2.5-**14B** Instruct (Q4_K_M) on the
5090, identical prompts/RAG, to isolate *size* (no fine-tune confound). Two axes:

### Resource cost of the bigger brain (MEASURED, 5090)
| task | 7B decode tok/s | 14B decode tok/s | slowdown | 7B VRAM | 14B VRAM |
|---|--:|--:|--:|--:|--:|
| spec_rag | 220.5 | 113.0 | **1.95×** | 9.1 GB | 18.2 GB |
| doc_brief | 179.7 | 99.5 | **1.81×** | 9.2 GB | 18.4 GB |

Decode slowdown (~1.9×) ≈ **parameter ratio (14/7 = 2.0×)** ≈ what bandwidth-bound decode predicts:
2× the weights to stream per token → ~2× slower. **The model-size axis and the memory-bandwidth
thesis are the same physics.** VRAM also ~2× — and that is the gating fact on edge: a 14B Q4 (~9 GB
weights + KV) does **not** fit iq9 or i.MX95; it's a Thor/Orin-and-up brain.

### Accuracy / agent-quality of the bigger brain (⏳ two-judge pending; observations)
- **Tool use — the decisive gap.** On the multi-tool-chain task, **7B *narrates*** ("Sure, let's
  break this down into steps…") and never calls a tool; **14B actually *invokes* the real tool**
  (`write_file(path='chain_notes.txt' …)`, halts at the confirm gate). On the email task, 7B writes
  prose; 14B reaches for `send_email(...)`. The bigger model *acts*; the smaller one *describes*.
- **Fabrication under free synthesis.** 7B's narrated spec list invents interfaces and gets facts
  wrong — "**Neutrino** NPU" (it's **Neutron**), "Dual Cortex-A55 + two Cortex-M7" (it's **6× A55**),
  "16 GB **LPDDR4X**" (it's **LPDDR5X**). 14B sidesteps this by calling the tool instead of
  free-generating, and on direct RAG Q&A stays closer to the retrieved source text.
- **Honesty on bad input.** Given empty/degraded web-search results, **14B said so** ("there was an
  issue with the provided search results"); 7B confabulates a confident summary from nothing.
- **Neither is perfect.** 14B over-read the NPU block as "four Neutron NPUs" on doc_brief — a count
  it should not assert. Size reduces, does not eliminate, hallucination.

**The VP takeaway:** the "invented interfaces" failures are a **7B-capacity artifact**, not a Skippy
bug — the historical Skippy that grounded well ran *bigger* models (Mixtral 8×7B, Qwen 14B). The
sizing tradeoff is concrete: **+1 model tier ≈ 2× decode latency + 2× memory, bought back as a
materially better agent (acts vs narrates, grounds vs fabricates).** Which tier an edge board can
*hold and feed* is the real constraint — and it's a bandwidth/capacity question, still not a TOPS one.

> Provenance: resource table MEASURED (5090, base GGUFs, 2026-10-07, warm, n=1 — n=3 pending).
> Accuracy observations are single-reader; **two-judge grading pending before any deck claim** (repo rule).

### Edge 14B rung — status (2026-10-07, honest)
Attempted to confirm the 7B→14B decode ratio ON edge silicon (not just the 5090):
- **Orin (sm_87):** 14B GGUF transferred (8 GB, verified), 7B already resident. `llama-bench`
  **SEGFAULTS on model load (exit 139), both -ngl 99 and -ngl 0**, right after CUDA init — the
  resident build (b11009, 2026-09-16) has drifted against the current Jetson CUDA runtime. The
  prior Orin 7B = 27.8 t/s was MEASURED on an earlier working state; **not reproducible tonight**.
  No clean same-board 7B/14B pair from Orin without a rebuild. Clocks also unpinnable (no
  passwordless sudo) — any Orin number would carry an "unpinned clocks" caveat regardless.
- **Thor (sm_110):** has a WORKING build (build-sm110, a live llama-cli proves it runs) — the right
  board for the edge pair — but is currently contended (load 5.2, a 3.2 h llama-cli job under
  kyle's user). Measuring there now would be contention-contaminated.
- **Headline is unaffected:** the size-vs-accuracy comparison is already MEASURED and solid on the
  5090 (7B 220 → 14B 113 t/s, ~1.9× ≈ param ratio). qualcomm's iq9 data independently confirms the
  mechanism (decode DDR-bound, not TOPS). The edge-14B rung is a *reinforcement*, not the headline;
  if not measured by Friday it ships as DERIVED (expected ~1.9× slower, matching 5090 + param ratio)
  clearly tagged, with the measured confirmation flagged as the immediate next step.

### Edge 14B rung — RESOLVED on Thor (MEASURED, 2026-10-07)
Thor (Jetson AGX, sm_110 Blackwell), llama.cpp build-sm110 (fb27a525), base GGUFs, -ngl 99,
-p 512 -n 128 -r 3. Board reserved hard; a stale 3.2 h llama-cli job was reaped first (Kyle-
authorized); census at run confirmed no competing compute tenant. Clocks UNPINNED (no passwordless
sudo) — but n=3 σ < 0.1%, so the rate is solid.

| model | prefill t/s | decode t/s |
|---|--:|--:|
| 7B base Q4_K_M | 1601 ± 18 | **40.85 ± 0.02** |
| 14B base Q4_K_M | 799 ± 8 | **20.89 ± 0.01** |

**Decode ratio 7B/14B = 1.96× on edge**, matching 5090 (1.95×) and the parameter ratio (1.94×).
The ~2× size penalty is **architecture-invariant across the whole ladder** (datacenter GPU →
Blackwell edge) — because decode is weight-streaming/bandwidth-bound, so time scales with params.
7B decode 40.85 reproduces the prior Thor 40.9 rung (independent confirmation; rate is
arch-determined, base≈fine-tune). Prefill ratio 2.00× (compute-bound, also ~linear in params here).

**Ladder now (7B / 14B decode t/s, Q4):** 5090 220.5 / 113.0 · Thor 40.85 / 20.89 · Orin 27.8(prior)
/ build-broken · iq9 8.375(qualcomm, w4a16, 7B-class; 14B won't fit 8 MB VTCM) / i.MX95 orb_slam-ARA240(dated).

### Task-success column — clean 7B-base (SINGLE-JUDGE DRAFT, 2026-10-07)
⚠️ Single-judge (Opus, in-context) draft — the FORMAL two-judge grade (Sonnet + GPT-4o, repo rule
for any result leaving the repo) is PENDING API keys and MUST run before this ships in the deck.
Graded against datasheet ground truth. **7 PASS / 2 PARTIAL / 1 FAIL** (up from 5 PASS pre-apparatus-fix).
PASS: email, spec_rag, file_ops, transcribe, ocr, run_script, doc_brief. PARTIAL: web_summarize
(generic, weak web results), meeting_summarize (correct summary but fabricated action-items/names
from a non-meeting clip). FAIL: multi_tool_chain (narrates instead of invoking tools + fabricates
specs). The FAIL + PARTIALs are genuine 7B capability limits, not apparatus — and the size study
shows 14B base FIXES the FAIL (invokes write_file). Detail: task_success_7b_draft.json.
