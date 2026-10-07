# Agentic-at-the-Edge — Measured Findings (deck source of truth)

*2026-10-06, 5090 source-of-truth measurements + deep-research grounding. Provenance tags per
Fleet Law: MEASURED (ran it) / DERIVED (computed) / SOURCED (literature). Harness:
`scripts/agentic_bench/trace.py`. Research: `docs/research/agentic-edge-deep-research-20261006.json`.*

## The frame (what "agentic at the edge" means)
- **(a) bounded skill-orchestrator** — fixed, pre-tested skills; intent→plan→execute→synthesize;
  no code-gen. **Shippable on edge. = Skippy.** SOURCED: *every* shipping on-device agent today is
  this model (Gemini Nano = 4 fixed LoRA skills; Apple 3B, "not a general chatbot," constrained
  tool-calling that can't hallucinate tool names). **No shipped product runs on-device code-gen.**
- **(b) autonomous code-gen agent** — writes/runs novel code; wants a frontier brain. = openclaw /
  the fleet. The CEILING; cloud or biggest-chip territory.

## ★ Headline (the VP's premise, corrected — SOURCED + MEASURED)
**Agentic-edge is memory-BANDWIDTH-bound, not TOPS-bound.** A big NPU (e.g. 500 eTOPS) is largely
idle during agentic work; the binding resource is LPDDR5X bandwidth.
- SOURCED: real agentic traces are decode-dominated (decode = 91–98.6% of LLM time, because high
  cross-turn context reuse means little re-prefill); decode is bandwidth-bound, prefill is
  compute-bound. No existing agentic benchmark (AgentBench/τ-bench/WebArena/AndroidWorld) measures
  on-device resource cost — only task success. That gap is the whitespace.
- MEASURED (ours, 5090, `trace.py decode_heavy`): on a generation task **decode = 1826 ms vs
  prefill 21 ms → decode is 98.8% of LLM time.** Matches the literature on our own stack.

## MEASURED — 5090 (source of truth)
| workload | wall | prefill | decode | decode % of LLM | note |
|---|--:|--:|--:|--:|---|
| decode_heavy (400 tok gen) | 2.24 s | 21 ms | 1826 ms | **98.8%** | decode dominates generation |
| prefill_heavy (big ctx, 1 tok) | 1.26 s | 997 ms | ~0 | 0% | prefill = compute-bound (SM 99% peak) |
| rag_multistep (WARM) | **2.5 s** | 520 ms | ~1100 ms | — | agentic task: ~**65% LLM / ~35% retrieval+orchestration** |
| rag_multistep (COLD, 1st call) | **89 s** | 519 ms | 312 ms | — | **cold-cache: ~88 s loading embed model + BM25 index + ChromaDB** |

- **Cold-start is a first-class edge metric.** 89 s cold → ~1.7–2.5 s warm (50×). A device that
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

## MEASURED — the decode ladder (same llama-bench instrument, 7B v4 Q4_K_M)
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
drops up to 66% under concurrent agent load. → **A single well-scoped bounded agent (Skippy model)
is the edge answer; multi-agent is justified only when the mission needs genuinely parallel
independent context.** Confirms the floor (Skippy) / ceiling (openclaw, the fleet) split.

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

**Why this validates the DERIVED ladder:** the retrieval embed+search layer slows ~6.6× on Thor's
ARM — about the same factor as decode (5.4×, 220→40.9 tok/s). Both layers scale down together, so
the per-task phase *proportions* are ~preserved across the ladder (the assumption the derived
projection rests on). And absolute retrieval stays negligible (33 ms) vs decode (~5 s for a
200-token answer) ⇒ **the agentic task is decode-dominated even harder at the edge** (~99% decode
on Thor vs ~65% on the 5090). The bandwidth-bound thesis gets *stronger* down the ladder.

**Scope (honest):** this measures the SEMANTIC embed+search component on Thor ARM — NOT the full
hybrid orchestration (BM25 + reranker + ChromaDB round-trips + agent-loop detection passes). Those
other parts are either light Python (BM25/rerank) or are themselves LLM calls (agent-loop detection)
that scale with decode — so the full orchestration would track the same ~5-6× band. Standing up the
complete Skippy stack on Thor (pip+chromadb+FastAPI) remains the fuller version; this is the key
NN component measured, reusing qualcomm's m6venv + the llama.cpp CUDA build (no llama-cpp-python compile).

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
