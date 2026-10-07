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
- i.MX95 = SOURCED/feasibility (ARA240 runs a 7B; Neutron feasibility gate exists). Not measured.
- Retrieval ANSWER-quality is a separate axis from resource cost: warm+skip_agent_loop IDs the
  right chip, but TOPS is sometimes wrong (said 0.5, truth is 2.0 TOP/s) and the agent-loop path
  degraded one answer to "no NPU"; plus residual i.MX93 leakage. Known over-grounding headroom —
  flagged, not fixed here. Pick clean demo phrasings; do not showcase a wrong-answer trace.
