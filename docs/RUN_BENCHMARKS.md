# Run the Agentic-Edge Benchmark Yourself

Measure what an **agentic AI workload actually costs on edge silicon** — decomposed into
CPU / memory-bandwidth / NPU — using Skippy (a bounded skill-orchestrator agent) as the
reference. This reproduces the ladder behind the finding that **agentic-edge is
memory-bandwidth-bound, not TOPS-bound**.

> **You do NOT need the private fine-tuned "Skippy" brain.** Resource cost is determined by the
> model *architecture* (Qwen 2.5 7B), not the fine-tuning — so the public base model reproduces
> the same numbers. (The fine-tune only changes voice/style and tool-calling propensity; see
> "Which brain" below.)

---

## Quick start — the whole system, one command

```bash
python3 scripts/run_benchmark.py
```

That sequences all four layers into one run and writes a single unified result
(`eval/results/system/latest.json`) plus a printed rollup:

1. **Workload** — the 10-task agentic harness on the 5090 (per-task resource profile)
2. **Ladder** — decode/prefill across boards (5090 from harness telemetry; Thor/Orin via llama-bench)
3. **Grade** — two-judge (Sonnet + GPT-4o) task-success
4. **Aggregate** — one manifest + a rollup table

It is **graceful**: if Skippy isn't up, a board is unreachable, or the API keys aren't set, that
phase is skipped with a clear note and the rest still runs. Default scope is **5090 / Thor / Orin**;
`--full` adds the small boards (iq9 + i.MX95, via their driver adapters — rungs tagged by model size
and provenance). Useful flags:
- `--model base-7b|prod-7b-v4|14b|<path>` — swap + restart the model as part of the run
- `--full` — all five boards (5090/thor/orin/iq9/imx95-cpu/imx95-ara)
- `--boards 5090,thor` · `--skip-grade` · `--skip-ladder` · `--only harness|ladder|grade|aggregate`

### Provisioning + the feasibility gate ("can this board run this model?")

Before running a model on a board you have to answer *can it run there* and *get the artifact onto
it*. That's `scripts/provision.py` + `eval/model_registry.json`:

```bash
python3 scripts/provision.py orin qwen2.5-14b            # gate -> download from HF if missing -> stage -> ready path
python3 scripts/provision.py orin qwen2.5-3b --check-only # just the gate: RUNNABLE / UNRUNNABLE + why
python3 scripts/bench_board.py orin --provision qwen2.5-7b # gate -> provision -> bench that model
```

The **feasibility gate** checks memory fit (weights + KV vs board RAM) and runtime/arch support, and
refuses with a reason (`UNRUNNABLE — won't fit: ~12 GB > 8 GB`) instead of failing obscurely. The
**provisioner** gets the GGUF in cost order: already-staged → download from HF (`hf download`) →
[Phase B: build/convert]. **Scope today: NVIDIA Jetsons (GGUF).** iq9 (Qualcomm Genie `.bin`) and
i.MX95 (Kinara `.dvm`) are the Phase-B *build* targets — the gate already refuses them here rather
than pretend; wiring their build/convert toolchains is the next phase.

**Prerequisites:** Skippy up on :8080 (`./run.sh start`) for the workload phase; SSH reach to the
Jetsons for the ladder; `~/.personal-ai/keys.env` populated for grading (see `api-keys-setup.md`).
Everything below is the per-piece detail the orchestrator drives.

---

## What it measures

A 10-task benchmark grounded in the capability set of a general agent framework, spanning
distinct resource profiles — so you see *different agentic tasks hit different silicon*:

| task | engine | dominant resource |
|---|---|---|
| email (draft) | LLM + RAG | decode / bandwidth |
| web-search + summarize | web + LLM | network I/O + decode |
| spec-RAG (knowledge Q&A) | RAG + LLM | retrieval + decode |
| file read→summarize→write | file tools + LLM | filesystem + decode |
| transcribe (ASR) | Whisper | audio compute |
| OCR (image→text) | Tesseract | **CPU (GPU idle)** |
| meeting→transcribe→summarize | Whisper → LLM | multi-stage |
| run sandboxed script | shell | process/compute |
| multi-tool chain | agent loop | multi-step orchestration |
| doc brief (long-context) | RAG + LLM | prefill-heavy |

Each run captures wall-time, prefill/decode split, tokens/sec, and a GPU sampler
(compute% / memory% / power). **Coherence gate:** read every output — a broken output is a
cheaper, unrepresentative computation ("broken is faster"), so its numbers don't count.

---

## Prerequisites

- **Hardware:** an NVIDIA box. Tested on RTX 5090 (desktop), Jetson AGX Thor (sm_110), Jetson
  AGX Orin (sm_87). Any CUDA GPU works; the *point* is comparing across tiers.
- **Skippy** running (the LLM server on :8080 + ChromaDB). See the main README for setup.
- **A model** (see below).
- **A RAG knowledge base** for the retrieval tasks (see below).
- `requests` in your Python env (for the harness uploads).

## Which brain (model)

Point `pipeline/config.yaml` → `model.path` at a GGUF:

- **Reproducible/public:** `Qwen2.5-7B-Instruct` Q4_K_M (download from Hugging Face). This is what
  the published numbers use — anyone can reproduce them.
- **Production Skippy:** the fine-tuned `kyle-qwen25-7b-v4` (not public). Same architecture → same
  resource rates; cleaner *voice* but with fine-tune quirks. It is a better *tool-caller* (it
  actually invokes tools on the multi-tool task, where the base model narrates instead) — the one
  place the two differ for this benchmark.

Restart the server after changing the path. Confirm which model is live:
```bash
curl -s -XPOST localhost:8080/generate -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"prompt":"hi","use_rag":false,"max_tokens":3}' | jq .model
```

## The knowledge base (RAG tasks)

The retrieval tasks query a datasheet corpus (we use NXP i.MX datasheets — relevant to the audience
and a realistic text-and-table workload). Options:
- **Bring your own:** ingest your docs via the upload/ingest endpoints (see main README), then point
  the task prompts at your domain.
- **Sample KB:** a small example corpus + ingest script is on the cleanup list (TODO). Until then,
  the RAG tasks need *a* populated ChromaDB; the resource *profile* (retrieve + decode) is the same
  regardless of corpus.

---

## Run it

```bash
# all 10 tasks (default)
python3 scripts/agentic_bench/trace_bench.py

# a subset
python3 scripts/agentic_bench/trace_bench.py email spec_rag transcribe ocr

# raw-decode ladder point for a board (same instrument across boards):
#   build llama.cpp for the board's arch, then:
~/llama.cpp/build/bin/llama-bench -m <model.gguf> -p 512 -n 128 -r 3 -ngl 99
```

### Turnkey per-board runs (`scripts/bench_board.py`)

One command per board — reserves it HARD, censuses (flags only processes actually using >15% CPU,
not idle name-matches), stages the base 7B if missing, verifies the runtime loads (rebuilds a
drifted Jetson `llama.cpp` with `--rebuild`), runs the decode/prefill bench, writes a standardized
JSON to `eval/results/ladder/board_runs/<board>.json`, releases the board and prints any process it
started:

```bash
python3 scripts/bench_board.py thor        # NVIDIA Jetson AGX Thor  (CUDA llama-bench)
python3 scripts/bench_board.py orin        # NVIDIA Jetson AGX Orin  (CUDA llama-bench; --rebuild if it segfaults)
python3 scripts/bench_board.py imx95-cpu   # NXP i.MX95 FRDM-PRO, A55 CPU (llama.cpp)
python3 scripts/bench_board.py imx95-ara   # NXP i.MX95 + Kinara Ara-2 (reports endpoint count; .dvm perf is dated)
python3 scripts/bench_board.py iq9          # Qualcomm IQ-9075 Hexagon (Genie — stub; coordinate w/ the qualcomm session)
python3 scripts/aggregate_boards.py        # combine board_runs/*.json -> the decode ladder
```

Per-board runtimes genuinely differ (CUDA llama.cpp / CPU llama.cpp / Kinara `.dvm` / Qualcomm
Genie), so each board has its own adapter; the driver standardizes the *invocation and the output*,
not the engine. iq9 is a stub because that toolchain + board belong to the qualcomm session.
**i.MX95 accelerator note:** the Kinara Ara-2 ("ARA240") exposes **independent endpoints** —
`hw_metrics` reports `count=N`; a 2nd Ara adds ~2× aggregate throughput, not single-query speed
(same shape as the iq9 dual-NSP). The on-SoC Neutron NPU is separate and is a prefill/TTFT engine
(8.4× offload), not a decode one.

Each task writes a JSON to `eval/results/ladder/agentic/bench_<task>_<ts>.json` with wall-time,
telemetry, the GPU signature, and the full output. **Read the outputs** — apply the coherence gate.

## Interpret

- **Decode dominates.** Generation-heavy tasks are ~95-99% decode time; decode is
  memory-bandwidth-bound. Across boards, decode tok/s tracks memory bandwidth, not TOPS.
- **The NPU is idle for the agent.** Perception tasks (OCR, ASR) are where an accelerator earns its
  area; the LLM agent rides on bandwidth + CPU. OCR here runs with the GPU at ~3%.
- **Orchestration + cold-start matter.** RAG/tool tasks carry real CPU/DDR cost; first-query
  cold-start (model + index load) can be ~50× the warm path — design to keep models resident.

Full measured results and methodology (with MEASURED / DERIVED / SOURCED provenance tags):
`eval/results/ladder/agentic/FINDINGS.md`. Context: `docs/agentic-edge-benchmark-roadmap.md`.

## Honesty notes
- Numbers are hardware-specific; report your board + model + date.
- The i.MX95 rung uses fleet-measured ARA240 numbers (dated); re-measure on your silicon.
- This measures *resource cost*, not answer quality — those are separate axes.
