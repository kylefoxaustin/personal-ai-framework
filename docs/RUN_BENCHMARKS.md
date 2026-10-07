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
