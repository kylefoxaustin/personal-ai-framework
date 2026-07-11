# RunPod Runbook — fp16 vs W8A8 INT8 v2+RAG (matched on non-Blackwell GPU)

Short runbook for measuring **W8A8 INT8 accuracy against fp16** on a
GPU that has working INT8 tensor-core kernels — needed because 5090
SM120 refuses `cutlass_scaled_mm` for INT8 at runtime (the
"ecosystem-blocked" finding in `eval/STORY_DRAFT.md`).

**Target pod:** A100 80GB (~$1.00/hr) or H100 80GB (~$1.75/hr). Both
expose SM80/SM90 INT8 tensor-core paths via CUTLASS. A100 is the
cheapest option that works; H100 is faster but price/perf-worse for
this one-off.

**Expected runtime:** ~35 min on H100, ~40 min on A100 → **~$0.75-$1.00
total**.

**Companion to:** `docs/runpod-fp8-eval-runbook.md` (same shape,
different quantization recipe).

---

## Part 1 — Provision the pod

1. Log into **https://runpod.io** → Secure Cloud → Deploy
2. Filter: GPU = **A100 80GB** (or H100 80GB)
3. Template: **PyTorch 2.4+ with CUDA 12.x** (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`)
4. Container Disk: **100 GB** (28 GB fp16 + ~15 GB W8A8 + temp + venv)
5. Expose SSH over TCP. No HTTP ports needed.
6. Launch. Wait ~2 min for boot.
7. Copy the SSH command from the pod's Connect tab.

## Part 2 — Upload the bundle

From your local machine (this repo root):

```bash
POD_HOST=<pod-ip>
POD_PORT=<ssh-port>   # from RunPod Connect tab

scp -P "$POD_PORT" \
  eval/runpod_bootstrap_int8_rag.sh \
  eval/run_accuracy_eval_vllm.py \
  eval/quantize_w8a8.py \
  eval/compare_accuracy_runs.py \
  eval/prompts_v2.json \
  eval/results/rag_chunks_for_v2.json \
  root@"$POD_HOST":/workspace/
```

Bundle is ~1.1 MB (the RAG chunk cache is the bulk at ~1 MB). Uploads
in seconds.

## Part 3 — Run

```bash
ssh -p "$POD_PORT" root@"$POD_HOST"
cd /workspace
bash runpod_bootstrap_int8_rag.sh 2>&1 | tee /root/runpod/run.log
```

The bootstrap script moves files to `/root/runpod/` first (the 100 GB
container volume — `/workspace` is only 20 GB and fills up during quant).

Wall-clock breakdown:

| Phase | Time | Notes |
|---|---|---|
| Install deps | 3-4 min | pip install vllm + llmcompressor |
| Download fp16 from HF | 8 min | 28 GB at cloud uplink speeds |
| Quantize fp16 → W8A8 | 20 min | SmoothQuant + GPTQ with 512 calibration samples |
| fp16 reference eval | 8 min | 44 prompts × 3 samples, RAG on |
| W8A8 candidate eval | 6 min | faster per-token than fp16 on H100 |
| Diff | 5 s | compare_accuracy_runs.py |
| **Total** | **~35-40 min** | → **~$1.00 on H100, ~$0.75 on A100** |

Headline diff prints at the end of the run.

## Part 4 — Bring results home

```bash
mkdir -p eval/results/runpod-int8
scp -P "$POD_PORT" -r root@"$POD_HOST":/root/runpod/results/ eval/results/runpod-int8/
```

Expected artifacts:
- `acc_reference-fp16-v2-rag-pure-int8pod_*.json`
- `acc_candidate-int8-v2-rag-pure_*.json`
- `acc_diff_fp16_vs_int8_v2_rag_pure.md`
- `acc_diff_fp16_vs_int8_v2_rag_pure.json`

## Part 5 — Terminate the pod

**IMPORTANT:** RunPod bills by the minute once the pod is running. Terminate
from the web UI (Pods → this pod → Stop → Terminate) as soon as scp'd results
are safe locally.

## What to do with the results

The diff's headline pass-rate Δ between fp16 and W8A8 is the INT8-on-14B
accuracy cost for the deck story. Prior run (2026-04-23, H100) showed
**Δ = -3.8pp, concentrated in refusal-specificity (not capability)**.
If the rerun confirms similar, the sizer precision ladder gets a clean
"W8A8 base + RAG = X%" row alongside the FP8 and fp16 references.

If Δ is meaningfully different from -3.8pp, something changed (different
calibration set, different vllm/llmcompressor versions, or the prior
measurement was noisy). Worth rerunning twice to check seed variance
before publishing.
