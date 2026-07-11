# RunPod Runbook — fp16 vs FP8 v2+RAG apples-to-apples

Short runbook for running the purity-check RAG comparison that won't
fit locally on the 5090 (28 GB fp16 model + ~7K KV cache > 32 GB).

**Why we need the pod:** our local FP8 RAG test had to use a shim
prompt-wrapping that differs from Q4_K_M's Skippy-native path. For a
fully methodologically pure fp16-vs-FP8 answer we need both models
going through identical wrapping. Both models need to fit on the same
GPU with headroom for 8 RAG chunks (~7K context). 5090 can't fit fp16
at that budget.

**Target pod:** A6000 48GB (~$0.50/hr) or L40S 48GB (~$0.80/hr).
A100 80GB works too but is overkill. Runtime estimate: **~25 min on
A6000 → ~$0.25 total**.

---

## Part 1 — Provision the pod

1. Log into **https://runpod.io** → Secure Cloud → Deploy
2. Filter: GPU = **RTX A6000 48GB** (or L40S 48GB)
3. Template: **PyTorch 2.4+ with CUDA 12.x** (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`)
4. Container Disk: **100 GB** (need room for 28 GB fp16 + 16 GB FP8 + temp + venv)
5. Expose SSH over TCP. No HTTP ports needed.
6. Launch. Wait ~2 min for boot.
7. Copy the SSH command from the pod's Connect tab. You'll use it for
   upload + running.

## Part 2 — Upload the bundle

From your local machine (this repo root):

```bash
POD="<user>@<pod-ip>"  # from RunPod Connect tab, e.g. root@1.2.3.4 -p 12345

# Push the scripts + prompt set + cached RAG chunks
scp -P <port> \
  eval/runpod_bootstrap_fp8_rag.sh \
  eval/run_accuracy_eval_vllm.py \
  eval/quantize_w8a8.py \
  eval/compare_accuracy_runs.py \
  eval/prompts_v2.json \
  eval/results/rag_chunks_for_v2.json \
  root@<pod-ip>:/workspace/
```

The whole bundle is ~200 KB — uploads in seconds.

## Part 3 — Run

SSH in and kick off the bootstrap:

```bash
ssh -p <port> root@<pod-ip>
cd /workspace
bash runpod_bootstrap_fp8_rag.sh 2>&1 | tee run.log
```

Total wall clock breakdown:

| Phase | Time | Notes |
|---|---|---|
| Install deps | 3 min | pip install vllm + friends |
| Download fp16 from HF | 8 min | 28 GB at cloud uplink speeds |
| Quantize FP8 | 1 min | GPTQ-less FP8_DYNAMIC is quick |
| fp16 eval | 8 min | 44 prompts × 3 samples, with RAG |
| FP8 eval | 6 min | faster than fp16 (the 1.57× speedup) |
| Diff | 5 s | compare_accuracy_runs.py |
| **Total** | **~26 min** | → **~$0.25 on A6000** |

Watch `run.log` tail for progress. The headline diff prints at the end.

## Part 4 — Bring results home

```bash
# From local machine:
mkdir -p eval/results/runpod
scp -P <port> -r root@<pod-ip>:/workspace/results/ eval/results/runpod/
```

Files of interest:
- `acc_diff_fp16_vs_fp8_v2_rag_pure.md` — headline + category + most-divergent
- `acc_diff_fp16_vs_fp8_v2_rag_pure.json` — structured for sizer ingest
- `acc_reference-fp16-v2-rag-pure_*.json` — fp16 raw outputs
- `acc_candidate-fp8-v2-rag-pure_*.json` — FP8 raw outputs

## Part 5 — Destroy the pod

**Don't forget this step** — RunPod charges by the minute while the pod
exists.

1. RunPod console → My Pods → Stop → Terminate.
2. Verify the pod disappears from your list.
3. Confirm with the Billing page that charges stopped.

## Gotchas worth knowing

- **Pod region affects download speed.** US pods get HF downloads faster
  than EU due to HF's CDN. Doesn't matter much for 28 GB but if you're
  bored during setup, pick US.
- **If the download stalls:** `pip install hf_transfer` on the pod and
  set `HF_HUB_ENABLE_HF_TRANSFER=1` — downloads become ~2-3x faster.
- **If vLLM OOMs during eval:** the A6000 48GB should comfortably fit
  both models at 16K context. If it errors, drop `--gpu-mem-utilization`
  to 0.80 and `--max-model-len` to 8192 (some prompts may truncate but
  most will fit).
- **Bootstrap re-runnable:** safe to re-run phases 4+5 without re-
  downloading. The model is cached in `/workspace/models/`.

## What the numbers mean when they come back

The diff markdown will show:
- **Pass-rate delta** — the headline. If FP8 is within 1pp of fp16,
  we have pure apples-to-apples evidence that FP8 doesn't damage
  Skippy's RAG flow at matched methodology.
- **Jaccard per category** — where outputs drift.
- **Most-divergent prompts** — which specific prompts to manually
  inspect.

Update `eval/STORY_DRAFT.md`'s RAG section with the new number — the
caveat about "different prompt wrapping" goes away and the fp16-vs-FP8
comparison becomes clean.
