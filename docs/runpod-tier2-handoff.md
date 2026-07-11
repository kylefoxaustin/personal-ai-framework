# Tier 2 RunPod handoff — MoE-router + 32B dense

Two recipe-taxonomy Tier 2 runs to kick off (priority order).

**Bundle:** `tier2-bundle.tar.gz` on Drive at
`gdrive:skippy_files/personal-ai-assistant/docs/runs/tier2-bundle/`
SHA256: `3fbf5b840ab4a214e0e2bf1a43d32fece40076f3b00062d8a27ab13c66d864e2`

Contains all of `training/pod/` + `training/data/train_alpaca.json` (same 6,517 examples as MoE v4).

---

## Run 1 — MoE + (attention + router) LoRA  ← do this first

**Hypothesis to test:** the MoE v4 capability regression (multihop 6/9 → 0/9) was caused by attention-only LoRA disrupting expert routing without re-conditioning the gate networks. Adding the per-layer `gate` (router) projection to the LoRA target set should let the router co-adapt with attention.

**If this works:** customer rule becomes "MoE bases need MoE-aware LoRA targets — include the router." Recipe taxonomy gains a validated MoE cell.

**If this still regresses:** the failure is deeper than LoRA targeting (maybe needs full FT, DPO, or expert-FFN LoRA). Move down the hypothesis list.

### Pod parameters

| Field | Value |
|---|---|
| Hardware | H100 80GB Secure Cloud |
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` (same as v4) |
| Expected wall | ~4–5 hr |
| Expected $ | ~$15–25 |
| Output Drive folder | `gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-router-v1/` |

### Pipeline (phone-Claude or direct)

```bash
# 1. Fetch base model
bash fetch_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507

# 2. (Optional but recommended) Smoke test — 30s feasibility check on the new LoRA targets
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 python3 smoke_test.py

# 3. Train (4–5 hr, use -u for unbuffered loss logs)
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 \
TRAIN_DATA=/workspace/training/data/train_alpaca.json \
OUTPUT_DIR=/workspace/training/output/moe-router-v1 \
nohup python3 -u train_moe_lora_router.py > /workspace/training/output/moe-router-v1.log 2>&1 &

# 4. Merge adapter back into base (~20–25 min)
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 \
ADAPTER=/workspace/training/output/moe-router-v1 \
OUTPUT_DIR=/workspace/training/output/moe-router-v1-merged \
python3 merge.py

# 5. Convert to GGUF Q4_K_M (~30 min, uses /dev/shm)
MERGED=/workspace/training/output/moe-router-v1-merged \
GGUF_NAME=kyle-30b-a3b-router-v1-q4_k_m.gguf \
bash convert_and_quantize.sh

# 6. Push results to Drive
rclone copy /workspace/training/output/moe-router-v1-merged-q4_k_m.gguf \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-router-v1/result/
rclone copy /workspace/training/output/moe-router-v1/adapter_config.json \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-router-v1/result/
rclone copy /workspace/training/output/moe-router-v1/trainer_state.json \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-router-v1/result/
rclone copy /workspace/training/output/moe-router-v1.log \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-router-v1/result/
```

### Critical gate at training start

The script prints `LoRA-wrapped module count: N (expected ~240)` before training begins. If `N > 500`, the regex matched expert FFN modules and trainable params will explode to ~6.6B. The script raises `RuntimeError` in that case — abort and do not proceed.

If trainable params print is in the **30–35M range** after `model.print_trainable_parameters()`, you're good.

### GGUF post-processing

Same `{% generation %}` chat-template gotcha as MoE v4. After pulling the GGUF home, run the in-place template patch from `feedback_qwen3_gguf_template_patch.md` before loading in llama-cpp-python:

```python
import mmap
path = 'models/qwen3-30b-a3b-kyle/kyle-30b-a3b-router-v1-q4_k_m.gguf'
needles = [
    (b'{% generation %}',    b' ' * len(b'{% generation %}')),
    (b'{% endgeneration %}', b' ' * len(b'{% endgeneration %}')),
]
with open(path, 'r+b') as fh:
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_WRITE)
    head = mm[:10_000_000]
    for old, new in needles:
        idx = 0
        while True:
            i = head.find(old, idx)
            if i < 0: break
            mm[i:i+len(old)] = new
            idx = i + len(old)
    mm.flush(); mm.close()
```

---

## Run 2 — Qwen2.5 32B dense + v4 recipe

**Hypothesis to test:** the v4 dense recipe (attention + FFN LoRA) scales up the dense Qwen2.5 size axis. If 7B (+3.1pp) and 14B (+5.3pp) both gained, 32B should too.

**If this works:** recipe taxonomy gains a third validated dense cell, and the customer story extends to "the recipe is proven from 7B all the way up to 32B on dense Qwen2.5."

### Pod parameters

| Field | Value |
|---|---|
| Hardware | H100 80GB Secure Cloud |
| Base model | `Qwen/Qwen2.5-32B-Instruct` |
| Expected wall | ~5–7 hr (more compute per step than MoE since all 32B params are active) |
| Expected $ | ~$15–25 |
| Output Drive folder | `gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v1/` |

### Pipeline

```bash
# 1. Fetch
bash fetch_model.sh Qwen/Qwen2.5-32B-Instruct

# 2. Smoke test (dense variant — uses standard QLoRA prep, attention+FFN targets)
MODEL_PATH=/workspace/models/qwen2.5-32b-instruct python3 smoke_test_dense.py

# 3. Train
MODEL_PATH=/workspace/models/qwen2.5-32b-instruct \
TRAIN_DATA=/workspace/training/data/train_alpaca.json \
OUTPUT_DIR=/workspace/training/output/dense-32b-v1 \
nohup python3 -u train_dense_lora.py > /workspace/training/output/dense-32b-v1.log 2>&1 &

# 4. Merge
MODEL_PATH=/workspace/models/qwen2.5-32b-instruct \
ADAPTER=/workspace/training/output/dense-32b-v1 \
OUTPUT_DIR=/workspace/training/output/dense-32b-v1-merged \
python3 merge_dense.py

# 5. Convert
MERGED=/workspace/training/output/dense-32b-v1-merged \
GGUF_NAME=kyle-qwen25-32b-v1-q4_k_m.gguf \
bash convert_and_quantize.sh

# 6. Push results
rclone copy /workspace/training/output/dense-32b-v1-merged-q4_k_m.gguf \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v1/result/
rclone copy /workspace/training/output/dense-32b-v1/adapter_config.json \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v1/result/
rclone copy /workspace/training/output/dense-32b-v1/trainer_state.json \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v1/result/
rclone copy /workspace/training/output/dense-32b-v1.log \
    gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v1/result/
```

No template patch needed for dense Qwen2.5 — the `{% generation %}` issue is Qwen3-specific.

---

## What I (Kyle) need to do

1. Spin up an H100 80GB on RunPod Secure Cloud
2. SSH in (from work PC won't reach due to corp firewall — use phone hotspot or phone-Claude)
3. Pull the bundle:
   ```bash
   rclone copy gdrive:skippy_files/personal-ai-assistant/docs/runs/tier2-bundle/tier2-bundle.tar.gz /workspace/
   cd /workspace && tar xzf tier2-bundle.tar.gz && cd tier2-bundle/training/pod
   chmod +x bootstrap.sh fetch_model.sh convert_and_quantize.sh
   bash bootstrap.sh
   ```
4. Run the pipeline for whichever run you're starting (either Run 1 or Run 2 — they're independent pods)

After both runs land, [docs] (Skippy session) will pull GGUFs, run the v2-RAG eval, run voice metrics, and write the recipe-taxonomy update.
