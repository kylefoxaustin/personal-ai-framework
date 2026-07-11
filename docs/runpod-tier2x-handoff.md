# Tier 2.x RunPod handoff — recipe-clean 32B + full-MoE

Two pods to spin up in parallel.

**Bundle:** `tier2-bundle.tar.gz` on Drive at `gdrive:skippy_files/personal-ai-assistant/docs/runs/tier2-bundle/`
SHA256: `61fd0e0cebcb8829eacc0908bf86c0e55371ba57831a4ed0e3cae10c1dc4970c`

The bundle is the SAME tarball name as before but updated — now contains `train_dense_lora_v4.py` (new) and `train_moe_lora_full.py` (new) alongside all prior scripts.

## Pod template (both pods, same)

| Setting | Value |
|---|---|
| Cloud | RunPod **Secure Cloud** |
| GPU | **H100 80GB SXM or PCIe** |
| Container Disk | 50 GB |
| Volume Disk | **150 GB** at `/workspace` |
| SSH | "SSH over exposed TCP" |
| Template | RunPod PyTorch 2.4.0 |

## Run A — Qwen2.5-32B v4 (CLEAN recipe, 2 epochs)

**Hypothesis:** "v4 recipe scales up the dense size axis" — to be tested cleanly with the v4 hyperparams (2 epochs + assistant_only_loss + messages-list format). The prior 32B run used the OLD pre-v4 recipe (3 epochs + full-sequence loss) and tanked rag_datasheet by over-fitting; this is the apples-to-apples test.

| Field | Value |
|---|---|
| Base | `Qwen/Qwen2.5-32B-Instruct` |
| Train script | `train_dense_lora_v4.py` |
| Merge script | `merge_dense.py` |
| Output GGUF name | `kyle-qwen25-32b-v4-q4_k_m.gguf` |
| Drive result folder | `gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v4/result/` |
| Expected wall | ~3-4 hr (2 epochs vs 3) |
| Cost | ~$15-25 |

### Pipeline

```bash
# Bootstrap (after pulling and extracting bundle)
bash /workspace/tier2-bundle/training/pod/bootstrap.sh

# Fetch base
bash /workspace/tier2-bundle/training/pod/fetch_model.sh Qwen/Qwen2.5-32B-Instruct

# Train (use train_dense_lora_v4.py, NOT train_dense_lora.py)
MODEL_PATH=/workspace/models/qwen2.5-32b-instruct \
TRAIN_DATA=/workspace/tier2-bundle/training/train_alpaca.json \
OUTPUT_DIR=/workspace/training/output/dense-32b-v4 \
nohup python3 -u /workspace/tier2-bundle/training/pod/train_dense_lora_v4.py \
  > /workspace/train.log 2>&1 < /dev/null &
disown
echo "pid=$!"
```

Then wire up the post-train chain (see template below).

## Run B — MoE attention + router + experts (full MoE LoRA)

**Hypothesis:** "Domain-knowledge regression on Qwen3-MoE fine-tunes lives in expert FFNs — adding LoRA to per-expert gate_proj/up_proj/down_proj at low rank should recover rag_datasheet precision." Router-v1 result already fixed the reasoning side; this tests the orthogonal failure mode.

| Field | Value |
|---|---|
| Base | `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| Train script | `train_moe_lora_full.py` |
| Merge script | `merge.py` |
| Output GGUF name | `kyle-30b-a3b-full-v1-q4_k_m.gguf` |
| Drive result folder | `gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-full-v1/result/` |
| Expected wall | ~6-8 hr (8× LoRA params vs router-v1) |
| Cost | ~$30-40 |

### Recipe specifics

- Attention (q/k/v/o): r=64, alpha=128 (standard)
- Router (gate.weight): ParamWrapper, r=64 (same as router-v1)
- Expert FFNs (gate_proj/up_proj/down_proj × 128 experts × 48 layers): **r=8, alpha=16**
  — low rank to keep trainable params tractable (~410M for experts)
- Total trainable: ~470M (~1.5% of 30B)
- 2 epochs, assistant_only_loss=True

### Pipeline

```bash
# Bootstrap (same as Run A — already done if you ran A first on the same pod, but
# Run B needs a SEPARATE pod since they run in parallel)
bash /workspace/tier2-bundle/training/pod/bootstrap.sh

# Fetch base
bash /workspace/tier2-bundle/training/pod/fetch_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507

# Patch chat template (Qwen3-specific gotcha)
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 \
  python3 /workspace/tier2-bundle/training/pod/patch_template.py

# Train (use train_moe_lora_full.py)
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 \
TRAIN_DATA=/workspace/tier2-bundle/training/train_alpaca.json \
OUTPUT_DIR=/workspace/training/output/moe-full-v1 \
nohup python3 -u /workspace/tier2-bundle/training/pod/train_moe_lora_full.py \
  > /workspace/train.log 2>&1 < /dev/null &
disown
echo "pid=$!"
```

### Sanity gates inside `train_moe_lora_full.py`

The script aborts before training if either:
- LoRA-wrapped module count is unexpectedly low (< 1000) → `target_modules` regex didn't match expert FFNs
- Trainable param count exceeds 1.5B → `rank_pattern` didn't apply r=8 to experts and they got r=64

If the script raises RuntimeError, paste the line and we'll fix it.

## Post-training chain (both pods)

Same shape as the prior tier2 chain — wait for "DONE. Adapter saved" then merge → quantize → push to Drive. Updated for the new run names below.

### Pod 1 (Run A — 32B dense v4 clean)

```bash
cat > /workspace/post_train.sh <<'POSTEOF'
#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/post_train.log) 2>&1

RUN=dense-32b-v4
ADAPTER=/workspace/training/output/$RUN
BASE=/workspace/models/qwen2.5-32b-instruct
MERGED=/workspace/models/qwen2.5-32b-instruct-merged-v4
GGUF_NAME=kyle-qwen25-32b-v4
DRIVE=gdrive:skippy_files/personal-ai-assistant/docs/runs/$RUN/result/
SCRIPTS=/workspace/tier2-bundle/training/pod

echo "[$(date -u +%H:%M:%SZ)] waiting for 'DONE. Adapter saved' in /workspace/train.log"
until grep -q "DONE. Adapter saved" /workspace/train.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%SZ)] training complete"

BASE_MODEL=$BASE ADAPTER=$ADAPTER OUT=$MERGED \
  python3 -u $SCRIPTS/merge_dense.py
echo "[$(date -u +%H:%M:%SZ)] merge complete"

MERGED=$MERGED OUT_NAME=$GGUF_NAME bash $SCRIPTS/convert_and_quantize.sh
echo "[$(date -u +%H:%M:%SZ)] convert + quantize complete"

GGUF=/dev/shm/${GGUF_NAME}-q4_k_m.gguf
SHA=$(sha256sum $GGUF | cut -d' ' -f1); SIZE=$(stat -c %s $GGUF)
rclone copy $GGUF $DRIVE
rclone copy $ADAPTER/adapter_config.json $DRIVE
[ -f $ADAPTER/trainer_state.json ] && rclone copy $ADAPTER/trainer_state.json $DRIVE
rclone copy /workspace/train.log $DRIVE
cat > /tmp/STATUS.md <<STATEOF
# dense-32b-v4 RunPod run — STATUS
Updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Pod: $(hostname)
Recipe: v4 (2 epochs + assistant_only_loss + messages format), attention + dense FFN LoRA
Base: Qwen2.5-32B-Instruct
COMPLETE. GGUF uploaded. SHA256=$SHA size=$SIZE bytes. Pod can be terminated.
STATEOF
rclone copy /tmp/STATUS.md $DRIVE
echo "[$(date -u +%H:%M:%SZ)] CHAIN DONE"
POSTEOF
chmod +x /workspace/post_train.sh
nohup /workspace/post_train.sh > /workspace/post_train_outer.log 2>&1 < /dev/null &
disown
```

### Pod 2 (Run B — MoE full)

Same template, with these substitutions:
- `RUN=moe-full-v1`
- `BASE=/workspace/models/qwen3-30b-a3b-instruct-2507`
- `MERGED=/workspace/models/qwen3-30b-a3b-instruct-merged-full`
- `GGUF_NAME=kyle-30b-a3b-full-v1`
- `DRIVE=gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-full-v1/result/`
- Use `merge.py` (NOT `merge_dense.py`) for MoE merge
- Recipe line: `Recipe: v4 + attention + router (gate.weight) + expert FFNs (r=8 via rank_pattern)`

## Drive folders to create before launch

```
gdrive:skippy_files/personal-ai-assistant/docs/runs/dense-32b-v4/result/
gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-full-v1/result/
```
