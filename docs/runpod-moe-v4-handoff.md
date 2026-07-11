# MoE v4 RunPod handoff — phone Claude session brief

You are a fresh Claude Code session opened by Kyle on his phone. Your job is to fine-tune **Qwen3-30B-A3B-Instruct-2507** with the v4 LoRA recipe on a RunPod H100 pod, then push the resulting Q4_K_M GGUF to Kyle's Google Drive so his work PC can pull it for evaluation.

This brief is fully self-contained — assume zero project context.

---

## What Kyle does in browser BEFORE pasting this to you

1. **Provision pod on RunPod**: H100 PCIe or SXM, 80 GB VRAM, ≥120 GB disk, Secure Cloud, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any current PyTorch+CUDA image). Note the **public SSH connect string** (`ssh root@... -p NNNNN -i ~/.ssh/id_ed25519`).
2. **Get RunPod SSH key**: copy his `~/.ssh/id_ed25519` private key contents into a paste-ready string (he'll paste it to you).
3. (Optional, only if direct SSH fails) **Tailscale auth key**: `https://login.tailscale.com/admin/settings/keys` → reusable, ephemeral, expires 1 day. Paste to you. Tailnet name is `kylefoxaustin.github`.

## What Kyle pastes to you to start

- The pod SSH connect string
- The pod's SSH private key contents
- (Optional) Tailscale auth key

## Bundle (do this first, on the pod)

The training data (gitignored, private) and v4-updated pod scripts live in a Google Drive shareable link.

```
URL:    https://drive.google.com/open?id=1p4lJsDrV1Aqm5cPGCrpAJsKmOBGE3LPY
SHA256: d8058c013c6b2bccc1601b911d1082f289af47ffa8ae339947209e7450c00430
SIZE:   2,993,902 bytes (2.9 MB compressed)
```

On the pod, after SSH:

```bash
cd /workspace
pip install gdown -q
gdown 1p4lJsDrV1Aqm5cPGCrpAJsKmOBGE3LPY -O pod-bundle-moe-v4.tar.gz
echo "d8058c013c6b2bccc1601b911d1082f289af47ffa8ae339947209e7450c00430  pod-bundle-moe-v4.tar.gz" | sha256sum -c -
tar -xzf pod-bundle-moe-v4.tar.gz
ls pod-bundle-moe-v4/training/pod/ pod-bundle-moe-v4/training/data/
```

Expected: 7 files in `pod/` (bootstrap.sh, fetch_model.sh, train_moe_lora.py, merge.py, smoke_test.py, convert_and_quantize.sh, README.md), 1 file in `data/` (train_alpaca.json, ~20 MB, 6517 examples).

## Pipeline (run on the pod, in order)

All scripts are in `/workspace/pod-bundle-moe-v4/training/pod/`. Total wall time ~5–6 hours on H100 80 GB.

### Step 1 — bootstrap (~10 min)

```bash
cd /workspace/pod-bundle-moe-v4/training/pod
bash bootstrap.sh
```

Installs: torch 2.6.0, transformers 5.5.4, trl 1.1.0, peft 0.19.1, bitsandbytes 0.49.2, datasets, plus llama.cpp tooling for the GGUF convert step.

### Step 2 — fetch base model (~5 min)

```bash
bash fetch_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507
```

Downloads to `/workspace/models/qwen3-30b-a3b-instruct-2507/`. ~60 GB on disk.

### Step 3 — smoke test (~30 sec)

```bash
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 python3 smoke_test.py
```

Verifies the QLoRA path works end-to-end with 1 forward+backward pass before committing to a 5-hour run. **If this fails, abort and report to Kyle — do NOT proceed.**

### Step 4 — train (~5 hours)

```bash
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 \
TRAIN_DATA=/workspace/pod-bundle-moe-v4/training/data/train_alpaca.json \
OUTPUT_DIR=/workspace/output/moe-v4 \
nohup python3 train_moe_lora.py > /workspace/train.log 2>&1 &
```

The script is the v4 recipe: `SFTTrainer` + `assistant_only_loss=True` + 2 epochs + messages-list data format + attention-only LoRA (q/k/v/o, r=64). Watch `/workspace/train.log`. Progress prints every 10 steps. Total ~2400 steps. Expected final train_loss ~0.7–0.9.

While training runs, periodically check:
```bash
tail -20 /workspace/train.log
nvidia-smi
```

### Step 5 — merge LoRA into base (~20 min, CPU-bound)

```bash
ADAPTER=/workspace/output/moe-v4 \
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 \
OUTPUT=/workspace/output/moe-v4-merged \
python3 merge.py
```

### Step 6 — convert + quantize to Q4_K_M GGUF (~30 min)

```bash
INPUT=/workspace/output/moe-v4-merged \
OUTPUT=/dev/shm/kyle-30b-a3b-v4-q4_k_m.gguf \
bash convert_and_quantize.sh
ls -lh /dev/shm/kyle-30b-a3b-v4-q4_k_m.gguf
```

Expected output size: ~17 GB.

## Pushing the result back to Kyle

Install rclone on the pod and configure with the same OAuth flow Kyle used (he can do the browser step on his phone):

```bash
curl -fsSL https://rclone.org/install.sh | sudo bash
rclone config
# n (new remote) → name: gdrive → drive → leave client_id/secret blank → scope 1 (full access)
# → service_account_file blank → advanced n → auto config: n
# → it prints a URL to paste into a browser → Kyle opens on phone, authorizes,
#   pastes back the verification code
```

Then upload:

```bash
rclone copy /dev/shm/kyle-30b-a3b-v4-q4_k_m.gguf \
  gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-v4/result/ \
  --progress
```

Verify upload:

```bash
rclone ls gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-v4/result/
```

Also push the training log + a `STATUS.md` so Kyle's work-PC session knows the run completed:

```bash
cp /workspace/train.log /tmp/moe-v4-train.log
cat > /tmp/STATUS.md <<EOF
# MoE v4 RunPod run — DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Pod: $(hostname)
- Final train_loss: $(grep -oP "train_loss': \K[\d.]+" /workspace/train.log | tail -1)
- Output GGUF: kyle-30b-a3b-v4-q4_k_m.gguf
- Output size: $(stat -c%s /dev/shm/kyle-30b-a3b-v4-q4_k_m.gguf 2>/dev/null) bytes
- SHA256: $(sha256sum /dev/shm/kyle-30b-a3b-v4-q4_k_m.gguf | cut -d' ' -f1)
EOF
rclone copy /tmp/moe-v4-train.log /tmp/STATUS.md \
  gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-v4/result/
```

## When everything is uploaded

Tell Kyle:
- "MoE v4 done — GGUF + train log + STATUS.md in `gdrive:skippy_files/personal-ai-assistant/docs/runs/moe-v4/result/`"
- The final train_loss number
- The SHA256 of the GGUF
- Remind him to **terminate the RunPod pod** to stop billing

## If something goes wrong

- **Smoke test fails** → don't proceed; copy the smoke_test stderr to Kyle, terminate pod
- **Training OOMs** → unlikely on H100 80 GB with attention-only LoRA, but if it does, check if any other processes are on the GPU; report to Kyle, do NOT switch to a different recipe without his go-ahead
- **GGUF convert fails** → the merged HF model is still at `/workspace/output/moe-v4-merged`; `rclone copy` that whole directory to `gdrive:.../result/merged-fallback/` so Kyle can finish the convert locally
- **Direct SSH to pod fails** for the phone (firewall or RunPod issues) → install Tailscale on the pod (`curl -fsSL https://tailscale.com/install.sh | sudo sh && sudo tailscale up --auth-key=<key Kyle pasted>`) and SSH via the pod's tailnet hostname instead

## Cost guardrail

- H100 PCIe Secure on RunPod: ~$2.79/hr → ~$15–18 for the full run
- Spot/community variants are cheaper if available: $1.50–2.00/hr
- DO NOT leave the pod running after the upload completes. Terminate it.
