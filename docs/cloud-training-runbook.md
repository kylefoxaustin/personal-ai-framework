# Cloud Training Runbook — MoE LoRA on a Rented H100

Step-by-step guide to training a MoE LoRA adapter (Qwen3-30B-A3B or similar), converting it to GGUF Q4_K_M, and deploying it locally.

This runbook was first written on 2026-04-15 and rewritten on 2026-04-17 after a full end-to-end training run exposed a dozen gotchas. Every `⚠️` is a real trap we hit on the first run — read them before copy-pasting.

**Target audience:** you (Kyle). You've done this once now. You're comfortable with SSH, bash, and `run.sh`. No Kubernetes, no Terraform — just a web console, one `ssh` session, one `scp` at the end.

**Rough timing** (30B MoE, 6K training examples, 3 epochs):

| Phase | Wall clock | Cost on H100 ($2.39/hr) |
|---|---|---|
| Account + pod setup | 10 min | — |
| Data prep (local) | 5 min | — |
| Pod bootstrap + model download | 15 min | $0.60 |
| Smoke test | 1 min | $0.04 |
| Training (3 epochs) | 5 h | $12 |
| Merge + convert + quantize | 35 min | $1.40 |
| scp home | 8 min | $0.30 |
| **Total** | **~6.5 h** | **~$15** |

Earlier versions of this runbook estimated ~$6 / 2.5h; reality on a 30B MoE is 2-3× that. A $20 credit is enough for one run plus a small safety margin.

**Privacy posture:** dedicated pod on RunPod Secure Cloud. Your training data (emails, writing samples) is uploaded to a pod you control, weights-in-VRAM during the run, destroyed when you terminate the pod. Good enough for personal writing style; not good enough for classified data.

---

## Part 0 — The scripts live in the repo

Reusable scripts are checked in at `training/pod/`:

| Script | What it does | Runs on |
|---|---|---|
| `bootstrap.sh` | Upgrades torch + installs all training deps + clones llama.cpp + pip installs cmake | pod |
| `fetch_model.sh <hf-repo>` | Downloads base weights via the new `hf` CLI | pod |
| `smoke_test.py` | Loads base + attaches LoRA + does forward+backward on mini batch | pod |
| `train_moe_lora.py` | The actual training loop, tuned for Qwen3 MoE | pod |
| `merge.py` | Merges LoRA adapter back into base weights (CPU, bf16) | pod |
| `convert_and_quantize.sh` | HF safetensors → GGUF f16 → Q4_K_M, routed through `/dev/shm` | pod |

The body of this runbook walks through when to call each.

---

## Part 1 — One-time account setup (10 min)

### 1.1 Create a RunPod account
1. Go to <https://runpod.io>
2. Sign up with email (or GitHub/Google SSO)
3. Verify email

### 1.2 Add credit
1. Billing → Add Balance → add $20 (covers one 30B MoE run with safety margin)
2. Credit card, or crypto if you prefer

### 1.3 Add your SSH public key
1. Settings → SSH Public Keys
2. Paste `~/.ssh/id_ed25519.pub` (or generate with `ssh-keygen -t ed25519 -C "kyle@skippy"` if you don't have one)

### 1.4 (Optional) Generate an API key for later automation
Not needed for your first run — web UI is faster to learn on.

⚠️ **`runpodctl` is not required.** Some tutorials push it, but the web console + SSH is enough. `runpodctl` saves a few clicks for repeat runs.

---

## Part 2 — Prepare training data locally (5 min)

### 2.1 Regenerate the alpaca JSON
```bash
cd ~/Documents/GitHub/personal-ai-framework
python3 training/collect_training_data.py
```

Output: `training/data/train_alpaca.json` (~20 MB for 6K conversations).

Run this each time before a new training run — it picks up conversations you've had with Skippy since last time.

### 2.2 Sanity check
```bash
wc -l training/data/train_alpaca.json
python3 -c "import json; d=json.load(open('training/data/train_alpaca.json')); print(f'examples: {len(d)}')"
```

⚠️ Need **≥ 1000 examples** for LoRA to produce a noticeable style shift. Below that, the signal is too weak.

### 2.3 Pack for upload
```bash
tar czf /tmp/skippy-training-$(date +%Y%m%d).tar.gz \
    training/data/train_alpaca.json \
    training/pod/train_moe_lora.py \
    pipeline/config.yaml
ls -lh /tmp/skippy-training-*.tar.gz
```
Tarball is typically 3-20 MB.

---

## Part 3 — Launch a pod (5 min)

### 3.1 Go to deploy
<https://runpod.io/console/deploy>

### 3.2 Choose cloud tier + GPU
- **Secure Cloud** (top tab). Community Cloud is cheaper but shared hosts — bad for personal data.
- Filter: **H100 80GB PCIe** (~$2.39/hr). A100 80GB (~$1.89/hr) works but is ~30% slower.

### 3.3 Configure the pod

| Field | Value |
|---|---|
| **Template** | `RunPod Pytorch 2.4 (CUDA 12.4)` |
| **Container Disk** | 50 GB |
| **Volume Disk** | 150 GB |
| **Volume Mount Path** | `/workspace` |
| **Expose HTTP ports** | default (8888 for Jupyter) |
| **Expose TCP ports** | default (22 for SSH) |

Click **Deploy On-Demand**. Pod boots in ~60 seconds.

### 3.4 Copy the SSH command
In **My Pods** → click the pod → **Connect** → copy the **SSH over exposed TCP** command. Looks like:
```
ssh root@64.247.201.49 -p 16439 -i ~/.ssh/id_ed25519
```

⚠️ Use the **SSH over exposed TCP** version, not the proxied `ssh.runpod.io` one. The proxy doesn't support scp/sftp, you'll need that later.

### 3.5 Verify SSH from your current network
```bash
ssh -o ConnectTimeout=10 -p <port> -i ~/.ssh/id_ed25519 root@<ip> 'echo OK; nvidia-smi --query-gpu=name --format=csv,noheader'
```

⚠️ **Corporate/work wifi often drops outbound high ports silently** — SSH will stall on banner exchange. If this hangs, switch to home network, phone hotspot, or Tailscale. (Kyle learned this the hard way: a 45-minute debug was pure firewall.)

---

## Part 4 — Bootstrap the pod (10 min)

All commands below run on the pod unless prefixed with `LOCAL:`.

### 4.1 scp the training tarball + pod scripts
```bash
# LOCAL:
cd ~/Documents/GitHub/personal-ai-framework
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
    /tmp/skippy-training-*.tar.gz \
    training/pod/*.sh training/pod/*.py \
    root@<pod-ip>:/workspace/
```

### 4.2 On the pod: extract + run bootstrap
```bash
cd /workspace
tar --no-same-owner -xzf skippy-training-*.tar.gz
bash bootstrap.sh
```

⚠️ **`tar --no-same-owner`** — without it, `tar` fails with "Cannot change ownership" on network storage.

What `bootstrap.sh` does (see the script for the reasoning in comments):
1. **Upgrades torch to 2.6+** (the RunPod 2.4.0 image ships torch 2.4.1; transformers 5.x calls `torch.nn.Module.set_submodule()` which only exists in torch ≥2.5).
2. Installs `transformers peft bitsandbytes accelerate datasets safetensors trl huggingface_hub`.
3. **Installs `cmake` via pip** (apt on the minimal pod image doesn't have a cmake package).
4. Clones llama.cpp.
5. Verifies everything imports cleanly.

⚠️ `huggingface_hub` 1.11+ **renamed `huggingface-cli` → `hf`**. Old scripts that use `huggingface-cli download` will fail with a usage-hint dump. Use `hf download` or the `fetch_model.sh` helper.

### 4.3 Download the base model
```bash
bash fetch_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507
```

⚠️ The correct repo is **`Qwen/Qwen3-30B-A3B-Instruct-2507`** — the earlier runbook said `Qwen/Qwen3-30B-A3B-Instruct` which 404s. Instruct variants are date-suffixed. Alternatives on HF: `-Thinking-2507`, `-Base`, `-GPTQ-Int4`, `-FP8`.

~57 GB download, 5-15 min on RunPod's network.

---

## Part 5 — Smoke test (1 min, $0.04 insurance)

Before committing to a multi-hour training run, validate that the config actually fits:

```bash
MODEL_PATH=/workspace/models/qwen3-30b-a3b-instruct-2507 python3 /workspace/smoke_test.py
```

Expected output (if things are fine):
```
loaded in ~18s, vram: ~60 GB
Linear class distribution: Counter({'Linear4bit': 192, 'Linear': 1})
trainable params: ~53M || all params: ~30.5B || trainable%: 0.17
PEAK VRAM: ~61 GB
=== SMOKE OK ===
```

⚠️ **Linear4bit count is 192, not 18,000+** — that's intentional. bitsandbytes can't quantize MoE fused-expert tensors (they aren't `nn.Linear` modules). Experts stay in bf16 taking ~55 GB, attention gets quantized. **Training works anyway** because we only LoRA attention; frozen experts don't need gradient memory.

⚠️ **If `prepare_model_for_kbit_training` is ever added back, it OOMs** — it tries to upcast layernorms to fp32, which on a 30B MoE with 60 GB already allocated just can't fit. The training script and smoke test use manual light prep instead (freeze + gradient checkpointing + enable_input_require_grads).

If smoke passes, proceed. If it errors, **stop and debug** — don't launch the full run hoping it'll work.

---

## Part 6 — Run the training (~5 h, hands-off, ~$12)

```bash
# Extract training data into expected location
cd /workspace
mkdir -p training/data
mv training/data/train_alpaca.json /workspace/training/data/   # if not already there

# Launch in nohup so SSH drops don't kill it
cd /workspace
(nohup python3 -u /workspace/train_moe_lora.py > /workspace/training.log 2>&1 &)

# Confirm it started
pgrep -af train_moe_lora
tail -f /workspace/training.log    # Ctrl-C to stop tailing; training continues
```

### 6.1 What to watch for

- **First loss line (`logging_steps=10` = every ~2.5 min)** should show loss around 1.3-1.7 and a non-NaN grad_norm.
- Loss should trend down to the 0.7-0.9 range by epoch 3.
- VRAM stable ~63 GB, GPU temp 30-40°C on an H100.
- ~14-15 s/step, 1200 total steps.

### 6.2 If SSH drops
Training is in `nohup` — it keeps running. Just SSH back in and `tail -f /workspace/training.log`.

### 6.3 Trouble-shooting

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM at step 1 | You re-added `prepare_model_for_kbit_training` or targeted expert FFN | Use the training script as-is (attention-only LoRA, manual prep) |
| Loss NaN | LR too high | Retry with `learning_rate=1e-4` |
| Loss flat around 1.5 for whole run | Training data too homogeneous | Add variety; LoRA has nothing to learn |
| `CUDA OOM at decoder layer N` | Activation spike, seq too long | Drop `max_length` to 1024 in `SFTConfig` |
| GPU util < 50% | Normal for batch=1 on H100 — throughput is IO-bound at this scale | Leave it |

---

## Part 7 — Merge + convert + quantize (~35 min)

The pipeline is a single script you invoke in two parts (merge first, then convert+quantize). Split this way because merge is CPU-heavy (~20 min) and convert/quantize has its own output-path gotcha.

### 7.1 Merge adapter into base
```bash
BASE_MODEL=/workspace/models/qwen3-30b-a3b-instruct-2507 \
ADAPTER=/workspace/training/output/moe-lora-v1 \
OUT=/workspace/models/qwen3-30b-a3b-kyle-merged \
python3 /workspace/merge.py
```
~20 min. Output is ~57 GB of bf16 sharded safetensors at `$OUT`.

### 7.2 Convert + quantize (routed through `/dev/shm`)
```bash
MERGED=/workspace/models/qwen3-30b-a3b-kyle-merged \
OUT_NAME=kyle-30b-a3b-v1 \
bash /workspace/convert_and_quantize.sh
```
~15 min total: 5 min f16 convert + 3 min cmake configure+build + 3 min quantize + buffer.

⚠️ **`/workspace` has a ~25 GB per-file size limit** on the RunPod MooseFS cluster. A 60+ GB f16 GGUF silently truncates there (the convert script appears to complete but the file stops growing at exactly 25 GB). The script routes through `/dev/shm` (tmpfs, 117 GB RAM-backed) to avoid this. **Final Q4_K_M is ~16 GB** so it'd fit on `/workspace`, but staying on `/dev/shm` means one less copy.

⚠️ **Don't use pipes like `convert... | tail -25` in a `set -e` bash script** — `set -e` doesn't propagate errors through pipelines by default, so a crashed convert looks like success. This was the root cause of a 15-minute silent-failure debug loop on the first run. `convert_and_quantize.sh` uses direct redirection instead.

---

## Part 8 — Bring the GGUF home (~8 min)

```bash
# LOCAL:
mkdir -p ~/Documents/GitHub/personal-ai-framework/models/qwen3-30b-a3b-kyle
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
    root@<pod-ip>:/dev/shm/kyle-30b-a3b-v1-q4_k_m.gguf \
    ~/Documents/GitHub/personal-ai-framework/models/qwen3-30b-a3b-kyle/
```
~16 GB at RunPod's ~300 Mbps egress = ~7-10 min.

---

## Part 9 — Deploy locally (5 min)

### 9.1 Update `pipeline/config.yaml`
```yaml
model:
  path: "/app/models/qwen3-30b-a3b-kyle/kyle-30b-a3b-v1-q4_k_m.gguf"
  context_length: 32768
  max_tokens: 512
  temperature: 0.7
  gpu_layers: -1
```

### 9.2 Clear any stale `active_model_path` in `~/.personal-ai/users/<user>/settings.json`
```bash
python3 -c "
import json
p = '/home/kyle/.personal-ai/users/kyle/settings.json'
d = json.load(open(p))
d.setdefault('model', {})['active_model_path'] = '/app/models/qwen3-30b-a3b-kyle/kyle-30b-a3b-v1-q4_k_m.gguf'
json.dump(d, open(p, 'w'), indent=2)
"
```
⚠️ `llm_server.py` persists the last-used model path per-user; this overrides `config.yaml` on startup. If you skip this step you'll see "Loading model from qwen2.5-14b/..." even though config points elsewhere.

### 9.3 Restart
```bash
cd ~/Documents/GitHub/personal-ai-framework
./run.sh stop && ./run.sh start
```

⚠️ **Rebuild the Docker image if any requirement was added since your last build:**
```bash
docker compose build llm-server   # ~5-10 min for the llama-cpp-python CUDA rebuild
```

### 9.4 Verify
```bash
docker logs personal-ai-framework-llm-server-1 2>&1 | grep -E "arch\s+=|general.name" | tail -2
```
Should show `arch = qwen3moe` and your merged name. If it shows `qwen2` you missed step 9.2.

---

## Part 10 — Tear down the pod (1 min — critical!)

**This is where the billing stops.**

1. <https://runpod.io/console/pods>
2. Click pod → **Stop** → **Terminate**

If you want to iterate with new training data quickly:
- **Stop** (not Terminate) keeps the volume at $0.10/GB/month (~$15/month for 150 GB)
- Resume preserves the 60 GB base model download (saves 10-15 min next time)
- **Terminate** when you're actually done experimenting

---

## Appendix A — Pod package list (what bootstrap.sh installs)

If you want to build your own bootstrap, this is the minimum set for the 30B MoE QLoRA pipeline:

```
# Python packages (pip)
torch>=2.6 torchvision torchaudio   # via pytorch cu124 index
transformers>=5.5
peft>=0.19
bitsandbytes>=0.49
accelerate>=1.13
datasets>=4.8
safetensors>=0.7
trl>=1.1
huggingface_hub>=1.11
cmake>=4.0

# System (already in RunPod PyTorch 2.4 template)
git
build-essential
python3.11
```

---

## Appendix B — Gotchas index (quick lookup)

Every ⚠️ in this doc, in one place:

1. **SSH stalls on banner exchange** → work/corporate wifi firewall; use home/hotspot/Tailscale
2. **`huggingface-cli` no longer exists** → `hf download` (huggingface_hub ≥1.11)
3. **Model repo is date-suffixed** → `Qwen/Qwen3-30B-A3B-Instruct-2507`, not `-Instruct`
4. **Torch 2.4 lacks `set_submodule`** → upgrade to torch ≥2.5 (2.6+ best)
5. **Torchvision version mismatch after torch upgrade** → upgrade torchvision+torchaudio same index
6. **`apt install cmake` fails on this image** → `pip install cmake`
7. **bnb doesn't quantize MoE fused experts** → expected; attention-only LoRA still works
8. **`prepare_model_for_kbit_training` OOMs on MoE** → use manual prep (freeze + gradient checkpointing)
9. **Targeting expert FFN with LoRA creates ~6.6B trainable params** → target attention only
10. **`trl 1.x` moved `max_seq_length` → `max_length`, `tokenizer` → `processing_class`** → use SFTConfig
11. **`/workspace` has ~25 GB per-file limit** → stage big GGUFs on `/dev/shm`
12. **`set -e` doesn't propagate through pipes** → don't pipe tool output to `tail` inside critical scripts
13. **`pgrep -f` self-matches** → use `kill -0 <pid>` or a specific basename for monitor loops
14. **`tar` fails on ownership** → `tar --no-same-owner -xzf ...`
15. **Config path gets overridden by per-user settings** → update `~/.personal-ai/users/<user>/settings.json` on deploy
16. **Docker image caches Python deps** → `docker compose build` after editing `requirements.txt`
