# Cloud Training Runbook — MoE LoRA on a Rented H100

Step-by-step guide to training a MoE LoRA adapter (Qwen 3 30B-A3B or Mixtral 8x7B) on a rented H100, converting it to GGUF, and deploying it locally.

Target audience: you (Kyle). You've never rented cloud GPU before, but you're comfortable with SSH, bash, and `run.sh`. No Kubernetes, no Terraform — just a web console, one `ssh` session, and a couple of `scp` commands.

**Rough timing:**
- One-time setup: ~30 min
- Per training run: ~3 h wall clock (15 min hands-on, 2.5 h hands-off training)
- Cost per run: ~$6 on an H100 80 GB

**Privacy posture:** dedicated pod on RunPod Secure Cloud. Your training data (emails, writing samples) is uploaded to a pod you control, weights-in-VRAM during the ~3-hour run, then destroyed when you terminate the pod. Good enough for personal writing style; not good enough for classified data.

---

## Part 1 — One-time account setup (30 min)

### 1.1 Create a RunPod account
1. Go to <https://runpod.io>
2. Sign up with email (or GitHub/Google SSO)
3. Verify email

### 1.2 Add credit
1. Billing → Add Balance
2. Add $20 (covers ~3 training runs + small storage buffer)
3. Credit card gets you started; crypto is available if you prefer

### 1.3 Generate an API key (for scripting later, optional now)
1. Settings → API Keys → Create
2. Name: `skippy-training`
3. Read-write permissions
4. Copy the key, save it to your password manager — it's shown once

### 1.4 Add your SSH public key
1. Settings → SSH Public Keys
2. Paste the contents of `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`)
3. If you don't have one: `ssh-keygen -t ed25519 -C "kyle@skippy"` then upload `~/.ssh/id_ed25519.pub`

### 1.5 Install runpodctl (optional but useful)
```bash
curl -sL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 -o runpodctl
chmod +x runpodctl && sudo mv runpodctl /usr/local/bin/
runpodctl config --apiKey <your-api-key>
```

You won't use runpodctl for the first run (web UI is easier to learn on), but it's there for later automation.

---

## Part 2 — Prepare training data locally (10 min)

### 2.1 Pull the latest training data
```bash
cd ~/Documents/GitHub/personal-ai-framework
source training/venv/bin/activate    # if you have a training venv
python3 training/collect_training_data.py
```

This should output something like `training/data/train_alpaca.json` with your ~10K conversation turns.

### 2.2 Sanity-check
```bash
wc -l training/data/train_alpaca.json
jq length training/data/train_alpaca.json   # if jq installed
head -1 training/data/train_alpaca.json | jq .
```

Look for ≥ 1000 examples. Fewer than that and MoE LoRA won't differentiate enough experts to produce a noticeable flavor change. If you're short, add more conversations in Skippy before training.

### 2.3 Pack for upload
```bash
cd ~/Documents/GitHub/personal-ai-framework
tar czf /tmp/skippy-training-$(date +%Y%m%d).tar.gz \
    training/data/train_alpaca.json \
    training/train_lora.py \
    pipeline/config.yaml
ls -lh /tmp/skippy-training-*.tar.gz
```

Tarball will be ~5–20 MB. You'll scp this onto the pod in Part 4.

---

## Part 3 — Launch a pod (5 min)

### 3.1 Go to deploy
Open <https://runpod.io/console/deploy>

### 3.2 Choose cloud tier
Click **Secure Cloud** (top tab). Community Cloud is cheaper but shared hosts — not what you want for personal data.

### 3.3 Pick a GPU
Filter: **H100 80GB PCIe**. Look for ~$2.39/hr on-demand. If none available, **A100 80GB** at ~$1.89/hr works too (slower by ~30% but adequate for 3B-active MoE).

Click **Deploy** on the cheapest available region.

### 3.4 Configure the pod

| Field | Value |
|---|---|
| **Template** | `RunPod Pytorch 2.4 (CUDA 12.4)` — pre-built PyTorch image |
| **Container Disk** | 50 GB |
| **Volume Disk** | 150 GB (persistent; survives pod stop) |
| **Volume Mount Path** | `/workspace` |
| **Expose HTTP ports** | (leave default — 8888 for Jupyter) |
| **Expose TCP ports** | (leave default — 22 for SSH) |

150 GB sounds like overkill but: base Qwen 3 30B-A3B weights (~60 GB fp16) + LoRA checkpoints (~5 GB) + GGUF output (~16 GB Q4) = ~80 GB, plus slack.

Click **Deploy On-Demand**. Pod provisions in ~60 seconds.

### 3.5 Copy the SSH command
Once the pod is running, go to **My Pods** → click the pod → find the **Connect** button. Copy the **SSH over exposed TCP** command. It looks like:

```bash
ssh root@123.456.78.9 -p 22145 -i ~/.ssh/id_ed25519
```

Open a terminal on your machine and paste. Answer `yes` to the host key prompt.

---

## Part 4 — Set up the training environment (10 min)

You are now SSH'd into the pod. Everything below runs ON the pod.

### 4.1 Sanity checks
```bash
nvidia-smi      # expect H100 80GB with 0 MiB used
python3 --version   # expect 3.10+
df -h /workspace    # expect ~150 GB free
```

### 4.2 Clone llama.cpp (for GGUF conversion later)
```bash
cd /workspace
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt
cd /workspace
```

### 4.3 Install training deps
```bash
pip install --upgrade transformers peft bitsandbytes accelerate datasets safetensors
```

Takes ~2 min. Expect lots of "already installed" for torch.

### 4.4 Upload your training tarball
On your LOCAL machine (new terminal):
```bash
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
    /tmp/skippy-training-*.tar.gz \
    root@<pod-ip>:/workspace/
```

Back on the pod:
```bash
cd /workspace
tar xzf skippy-training-*.tar.gz
ls training/data/  # should show train_alpaca.json
```

### 4.5 Download the base model
Pick one:

**Qwen 3 30B-A3B** (the MoE you've been evaluating):
```bash
cd /workspace
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct \
    --local-dir /workspace/models/qwen3-30b-a3b \
    --local-dir-use-symlinks False
```

~60 GB download, 10–15 min on the pod's network.

**Mixtral 8x7B Instruct** (alternative):
```bash
huggingface-cli download mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --local-dir /workspace/models/mixtral-8x7b \
    --local-dir-use-symlinks False
```

~93 GB download, 20+ min. Bigger, so prefer Qwen 3 for your first run.

Some models are gated — you'll need to `huggingface-cli login` with an HF token first if so.

---

## Part 5 — Run the training (2.5 h, hands-off)

### 5.1 Create the training script

Your existing `training/train_lora.py` is dense-model-oriented. For MoE, write a new entry point on the pod:

```bash
cat > /workspace/train_moe_lora.py << 'PYEOF'
"""
MoE LoRA training — QLoRA style, Qwen 3 MoE or Mixtral.
Adapted from HF PEFT docs + Mixtral QLoRA recipes.
"""
import os, json, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

MODEL = os.environ.get("MODEL_PATH", "/workspace/models/qwen3-30b-a3b")
DATA  = os.environ.get("TRAIN_DATA", "/workspace/training/data/train_alpaca.json")
OUT   = os.environ.get("OUTPUT_DIR", "/workspace/training/output/moe-lora-v1")

# 4-bit base (QLoRA) — makes 30B MoE fit in 80 GB with room for activations
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# LoRA targets: attention + FFN expert weights. For Qwen3/Mixtral these are
# named things like 'q_proj', 'k_proj', 'v_proj', 'o_proj',
# 'w1', 'w2', 'w3' (Mixtral) or 'gate_proj', 'up_proj', 'down_proj' (Qwen).
# We target attention on all layers + FFN on all layers → PEFT figures out
# which modules match.
lora = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj",       # Qwen names
                    "w1","w2","w3"],                          # Mixtral names
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

raw = json.load(open(DATA))
def fmt(ex):
    return {"text": f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{ex['output']}<|im_end|>"}
ds = Dataset.from_list([fmt(x) for x in raw])

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,      # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10, save_steps=200,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=2,
)
trainer = SFTTrainer(
    model=model, train_dataset=ds, tokenizer=tok,
    args=args, max_seq_length=2048,
    dataset_text_field="text",
)
trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"Done. Adapter saved to {OUT}")
PYEOF
```

### 5.2 Kick it off
```bash
cd /workspace
nohup python3 train_moe_lora.py > training.log 2>&1 &
echo $! > train.pid
tail -f training.log
```

- Epoch 1 log lines should start appearing within 2 min
- `nvidia-smi` in another SSH window should show ~55–70 GB VRAM used
- Expect ~2.5 h total wall clock for 3 epochs on 10K examples with H100

Press **Ctrl-C** to stop tailing (training keeps running in background thanks to nohup).

### 5.3 Reconnect if your SSH drops
Training runs in `nohup`, so even if your laptop sleeps or your Wi-Fi dies, the pod keeps training. Just SSH back in and `tail -f /workspace/training.log`.

### 5.4 Watch for trouble

| Symptom | Fix |
|---|---|
| OOM in epoch 1 | Drop `per_device_train_batch_size` to… it's already 1. Reduce `max_seq_length` to 1024. |
| Loss doesn't drop | Check your `train_alpaca.json` — if all entries are nearly identical, LoRA has nothing to learn. |
| Loss goes NaN | `learning_rate` too high — retry with `1e-4`. |
| GPU util < 50% | Normal for small batches on H100. Throughput is IO-bound at this scale. |
| `CUDA out of memory at decoder layer N` | Peak VRAM is fine but activations are spiking — enable `gradient_checkpointing_enable()` (already in script). |

---

## Part 6 — Convert adapter to GGUF (15 min)

The training run outputs a PEFT adapter (~5 GB at rank 64). You need to:
1. Merge adapter into base weights (full model, fp16)
2. Quantize to Q4_K_M and emit GGUF

### 6.1 Merge adapter
```bash
cat > /workspace/merge.py << 'PYEOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/workspace/models/qwen3-30b-a3b"
ADAPTER = "/workspace/training/output/moe-lora-v1"
OUT = "/workspace/models/qwen3-30b-a3b-kyle-merged"

base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
peft = PeftModel.from_pretrained(base, ADAPTER)
merged = peft.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print(f"Merged model at {OUT}")
PYEOF

python3 /workspace/merge.py
```

Takes ~10 min (CPU-bound). Output: ~60 GB of fp16 sharded safetensors.

### 6.2 Convert to GGUF
```bash
cd /workspace/llama.cpp
python3 convert_hf_to_gguf.py \
    /workspace/models/qwen3-30b-a3b-kyle-merged \
    --outfile /workspace/kyle-30b-a3b-v1-f16.gguf \
    --outtype f16
```

~5 min. Output: ~60 GB f16 GGUF.

### 6.3 Quantize to Q4_K_M
```bash
# Build llama.cpp quantizer
cd /workspace/llama.cpp
make quantize -j   # ~2 min

./quantize /workspace/kyle-30b-a3b-v1-f16.gguf \
           /workspace/kyle-30b-a3b-v1-q4_k_m.gguf \
           Q4_K_M
```

~5 min. Output: ~16 GB Q4_K_M GGUF. **This is the file you ship.**

---

## Part 7 — Bring the GGUF home (5–15 min)

On your LOCAL machine:
```bash
cd ~/Documents/GitHub/personal-ai-framework
mkdir -p models/qwen3-30b-a3b-kyle
scp -P <pod-port> -i ~/.ssh/id_ed25519 \
    root@<pod-ip>:/workspace/kyle-30b-a3b-v1-q4_k_m.gguf \
    models/qwen3-30b-a3b-kyle/
ls -lh models/qwen3-30b-a3b-kyle/
```

16 GB at ~100 Mbps = ~20 min. RunPod outbound is usually faster (~300 Mbps), so more like 7 min.

---

## Part 8 — Deploy locally (5 min)

### 8.1 Point `config.yaml` at the new model
Edit `pipeline/config.yaml`:
```yaml
model:
  path: /app/models/qwen3-30b-a3b-kyle/kyle-30b-a3b-v1-q4_k_m.gguf
  context_length: 16384
```

### 8.2 Restart the container
```bash
./run.sh stop && ./run.sh start
```

### 8.3 Smoke test
1. Open <http://localhost:8765>
2. Log in
3. Ask a question that matches your training distribution ("Draft an email about X")
4. Compare response to the previous dense model — you should see flavor differences, not capability differences

---

## Part 9 — Tear down the pod (1 min)

**Critical — this is where you stop the billing.**

1. Go to <https://runpod.io/console/pods>
2. Click the pod → **Stop**
3. Then **Terminate** (confirms the delete)

Alternatively, if you want to iterate (train again with new data):
- **Stop** the pod instead of terminate — volume persists at $0.10/GB/month (~$15/month for 150 GB)
- Next run, **Resume** the pod — skips the base-model download (huge time win)
- Terminate for real when you're done experimenting

---

## Part 10 — What to look for in the result

First session with the new MoE model, watch for:

**Good signs:**
- Response style matches your writing more closely (phrasing, sentence length, tone)
- Tool use still works (the detection pass, agent loop, email composing)
- Throughput is comparable or better than 14B dense — MoE with 3B active should feel snappier

**Bad signs (report back):**
- Gibberish or garbled output — GGUF conversion issue; re-quantize or check llama.cpp version
- Much slower than dense 14B — MoE routing overhead, or quantization is bad
- Refuses tool calls — fine-tune may have overfit on non-tool prose; reduce epochs to 2 next time
- Worse answers across the board — too few training examples, or LoRA rank too low

---

## Cheat sheet — one-page summary

```
LOCAL                              CLOUD POD (H100 80GB)
─────                              ────────────────────
1. collect_training_data.py  ────► 2. tar → scp to /workspace/
                                   3. clone llama.cpp
                                   4. pip install peft bnb accelerate
                                   5. huggingface-cli download <model>
                                   6. nohup train_moe_lora.py &
                                     (~2.5 h, hands-off)
                                   7. merge + convert + quantize
                                     (~25 min)
8. scp GGUF home  ◄────────────── (16 GB)
9. config.yaml model path updated
10. ./run.sh restart
11. smoke test
12. terminate pod (STOP THE BILLING)
```

**Cost target:** $6 per run if you stay under 3 hours. Check the RunPod billing page when you terminate — you should see ~$6 deducted.

**Keep this runbook up to date** when we discover gotchas. The first run will find things I missed here.
