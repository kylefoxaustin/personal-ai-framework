# RunPod training scripts

These scripts run **on the pod**, not locally. See `docs/cloud-training-runbook.md` for the full procedure (pod setup → training → deploy).

| Script | Purpose |
|---|---|
| `bootstrap.sh` | Fresh-pod setup (torch upgrade, training deps, cmake, llama.cpp) |
| `fetch_model.sh <hf-repo>` | Download base weights via the `hf` CLI |
| `smoke_test.py` | Pre-flight feasibility check before training |
| `train_moe_lora.py` | QLoRA training loop for MoE bases (attention-only) |
| `merge.py` | Merge trained adapter back into base weights |
| `convert_and_quantize.sh` | HF safetensors → GGUF f16 → Q4_K_M, via `/dev/shm` |

Run `bash bootstrap.sh` on a fresh pod before anything else.
