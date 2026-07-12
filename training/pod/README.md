# RunPod training scripts

These scripts run **on the pod**, not locally. See `docs/cloud-training-runbook.md` for the full procedure (pod setup → training → deploy).

| Script | Purpose |
|---|---|
| `bootstrap.sh` | Fresh-pod setup (torch upgrade, training deps, cmake, llama.cpp) |
| `fetch_model.sh <hf-repo>` | Download base weights via the `hf` CLI |
| `smoke_test.py` | Pre-flight feasibility check (MoE base — attention-only LoRA) |
| `smoke_test_dense.py` | Pre-flight feasibility check (dense base — attention+FFN LoRA) |
| `train_moe_lora.py` | QLoRA training loop for MoE bases (attention-only) |
| `train_moe_lora_router.py` | QLoRA training loop for MoE bases (attention + router) — recipe-taxonomy Tier 2 hypothesis test |
| `train_moe_lora_full.py` | QLoRA training loop for MoE bases (attention + router + expert FFNs at low rank) — Tier 2.x extension test |
| `train_dense_lora.py` | QLoRA training loop for dense bases (attention+FFN) — **OLD recipe (3 epochs, full-sequence loss)**, kept for reference |
| `train_dense_lora_v4.py` | QLoRA training loop for dense bases — **v4 recipe (2 epochs + assistant_only_loss + messages format)**, use this for new dense runs |
| `merge.py` | Merge trained adapter back into MoE base weights |
| `merge_dense.py` | Merge trained adapter back into dense base weights |
| `convert_and_quantize.sh` | HF safetensors → GGUF f16 → Q4_K_M, via `/dev/shm` (model-agnostic) |

Run `bash bootstrap.sh` on a fresh pod before anything else.

**Pick the right pair for your base model:**
- MoE bases (Qwen3-30B-A3B-class): `smoke_test.py` + `train_moe_lora.py` + `merge.py`
- Dense bases (Qwen 2.5 32B-class): `smoke_test_dense.py` + `train_dense_lora.py` + `merge_dense.py`

The dense and MoE recipes differ in two ways: (1) dense targets attention AND FFN
LoRA modules; MoE only attention because expert FFN is 128× duplicated and would
balloon trainable params. (2) dense uses `prepare_model_for_kbit_training`; MoE
uses manual prep because the standard prep upcasts layernorms to fp32 and OOMs
on MoE due to fused-expert tensors not being quantized.
