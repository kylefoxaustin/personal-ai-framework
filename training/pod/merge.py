"""
Merge a trained LoRA adapter back into the base model weights (CPU, bf16).
Produces a standalone safetensors directory that can be passed to
convert_hf_to_gguf.py.

Wall clock: ~20 min for a 30B MoE (CPU-bound merge + 17 shards × 15-30s each).

Usage:
  BASE_MODEL=/workspace/models/<base> \
  ADAPTER=/workspace/training/output/moe-lora-v1 \
  OUT=/workspace/models/<base>-merged \
  python3 merge.py
"""
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.environ.get("BASE_MODEL", "/workspace/models/qwen3-30b-a3b-instruct-2507")
ADAPTER = os.environ.get("ADAPTER", "/workspace/training/output/moe-lora-v1")
OUT = os.environ.get("OUT", BASE.rstrip("/") + "-merged")

t0 = time.time()
print(f"=== loading base (cpu, bf16) from {BASE} ===")
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print(f"  loaded in {time.time()-t0:.0f}s")

print(f"=== loading adapter from {ADAPTER} ===")
peft = PeftModel.from_pretrained(base, ADAPTER)
print(f"  loaded at {time.time()-t0:.0f}s")

print("=== merging adapter into base ===")
merged = peft.merge_and_unload()
print(f"  merged at {time.time()-t0:.0f}s")

print(f"=== saving merged → {OUT} ===")
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print(f"=== DONE in {time.time()-t0:.0f}s. Merged at {OUT} ===")
