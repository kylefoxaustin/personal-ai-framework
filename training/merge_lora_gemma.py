#!/usr/bin/env python3
"""Merge Gemma-2-9B v4 LoRA adapters into base model for GGUF conversion."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/kyle/Documents/GitHub/personal-ai-framework/models/gemma-2-9b-it-hf",
)
LORA_PATH = os.environ.get("LORA_PATH", "training/output/gemma-v4/final")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "models/gemma-2-9b-kyle")

print("=" * 60)
print("Merging Gemma-2-9B v4 LoRA adapters into base model")
print("=" * 60)
print(f"Base:   {MODEL_PATH}")
print(f"LoRA:   {LORA_PATH}")
print(f"Output: {OUTPUT_PATH}")

print("\n📥 Loading base model to CPU (bf16, ~18 GB RAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

print("📎 Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cpu")

print("🔀 Merging and unloading...")
model = model.merge_and_unload()

os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"💾 Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

print("💾 Saving tokenizer (stock template — Skippy pipeline uses GGUF's embedded template)...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tok.save_pretrained(OUTPUT_PATH)

print("✅ Merge complete. Next: convert to GGUF with llama.cpp convert_hf_to_gguf.py")
