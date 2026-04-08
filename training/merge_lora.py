#!/usr/bin/env python3
"""Merge LoRA adapters into base model for deployment."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

MODEL_PATH = "Qwen/Qwen2.5-14B-Instruct"
LORA_PATH = "training/output/final"
OUTPUT_PATH = "training/output/merged"

print("="*60)
print("Merging LoRA adapters into base model")
print("="*60)

print("\n📥 Loading base model (this needs ~30GB RAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",  # Load to CPU for merging
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("📥 Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cpu")

print("🔧 Merging weights...")
model = model.merge_and_unload()

print(f"💾 Saving merged model to {OUTPUT_PATH}...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n✅ Merged model saved!")
print(f"Size: {sum(f.stat().st_size for f in __import__('pathlib').Path(OUTPUT_PATH).rglob('*') if f.is_file()) / 1e9:.1f} GB")
print("\nNext step: Convert to GGUF format")
