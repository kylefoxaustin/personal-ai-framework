#!/usr/bin/env python3
"""Merge LoRA adapters into Mistral-7B-Instruct-v0.3 base for deployment."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

MODEL_PATH = "/home/kyle/Documents/GitHub/personal-ai-framework/models/mistral-7b-instruct-v0.3-hf"
LORA_PATH = "training/output/mistral-v4/final"
OUTPUT_PATH = "training/output/mistral-v4/merged"

print("="*60)
print("Merging Mistral v4 LoRA adapters into base model")
print("="*60)

print("\n📥 Loading base model (this needs ~30GB RAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("📥 Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_PATH, device_map="cpu")

print("🔧 Merging weights...")
model = model.merge_and_unload()

print(f"💾 Saving merged model to {OUTPUT_PATH}...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="4GB")
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n✅ Merged model saved!")
print("Next: Convert to GGUF via llama.cpp's convert_hf_to_gguf.py")
