#!/usr/bin/env python3
"""Quick test of the trained LoRA adapters."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "training/output/final"

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("\n" + "="*60)
print("🧠 YOUR PERSONALIZED AI IS READY!")
print("="*60 + "\n")

# Test prompts
prompts = [
    "Write a professional email about: Project status update",
    "Write a professional email about: Running late to meeting",
    "Help me write about embedded systems development",
]

for prompt in prompts:
    print(f"📝 Prompt: {prompt}\n")
    
    inputs = tokenizer(f"### Instruction:\n{prompt}\n\n### Response:\n", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    
    print(f"🤖 Response:\n{response}\n")
    print("-"*60 + "\n")
