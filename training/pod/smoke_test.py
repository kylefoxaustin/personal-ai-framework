"""
Pre-flight memory feasibility check before committing to a multi-hour training run.
Loads base in 4-bit, attaches attention-only LoRA, runs forward+backward on a
tiny batch, prints peak VRAM. Should take ~30s on an H100 for a 30B MoE.

Cheap insurance: if this OOMs or errors, you know BEFORE burning 2+ hours of pod time.

Usage: MODEL_PATH=/workspace/models/xxx python3 smoke_test.py
"""
import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("MODEL_PATH", "/workspace/models/qwen3-30b-a3b-instruct-2507")
t0 = time.time()

print(f"=== loading tokenizer from {MODEL} ===")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  vocab: {tok.vocab_size}, pad={tok.pad_token}")

print("=== loading 4-bit base model (bnb nf4, bf16 compute) ===")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
print(f"  loaded in {time.time()-t0:.1f}s, vram: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Note on MoE + bnb: bitsandbytes replaces nn.Linear with Linear4bit, but MoE
# expert FFNs are usually stored as a single fused tensor per layer (not as
# individual nn.Linear modules). Those won't be quantized — they stay bf16.
# For Qwen3-30B-A3B this leaves ~60 GB loaded instead of ~15 GB. Still OK on
# an H100 80 GB because we LoRA only attention (frozen experts don't need
# gradient memory).
from collections import Counter
cls_dist = Counter(mod.__class__.__name__ for _, mod in m.named_modules()
                   if "inear" in mod.__class__.__name__)
print(f"  Linear module class distribution: {cls_dist}")

print("=== manual lightweight prep (skip prepare_model_for_kbit_training — upcasts norms to fp32 and OOMs on MoE) ===")
for p in m.parameters():
    p.requires_grad_(False)
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.enable_input_require_grads()

print("=== applying attention-only LoRA (MoE would balloon trainable params if FFN targeted) ===")
lora = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
m = get_peft_model(m, lora)
m.print_trainable_parameters()
print(f"  after LoRA vram: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("=== forward + backward on a mini batch ===")
inp = tok(
    "<|im_start|>user\nwrite a short email.<|im_end|>\n"
    "<|im_start|>assistant\nSure! Here is a draft.<|im_end|>",
    return_tensors="pt", truncation=True, max_length=128,
).to(m.device)
inp["labels"] = inp["input_ids"].clone()
m.train()
out = m(**inp)
print(f"  loss: {out.loss.item():.3f}")
out.loss.backward()
print(f"  backward OK, PEAK VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print(f"=== SMOKE OK in {time.time()-t0:.1f}s — safe to launch full training ===")
