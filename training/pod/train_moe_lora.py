"""
MoE LoRA training for Qwen3-30B-A3B (or similar MoE bases) with QLoRA.

Tested configuration (2026-04-17):
  - Base: Qwen/Qwen3-30B-A3B-Instruct-2507 (30B total, 3B active, 128 experts)
  - Hardware: single H100 80GB (RunPod Secure Cloud)
  - Deps: torch 2.6.0, transformers 5.5.4, trl 1.1.0, peft 0.19.1, bnb 0.49.2
  - 6,417 alpaca-format training examples, 3 epochs, effective batch 16
  - Wall clock: ~5 h, final loss 0.67, train_loss avg 0.88, token_acc 68.9%

LoRA strategy: ATTENTION ONLY. Do NOT target expert FFN modules.
  Qwen3-30B-A3B has 48 layers × 128 experts × 3 FFN projections = 18,432
  expert linears. Targeting them with rank 64 would create ~6.6B trainable
  LoRA params — defeats the purpose of LoRA and OOMs an H100. Attention-only
  gives ~53M LoRA params (0.17% of model) and fits comfortably with ~19 GB VRAM
  headroom for activations + optimizer.

Usage:
  MODEL_PATH=/workspace/models/<base> \
  TRAIN_DATA=/workspace/training/data/train_alpaca.json \
  OUTPUT_DIR=/workspace/training/output/moe-lora-v1 \
  python3 train_moe_lora.py
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

MODEL = os.environ.get("MODEL_PATH", "/workspace/models/qwen3-30b-a3b-instruct-2507")
DATA = os.environ.get("TRAIN_DATA", "/workspace/training/data/train_alpaca.json")
OUT = os.environ.get("OUTPUT_DIR", "/workspace/training/output/moe-lora-v1")

print(f"=== loading base model (4-bit QLoRA) from {MODEL} ===")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)

# Manual light prep — prepare_model_for_kbit_training upcasts layernorms to
# fp32 and causes OOM on MoE bases because the "4-bit" load still leaves
# fused expert tensors in bf16 (see smoke_test.py comment).
for p in model.parameters():
    p.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("=== applying LoRA adapters (attention only) ===")
lora = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

print(f"=== loading + formatting dataset from {DATA} ===")
raw = json.load(open(DATA))
def fmt(ex):
    instr = ex["instruction"]
    if ex.get("input"):
        instr = instr + "\n\n" + ex["input"]
    return {"text": f"<|im_start|>user\n{instr}<|im_end|>\n"
                    f"<|im_start|>assistant\n{ex['output']}<|im_end|>"}
ds = Dataset.from_list([fmt(x) for x in raw])
print(f"  examples: {len(ds)}")
print(f"  first: {ds[0]['text'][:200]}...")

# trl 1.x API: SFTConfig replaces inline kwargs; `max_seq_length` → `max_length`;
# `tokenizer=` → `processing_class=`.
cfg = SFTConfig(
    output_dir=OUT,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    save_total_limit=2,
    max_length=2048,
    dataset_text_field="text",
    packing=False,
)
trainer = SFTTrainer(
    model=model, args=cfg, train_dataset=ds, processing_class=tok,
)

print("=== starting training ===")
trainer.train()
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"=== DONE. Adapter saved to {OUT} ===")
