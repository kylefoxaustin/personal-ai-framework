"""
v4 LoRA training applied to Gemma-2-9B-IT — cross-family validation (SK-P2-003).

Forked from train_lora_llama_v4.py 2026-05-08. Third non-Qwen family in the
gotcha-#7 validation arc. Tests whether the Skippy v4 recipe (SFTTrainer +
assistant_only_loss + 100 refusal exemplars + 2 epochs) transfers across the
Qwen→Gemma architecture-family boundary.

Key notes vs Llama script:
  - Base model: google/gemma-2-9b-it (local HF snapshot, 9B params)
  - LoRA targets: same layer names as Qwen/Llama/Mistral — Gemma-2 uses
    q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/down_proj
  - Chat template: Gemma-2-it uses a role-renaming approach — sets role='model'
    for assistant turns then renders all roles via a single line. trl ships
    gemma_training_chat_template with the required {% generation %} markers.
    Assign directly (same bypass as Llama — avoids get_training_chat_template's
    exact-string match failing on the local tokenizer's template).
  - pad_token: Gemma tokenizer uses eos_token as pad by default in some versions;
    set explicitly to be safe.
  - No template patch needed — unlike Mistral, no {% generation %} surgery
    required. This makes Gemma as clean a data point as Llama for gotcha #7.

Hardware target (5090 32GB):
  - 9B bf16 ≈ 18 GB + activations (~2 GB) + LoRA adapter (~200M, ~1.5 GB)
  - Should fit on 32GB without QLoRA
  - Wall clock estimate: ~60-80 min

Usage:
  python3 training/train_lora_gemma_v4.py
"""
import os
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.chat_template_utils import gemma_training_chat_template


MODEL_PATH = "/home/kyle/Documents/GitHub/personal-ai-framework/models/gemma-2-9b-it-hf"
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output/gemma-v4")
LOG_DIR = Path(os.environ.get("TRAINING_LOG_DIR", "training/logs"))


def load_dataset(path: str) -> Dataset:
    print(f"Loading data from {path}")
    raw = json.load(open(path))
    print(f"Loaded {len(raw)} examples")

    def to_messages(ex):
        instr = ex["instruction"]
        if ex.get("input"):
            instr = instr + "\n\n" + ex["input"]
        return {"messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": ex["output"]},
        ]}

    return Dataset.from_list([to_messages(x) for x in raw])


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_gemma_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"

    print("=" * 60)
    print("Cross-family v4 LoRA: Gemma-2-9B-IT (SK-P2-003)")
    print("=" * 60)
    print(f"Start: {datetime.now()}")
    print(f"Log:   {log_path}")
    print(f"Model: {MODEL_PATH}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("❌ NO GPU AVAILABLE")
        return

    print("\n📥 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print(f"   pad_token set to eos_token: {tok.eos_token}")

    print("\n🩹 Applying trl gemma_training_chat_template directly...")
    tok.chat_template = gemma_training_chat_template
    print("   Done.")

    print("\n📥 Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("\n🔧 Setting up LoRA (attention + FFN)...")
    lora = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print("\n📚 Loading training data...")
    train_ds = load_dataset(f"{DATA_DIR}/train_alpaca.json")
    print(f"Training examples: {len(train_ds)}")

    print("\n🏃 Configuring SFTTrainer with assistant_only_loss=True (v4 recipe)...")
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="no",
        bf16=True,
        optim="adamw_torch_fused",
        report_to="none",
        max_length=2048,
        packing=False,
        assistant_only_loss=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )

    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    print("=" * 60)
    trainer.train()

    print("\n💾 Saving LoRA adapters...")
    final_path = Path(OUTPUT_DIR) / "final"
    model.save_pretrained(str(final_path))
    tok.save_pretrained(str(final_path))
    print(f"✅ Done. Adapters at: {final_path}")


if __name__ == "__main__":
    main()
