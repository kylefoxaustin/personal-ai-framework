"""
v4 LoRA training applied to Yi-1.5-9B-Chat — N=6 cross-family validation.

Forked from train_lora_llama_v4.py 2026-05-10. Yi-1.5-9B-Chat is the
reviewer-recommended N=6 candidate from the cross-judge cycle (Phi-3-mini-4k
disqualified for 4K context-window saturation; Yi has solid stock baseline at
68.3% post-regrade with intermediate reasoning floor 3/6).

Yi-1.5-9B-Chat uses ChatML format (`<|im_start|>role\\n...<|im_end|>`). The
trl-shipped qwen2_5_training_chat_template is designed for the same ChatML
format and includes the {% generation %} markers needed for
assistant_only_loss=True. We assign it directly — same approach as Llama
which uses llama3_training_chat_template, and Gemma which uses
gemma_training_chat_template.

Tokenizer notes vs Qwen2.5:
  - Yi vocab size 64000 (vs Qwen2.5's 152064 — different vocabulary, but
    the ChatML control tokens <|im_start|>/<|im_end|> are present)
  - <|im_end|> id 7 == eos_token (Yi)
  - <|im_start|> id 6
  - pad_token already set to <unk> by tokenizer (no override needed)

LoRA targets: same naming as Qwen/Llama/Mistral/Gemma — q_proj/k_proj/v_proj/
o_proj + gate_proj/up_proj/down_proj.

Hardware target (5090 32GB):
  - 9B bf16 ≈ 18 GB + activations + LoRA adapter (~150M trainable, ~1 GB)
  - Should fit without QLoRA, same envelope as Gemma 2 9B v4
  - Wall clock estimate: ~60-80 min based on Llama 47.7 min / Gemma observed timing

Usage:
  python3 training/train_lora_yi_v4.py
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
from trl.chat_template_utils import qwen2_5_training_chat_template


MODEL_PATH = "/home/kyle/Documents/GitHub/personal-ai-framework/models/yi-1.5-9b-chat-hf"
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output/yi-v4")
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
    log_path = LOG_DIR / f"train_yi_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"

    print("=" * 60)
    print("Cross-family v4 LoRA: Yi-1.5-9B-Chat (N=6)")
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
    print(f"   eos_token: {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"   pad_token: {tok.pad_token!r}")

    # Yi-1.5-Chat uses ChatML (<|im_start|>/<|im_end|>) — same control tokens
    # as Qwen2.5. trl's qwen2_5_training_chat_template includes the
    # {% generation %} markers needed for assistant_only_loss=True. Assign
    # directly (same bypass pattern as Llama and Gemma scripts — avoids
    # get_training_chat_template's exact-string match failing on Yi's
    # in-house template variant).
    print("\n🩹 Applying trl qwen2_5_training_chat_template directly (ChatML-compatible)...")
    tok.chat_template = qwen2_5_training_chat_template
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
