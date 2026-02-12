#!/usr/bin/env python3
"""
LoRA Fine-tuning for Personal AI
Trains style adapters on your writing samples.
"""
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Configuration
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.2"  # HF model for tokenizer/config
GGUF_PATH = "models/mixtral/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
OUTPUT_DIR = "training/output"
DATA_DIR = "training/data"

# LoRA Config
LORA_R = 64          # Rank - higher = more capacity but more memory
LORA_ALPHA = 128     # Scaling factor
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Config  
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
WARMUP_RATIO = 0.03


def load_training_data(data_path: str, tokenizer, max_samples: int = None):
    """Load and tokenize training data."""
    print(f"Loading data from {data_path}")
    
    with open(data_path) as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} examples")
    
    # Format as instruction-following
    formatted = []
    for item in data:
        # Alpaca format: instruction + input + output
        instruction = item.get('instruction', '')
        inp = item.get('input', '')
        output = item.get('output', '')
        
        if inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        formatted.append({"text": text})
    
    dataset = Dataset.from_list(formatted)
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"].copy()})
    
    return tokenized


def main():
    print("=" * 60)
    print("LoRA Fine-tuning for Personal AI")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Training requires GPU.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Quantization config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print("📥 Loading model (this takes a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    print("🔧 Setting up LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("\n📚 Loading training data...")
    train_data = load_training_data(f"{DATA_DIR}/train_alpaca.json", tokenizer)
    val_data = load_training_data(f"{DATA_DIR}/val_alpaca.json", tokenizer, max_samples=500)
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    print("=" * 60)
    print(f"This will take several hours. Progress will be shown below.")
    print()
    
    trainer.train()
    
    # Save final model
    print("\n💾 Saving LoRA adapters...")
    final_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n✅ Training complete!")
    print(f"LoRA adapters saved to: {final_path}")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
