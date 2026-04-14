#!/usr/bin/env python3
"""
LoRA Fine-tuning for Personal AI
Trains style adapters on your writing samples.
"""
import os
import sys
import json
import torch
import psutil
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

LOG_DIR = Path(os.environ.get("TRAINING_LOG_DIR", "training/logs"))


class _Tee:
    """Duplicate writes to multiple streams (e.g. stdout + a log file).

    Used so training output survives even if the controlling terminal
    is killed mid-run (e.g. by systemd-oomd taking out the vte-spawn scope).
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _setup_logging() -> Path:
    """Tee stdout and stderr to a timestamped log file under training/logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_path = LOG_DIR / f"train_{ts}.log"
    log_file = open(log_path, "a", buffering=1)  # line-buffered text mode
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


class MemoryLoggingCallback(TrainerCallback):
    """Log host RSS + available RAM + GPU memory every N steps.

    Leaves a forensic trail so future oomd kills are easy to post-mortem.
    """

    def __init__(self, every_n_steps: int = 50):
        self.every_n_steps = every_n_steps
        self._proc = psutil.Process(os.getpid())

    def _snapshot(self, label: str):
        rss_gb = self._proc.memory_info().rss / (1024 ** 3)
        vm = psutil.virtual_memory()
        gpu = ""
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu = f" | gpu_alloc={gpu_alloc:.2f}GB reserved={gpu_reserved:.2f}GB"
        print(
            f"[mem] {label} rss={rss_gb:.2f}GB "
            f"host_avail={vm.available / (1024 ** 3):.1f}GB "
            f"pct_used={vm.percent:.1f}%{gpu}",
            flush=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._snapshot("train_begin")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % self.every_n_steps == 0:
            self._snapshot(f"step={state.global_step}")

    def on_epoch_end(self, args, state, control, **kwargs):
        self._snapshot(f"epoch_end={state.epoch}")

    def on_train_end(self, args, state, control, **kwargs):
        self._snapshot("train_end")


# Configuration (overridable via environment variables)
MODEL_PATH = "Qwen/Qwen2.5-14B-Instruct"  # HF model for tokenizer/config
GGUF_PATH = "models/qwen2.5-14b/qwen2.5-14b-instruct-q4_k_m.gguf"
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output")
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data")

# LoRA Config
LORA_R = int(os.environ.get("TRAINING_LORA_RANK", "64"))
LORA_ALPHA = LORA_R * 2     # Scaling factor (2x rank)
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Config
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = int(os.environ.get("TRAINING_EPOCHS", "3"))
LEARNING_RATE = 2e-4
MAX_LENGTH = 512
WARMUP_RATIO = 0.03
RESUME_CHECKPOINT = os.environ.get("TRAINING_RESUME_CHECKPOINT", None)


def load_training_data(data_path: str, tokenizer, max_samples: int = None):
    """Load and tokenize training data."""
    print(f"Loading data from {data_path}")
    
    with open(data_path) as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} examples")
    
    # Format as ChatML (Qwen 2.5 format)
    formatted = []
    for item in data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')

        text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
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
    log_path = _setup_logging()
    print("=" * 60)
    print("LoRA Fine-tuning for Personal AI")
    print("=" * 60)
    print(f"Start time: {datetime.now()}")
    print(f"Log file:   {log_path}")
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
        bnb_4bit_compute_dtype=torch.bfloat16,
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
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Reduce activation memory. use_reentrant=False is the modern path and
    # plays nicely with LoRA + PEFT.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

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
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        group_by_length=True,
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
        callbacks=[MemoryLoggingCallback(every_n_steps=50)],
    )
    
    # Train!
    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    print("=" * 60)
    print(f"This will take several hours. Progress will be shown below.")
    print()
    
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"📂 Resuming from checkpoint: {RESUME_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
    else:
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
