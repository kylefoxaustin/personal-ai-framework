"""
v4 LoRA training applied to Mistral-7B-Instruct-v0.3 — cross-family validation.

Forked from train_lora_v4.py 2026-05-07. Tests whether the Skippy v4 recipe
(SFTTrainer + assistant_only_loss + 100 refusal exemplars + 2 epochs) transfers
across the Qwen→Mistral architecture-family boundary. The recipe is validated
on dense Qwen2.5 7B (+3.1pp) and 14B (+5.3pp); this run answers whether
non-Qwen bases can be expected to gain similarly.

Key adjustments from train_lora_v4.py:
  - Base model path: mistralai/Mistral-7B-Instruct-v0.3
  - LoRA targets: same naming convention as Qwen2.5 (q_proj/k_proj/v_proj/o_proj
    + gate_proj/up_proj/down_proj — Mistral uses the same Linear layer names)
  - Mistral chat template ships with [INST] markers; trl's assistant_only_loss=True
    needs `{% generation %}` markers in the chat template. If Mistral's template
    doesn't have them (likely), the training will abort at SFTTrainer init with the
    same error we saw on Qwen3-MoE. Fix: patch the tokenizer's chat_template before
    training (see patch_template logic at the bottom of this file).

Hardware target (5090 32GB):
  - 7B bf16 ≈ 14 GB + activations + LoRA adapter (~150M trainable, ~1 GB)
  - Should fit without QLoRA, same as Qwen2.5 7B v4
  - Wall clock estimate: ~50-60 min (similar to Qwen2.5 7B v4 at 46 min)

Usage:
  python3 training/train_lora_mistral_v4.py
"""
import os
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


MODEL_PATH = "/home/kyle/Documents/GitHub/personal-ai-framework/models/mistral-7b-instruct-v0.3-hf"
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output/mistral-v4")
LOG_DIR = Path(os.environ.get("TRAINING_LOG_DIR", "training/logs"))


def patch_chat_template_for_assistant_only_loss(tok):
    """Inject `{% generation %}` markers around the assistant block in the
    chat_template so trl's assistant_only_loss=True can identify the loss-mask region.
    Mistral's stock template doesn't ship with these markers (similar to Qwen3-MoE).
    Idempotent — re-running on a patched template prints 'already patched'."""
    template = tok.chat_template
    if template is None:
        raise ValueError("no chat_template on tokenizer")
    if "{% generation %}" in template:
        print("=== chat_template already has {% generation %} markers ===")
        return tok

    # Mistral-7B-Instruct-v0.3 uses [INST] ... [/INST] style. The assistant turn
    # is what comes between [/INST] and the next [INST]. We need to wrap that.
    # Common Mistral v0.3 template structure:
    #   {%- for message in messages %}
    #       {%- if message.role == "user" %}
    #           [INST] ... [/INST]
    #       {%- elif message.role == "assistant" %}
    #           ... + eos
    #       {%- endif %}
    #   {%- endfor %}

    opener_re = re.compile(
        r'(\{%-?\s*elif\s+message\[?[\'"]?role[\'"]?\]?\s*==\s*[\'"]assistant[\'"]\s*%\}\s*\n)',
        re.MULTILINE,
    )
    m = opener_re.search(template)
    if not m:
        # Maybe the template uses a different conditional structure
        # (e.g. "if message.role == 'assistant'" without elif)
        opener_re2 = re.compile(
            r'(\{%-?\s*if\s+message\[?[\'"]?role[\'"]?\]?\s*==\s*[\'"]assistant[\'"]\s*%\}\s*\n)',
            re.MULTILINE,
        )
        m = opener_re2.search(template)
    if not m:
        print("=== template structure not recognized — printing first 800 chars ===")
        print(template[:800])
        raise ValueError("could not find assistant elif/if in chat_template")

    template_p = (
        template[: m.end()]
        + "        {% generation %}\n"
        + template[m.end() :]
    )

    # Find the closing point — the next elif (e.g. 'tool' branch) OR the end of
    # the assistant block (endif followed by endfor or another role check).
    closer_re = re.compile(
        r'(\s*\{%-?\s*el(if|se)\b)',
        re.MULTILINE,
    )
    m2 = None
    # Search starting AFTER the assistant opener we just patched
    search_start = m.end() + len("        {% generation %}\n")
    m2 = closer_re.search(template_p, pos=search_start)
    if not m2:
        # Fall back to inserting before the next {%- endif %}
        endif_re = re.compile(r'(\s*\{%-?\s*endif\s*%\})', re.MULTILINE)
        m2 = endif_re.search(template_p, pos=search_start)
    if not m2:
        raise ValueError("could not find closing point for assistant block")

    template_p = (
        template_p[: m2.start()]
        + "\n        {% endgeneration %}"
        + template_p[m2.start() :]
    )

    tok.chat_template = template_p
    print(f"=== chat_template patched: gen={template_p.count('{% generation %}')} endgen={template_p.count('{% endgeneration %}')} ===")
    return tok


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
    log_path = LOG_DIR / f"train_mistral_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"

    print("=" * 60)
    print("Cross-family v4 LoRA: Mistral-7B-Instruct-v0.3")
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

    print("\n🩹 Patching chat_template for assistant_only_loss compatibility...")
    tok = patch_chat_template_for_assistant_only_loss(tok)

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
        optim="adamw_torch_fused",  # was paged_adamw_8bit; bnb cu130 libs not present locally
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
