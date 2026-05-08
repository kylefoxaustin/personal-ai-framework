"""
v4 LoRA training applied to Llama-3.1-8B-Instruct — cross-family validation.

Forked from train_lora_mistral_v4.py 2026-05-08. Tests whether the Skippy v4
recipe (SFTTrainer + assistant_only_loss + 100 refusal exemplars + 2 epochs)
transfers across the Qwen→Llama architecture-family boundary.

Key adjustments from train_lora_mistral_v4.py:
  - Base model path: meta-llama/Llama-3.1-8B-Instruct (local HF snapshot)
  - LoRA targets: same layer names as Qwen2.5/Mistral (q_proj/k_proj/v_proj/
    o_proj + gate_proj/up_proj/down_proj — Llama 3.1 uses the same naming)
  - Llama 3.1 chat template uses a generic role loop rather than role-specific
    elif branches. The assistant content is rendered inline:
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'
            + message['content'] | trim + '<|eot_id|>' }}
    We split this into a role-conditional that adds {% generation %} markers
    only around the assistant content block.
  - pad_token is None on Llama tokenizer — set to eos_token before training.

Hardware target (5090 32GB):
  - 8B bf16 ≈ 16 GB + activations + LoRA adapter (~150M trainable, ~1 GB)
  - Should fit without QLoRA, same as Qwen2.5 7B v4
  - Wall clock estimate: ~50-65 min

Usage:
  python3 training/train_lora_llama_v4.py
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
from trl.chat_template_utils import llama3_training_chat_template


MODEL_PATH = "/home/kyle/Documents/GitHub/personal-ai-framework/models/llama-3.1-8b-instruct-hf"
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output/llama-v4")
LOG_DIR = Path(os.environ.get("TRAINING_LOG_DIR", "training/logs"))


def patch_chat_template_for_assistant_only_loss(tok):
    """Inject {% generation %} markers around the assistant content block.

    Llama 3.1 renders all non-tool roles via a single generic line:
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'
            + message['content'] | trim + '<|eot_id|>' }}

    trl's assistant_only_loss=True needs {% generation %}/{% endgeneration %}
    around just the assistant content. We replace the generic line with a
    role-conditional that adds markers only for the assistant turn.
    Idempotent — re-running on a patched template prints 'already patched'."""
    template = tok.chat_template
    if template is None:
        raise ValueError("no chat_template on tokenizer")
    if "{% generation %}" in template:
        print("=== chat_template already has {% generation %} markers ===")
        return tok

    # Target: the generic role rendering line in the Llama 3.1 message loop.
    # Use regex to be robust to minor whitespace variation.
    generic_re = re.compile(
        r"(\{\{-?\s*'<\|start_header_id\|>'\s*\+\s*message\['role'\]\s*\+"
        r"\s*'<\|end_header_id\|>\\n\\n'\s*\+\s*message\['content'\]\s*\|\s*trim"
        r"\s*\+\s*'<\|eot_id\|>'\s*\}\})",
        re.MULTILINE,
    )
    m = generic_re.search(template)
    if m:
        # Build the role-split replacement.
        # Preserve the indentation of the matched line.
        line_start = template.rfind("\n", 0, m.start()) + 1
        indent = ""
        for ch in template[line_start:]:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        replacement = (
            "{%- if message.role == 'assistant' %}\n"
            + indent + "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{%- generation %}{{- message['content'] | trim }}{%- endgeneration %}"
            "{{- '<|eot_id|>' }}\n"
            + indent[:-4] + "{%- else %}\n"
            + indent + "{{- '<|start_header_id|>' + message['role'] + "
            "'<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n"
            + indent[:-4] + "{%- endif %}"
        )
        template_p = template[: m.start()] + replacement + template[m.end():]
        tok.chat_template = template_p
        gen_count = template_p.count("generation %}")
        print(f"=== Llama chat_template patched (generic-loop strategy): "
              f"generation markers={gen_count // 2} pair(s) ===")
        return tok

    # Fallback: Mistral-style elif branch (shouldn't be needed for Llama 3.1)
    opener_re = re.compile(
        r'(\{%-?\s*elif\s+message\[?[\'"]?role[\'"]?\]?\s*==\s*[\'"]assistant[\'"]\s*%\}\s*\n)',
        re.MULTILINE,
    )
    m = opener_re.search(template)
    if not m:
        opener_re2 = re.compile(
            r'(\{%-?\s*if\s+message\[?[\'"]?role[\'"]?\]?\s*==\s*[\'"]assistant[\'"]\s*%\}\s*\n)',
            re.MULTILINE,
        )
        m = opener_re2.search(template)
    if not m:
        print("=== template structure not recognized — printing first 800 chars ===")
        print(template[:800])
        raise ValueError("could not find assistant block in chat_template")

    template_p = template[: m.end()] + "        {% generation %}\n" + template[m.end():]
    closer_re = re.compile(r'(\s*\{%-?\s*el(if|se)\b)', re.MULTILINE)
    search_start = m.end() + len("        {% generation %}\n")
    m2 = closer_re.search(template_p, pos=search_start)
    if not m2:
        endif_re = re.compile(r'(\s*\{%-?\s*endif\s*%\})', re.MULTILINE)
        m2 = endif_re.search(template_p, pos=search_start)
    if not m2:
        raise ValueError("could not find closing point for assistant block")
    template_p = (
        template_p[: m2.start()]
        + "\n        {% endgeneration %}"
        + template_p[m2.start():]
    )
    tok.chat_template = template_p
    print(f"=== chat_template patched (elif strategy): "
          f"gen={template_p.count('{% generation %}')} "
          f"endgen={template_p.count('{% endgeneration %}')} ===")
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
    log_path = LOG_DIR / f"train_llama_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"

    print("=" * 60)
    print("Cross-family v4 LoRA: Llama-3.1-8B-Instruct")
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

    # trl's get_training_chat_template does an exact-string match against known templates.
    # Our tokenizer's template may differ slightly from trl's stored copy. Bypass the
    # auto-detection by assigning the training template directly — it was built for
    # Llama 3.x and includes the required {% generation %} markers.
    print("\n🩹 Applying trl llama3_training_chat_template directly...")
    tok.chat_template = llama3_training_chat_template
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
