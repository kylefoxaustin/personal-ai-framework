"""
v4 LoRA training applied to Phi-4 — N=7 cross-family validation.

Forked from train_lora_yi_v4.py 2026-05-10. Phi-4 is the reviewer-recommended
N=7 candidate from the cross-judge cycle (Microsoft/Phi family, distinct from
anything in N=6; recent enough for NXP-credibility on 'modern small models';
16K context — close to Yi's 128K, controls for context-saturation confound).
Stock baseline measured 2026-05-10: 90/126 = 71.4%, reasoning 3/6 (intermediate
band, same as Yi).

Per the reviewer-blessed two-factor model, Phi-4 should regress (cross-family,
not ceiling-reasoning). If it regresses → two-factor model corroborated at N=3
cross-family. If it lifts → two-factor breaks, "Yi-specific quirk" framing
returns.

Template note: Phi-4 uses ChatML-with-<|im_sep|> (variant of standard ChatML —
'<|im_start|>role<|im_sep|>content<|im_end|>' instead of
'<|im_start|>role\\ncontent<|im_end|>'). trl's phi3_training_chat_template uses
the OLD Phi-3 format (<|system|>, <|user|>, <|end|>) — incompatible with Phi-4.
We patch Phi-4's own template by wrapping the assistant content block with
{% generation %} / {% endgeneration %} markers. Same approach as the Llama
script's regex patching, but Phi-4's template structure is simpler.

LoRA targets: same naming as Qwen/Llama/Mistral/Gemma/Yi — q_proj/k_proj/
v_proj/o_proj + gate_proj/up_proj/down_proj. Phi-4 uses Phi3ForCausalLM
architecture which has these standard module names.

Hardware target (5090 32GB) — QLoRA (nf4 base):
  - 14B bf16 ≈ 28 GB doesn't fit on 5090 with activations + adapter + grad buffers
    (first attempt 2026-05-10 crashed with MmBackward0 meta-device error from
     accelerate's auto-offload). QLoRA nf4 brings base down to ~7 GB.
  - Following the same approach as train_lora_14b_v4.py (Qwen 14B v4 precedent
    on this hardware).
  - Wall clock estimate: ~80-120 min based on Qwen 14B v4

Usage:
  python3 training/train_lora_phi4_v4.py
"""
import os
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


MODEL_PATH = "/home/kyle/Documents/GitHub/personal-ai-framework/models/phi-4-hf"
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output/phi4-v4")
LOG_DIR = Path(os.environ.get("TRAINING_LOG_DIR", "training/logs"))


def patch_phi4_template_for_assistant_only_loss(tok):
    """Wrap Phi-4's assistant content block with {% generation %} markers.

    Phi-4 template (single-line, no whitespace):
        ...{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}...

    Patched:
        ...{% elif (message['role'] == 'assistant') %}{{'<|im_start|>assistant<|im_sep|>'}}{% generation %}{{ message['content'] }}{% endgeneration %}{{'<|im_end|>'}}...

    Same rendered output, but the assistant content is marked as the loss target
    for trl's assistant_only_loss=True. Idempotent — re-running on a patched
    template prints 'already patched'.
    """
    template = tok.chat_template
    if template is None:
        raise ValueError("no chat_template on tokenizer")
    if "{% generation %}" in template:
        print("=== chat_template already has {% generation %} markers ===")
        return tok

    original_assistant = (
        "{% elif (message['role'] == 'assistant') %}"
        "{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}"
    )
    patched_assistant = (
        "{% elif (message['role'] == 'assistant') %}"
        "{{'<|im_start|>assistant<|im_sep|>'}}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{{'<|im_end|>'}}"
    )

    if original_assistant not in template:
        print("=== template structure not recognized — printing first 600 chars ===")
        print(template[:600])
        raise ValueError("could not find assistant block in Phi-4 chat_template")

    tok.chat_template = template.replace(original_assistant, patched_assistant)
    gen_count = tok.chat_template.count("{% generation %}")
    end_count = tok.chat_template.count("{% endgeneration %}")
    print(f"=== Phi-4 chat_template patched: generation={gen_count} "
          f"endgeneration={end_count} ===")
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
    log_path = LOG_DIR / f"train_phi4_v4_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"

    print("=" * 60)
    print("Cross-family v4 LoRA: Phi-4 (N=7)")
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
    print(f"   eos_token: {tok.eos_token!r} (id={tok.eos_token_id})")
    print(f"   pad_token: {tok.pad_token!r}")

    print("\n🩹 Patching Phi-4 chat_template for assistant_only_loss...")
    tok = patch_phi4_template_for_assistant_only_loss(tok)

    print("\n📥 Loading model in nf4 (QLoRA)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
