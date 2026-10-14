"""
v5 "clean" LoRA: Skippy's first fine-tune with NO personal email (the email-trained v1-v4 line was
archived 2026-10-13 to ~/skippy-archive/2026-10-13-email-brain after the models were found
reproducing Kyle's real signature block, phone number included).

Recipe = v4 (r=64/alpha=128 attention+FFN, 2 epochs, lr 2e-4 cosine, eff. batch 16,
paged_adamw_8bit, assistant-only loss, bf16 on the 5090), with two changes:
- Data: training/data/clean_v5 (build_grounded_dataset.py): grounded datasheet answers,
  explicit "the excerpts don't cover that" abstentions, general-knowledge answers, refusals.
- Format: full system + user + assistant messages using Skippy's REAL inference prompt (persona +
  RAG directive + retrieved excerpts). v4 never saw a system turn or context in training, which is
  consistent with it overriding good context with memorised specs.
max_length 6144 (v4: 2048): the excerpts make prompts median ~3.6k tokens, p95 ~4.8k. Rows over it are
DROPPED, not truncated: truncation cuts the assistant turn off the end, the only part that carries loss.

    python3 training/train_lora_v5_clean.py     # needs a HARD GPU lease and llm-server stopped
"""
import json, os
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "training/data/clean_v5")
OUTPUT_DIR = os.environ.get("TRAINING_OUTPUT_DIR", "training/output/v5-clean")
MAX_LEN = 6144


def load_messages(path, tok):
    rows = json.load(open(path))
    lens = [len(tok.apply_chat_template(r["messages"], tokenize=True)) for r in rows]
    s = sorted(lens)
    keep = [r for r, n in zip(rows, lens) if n <= MAX_LEN]
    print(f"Loaded {len(rows)} examples from {path}; token lengths median {s[len(s) // 2]}, p95 {s[int(len(s) * .95)]}, "
          f"max {s[-1]}; dropped {len(rows) - len(keep)} over {MAX_LEN}")
    return Dataset.from_list([{"messages": r["messages"]} for r in keep])


def main():
    print(f"v5-clean LoRA — start {datetime.now()}  model={MODEL_PATH}  data={DATA_DIR}")
    assert torch.cuda.is_available(), "no GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_ds = load_messages(f"{DATA_DIR}/train_messages.json", tok)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_steps=50, save_total_limit=2, eval_strategy="no", bf16=True,
        optim="paged_adamw_8bit", report_to="none", max_length=MAX_LEN, packing=False,
        assistant_only_loss=True,
    )
    SFTTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tok).train()

    final = Path(OUTPUT_DIR) / "final"
    model.save_pretrained(str(final)); tok.save_pretrained(str(final))
    print(f"done {datetime.now()} — adapters at {final}")


if __name__ == "__main__":
    main()
