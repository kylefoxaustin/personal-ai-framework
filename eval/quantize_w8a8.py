#!/usr/bin/env python3
"""Produce a W8A8 INT8 quantization of a HuggingFace fp16 model.

Uses llm-compressor's reference recipe:
  - SmoothQuant (strength=0.7) — migrates activation outliers into weights
  - GPTQ (W8A8) — calibrated 8-bit weight + 8-bit activation quantization

Output is compressed-tensors format, loadable via transformers +
compressed-tensors library. Runs activation int8 matmul on Tensor
cores when available (Ampere+, including Blackwell).

Usage:
  python3 eval/quantize_w8a8.py \\
    --source models/qwen2.5-14b-hf \\
    --output models/qwen2.5-14b-w8a8 \\
    --calib-samples 512

Calibration default (512 samples of ultrachat) is tuned to give stable
quantization scales without over-fitting to any single corpus. For
Skippy-specific calibration, pass --calib-dataset to a custom dataset
of conversations.
"""
import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Path to fp16 HF model dir (or HF repo ID)")
    parser.add_argument("--output", required=True,
                        help="Output directory for W8A8 model")
    parser.add_argument("--calib-samples", type=int, default=512,
                        help="Number of calibration samples (default 512)")
    parser.add_argument("--calib-dataset", default="HuggingFaceH4/ultrachat_200k",
                        help="HF dataset for calibration (default ultrachat)")
    parser.add_argument("--calib-split", default="train_sft",
                        help="Dataset split to use for calibration")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length during calibration")
    parser.add_argument("--smoothing", type=float, default=0.7,
                        help="SmoothQuant strength (0.0-1.0, default 0.7)")
    parser.add_argument("--scheme", default="W8A8",
                        choices=["W8A8", "FP8_DYNAMIC", "FP8", "W4A16", "W8A16"],
                        help="Quantization scheme (W8A8 int8; FP8_DYNAMIC for Blackwell-native FP8)")
    parser.add_argument("--skip-smoothquant", action="store_true",
                        help="Skip SmoothQuant (recommended for FP8 — its dynamic range handles outliers natively)")
    args = parser.parse_args()

    print(f"▶ Loading fp16 source model from {args.source}...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.source,
        torch_dtype=torch.bfloat16,  # avoid fp16 overflow during stats
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    print(f"  loaded in {time.time()-t0:.1f}s")

    print(f"▶ Building calibration dataset ({args.calib_samples} samples from "
          f"{args.calib_dataset})...")
    from datasets import load_dataset
    ds = load_dataset(args.calib_dataset, split=args.calib_split)
    ds = ds.shuffle(seed=42).select(range(args.calib_samples))

    # ultrachat gives us 'messages' in chat format — apply the model's chat
    # template so calibration sees realistic role-wrapped input.
    def preprocess(example):
        if "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )
        elif "text" in example:
            text = example["text"]
        else:
            raise KeyError(f"unknown example shape: {list(example.keys())}")
        return tokenizer(text, truncation=True, max_length=args.max_seq_len,
                         padding=False, return_tensors=None)

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    print(f"  prepared {len(ds)} calibration samples")

    print(f"▶ Building recipe (scheme={args.scheme}, "
          f"smoothquant={'on' if not args.skip_smoothquant else 'off'})...")
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    recipe = []
    if not args.skip_smoothquant:
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        recipe.append(SmoothQuantModifier(smoothing_strength=args.smoothing))
    # For FP8 schemes, QuantizationModifier (PTQ) is the standard path;
    # GPTQ is overkill since FP8 dynamic range already handles outliers.
    # For INT8 (W8A8), GPTQ improves quality over bare PTQ.
    if args.scheme in ("FP8_DYNAMIC", "FP8"):
        recipe.append(QuantizationModifier(
            targets="Linear",
            scheme=args.scheme,
            ignore=["lm_head"],
        ))
    else:
        recipe.append(GPTQModifier(
            targets="Linear",
            scheme=args.scheme,
            ignore=["lm_head"],
        ))

    print(f"▶ Running quantization (this uses GPU + RAM; expect 30-90 min for 14B)...")
    t0 = time.time()
    from llmcompressor import oneshot
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,
        output_dir=args.output,
    )
    print(f"  quantized in {time.time()-t0:.1f}s")

    print(f"\n✅ W8A8 model saved to {args.output}")
    # Show the size
    import os
    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(args.output) for f in files)
    print(f"   total size on disk: {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
