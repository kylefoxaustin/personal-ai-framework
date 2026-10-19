#!/bin/bash
# Download the public base model a fresh install runs on: Qwen 2.5 7B Instruct, Q4_K_M GGUF (~4.7 GB).
# The fine-tuned Skippy model is NOT distributed (kept local by policy); this is the stock base it was
# trained from. setup.sh offers the same download plus a 14B option and updates pipeline/config.yaml.
set -e
cd "$(dirname "$0")/.."
mkdir -p models
FILE=models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
echo "Downloading Qwen 2.5 7B Instruct (Q4_K_M, ~4.7 GB) -> $FILE"
wget -c "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" -O "$FILE" --show-progress
echo "✅ Done. Set pipeline/config.yaml model.path to: /app/$FILE"
ls -lh "$FILE"
