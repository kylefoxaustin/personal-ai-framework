#!/bin/bash
# Fresh-pod bootstrap for RunPod PyTorch 2.4.0 template.
# Run once on a new pod before any training/merge/quantize work.
#
# Idempotent — safe to re-run. Exits non-zero on any failure.
set -euo pipefail

echo "=== [1/5] Upgrading torch to 2.6+ (transformers 5.x requires set_submodule) ==="
# The default RunPod PyTorch 2.4.0 image ships torch 2.4.1, which lacks
# torch.nn.Module.set_submodule() that transformers 5.x calls during bnb 4-bit
# quantization. Upgrade to 2.6+ for CUDA 12.4 and matching torchvision/audio.
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== [2/5] Installing training deps (transformers, peft, bnb, accelerate, datasets, trl) ==="
pip install --upgrade \
  transformers \
  peft \
  bitsandbytes \
  accelerate \
  datasets \
  safetensors \
  trl \
  huggingface_hub

echo "=== [3/5] Installing cmake via pip (apt cmake is not in this minimal image) ==="
pip install cmake

echo "=== [4/5] Cloning llama.cpp (for GGUF conversion + quantization) ==="
if [ ! -d /workspace/llama.cpp ]; then
  git clone https://github.com/ggerganov/llama.cpp /workspace/llama.cpp
fi
cd /workspace/llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt

echo "=== [5/5] Verifying GPU + tool availability ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, set_submodule: {hasattr(torch.nn.Module, \"set_submodule\")}')"
python3 -c "from trl import SFTTrainer, SFTConfig; from peft import LoraConfig; import bitsandbytes; print('imports OK')"
which cmake && cmake --version | head -1
which hf && hf --version 2>&1 | head -1

echo "=== BOOTSTRAP DONE ==="
echo "Next: upload your training tarball and run: bash training/pod/fetch_model.sh <hf-repo>"
