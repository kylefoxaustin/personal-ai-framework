#!/usr/bin/env bash
# Runs the fp16-vs-FP8 v2+RAG apples-to-apples comparison on a RunPod
# cloud pod with enough VRAM to fit both models at full RAG context.
#
# Purpose: the 5090 (32 GB) can't hold fp16 Qwen 2.5 14B + ~7K context,
# so we can't do the methodologically-pure fp16-vs-FP8 RAG comparison
# locally. This script does it on a rented GPU (≥48 GB — A6000 or L40S
# or A100 80GB).
#
# Expected runtime: ~25 min on A6000 48GB (~$0.25).
# Expected artifacts saved to /workspace/results/:
#   acc_reference-fp16-v2-rag-pure_*.json
#   acc_candidate-fp8-v2-rag-pure_*.json
#   acc_diff_fp16_vs_fp8_v2_rag_pure.{md,json}
#
# Prereqs on the pod (see docs/runpod-fp8-eval-runbook.md for setup):
#   - RunPod pod with CUDA 12.x base image (PyTorch 2.4+)
#   - This script uploaded to /workspace/
#   - prompts_v2.json + rag_chunks_for_v2.json uploaded to /workspace/
#   - The three eval scripts uploaded to /workspace/:
#       run_accuracy_eval_vllm.py
#       quantize_w8a8.py
#       compare_accuracy_runs.py
set -euo pipefail

cd /workspace
mkdir -p results

echo "=== Phase 1: install deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet \
  'torch>=2.6,<2.11' transformers accelerate datasets huggingface_hub \
  llmcompressor compressed-tensors vllm
python3 -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "=== Phase 2: download Qwen 2.5 14B Instruct fp16 (~28 GB, ~10 min) ==="
mkdir -p /workspace/models/qwen2.5-14b-hf
python3 << 'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    local_dir="/workspace/models/qwen2.5-14b-hf",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
)
PY

echo
echo "=== Phase 3: quantize fp16 → FP8_DYNAMIC (~1 min) ==="
python3 /workspace/quantize_w8a8.py \
  --source /workspace/models/qwen2.5-14b-hf \
  --output /workspace/models/qwen2.5-14b-fp8 \
  --scheme FP8_DYNAMIC \
  --skip-smoothquant \
  --calib-samples 256

echo
echo "=== Phase 4: fp16 eval on v2+RAG shim path (~8 min) ==="
python3 /workspace/run_accuracy_eval_vllm.py \
  --model-path /workspace/models/qwen2.5-14b-hf \
  --name reference-fp16-v2-rag-pure \
  --prompts /workspace/prompts_v2.json \
  --rag-chunks /workspace/rag_chunks_for_v2.json \
  --max-rag-chunks 8 \
  --samples 3 \
  --max-model-len 16384 \
  --max-tokens 400 \
  --gpu-mem-utilization 0.85

mv /workspace/eval/results/acc_reference-fp16-v2-rag-pure_*.json /workspace/results/ 2>/dev/null || \
  mv /workspace/acc_reference-fp16-v2-rag-pure_*.json /workspace/results/ 2>/dev/null || true

echo
echo "=== Phase 5: FP8 eval on v2+RAG shim path (~6 min) ==="
python3 /workspace/run_accuracy_eval_vllm.py \
  --model-path /workspace/models/qwen2.5-14b-fp8 \
  --name candidate-fp8-v2-rag-pure \
  --prompts /workspace/prompts_v2.json \
  --rag-chunks /workspace/rag_chunks_for_v2.json \
  --max-rag-chunks 8 \
  --samples 3 \
  --max-model-len 16384 \
  --max-tokens 400 \
  --gpu-mem-utilization 0.85

mv /workspace/eval/results/acc_candidate-fp8-v2-rag-pure_*.json /workspace/results/ 2>/dev/null || \
  mv /workspace/acc_candidate-fp8-v2-rag-pure_*.json /workspace/results/ 2>/dev/null || true

echo
echo "=== Phase 6: diff fp16 vs FP8 ==="
REF=$(ls -t /workspace/results/acc_reference-fp16-v2-rag-pure_*.json | head -1)
CAND=$(ls -t /workspace/results/acc_candidate-fp8-v2-rag-pure_*.json | head -1)
python3 /workspace/compare_accuracy_runs.py \
  --reference "$REF" \
  --candidate "$CAND" \
  --out /workspace/results/acc_diff_fp16_vs_fp8_v2_rag_pure.md

echo
echo "=== DONE ==="
echo "Results in /workspace/results/:"
ls -lh /workspace/results/
echo
echo "Headline from the diff:"
head -20 /workspace/results/acc_diff_fp16_vs_fp8_v2_rag_pure.md
echo
echo "scp everything home:"
echo "  scp -r <pod-user>@<pod-ip>:/workspace/results ./eval/results/runpod/"
