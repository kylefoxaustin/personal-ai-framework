#!/usr/bin/env bash
# Rents a GPU with working INT8 tensor cores (A100 SM80 or H100 SM90 — NOT
# Blackwell SM120, which has tensor-core INT8 removed) and runs the W8A8
# INT8 accuracy eval on Skippy's v2 prompt set.
#
# This closes the accuracy gap in the silicon story: we've shown FP8 matches
# fp16, but never measured W8A8 INT8 vs fp16 because Blackwell 5090 can't
# load a W8A8 model via vLLM. A100/H100 can.
#
# Expected artifacts in /root/runpod/results/:
#   acc_candidate-int8-v2-rag-pure_*.json
#   acc_reference-fp16-v2-rag-pure-int8pod_*.json  (rerun on same pod for
#       strict apples-to-apples, cheap)
#   acc_diff_fp16_vs_int8_v2_rag_pure.{md,json}
#
# Expected runtime: ~40 min on A100 80GB (~$1.00) or ~35 min on H100 80GB
# (~$1.75).
#
# Prereqs on pod:
#   - PyTorch/CUDA base image (same as fp8 run)
#   - This script + eval scripts + prompts + rag chunks uploaded to /workspace/
#     via scp — the bundle is identical to the fp8 runbook's bundle
set -euo pipefail

# Work on container root (100 GB) not /workspace (20 GB default)
mkdir -p /root/runpod
cd /workspace
mv -t /root/runpod/ \
  runpod_bootstrap_int8_rag.sh \
  run_accuracy_eval_vllm.py \
  quantize_w8a8.py \
  compare_accuracy_runs.py \
  prompts_v2.json \
  rag_chunks_for_v2.json 2>/dev/null || true
cd /root/runpod
mkdir -p results

export HF_HUB_DISABLE_XET=1  # avoid the xet segfault from last pod

echo "=== Phase 1: install deps ==="
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet \
  'torch>=2.6,<2.11' transformers accelerate datasets huggingface_hub \
  llmcompressor compressed-tensors vllm hf_transfer
python3 -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv | head -3

echo
echo "=== Phase 2: download Qwen 2.5 14B Instruct fp16 (~28 GB) ==="
mkdir -p /root/runpod/models/qwen2.5-14b-hf
python3 -c "
import os; os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen2.5-14B-Instruct',
    local_dir='/root/runpod/models/qwen2.5-14b-hf',
    allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*'])
print('download ok')
"

echo
echo "=== Phase 3: quantize fp16 → W8A8 INT8 (SmoothQuant + GPTQ, ~20 min) ==="
python3 quantize_w8a8.py \
  --source models/qwen2.5-14b-hf \
  --output models/qwen2.5-14b-w8a8 \
  --scheme W8A8 \
  --calib-samples 512

echo
echo "=== Phase 4: fp16 reference eval via vLLM shim (same as fp8 pod, ~8 min) ==="
python3 run_accuracy_eval_vllm.py \
  --model-path models/qwen2.5-14b-hf \
  --name reference-fp16-v2-rag-pure-int8pod \
  --prompts prompts_v2.json \
  --rag-chunks rag_chunks_for_v2.json \
  --max-rag-chunks 8 \
  --samples 3 \
  --max-model-len 16384 \
  --max-tokens 400 \
  --gpu-mem-utilization 0.85
find /root/runpod -name "acc_reference-fp16-v2-rag-pure-int8pod_*.json" -exec mv {} /root/runpod/results/ \;

echo
echo "=== Phase 5: INT8 W8A8 eval via vLLM shim (~8 min) ==="
python3 run_accuracy_eval_vllm.py \
  --model-path models/qwen2.5-14b-w8a8 \
  --name candidate-int8-v2-rag-pure \
  --prompts prompts_v2.json \
  --rag-chunks rag_chunks_for_v2.json \
  --max-rag-chunks 8 \
  --samples 3 \
  --max-model-len 16384 \
  --max-tokens 400 \
  --gpu-mem-utilization 0.85
find /root/runpod -name "acc_candidate-int8-v2-rag-pure_*.json" -exec mv {} /root/runpod/results/ \;

echo
echo "=== Phase 6: diff fp16 vs INT8 ==="
REF=$(ls -t /root/runpod/results/acc_reference-fp16-v2-rag-pure-int8pod_*.json | head -1)
CAND=$(ls -t /root/runpod/results/acc_candidate-int8-v2-rag-pure_*.json | head -1)
python3 compare_accuracy_runs.py \
  --reference "$REF" \
  --candidate "$CAND" \
  --out /root/runpod/results/acc_diff_fp16_vs_int8_v2_rag_pure.md

echo
echo "=== DONE ==="
ls -lh /root/runpod/results/
echo
head -30 /root/runpod/results/acc_diff_fp16_vs_int8_v2_rag_pure.md
