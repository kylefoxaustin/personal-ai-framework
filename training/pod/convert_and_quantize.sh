#!/bin/bash
# Convert merged HF safetensors → GGUF f16 → Q4_K_M, writing to /dev/shm.
#
# CRITICAL: /workspace on RunPod Secure Cloud is MooseFS network storage with
# a per-file size limit observed around 25 GB. A 60+ GB f16 GGUF silently
# truncates there. /dev/shm is tmpfs (RAM, 117 GB on a standard pod), no such
# limit — and it's fast.
#
# The final Q4_K_M GGUF is ~16 GB so it fits fine anywhere; we scp it home
# directly from /dev/shm to avoid another round trip.
#
# Usage: MERGED=/workspace/models/<base>-merged OUT_NAME=kyle-v1 bash convert_and_quantize.sh
set -euo pipefail
MERGED="${MERGED:?must set MERGED=/path/to/merged-safetensors-dir}"
OUT_NAME="${OUT_NAME:-kyle-v1}"

F16="/dev/shm/${OUT_NAME}-f16.gguf"
Q4="/dev/shm/${OUT_NAME}-q4_k_m.gguf"

cd /workspace/llama.cpp

echo "=== [1/3] convert merged → f16 GGUF (output to tmpfs) ==="
# Use unbuffered python so progress bars stream if we're watching.
python3 -u convert_hf_to_gguf.py "$MERGED" \
  --outfile "$F16" \
  --outtype f16

[ ! -f "$F16" ] && { echo "convert failed: no output file"; exit 1; }
echo "f16 size: $(du -h "$F16" | cut -f1)"

echo "=== [2/3] build llama-quantize if needed ==="
if [ ! -x ./build/bin/llama-quantize ]; then
  # cmake is installed via pip (see bootstrap.sh).
  # Turn off CUDA + curl — quantize doesn't need them, saves ~5 min build time.
  cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
  cmake --build build --target llama-quantize -j
fi

echo "=== [3/3] quantize → Q4_K_M ==="
./build/bin/llama-quantize "$F16" "$Q4" Q4_K_M

[ ! -f "$Q4" ] && { echo "quantize failed: no output file"; exit 2; }
echo "=== DONE ==="
ls -lh "$Q4"
echo
echo "Next: from your LOCAL machine, run:"
echo "  scp -P <pod-port> -i ~/.ssh/id_ed25519 root@<pod-ip>:$Q4 \\"
echo "    ~/Documents/GitHub/personal-ai-framework/models/<dest>/"
echo
echo "Then: delete the f16 to reclaim tmpfs (optional, pod is terminating anyway):"
echo "  rm -f $F16"
