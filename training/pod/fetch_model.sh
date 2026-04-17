#!/bin/bash
# Download a base model from HuggingFace to /workspace/models/<name>.
#
# Usage: bash fetch_model.sh Qwen/Qwen3-30B-A3B-Instruct-2507
#
# Note: huggingface_hub 1.11+ renamed `huggingface-cli` → `hf`. Old runbooks
# that say `huggingface-cli download ...` will fail with usage hints.
set -euo pipefail
REPO="${1:?Usage: fetch_model.sh <hf-repo>, e.g. Qwen/Qwen3-30B-A3B-Instruct-2507}"
NAME="$(basename "$REPO" | tr '[:upper:]' '[:lower:]')"
OUT="/workspace/models/${NAME}"
mkdir -p "$OUT"
echo "=== downloading $REPO → $OUT ==="
hf download "$REPO" --local-dir "$OUT"
echo "=== DONE. Total: $(du -sh "$OUT" | cut -f1) ==="
ls -lh "$OUT" | head -20
