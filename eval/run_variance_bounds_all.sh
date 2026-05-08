#!/usr/bin/env bash
# SK-P0-002 orchestrator. For each of 5 anchored models:
#   1. Edit pipeline/config.yaml model.path
#   2. docker compose restart llm-server
#   3. Wait for /health
#   4. Run eval/run_variance_bounds.sh <label> 5 (5 reps at temp 0.3)
# At end: restore config.yaml to Qwen 7B v4 production, restart, verify.
#
# All runs land in eval/results/acc_variance_<label>_run<N>_*.json. The
# build_data_bundle.py script can later aggregate them into a
# variance_bounds sheet.
#
# Total wall-clock: ~5 model swaps * ~25 min/model = ~2 hours.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="pipeline/config.yaml"
PRODUCTION_PATH="/app/models/kyle-7b-v4-q4_k_m.gguf"

# Model definitions: label, in-container model path
MODELS=(
    "qwen-7b-base|/app/models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf"
    "mistral-7b-base|/app/models/mistral-7b-instruct-v0.3/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
    "skippy-7b-v4|/app/models/kyle-7b-v4-q4_k_m.gguf"
    "skippy-mistral-v4|/app/models/mistral-7b-kyle/kyle-mistral-7b-v4-q4_k_m.gguf"
    "qwen-32b-base|/app/models/qwen2.5-32b-instruct/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
)

NUM_RUNS="${NUM_RUNS:-5}"

swap_model() {
    local target_path="$1"
    echo
    echo "============================================================"
    echo "Swapping config.yaml → $target_path"
    echo "============================================================"
    python3 eval/_swap_model_path.py "$target_path" || {
        echo "  ❌ config swap failed — aborting"
        exit 1
    }
    docker compose restart llm-server > /dev/null 2>&1
    echo "  docker restart issued, waiting for /health..."
    until curl -sf http://localhost:8080/health 2>/dev/null | grep -q '"healthy"'; do sleep 3; done
    echo "  ✅ /health OK"
}

START=$(date +%s)

for entry in "${MODELS[@]}"; do
    label="${entry%%|*}"
    target="${entry##*|}"

    swap_model "$target"
    ./eval/run_variance_bounds.sh "$label" "$NUM_RUNS" 2>&1 | tee -a "training/logs/variance_${label}_orchestrator.log"
done

# Restore production
echo
echo "============================================================"
echo "Restoring production: $PRODUCTION_PATH"
echo "============================================================"
swap_model "$PRODUCTION_PATH"

END=$(date +%s)
echo
echo "TOTAL ORCHESTRATOR WALL TIME: $(( (END - START) / 60 )) min"
echo "ALL VARIANCE BOUNDS COMPLETE. Production restored to Qwen 7B v4."
