#!/usr/bin/env bash
# SK-P0-002 variance bounds wrapper. For each anchored model, run the v2-rag
# eval N times at non-zero temperature to capture sampling variance. Output
# headline pass rates → eval/results/acc_variance_<model>_run<N>.json.
#
# Run this AFTER swapping the LLM server's loaded model to the anchored
# candidate (config.yaml model.path → corresponding GGUF, ./run.sh start).
#
# Usage:
#   ./eval/run_variance_bounds.sh <model_label> [num_runs=5]
#
# Example:
#   ./eval/run_variance_bounds.sh skippy-7b-v4 5
#   ./eval/run_variance_bounds.sh mistral-stock 5
#
# After all 5 anchored models are run, build the variance_bounds sheet via
# scripts/build_data_bundle.py (the script will detect acc_variance_* JSONs
# and aggregate them).
set -euo pipefail

MODEL_LABEL="${1:?usage: $0 <model_label> [num_runs=5]}"
NUM_RUNS="${2:-5}"

# Sampling variance comes from temperature > 0 (default eval is temp=0
# deterministic). 0.3 is enough to surface sampling spread without making
# the model unhinged.
TEMPERATURE="${TEMPERATURE:-0.3}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-3}"

# Auth (kyle/123456 admin)
SKIPPY_USER="${SKIPPY_USER:-kyle}"
SKIPPY_PASSWORD="${SKIPPY_PASSWORD:-123456}"
export SKIPPY_USER SKIPPY_PASSWORD

cd "$(dirname "$0")/.."

# Verify server is up + healthy
if ! curl -sf http://localhost:8080/health | grep -q '"healthy"'; then
    echo "❌ LLM server not healthy at localhost:8080. Start with ./run.sh start first."
    exit 1
fi

echo "============================================================"
echo "Variance bounds for: $MODEL_LABEL"
echo "  num_runs: $NUM_RUNS"
echo "  temperature: $TEMPERATURE"
echo "  samples_per_prompt: $SAMPLES_PER_PROMPT"
echo "============================================================"

for i in $(seq 1 "$NUM_RUNS"); do
    NAME="acc_variance_${MODEL_LABEL}_run${i}"
    echo
    echo "--- Run $i / $NUM_RUNS → eval/results/${NAME}_*.json ---"
    python3 eval/run_accuracy_eval.py \
        --name "${NAME}" \
        --samples "$SAMPLES_PER_PROMPT" \
        --with-rag \
        --prompts eval/prompts_v2.json \
        --temperature "$TEMPERATURE" \
        2>&1 | tee "training/logs/variance_${MODEL_LABEL}_run${i}.log" | tail -3
done

echo
echo "✅ All $NUM_RUNS runs complete for $MODEL_LABEL"
echo "Headline pass rates:"
python3 -c "
import glob, json
paths = sorted(glob.glob(f'eval/results/acc_variance_${MODEL_LABEL}_run*_*.json'))
rates = []
for p in paths:
    d = json.loads(open(p).read())
    s = d.get('summary', {})
    pct = s.get('pass_rate', 0) * 100
    rates.append(pct)
    print(f'  {p.split(\"/\")[-1]:80s} {s.get(\"passed\")}/{s.get(\"total\")} = {pct:.1f}%')
import statistics
if len(rates) >= 2:
    print(f'  mean: {statistics.mean(rates):.2f}%, stddev: {statistics.stdev(rates):.2f}pp, range: [{min(rates):.1f}, {max(rates):.1f}]')
"
