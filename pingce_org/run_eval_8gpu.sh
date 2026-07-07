#!/usr/bin/env bash
# Run 8 embedding models on GPUs 4/5/6/7 (2 models per GPU, sequential within GPU).
#
# Each GPU worker writes a partial report:
#   output/reports/partial_gpu4.json
#   output/reports/partial_gpu5.json
#   ...
# Then merge into:
#   output/reports/intent_retrieval_eval_msa.json
#
# Usage:
#   ./run_eval_8gpu.sh
#   ./run_eval_8gpu.sh --no-cache
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

REPORT_DIR="output/reports"
mkdir -p "$REPORT_DIR/logs"

EXTRA_ARGS=("$@")
PARTIAL_REPORTS=()
PIDS=()

run_gpu_worker() {
  local gpu="$1"
  shift
  local models=("$@")
  local report="partial_gpu${gpu}.json"
  local log="$REPORT_DIR/logs/gpu${gpu}.log"

  echo "[gpu${gpu}] models: ${models[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/eval_intent_retrieval.py \
    --device cuda:0 \
    --models "${models[@]}" \
    --report-file "${report}" \
    "${EXTRA_ARGS[@]}" >"${log}" 2>&1 &
  PIDS+=("$!")
  PARTIAL_REPORTS+=("${REPORT_DIR}/${report}")
}

# 2 models per GPU
run_gpu_worker 4 bge_m3 arabic_english_bge_m3
run_gpu_worker 5 multilingual_e5_large_instruct snowflake_arctic_l_v2
run_gpu_worker 6 gate_arabert_v1 arabic_triplet_matryoshka_v2
run_gpu_worker 7 qwen3_embedding_0_6b embeddinggemma_300m

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more GPU workers failed. Check logs under ${REPORT_DIR}/logs/"
  exit 1
fi

python src/merge_reports.py \
  "${PARTIAL_REPORTS[@]}" \
  -o "${REPORT_DIR}/intent_retrieval_eval_msa.json"

echo "All 8 models finished. Final report -> ${REPORT_DIR}/intent_retrieval_eval_msa.json"
