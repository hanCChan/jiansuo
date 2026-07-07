#!/usr/bin/env bash
# BGE-M3 multi-mode benchmark on GPUs 4/5/6/7 (one mode group per GPU).
#
# Config:  config_multimode.yaml
# Output:  output/reports/intent_retrieval_eval_msa_multimode.json
# Logs:    output/reports/logs/multimode_gpu*.log
#
# Modes:
#   GPU4 -> dense, sparse
#   GPU5 -> colbert
#   GPU6 -> hybrid
#   GPU7 -> dense+sparse
#
# Usage:
#   ./run_eval_multimode_4gpu.sh
#   ./run_eval_multimode_4gpu.sh --no-cache
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

CONFIG="config_multimode.yaml"
REPORT_DIR="output/reports"
mkdir -p "$REPORT_DIR/logs"

EXTRA_ARGS=("$@")
PARTIAL_REPORTS=()
PIDS=()

run_mode_worker() {
  local gpu="$1"
  local report="$2"
  shift 2
  local modes=("$@")
  local log="$REPORT_DIR/logs/multimode_gpu${gpu}.log"

  echo "[gpu${gpu}] bge_m3 modes: ${modes[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/eval_intent_retrieval.py \
    --config "${CONFIG}" \
    --device cuda:0 \
    --models bge_m3 \
    --modes "${modes[@]}" \
    --report-file "${report}" \
    "${EXTRA_ARGS[@]}" >"${log}" 2>&1 &
  PIDS+=("$!")
  PARTIAL_REPORTS+=("${REPORT_DIR}/${report}")
}

run_mode_worker 4 partial_multimode_gpu4.json dense sparse
run_mode_worker 5 partial_multimode_gpu5.json colbert
run_mode_worker 6 partial_multimode_gpu6.json hybrid
run_mode_worker 7 partial_multimode_gpu7.json "dense+sparse"

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more multimode workers failed. Check logs under ${REPORT_DIR}/logs/"
  exit 1
fi

python src/merge_reports.py \
  "${PARTIAL_REPORTS[@]}" \
  -o "${REPORT_DIR}/intent_retrieval_eval_msa_multimode.json"

echo "BGE-M3 multimode eval finished -> ${REPORT_DIR}/intent_retrieval_eval_msa_multimode.json"
