#!/usr/bin/env bash
# Cluster-avg multi-mode: BGE-M3 (5 modes) + GTE (3 modes) on MSA + qa_msa.json
#
# Config:  config_cluster_multimode_msa.yaml
# Output:  output/reports/intent_retrieval_cluster_eval_msa_multimode.json
# Logs:    output/reports/logs/cluster_multimode_msa_gpu*.log
#
# GPU plan:
#   GPU4 -> bge_m3: dense, sparse
#   GPU5 -> bge_m3: colbert
#   GPU6 -> bge_m3: hybrid
#   GPU7 -> bge_m3: dense+sparse
#   GPU2 -> gte_multilingual_base: dense, sparse, hybrid
#
# Usage:
#   ./run_eval_cluster_multimode_msa.sh
#   ./run_eval_cluster_multimode_msa.sh --no-cache
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

if ! python -c "import transformers; from packaging.version import parse; import sys; sys.exit(0 if parse(transformers.__version__) >= parse('4.51.0') else 1)"; then
  pip install -q 'transformers>=4.51.0,<5'
fi

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

CONFIG="config_cluster_multimode_msa.yaml"
REPORT_DIR="output/reports"
mkdir -p "$REPORT_DIR/logs"

EXTRA_ARGS=("$@")
PARTIAL_REPORTS=()
PIDS=()

run_cluster_mode_worker() {
  local gpu="$1"
  local report="$2"
  local model="$3"
  shift 3
  local modes=("$@")
  local log="$REPORT_DIR/logs/cluster_multimode_msa_gpu${gpu}.log"

  echo "[gpu${gpu}] ${model} modes: ${modes[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/eval_intent_retrieval_cluster.py \
    --config "${CONFIG}" \
    --device cuda:0 \
    --models "${model}" \
    --modes "${modes[@]}" \
    --report-file "${report}" \
    "${EXTRA_ARGS[@]}" >"${log}" 2>&1 &
  PIDS+=("$!")
  PARTIAL_REPORTS+=("${REPORT_DIR}/${report}")
}

run_cluster_mode_worker 4 partial_cluster_multimode_msa_gpu4.json bge_m3 dense sparse
run_cluster_mode_worker 5 partial_cluster_multimode_msa_gpu5.json bge_m3 colbert
run_cluster_mode_worker 6 partial_cluster_multimode_msa_gpu6.json bge_m3 hybrid
run_cluster_mode_worker 7 partial_cluster_multimode_msa_gpu7.json bge_m3 "dense+sparse"
run_cluster_mode_worker 2 partial_cluster_multimode_msa_gpu2.json gte_multilingual_base dense sparse hybrid

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more cluster multimode workers failed. Check logs under ${REPORT_DIR}/logs/"
  exit 1
fi

python src/merge_reports.py \
  "${PARTIAL_REPORTS[@]}" \
  -o "${REPORT_DIR}/intent_retrieval_cluster_eval_msa_multimode.json"

echo "Cluster multimode MSA eval finished -> ${REPORT_DIR}/intent_retrieval_cluster_eval_msa_multimode.json"
