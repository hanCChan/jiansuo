#!/usr/bin/env bash
# Cluster-avg retrieval on MSA eval + qa_msa.json, 10 models on GPUs 4/5/6/7.
#
# Config:  config_cluster_msa.yaml
# Output:  output/reports/intent_retrieval_cluster_eval_msa.json
# Logs:    output/reports/logs/cluster_msa_gpu*.log
#
# Usage:
#   ./run_eval_cluster_msa_4gpu.sh
#   ./run_eval_cluster_msa_4gpu.sh --no-cache
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

# Qwen3 needs transformers>=4.51; GTE needs transformers<5
if ! python -c "import transformers; from packaging.version import parse; import sys; sys.exit(0 if parse(transformers.__version__) >= parse('4.51.0') else 1)"; then
  pip install -q 'transformers>=4.51.0,<5'
fi

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

CONFIG="config_cluster_msa.yaml"
REPORT_DIR="output/reports"
mkdir -p "$REPORT_DIR/logs"

EXTRA_ARGS=("$@")
PARTIAL_REPORTS=()
PIDS=()

run_cluster_worker() {
  local gpu="$1"
  local report="$2"
  shift 2
  local models=("$@")
  local log="$REPORT_DIR/logs/cluster_msa_gpu${gpu}.log"

  echo "[gpu${gpu}] models: ${models[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/eval_intent_retrieval_cluster.py \
    --config "${CONFIG}" \
    --device cuda:0 \
    --models "${models[@]}" \
    --report-file "${report}" \
    "${EXTRA_ARGS[@]}" >"${log}" 2>&1 &
  PIDS+=("$!")
  PARTIAL_REPORTS+=("${REPORT_DIR}/${report}")
}

run_cluster_worker 4 partial_cluster_msa_gpu4.json \
  bge_m3 arabic_english_bge_m3 gte_multilingual_base
run_cluster_worker 5 partial_cluster_msa_gpu5.json \
  multilingual_e5_large_instruct snowflake_arctic_l_v2
run_cluster_worker 6 partial_cluster_msa_gpu6.json \
  gate_arabert_v1 arabic_triplet_matryoshka_v2 embeddinggemma_300m
run_cluster_worker 7 partial_cluster_msa_gpu7.json \
  qwen3_embedding_0_6b qwen3_embedding_4b

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more cluster workers failed. Check logs under ${REPORT_DIR}/logs/"
  exit 1
fi

python src/merge_reports.py \
  "${PARTIAL_REPORTS[@]}" \
  -o "${REPORT_DIR}/intent_retrieval_cluster_eval_msa.json"

echo "Cluster MSA eval finished -> ${REPORT_DIR}/intent_retrieval_cluster_eval_msa.json"
