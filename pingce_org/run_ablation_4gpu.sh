#!/usr/bin/env bash
# MSA FAQ retrieval ablation study on GPUs 4/5/6/7.
#
# Config:  config_ablation.yaml
# Output:  output/reports/intent_retrieval_eval_msa_ablation.json
# Logs:    output/reports/logs/ablation_gpu*.log
#
# Usage:
#   ./run_ablation_4gpu.sh
#   ./run_ablation_4gpu.sh --no-cache
#   ./run_ablation_4gpu.sh --retrieval-detail
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

# Qwen3 needs transformers>=4.51; GTE needs transformers<5
if ! python -c "import transformers; from packaging.version import parse; import sys; sys.exit(0 if parse(transformers.__version__) >= parse('4.51.0') else 1)"; then
  echo "Installing transformers>=4.51,<5 for Qwen3 ..."
  pip install -q 'transformers>=4.51.0,<5'
fi

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

CONFIG="config_ablation.yaml"
REPORT_DIR="output/reports"
mkdir -p "$REPORT_DIR/logs"

EXTRA_ARGS=("$@")
PARTIAL_REPORTS=()
PIDS=()

run_ablation_worker() {
  local gpu="$1"
  local report="$2"
  shift 2
  local models=("$@")
  local log="$REPORT_DIR/logs/ablation_gpu${gpu}.log"

  echo "[gpu${gpu}] models: ${models[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/eval_intent_retrieval.py \
    --config "${CONFIG}" \
    --device cuda:0 \
    --batch-size 64 \
    --models "${models[@]}" \
    --report-file "${report}" \
    "${EXTRA_ARGS[@]}" >"${log}" 2>&1 &
  PIDS+=("$!")
  PARTIAL_REPORTS+=("${REPORT_DIR}/${report}")
}

# GPU4: Qwen3-4B prompt ablation
run_ablation_worker 4 partial_ablation_gpu4.json \
  qwen3_4b_default qwen3_4b_faq_en qwen3_4b_faq_ar qwen3_4b_no_instruct

# GPU5: Qwen3-0.6B + E5 prompt ablation
run_ablation_worker 5 partial_ablation_gpu5.json \
  qwen3_06b_faq_en e5_current e5_enhanced e5_short

# GPU6: BGE-M3 hybrid weight grid
run_ablation_worker 6 partial_ablation_gpu6.json \
  bge_hybrid_d80_s10_c10 bge_hybrid_d70_s10_c20 bge_hybrid_d70_s05_c25 \
  bge_hybrid_d60_s10_c30 bge_hybrid_d90_s05_c05 bge_hybrid_d100_s00_c00 \
  bge_hybrid_baseline

# GPU7: GTE hybrid sparse weight grid
run_ablation_worker 7 partial_ablation_gpu7.json \
  gte_hybrid_s00 gte_hybrid_s05 gte_hybrid_s10 gte_hybrid_s20 gte_hybrid_s30

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "One or more ablation workers failed. Check logs under ${REPORT_DIR}/logs/"
  exit 1
fi

python src/merge_reports.py \
  "${PARTIAL_REPORTS[@]}" \
  -o "${REPORT_DIR}/intent_retrieval_eval_msa_ablation.json"

python src/compare_bge_cache.py \
  --cache-root "${REPORT_DIR}/score_cache_multimode" \
  -o "${REPORT_DIR}/bge_mode_top10_check.json"

python src/analyze_ablation.py \
  --ablation-report "${REPORT_DIR}/intent_retrieval_eval_msa_ablation.json" \
  --multimode-report "${REPORT_DIR}/intent_retrieval_eval_msa_multimode.json" \
  -o "${REPORT_DIR}/ablation_analysis.json"

echo "Ablation finished -> ${REPORT_DIR}/intent_retrieval_eval_msa_ablation.json"
echo "Analysis       -> ${REPORT_DIR}/ablation_analysis.json"
