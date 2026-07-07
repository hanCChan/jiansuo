#!/usr/bin/env bash
# Intent retrieval benchmark with cluster-averaged candidate scores
#
# Each candidate FAQ score = mean(query vs [Question] + 5x Question_cluster from qa.json)
# Input:  config eval.input_json -> output/cluster_retrieval_intent_eval.json
# QA:     config eval.cluster.qa_json -> output/qa.json
# Output: output/reports/intent_retrieval_cluster_eval.json
#
# Extra args are forwarded, e.g.:
#   ./run_eval_cluster.sh --device cuda:4
#   ./run_eval_cluster.sh --device 4,5,6,7
#   ./run_eval_cluster.sh --device all
#   ./run_eval_cluster.sh --no-cache
#   ./run_eval_cluster.sh --models bge_m3 --modes dense
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)/../faq_eval/src"

python src/eval_intent_retrieval_cluster.py \
  --models bge_m3 gte qwen3_4b \
  "$@"
