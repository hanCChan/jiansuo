#!/usr/bin/env bash
# Intent retrieval benchmark with cluster-averaged candidate scores
#
# Each candidate FAQ score = mean(query vs [Question] + 5x Question_cluster from qa.json)
# Input:  config eval.input_json (MSA full eval)
# QA:     config eval.cluster.qa_json -> qa_msa.json
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

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

python src/eval_intent_retrieval_cluster.py \
  --config config_cluster_msa.yaml \
  "$@"
