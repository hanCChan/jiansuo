#!/usr/bin/env bash
# Standard intent retrieval benchmark (single process).
#
# Input:  ../results/full_run/cluster_retrieval_intent_eval_msa_full.json
# Output: output/reports/intent_retrieval_eval_msa.json
#
# Examples:
#   ./run_eval.sh
#   ./run_eval.sh --models bge_m3 --device cuda:4
#   ./run_eval.sh --max-queries 2
#   ./run_eval.sh --no-cache
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

source /data1/hcc/jiansuo/setup_conda.sh

python src/eval_intent_retrieval.py "$@"
