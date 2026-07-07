#!/usr/bin/env bash
# GTE multilingual-base: dense / sparse / hybrid on MSA eval set.
#
# Config:  config_gte.yaml
# Output:  output/reports/intent_retrieval_eval_msa_gte.json
#
# Usage:
#   ./run_eval_gte.sh
#   ./run_eval_gte.sh --no-cache
#   CUDA_VISIBLE_DEVICES=5 ./run_eval_gte.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

# GTE custom code requires transformers < 5 (model built for 4.39.x)
tf_ver=$(python -c "import transformers; print(transformers.__version__)")
tf_major=$(echo "$tf_ver" | cut -d. -f1)
if [ "$tf_major" -ge 5 ]; then
  echo "WARN: transformers $tf_ver incompatible with GTE; installing 4.46.3 ..."
  pip install -q 'transformers==4.46.3'
fi

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

python src/eval_intent_retrieval.py \
  --config config_gte.yaml \
  --models gte_multilingual_base \
  "$@"
