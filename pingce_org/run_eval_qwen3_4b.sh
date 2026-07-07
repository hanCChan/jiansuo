#!/usr/bin/env bash
# Qwen3-Embedding-4B dense retrieval on MSA eval set.
#
# Config:  config_qwen3_4b.yaml
# Output:  output/reports/intent_retrieval_eval_msa_qwen3_4b.json
#
# Usage:
#   ./run_eval_qwen3_4b.sh
#   ./run_eval_qwen3_4b.sh --no-cache
#   CUDA_VISIBLE_DEVICES=5 ./run_eval_qwen3_4b.sh
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
source /data1/hcc/jiansuo/setup_conda.sh

# Qwen3 requires transformers >= 4.51
tf_ver=$(python -c "import transformers; from packaging.version import parse; print(parse(transformers.__version__))")
need=$(python -c "from packaging.version import parse; print(parse('4.51.0'))")
if python -c "import transformers; from packaging.version import parse; import sys; sys.exit(0 if parse(transformers.__version__) >= parse('4.51.0') else 1)"; then
  :
else
  echo "WARN: transformers too old for Qwen3; installing >=4.51,<5 ..."
  pip install -q 'transformers>=4.51.0,<5'
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

python src/eval_intent_retrieval.py \
  --config config_qwen3_4b.yaml \
  --models qwen3_embedding_4b \
  "$@"
