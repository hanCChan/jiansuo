#!/usr/bin/env bash
# Fill XXX placeholders in test dialogue MSA using Kimi 2.6 + thinking.
#
# Usage:
#   ./run_fill_test.sh
#   ./run_fill_test.sh --max-items 5
#   ./run_fill_test.sh --resume
set -euo pipefail
cd "$(dirname "$0")"

source /data1/hcc/jiansuo/setup_conda.sh
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

python fill_xxx_msa.py \
  --enable-thinking \
  --concurrency 4 \
  --resume \
  "$@"
