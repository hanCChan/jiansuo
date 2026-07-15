#!/usr/bin/env bash
# Translate dialogue test JSON: content (IDN) -> content_msa (MSA)
#
# Input:  dialogue_20260615_BCA_clean_test(1).json
# Output: dialogue_20260615_BCA_clean_test_msa.json
#
# Usage:
#   ./run_translate_dialogue_test.sh
#   ./run_translate_dialogue_test.sh --max-items 20
#   ./run_translate_dialogue_test.sh --resume
#   ./run_translate_dialogue_test.sh --retry-failed --enable-thinking --concurrency 32
set -euo pipefail
cd "$(dirname "$0")"

source /data1/hcc/jiansuo/setup_conda.sh
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

python ../translate/scripts/translate_dialogue_json.py "$@"
