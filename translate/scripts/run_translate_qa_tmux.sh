#!/usr/bin/env bash
set -euo pipefail
cd /data1/hcc/jiansuo/translate
mkdir -p output/logs
exec /data1/hcc/jiansuo/miniconda3/envs/jiansuo-embed/bin/python scripts/translate_qa_json.py \
  --resume \
  --wave-size 1200 \
  --batch-size 60 \
  --concurrency 16 \
  2>&1 | tee -a output/logs/translate_qa_json.log
