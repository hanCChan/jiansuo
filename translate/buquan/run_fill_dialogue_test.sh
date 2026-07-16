#!/usr/bin/env bash
# XXX semantic fill on dialogue test MSA (local cluster based).
#
# Usage:
#   ./run_fill_dialogue_test.sh scan
#   ./run_fill_dialogue_test.sh pilot        # dialogue 1019 only
#   ./run_fill_dialogue_test.sh full --resume
set -euo pipefail
cd "$(dirname "$0")"

source /data1/hcc/jiansuo/setup_conda.sh
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1

MODE="${1:-pilot}"
shift || true

case "$MODE" in
  scan)
    python scan_xxx_dialogues.py "$@"
    ;;
  pilot)
    python fill_xxx_dialogue.py \
      --dialogue-id 1019 \
      --enable-thinking \
      --concurrency 2 \
      --resume \
      "$@"
    ;;
  full)
    python fill_xxx_dialogue.py \
      --enable-thinking \
      --concurrency 4 \
      --resume \
      "$@"
    ;;
  *)
    echo "Usage: $0 {scan|pilot|full} [extra args...]"
    exit 1
    ;;
esac
