#!/usr/bin/env bash
# Deprecated: use run_fill_dialogue_test.sh (local-cluster based).
echo "Deprecated. Use: ./run_fill_dialogue_test.sh {scan|pilot|full}"
exec "$(dirname "$0")/run_fill_dialogue_test.sh" pilot "$@"
