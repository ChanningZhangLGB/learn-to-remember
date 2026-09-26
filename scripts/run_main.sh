#!/usr/bin/env bash
# Table 1: LeRe on all twelve streams for one backbone (or all three), paper defaults.
#   scripts/run_main.sh [model ...]        default: all three backbones
# Streams run sequentially per backbone (the memory is per stream; sequential runs also
# avoid provider rate limits). Logs go to runs/<model>/<stream>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(gemini-3.1-flash-lite gpt-4.1-mini gpt-4o-mini)
STREAMS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/streams.yaml'))))")
for M in "${MODELS[@]}"; do
  mkdir -p "runs/$M"
  for S in $STREAMS; do
    echo "[$(date '+%F %T')] $M / $S"
    scripts/run_stream.sh "$M" "$S" > "runs/$M/$S.log" 2>&1 \
      && python scripts/eval/score_runs.py "runs/$M/$S" --brief || echo "  FAILED, see runs/$M/$S.log"
  done
done
