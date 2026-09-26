#!/usr/bin/env bash
# Table 2: remove one component at a time on the nine model-stream pairs of the paper.
#   scripts/run_ablations.sh [no_planner|no_ccme|no_exec ...]   default: all three
set -uo pipefail
cd "$(dirname "$0")/.."
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(no_planner no_ccme no_exec)
PAIRS=(
  "gemini-3.1-flash-lite AIME_2020_2025"  "gemini-3.1-flash-lite MMMU_Pro_Standard_10"
  "gemini-3.1-flash-lite HLE_Exact"
  "gpt-4.1-mini AIME_2020_2025"  "gpt-4.1-mini MMMU_Pro_Vision"  "gpt-4.1-mini MMLU_Pro_Engineering"
  "gpt-4o-mini AIME_2020_2025"   "gpt-4o-mini MMMU_Pro_Standard_4"  "gpt-4o-mini GPQA_Diamond"
)
for A in "${ARMS[@]}"; do
  for P in "${PAIRS[@]}"; do
    set -- $P; M=$1; S=$2; mkdir -p "runs/$M"
    echo "[$(date '+%F %T')] $A / $M / $S"
    scripts/run_stream.sh "$M" "$S" "--${A//_/-}" > "runs/$M/${S}_$A.log" 2>&1 \
      && python scripts/eval/score_runs.py "runs/$M/${S}_$A" --brief || echo "  FAILED"
  done
done
