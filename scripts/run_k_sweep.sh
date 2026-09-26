#!/usr/bin/env bash
# Table 8 / Appendix C.3.1: retrieved entries K in {1, 5, 10} (K = 3 is run_main.sh).
#   scripts/run_k_sweep.sh <model> [K ...]      default K: 1 5 10
set -uo pipefail
cd "$(dirname "$0")/.."
M="${1:?model}"; shift; KS=("$@"); [ ${#KS[@]} -eq 0 ] && KS=(1 5 10)
STREAMS=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('configs/streams.yaml'))))")
mkdir -p "runs/$M"
for K in "${KS[@]}"; do
  for S in $STREAMS; do
    echo "[$(date '+%F %T')] $M / $S / K=$K"
    scripts/run_stream.sh "$M" "$S" --top-k "$K" > "runs/$M/${S}_k$K.log" 2>&1 \
      && python scripts/eval/score_runs.py "runs/$M/${S}_k$K" --brief || echo "  FAILED"
  done
done
