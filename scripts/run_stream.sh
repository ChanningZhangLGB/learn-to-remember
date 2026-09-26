#!/usr/bin/env bash
# Run LeRe on one stream: thin wrapper around run_lere.py that sets the environment.
#   scripts/run_stream.sh <model> <stream> [run_lere.py options ...]
#   scripts/run_stream.sh gpt-4.1-mini AIME_2025
#   scripts/run_stream.sh gemini-3.1-flash-lite MathVista --top-k 5
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?model}"; STREAM="${2:?stream}"; shift 2
# sentence-transformers pulls pandas, which can need a newer libstdc++ than the system one.
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false   # the code executor forks per tool call
export TF_CPP_MIN_LOG_LEVEL=3
exec python -u scripts/run_lere.py --model "$MODEL" --stream "$STREAM" "$@"
