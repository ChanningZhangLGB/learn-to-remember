#!/bin/bash
# Run LeRe experiment
# Usage: bash test_framework/run_experiment.sh [config_path] [key_id]
#   key_id (optional): 0=KEY_0, 1=KEY_1, 2=KEY_2
#   If omitted, key is selected automatically by run_id (1→0, 2→1, 3→2)

set -e

# Resolve repo root (the directory containing this script's parent)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

set -a
source ./config.env
set +a

CONFIG="${1:-configs/GPQA_Diamond/gemini-2.5-flash-lite/ccme_topk/gpqa_gemini_ccme_topk_run1.json}"

# Detect model name from config
MODEL_NAME=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('llm',{}).get('model_name',''))" 2>/dev/null || echo "")

# Gemini / NVIDIA models: OPENAI_API_KEY is only needed for embeddings → use EMBED key
# OpenAI models: select LLM key by run_id (explicit override takes priority)
if echo "$MODEL_NAME" | grep -qiE "gemini|nvidia"; then
    export OPENAI_API_KEY="${OPENAI_API_KEY_EMBED}"
else
    if [ -n "${2:-}" ]; then
        API_KEY_ID="$2"
    else
        RUN_ID=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('experiment',{}).get('run_id',1))" 2>/dev/null || echo 1)
        if [ "$RUN_ID" = "2" ]; then
            API_KEY_ID=1
        elif [ "$RUN_ID" = "3" ]; then
            API_KEY_ID=2
        else
            API_KEY_ID=0
        fi
    fi
    case "$API_KEY_ID" in
        1) export OPENAI_API_KEY="${OPENAI_API_KEY_1}" ;;
        2) export OPENAI_API_KEY="${OPENAI_API_KEY_2}" ;;
        *) export OPENAI_API_KEY="${OPENAI_API_KEY_0}" ;;
    esac
fi

PYTHONPATH="$REPO_ROOT" python3 main/run_lere_experiment.py --config "$CONFIG"
