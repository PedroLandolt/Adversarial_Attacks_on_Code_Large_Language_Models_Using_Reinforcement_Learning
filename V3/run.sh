#!/usr/bin/env bash

# Unified launcher for task presets in adversarial_attack.py
# Usage:
#   bash V3/run.sh                 # default: mbpp react task
#   bash V3/run.sh mbpp            # run adversarial_code_llm preset
#   bash V3/run.sh gitea           # run adversarial_gitea_react_attack preset
#   bash V3/run.sh mbpp --quick
#   bash V3/run.sh gitea --quick

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="mbpp"
EXTRA_ARGS=()

if [[ $# -ge 1 ]]; then
    case "$1" in
        mbpp|gitea)
            TASK="$1"
            shift
            ;;
    esac
fi

if [[ $# -gt 0 ]]; then
    EXTRA_ARGS=("$@")
fi

echo "============================================================"
echo "Adversarial Attack Launcher"
echo "============================================================"
echo "Task preset: $TASK"

action_mbpp() {
    bash "$SCRIPT_DIR/scripts/run_adversarial_code_llm.sh" "${EXTRA_ARGS[@]}"
}

action_gitea() {
    bash "$SCRIPT_DIR/scripts/run_adversarial_gitea_react_attack.sh" "${EXTRA_ARGS[@]}"
}

case "$TASK" in
    mbpp)
        action_mbpp
        ;;
    gitea)
        action_gitea
        ;;
    *)
        echo "Unknown task preset: $TASK"
        exit 1
        ;;
esac
