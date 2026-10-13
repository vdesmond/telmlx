#!/bin/bash
# Base vs LoRA at 8 bit, closed-book on all three benchmarks.
# RAG arms (RAG=1 scripts/eval.sh ... teleqna -T subject="Standards specifications") run at ~1 question/min on an M4.
cd "$(dirname "$0")/.."
MODELS=("models/gemma4-e4b-8bit gemma4-e4b-8bit" "models/gemma4-e4b-teleqna-8bit lora-8bit")
for m in "${MODELS[@]}"; do
  read -r path tag <<< "$m"
  for b in teleqna teletables three_gpp; do
    echo "== $tag $b $(date +%T)"
    scripts/eval.sh "$path" "$tag" "$b" 2>&1 | grep -E '^accuracy|Traceback|Error'
  done
done
.venv/bin/python scripts/rescore.py logs results/scores.csv
echo "DONE $(date +%T)"
