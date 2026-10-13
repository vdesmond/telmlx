#!/bin/bash
# Closed-book (or RAG=1) GSMA ot-lite eval of one MLX model through mlx_lm.server + Inspect AI.
#
#   scripts/eval.sh models/gemma4-e4b-8bit gemma4-e4b-8bit teleqna
#   scripts/eval.sh models/gemma4-e4b-teleqna-8bit lora-8bit teletables
#   RAG=1 scripts/eval.sh models/gemma4-e4b-8bit gemma4-e4b-8bit teleqna -T subject="Standards specifications"
#
# Settings match the September runs in logs/ (strict system prompt, temperature 0, 512 tokens, 4 connections).
# Logs land in logs/<tag>[-rag]/<bench>/; scripts/rescore.py turns them into results/scores.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=$1 TAG=$2 BENCH=$3; shift 3
SYS="You are taking a multiple-choice exam. Do not explain or reason. Reply with exactly one line of the form 'ANSWER: X' where X is the letter of the correct option."
export KMP_DUPLICATE_LIB_OK=TRUE

PORT=8080
.venv/bin/mlx_lm.server --model "$MODEL" --port 8080 --chat-template-args '{"enable_thinking":false}' > /tmp/telmlx-server.log 2>&1 &
SRV=$!
PIDS=($SRV)
trap 'kill ${PIDS[@]} 2>/dev/null || true' EXIT
until curl -s localhost:8080/v1/models > /dev/null; do sleep 2; done
if [ "${RAG:-0}" = 1 ]; then
  TAG=$TAG-rag PORT=8081
  .venv/bin/python rag/proxy.py --port 8081 > /tmp/telmlx-proxy.log 2>&1 &
  PIDS+=($!)
  until curl -s localhost:8081/v1/models > /dev/null 2>&1 || curl -s localhost:8081/docs > /dev/null 2>&1; do sleep 2; done
fi

mkdir -p logs/$TAG/$BENCH
cp /tmp/telmlx-server.log logs/$TAG/$BENCH/server.log 2>/dev/null || true
# three_gpp's prompt carries its own instructions; the September run used no system message for it
SYSARG=(--system-message "$SYS"); [ "$BENCH" = three_gpp ] && SYSARG=()
MLX_BASE_URL=http://localhost:$PORT/v1 MLX_API_KEY=local .venv/bin/inspect eval evals/$BENCH \
  --model "openai-api/mlx/$MODEL" ${SYSARG[@]+"${SYSARG[@]}"} --temperature 0 --max-tokens 512 \
  --max-connections 4 --log-dir logs/$TAG/$BENCH --display plain "$@"
