#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PATH=$PWD/.venv/bin:$PATH KMP_DUPLICATE_LIB_OK=TRUE
if [ "${1:-}" = stop ]; then pkill -f "mlx_lm server"; pkill -f rag/proxy.py; pkill -f "vite"; exit 0; fi
mkdir -p logs
python -m mlx_lm server --model mlx-community/gemma-4-e4b-it-4bit --port 8080 \
  --chat-template-args '{"enable_thinking": false}' --allowed-origins http://localhost:5173 \
  --log-level WARNING > logs/demo-server.log 2>&1 &
python rag/proxy.py --port 8081 --top_n 10 --top_k 3 --rerank_chars 600 > logs/demo-proxy.log 2>&1 &
(cd ui && bun dev --port 5173 > ../logs/demo-ui.log 2>&1 &)
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8081/health >/dev/null && break; sleep 1; done
echo "ready: http://localhost:5173"
