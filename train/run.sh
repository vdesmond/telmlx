#!/bin/bash
# Entry point inside the GPULab container: /project_ghent/telmlx holds train/ and data/ (see submit.sh).
# If HF_TOKEN is set, whatever lands in --out is synced to the private HF bucket at the end, even on failure.
set -euo pipefail
cd /project_ghent/telmlx
# the image's Python is Debian-managed and the job is not root: install on top of it in a user prefix
export HF_HOME=/project_ghent/hf PYTHONUNBUFFERED=1 PYTHONUSERBASE=/tmp/pyuser PATH=/tmp/pyuser/bin:$PATH
BUCKET=hf://buckets/vdesmond/telmlx-runs
python3 -m pip install -q --user --break-system-packages --no-cache-dir -r train/requirements.txt
id; python3 -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# training data too big for the job env: pull it from the bucket (submit.sh only ships code + smoke data)
[ -n "${HF_TOKEN:-}" ] && hf sync "$BUCKET/data" data
out=$(python3 -c 'import sys; a=sys.argv[1:]; print(a[a.index("--out")+1])' "$@")
sync_out() { [ -n "${HF_TOKEN:-}" ] && [ -d "$out" ] && hf sync "$out" "$BUCKET/$(basename "$out")" || true; }
trap sync_out EXIT
python3 train/lora.py "$@"
