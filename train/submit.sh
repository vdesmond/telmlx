#!/bin/bash
# Submit a GPULab job that unpacks train/ (+ small data files) into project storage and runs train/run.sh.
# No SSH needed: the files travel base64-encoded in the job's environment.
#
#   train/submit.sh "--smoke --out /project_ghent/telmlx/runs/smoke" data/smoke-chunks.jsonl
#   train/submit.sh "--train data/teleqna/train.jsonl --val data/teleqna/val.jsonl --out /project_ghent/telmlx/runs/r16"
#   (training data is pulled from hf://buckets/vdesmond/telmlx-runs/data by run.sh; upload it there first with hf sync)
#
# Reads GPULAB_CERT, GPULAB_PASSWORD and HF_TOKEN (fine-grained, write to the telmlx-runs bucket only)
# from .env (gitignored). Results: hf sync hf://buckets/vdesmond/telmlx-runs/<run> runs/<run>
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
args=$1; shift

bundle=$(COPYFILE_DISABLE=1 tar --no-xattrs -czf - train/lora.py train/run.sh train/requirements.txt "$@" | base64 | tr -d '\n')
# job spec goes straight to gpulab-cli (it holds the token, so no temp file)
python3 - "$args" "$bundle" <<'PY' | gpulab-cli --cert "$GPULAB_CERT" -p "$GPULAB_PASSWORD" submit --project basic
import json, os, sys
job = json.load(open("train/job.json"))
job["request"]["docker"]["environment"] = {"BUNDLE": sys.argv[2], "HF_TOKEN": os.environ.get("HF_TOKEN", "")}
job["request"]["docker"]["command"] = ["bash", "-c",
    "mkdir -p /project_ghent/telmlx && cd /project_ghent/telmlx && echo \"$BUNDLE\" | base64 -d | tar -xzf - "
    f"&& bash train/run.sh {sys.argv[1]}"]
print(json.dumps(job))
PY
