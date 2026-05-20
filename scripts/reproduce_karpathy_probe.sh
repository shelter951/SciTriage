#!/usr/bin/env bash
set -euo pipefail

SCITRIAGE_ROOT="${SCITRIAGE_ROOT:-$HOME/projects/scitriage}"
AUTORESEARCH_REPO="${AUTORESEARCH_REPO:-$HOME/projects/third_party/autoresearch}"
PROBE_REPO="${PROBE_REPO:-$HOME/projects/workspaces/autoresearch_probe_4090_v1}"

cd "$SCITRIAGE_ROOT"

python3 -m scitriage.cli materialize-autoresearch-probe \
  --source-repo "$AUTORESEARCH_REPO" \
  --target-dir "$PROBE_REPO"

cd "$PROBE_REPO"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
uv sync --locked

for seed in 1 2 3; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" SCITRIAGE_SEED="$seed" \
    timeout 700 uv run train.py > "runs/seed_${seed}.log" 2>&1
done

cd "$SCITRIAGE_ROOT"
python3 -m scitriage.cli aggregate-seeds \
  "$PROBE_REPO"/runs/seed_*.log \
  --metric val_bpb \
  --out runs/autoresearch_probe_4090_v1_seed_summary.json

