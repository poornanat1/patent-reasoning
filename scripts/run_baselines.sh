#!/usr/bin/env bash
# Generate the frozen-model baselines (zero-shot, few-shot) on every evaluation split.
# Each run is resumable, so a stopped script picks up where it left off.
#
# Usage: scripts/run_baselines.sh  (log: data/interim/run_baselines.log when run detached)
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
for split in val test ood_test; do
  for variant in base_zero_shot base_few_shot; do
    echo "== $variant $split $(date '+%F %T')"
    uv run --extra infer python scripts/generate.py "configs/generation/$variant.yaml" --split "$split"
  done
done
echo "== done $(date '+%F %T')"
